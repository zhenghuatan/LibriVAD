#!/bin/bash

export LC_ALL=C

trndatabase=LibriSpeech # training
database=LibriSpeech    # test database

Label=../Files/Labels/${database}
wav=../Results/${database}
clnwav=../Files/Datasets/${database}

Feat=feat_${database}

rm -f *.txt *.lst 

# Generate lists for all phases: training, development, and testing
for phase in train-clean-100 dev-clean test-clean; do
    echo "Preparing lists for $phase..."
    find -L ${Label}/${phase} -name "*.npy" > lab_temp
    find -L ${clnwav}/${phase} -name "*.wav" > phase_wav.lst
    cat phase_wav.lst >> wav.lst # Accumulate all wavs for MFCC extraction
      
    rm -rf ${phase}; mkdir -p ${phase}
    
    # --- 1. NOISY FILES LOOP ---
    for tx in SSN_noise Domestic_noise Nature_noise Office_noise Public_noise \
              Street_noise Transport_noise Babble_noise City_noise; do
        for db in -5 0 5 10 15 20; do
            
            # Pipe all wav files into awk and process them simultaneously
            find ${wav}/${phase}/${tx}/${db} -name "*.wav" 2>/dev/null | awk -v feat="${Feat}" '
            NR==FNR {
                # Read lab_temp into memory
                n=split($0, a, "/"); id=a[n]; sub(/\.npy$/, "", id);
                lab[id] = $0;
                next;
            }
            {
                # Match noisy wav streams against the labels in memory
                wavpath=$0; n=split(wavpath, a, "/"); id=a[n]; sub(/\.wav$/, "", id);
                if (id in lab) {
                    ft = feat "/" wavpath; sub(/\.wav$/, "", ft);
                    print wavpath "," lab[id] "," ft;
                }
            }' lab_temp - > tmp_list.lst
            
            cat tmp_list.lst >> ${phase}.lst
            if [ "$phase" != "train-clean-100" ]; then
                cat tmp_list.lst >> ${phase}/${tx}_${db}.lst
            fi
            rm -f tmp_list.lst
        done
    done

    # --- 2. CLEAN FILES LOOP ---
    awk -v feat="${Feat}" -v phaselist="${phase}.lst" '
    # 1st file: Load all labels into memory
    FILENAME=="lab_temp" {
        n=split($0, a, "/"); id=a[n]; sub(/\.npy$/, "", id);
        lab[id] = $0;
        next;
    }
    # 2nd file: Load the allowed IDs from the small-case noisy files
    FILENAME==phaselist {
        split($0, parts, ",");
        n=split(parts[1], a, "/"); id=a[n]; sub(/\.wav$/, "", id);
        wanted[id] = 1;
        next;
    }
    # 3rd file: Process the clean wavs, strictly applying the filter
    FILENAME=="phase_wav.lst" {
        wavpath=$0; n=split(wavpath, a, "/"); id=a[n]; sub(/\.wav$/, "", id);
        # Only add the clean file if it was used in the noisy dataset AND has a label
        if ((id in wanted) && (id in lab)) {
            ft = feat "/" wavpath; sub(/\.wav$/, "", ft);
            print wavpath "," lab[id] "," ft;
        }
    }' lab_temp ${phase}.lst phase_wav.lst | sort | uniq > ${phase}.clean.lst
    
    # --- Finalize phase lists ---
    if [ "$phase" == "train-clean-100" ]; then
        # Implementation of "Unseen Noise" protocol for training
        for unseen in Babble_noise SSN_noise Domestic_noise; do
            grep -v "${unseen}" ${phase}.lst > xyz
            mv xyz ${phase}.lst
        done     
        cp ${phase}.lst DNN.trn.scp
        cat ${phase}.clean.lst >> DNN.trn.scp
        sort DNN.trn.scp | uniq > xyz; mv xyz DNN.trn.scp
    elif [ "$phase" == "dev-clean" ]; then
        mv ${phase}.clean.lst ${phase}/clean_clean.lst
    elif [ "$phase" == "test-clean" ]; then
        mv ${phase}.clean.lst ${phase}/clean_clean.lst
    fi

    rm -f lab_temp phase_wav.lst
done

# Step 1: MFCC feature extraction & label generation for everything
echo "Extracting MFCC features..."
# Build wav.lst from all comma-separated lists
cat DNN.trn.scp > wav.lst
find dev-clean -name "*.lst" -exec cat {} + >> wav.lst
find test-clean -name "*.lst" -exec cat {} + >> wav.lst
sort wav.lst | uniq > xyz ; mv xyz wav.lst

awk -F ',' '{print $NF}' wav.lst 2>/dev/null | sed 's|\(.*\)/.*|\1|' | sort | uniq | xargs mkdir -p
python mfcc.py wav.lst

# Step 2: Loop through model sizes (Small, Medium, Large)
for size in small; do
    Nepoch=50
    if [ $size == "small" ]; then
        ExpDir=small-models/${trndatabase}_vit_MFCC
    elif [ $size == "medium" ]; then
        ExpDir=medium-models/${trndatabase}_vit_MFCC
    elif [ $size == "large" ]; then 
        ExpDir=large-models/${trndatabase}_vit_MFCC    
    fi

    echo "--- Training $size model ---"
    echo $Nepoch > epoch.txt
    echo $ExpDir > dir.txt
    
    # RUN TRAINING
    echo "Train" > phase.txt
    python KWT3_vad.py

    # Step 3: RUN EVALUATION
    echo "--- Evaluating $size model ---"
    echo "Test" > phase.txt
    echo "tr_${trndatabase}_to_${database}" > EvalFile.txt
    python KWT3_vad.py

    # Individual noise evaluation
    echo "Test_ind" > phase.txt
    python KWT3_vad.py

    # Calculate EER
    score=${ExpDir}/tr_${trndatabase}_to_${database}.testdata.score 
    label=${ExpDir}/tr_${trndatabase}_to_${database}.testdata.label 
    echo "Calculating EER for $size model..."
    python eer.py $score $label
done