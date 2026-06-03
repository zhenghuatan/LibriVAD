#!/bin/bash

export LC_ALL=C

trndatabase=LibriSpeech # training
database=LibriSpeech    # test database

Label=../Files/Labels/${database}
wav=../Results/${database}
clnwav=../Files/Datasets/${database}

Feat=feat_${database}

rm -f *.txt *.lst 
rm -f wav.lst # Ensure we start fresh

# Generate lists for all phases: training, development, and testing
for phase in train-clean-100 dev-clean test-clean; do
    echo "Preparing lists for $phase..."
    find -L ${Label}/${phase} -name "*.npy" > lab_temp
    find -L ${clnwav}/${phase} -name "*.wav" > phase_wav.lst
    cat phase_wav.lst >> wav.lst # Accumulate all wavs for MFCC extraction
      
    rm -rf ${phase}; mkdir -p ${phase}
    for tx in SSN_noise Domestic_noise Nature_noise Office_noise Public_noise \
              Street_noise Transport_noise Babble_noise City_noise; do
        for db in -5 0 5 10 15 20; do
            # Build the list of noisy files and their corresponding labels
            for xin in `find ${wav}/${phase}/${tx}/${db} -name "*.wav"`; do
                id=`echo $xin | awk -F '/' '{print $NF}' | sed 's|\.wav||g'`
                ft=`echo $xin | sed 's|\.wav||g' | sed 's|^|'${Feat}/'|g'`
                # Match noisy file with its label and feature path
                grep "/${id}.npy" lab_temp | awk -v y=$xin -v ft=$ft '{print y","$1","ft}' >> ${phase}.lst
                if [ "$phase" != "train-clean-100" ]; then
                    grep "/${id}.npy" lab_temp | awk -v y=$xin -v ft=$ft '{print y","$1","ft}' >> ${phase}/${tx}_${db}.lst
                fi
            done
        done
    done

    # Handle clean data for the current phase
    for x in `awk -F ',' '{print $1}' ${phase}.lst | awk -F '/' '{print $NF}' | sort | uniq | sed 's|\.wav||g'`; do
        wfile=`grep "/${x}.wav" phase_wav.lst`
        ft=`echo $wfile | sed 's|\.wav||g' | sed 's|^|'${Feat}/'|g'`
        grep "/${x}.npy" lab_temp | awk -v y=$wfile -v ft=$ft '{print y","$1","ft}' >> ${phase}.clean.lst
    done    
    sort ${phase}.clean.lst | uniq > xyz ; mv xyz ${phase}.clean.lst

    # Finalize phase lists
    if [ "$phase" == "train-clean-100" ]; then
        # Implementation of "Unseen Noise" protocol for training
        for unseen in Babble_noise SSN_noise Domestic_noise; do
            grep -v "${unseen}" train-clean-100.lst > xyz
            mv xyz train-clean-100.lst
        done     
        cp train-clean-100.lst DNN.trn.scp
        cat train-clean-100.clean.lst >> DNN.trn.scp
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
for size in small medium large; do
    Nepoch=50
    if [ $size == "small" ]; then
        ExpDir=small-models/${trndatabase}_vit_MFCC
    elif [ $size == "medium" ]; then
        ExpDir=medium-models/${trndatabase}_vit_MFCC
    elif [ $size == "large" ]; then 
        ExpDir=large-models/${trndatabase}_vit_MFCC    
        Nepoch=49
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
