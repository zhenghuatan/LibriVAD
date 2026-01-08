#!/bin/bash

export LC_ALL=C

trndatabase=LibriSpeech #trning
database=LibriSpeech #test database

Label=LibriVAD/Files/Labels/${database}
wav=LibriVAD/Results/${database}
clnwav=LibriVAD/Files/Datasets/${database}

Feat=feat_${database}  #feat_LibriSpeech


rm -f *.txt *.lst 
#wav -- labels
for phase in test-clean;do
	find -L ${Label}/${phase} -name "*.npy" > lab_temp
	find -L ${clnwav}/${phase} -name "*.wav" > wav.lst
      
        rm -rf ${phase}; mkdir -p ${phase}
        for tx in SSN_noise Domestic_noise Nature_noise Office_noise Public_noise\
	       	Street_noise Transport_noise Babble_noise City_noise;do
	      for db in -5 0 5 10 15 20;do
                 #wav
		 if [ "$phase" == "train-clean-100" ] ; then

                       for xin in `find ${wav}/${phase}/${tx}/${db} -name "*.wav"`;do
		          id=`echo $xin |awk -F '/' '{print $NF}' |sed 's|\.wav||g'`
			  ft=`echo $xin |sed 's|\.wav||g' | sed 's|^|'${Feat}/'|g'`
                          grep "/${id}.npy" lab_temp |awk -v y=$xin -v ft=$ft '{print y","$1","ft}' >> ${phase}.lst
                        done
		else
			for xin in `find ${wav}/${phase}/${tx}/${db} -name "*.wav"`;do
                          id=`echo $xin |awk -F '/' '{print $NF}' |sed 's|\.wav||g'`
			  ft=`echo $xin |sed 's|\.wav||g' | sed 's|^|'${Feat}/'|g'`
                          grep "/${id}.npy" lab_temp |awk -v y=$xin -v ft=$ft '{print y","$1","ft}' >>  ${phase}/${tx}_${db}.lst
                          grep "/${id}.npy" lab_temp |awk -v y=$xin -v ft=$ft '{print y","$1","ft}' >> ${phase}.lst			                                                        
                        done
			 

		 fi

              done
         done
         #clean data
         for x in `awk -F ',' '{print $1}' ${phase}.lst|awk -F '/' '{print $NF}' |sort|uniq|sed 's|\.wav||g'`;do
             wfile=`grep "/${x}.wav" wav.lst`
	     ft=`echo $wfile |sed 's|\.wav||g' | sed 's|^|'${Feat}/'|g'`
             grep "/${x}.npy" lab_temp |awk -v y=$wfile -v ft=$ft '{print y","$1","ft}' >> ${phase}.clean.lst
         done    
	 sort ${phase}.clean.lst |uniq > xyz ; mv xyz ${phase}.clean.lst

	 if [ "$phase" != "train-clean-100" ] ; then
		 mv ${phase}.clean.lst ${phase}/clean_clean.lst
		 rm -f ${phase}.lst 
	 fi

	 rm -f lab_temp  wav.lst
 done
#

if [ $phase == "train-clean-100" ] ; then
	#unsceen "Babble, SSN_noise Domestic_noise"
        for unseen in Babble_noise SSN_noise Domestic_noise;do
            grep -v "${unseen}" train-clean-100.lst > xyz
            mv xyz train-clean-100.lst
        done     
    
       mv train-clean-100.lst  DNN.trn.scp
       cat train-clean-100.clean.lst >> DNN.trn.scp
       #uniq
       sort DNN.trn.scp|uniq > xyz; mv xyz DNN.trn.scp
       cat DNN.trn.scp > wav.lst

       #MFCC feature extraction & label generation
       awk -F ',' '{print $NF}' DNN.trn.scp | sed 's|\(.*\)/.*|\1|' |sort| uniq| xargs mkdir -p


elif [ $phase == "dev-clean" ] ; then
	find dev-clean -type f |xargs cat - | awk -F ',' '{print $NF}' | sed 's|\(.*\)/.*|\1|' |sort| uniq| xargs mkdir -p
	find dev-clean -type f |xargs cat - >> wav.lst
elif [ $phase == "test-clean" ]; then
	find test-clean -type f |xargs cat - | awk -F ',' '{print $NF}' | sed 's|\(.*\)/.*|\1|' |sort| uniq| xargs mkdir -p
        find test-clean -type f |xargs cat - >> wav.lst
fi	

python mfcc.py wav.lst




for size in small  medium large ;do

  Nepoch=50

 if [ $size == "small" ] ; then
     ExpDir=small-models/${trndatabase}_vit_MFCC
 elif [ $size == "medium" ] ; then
     ExpDir=medium-models/${trndatabase}_vit_MFCC
 elif [ $size == "large" ] ; then 
   ExpDir=large-models/${trndatabase}_vit_MFCC    
   Nepoch=49
 fi

  echo $Nepoch > epoch.txt
  echo $ExpDir > dir.txt
  echo "Test" > phase.txt
  echo "tr_${trndatabase}_to_${database}" >EvalFile.txt
  cat EvalFile.txt
  python KWT3_vad.py

  #for eer and mindcf
  echo "Test_ind" > phase.txt
  echo "tr_${trndatabase}_to_${database}" >EvalFile.txt
  python KWT3_vad.py

  #test score 
  file=${ExpDir}/tr_${trndatabase}_to_${database}.testdata.auc.csv
  echo $acc $file
  echo ""
  rm -rf temp
  sed 's|\[||g' ${file}|sed 's|\]||g'|sed "s|'|,|g"| sed 's|\,| |g'|sed 's/^[ \t]*//;s/[ \t]*$//'|\
        awk '{print $1,$2, $NF}'> temp #noise, snr, auc

  #category
  awk '{print $1}' temp|sort|uniq| grep -v "clean" > category
  awk '{print $2}' temp|grep -v "clean"| sort|uniq|sort -n > snr

 #score arrange for the plot
 rm -f score.txt
 for noise in `cat category`;do
     rm -f tmp
     for db in `cat snr`;do
           grep -w "^${noise}" temp | grep -E "(^|\s)${db}($|\s)" |awk '{print $NF}' >> tmp
     done
     #
     cat tmp | tr '\n' ' ' >> score.txt
     echo "">> score.txt
 done

  paste  category score.txt

 
  #calculate EER
  score=${ExpDir}/tr_${trndatabase}_to_${database}.testdata.score 
  label=${ExpDir}/tr_${trndatabase}_to_${database}.testdata.label 
  echo "EER .."
  python   eer.py  $score $label

done 



