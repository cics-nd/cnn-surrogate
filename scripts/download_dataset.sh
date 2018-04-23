#!/bin/bash

KLE=$1


function gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}



case ${KLE} in
	'4225')
		GDRIVE_ID=1s9NfdeQ0HFdz5GzbhjsNxRyHwBMf1Y1u
		;;
	'500')
		GDRIVE_ID=17DQDIS8ywwtDjr0nFFylG56P9_R1yXOp
		;;
	'50')
		GDRIVE_ID=1TRYpsoE1dkt0V_ao-I9_aLUQuwRapn6C
		;;
esac

TARGET_DIR=./dataset/kle$KLE/
mkdir -p $TARGET_DIR

TAR_FILE=./dataset/kle$KLE.tar

gdrive_download $GDRIVE_ID $TAR_FILE
tar -xvf $TAR_FILE -C ./dataset/
mv ./dataset/kle$KLE/* ./dataset 
rm $TAR_FILE
rm -r $TARGET_DIR
