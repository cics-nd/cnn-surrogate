#!/bin/bash

KLE=$1
NUM_TRAIN=$2

function gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}


case ${KLE} in
	'4225')
		case ${NUM_TRAIN} in
			'128')
				GDRIVE_ID=1St84nsSySdV6HrrvMGa7wx2tYAsrgMxH
				;;
			'256')
				GDRIVE_ID=1c2vDa1hLUSalo7LLQnvfXZTq-Blgc5sI
				;;
			'512')
				GDRIVE_ID=15ovT8V_dqVMLnyftw_Dmr0lPqgZQQ4Mo
				;;
			'1024')
				GDRIVE_ID=1jdnxwpySK9SoTWbX7QzCNHXH7Jimce4M
				;;
		esac
		;;
	'500')
		case ${NUM_TRAIN} in
			'64')
				GDRIVE_ID=1VPv0pmlTBHpjIo9gY-kmeAIUhmBOPIrv
				;;
			'128')
				GDRIVE_ID=1ZRZvheyCxZX-2sdkOtD4BZZjQjrnRDhD
				;;
			'256')
				GDRIVE_ID=17uP88akvt1IU2di14kgUx_2piNw7LsDt
				;;
			'512')
				GDRIVE_ID=1c6RS8xVbc8newrV_yqJKg_VLJVH9zRDF
				;;
		esac
		;;
	'50')
		case ${NUM_TRAIN} in
			'32')
				GDRIVE_ID=1P4Q4qkWxrhVvwx8OQ7SBnbEta_zMfG6k
				;;
			'64')
				GDRIVE_ID=19taFYtNFbwSIopc9DD9dcTtc8v1rrM9C
				;;
			'128')
				GDRIVE_ID=12JxAaVenrwcmMvZIFUW9S1UhaPsNP799
				;;
			'256')
				GDRIVE_ID=1rQ62vkHw1oD7g6r3-joXasXoiIQpgpAm
				;;
		esac
		;;
esac


TARGET_DIR=./experiments/Bayesian/kle$KLE
mkdir -p $TARGET_DIR

TAR_FILE=./experiments/Bayesian/kle$KLE/kle${KLE}n${NUM_TRAIN}.tar

gdrive_download $GDRIVE_ID $TAR_FILE
tar -xvf $TAR_FILE -C $TARGET_DIR

rm $TAR_FILE

