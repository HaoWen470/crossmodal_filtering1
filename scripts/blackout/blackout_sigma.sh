#!/bin/bash

if [ -z "$1" ]
then
	echo "Please provide exactly 1 argument blackout amount"
else
	ratio=$1

	python train_fusion.py --data_size 1000 --batch 128 --epochs 1 --fusion_type sigma \
	--experiment_name fusion_sigma_blackout_$ratio_0 --pretrain 5 --blackout $ratio \
	 --lr 1e-5 

	python train_fusion.py --data_size 1000 --batch 128 --epochs 4 --fusion_type sigma \
	--experiment_name fusion_sigma_blackout_$ratio_0 --pretrain 5 --blackout $ratio \
	--train fusion  --lr 1e-5 

	python train_fusion.py --data_size 1000 --batch 128 --epochs 5 --fusion_type sigma \
	--experiment_name fusion_sigma_blackout_$ratio_0 --pretrain 5 --blackout $ratio \
	--train fusion  --lr 1e-5 


	python train_fusion.py --data_size 1000 --batch 128 --epochs 5 --fusion_type sigma \
	--experiment_name fusion_sigma_blackout_$ratio_0 --pretrain 5 --blackout $ratio \
	--train fusion  --lr 1e-5\
	--init_state_noise 0.3 


	python train_fusion.py --data_size 1000 --batch 128 --epochs 5 --fusion_type sigma \
	--experiment_name fusion_sigma_blackout_$ratio_0 --pretrain 5 --blackout $ratio \
	--train fusion  --lr 1e-5\
	--init_state_noise 0.4 

fi 

