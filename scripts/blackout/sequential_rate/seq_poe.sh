#!/bin/bash


name="fusion_poe_seq5_0"
echo $name

python train_fusion.py --data_size 1000 --batch 128 --epochs 1 --fusion_type poe \
--experiment_name $name --pretrain 5  --sequential_image 5  \
--lr 1e-5 

python train_fusion.py --data_size 1000 --batch 128 --epochs 4 --fusion_type poe \
--experiment_name $name --pretrain 5  --sequential_image 5 \
--lr 1e-5 	--train fusion 

python train_fusion.py --data_size 1000 --batch 128 --epochs 5 --fusion_type poe \
--experiment_name $name --pretrain 5  --sequential_image 5 \
--lr 1e-5 	--train fusion 

	python train_fusion.py --data_size 1000 --batch 128 --epochs 5 --fusion_type poe \
--experiment_name $name --pretrain 5  --sequential_image 5 \
--lr 1e-5 	--train fusion 


# python train_fusion.py --data_size 1000 --batch 128 --epochs 5 --fusion_type poe \
# --experiment_name $name --pretrain 5  --sequential_image 5 \
# --train fusion  --lr 1e-5 \
# --init_state_noise 0.3 


# python train_fusion.py --data_size 1000 --batch 128 --epochs 5 --fusion_type poe \
# --experiment_name $name --pretrain 5  --sequential_image 5 \
# --train fusion  --lr 1e-5 \
# --init_state_noise 0.4 



