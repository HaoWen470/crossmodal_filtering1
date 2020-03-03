#!/bin/bash


name="fusion_weighted_seq5_0"
echo $name
load="checkpoints/fusion_poe_blackout_0.8_1-phase_3_e2e.ckpt"

# python train_fusion.py --data_size 1000 --batch 128 --epochs 1 --fusion_type weighted \
# --experiment_name $name --pretrain 5  --sequential_image 5  \
# --lr 1e-5 

python train_fusion.py --data_size 1000 --batch 128 --epochs 5 --fusion_type weighted \
--experiment_name $name --pretrain 5  --sequential_image 5 \
--lr 1e-5 	--train fusion --load_checkpoint $load --module_type ekf

python train_fusion.py --data_size 1000 --batch 128 --epochs 5 --fusion_type weighted \
--experiment_name $name --pretrain 5  --sequential_image 5 \
--lr 1e-5 	--train fusion 

	python train_fusion.py --data_size 1000 --batch 128 --epochs 5 --fusion_type weighted \
--experiment_name $name --pretrain 5  --sequential_image 5 \
--lr 1e-5 	--train fusion 


# python train_fusion.py --data_size 1000 --batch 128 --epochs 5 --fusion_type weighted \
# --experiment_name $name --pretrain 5  --sequential_image 5 \
# --train fusion  --lr 1e-5 \
# --init_state_noise 0.3 


# python train_fusion.py --data_size 1000 --batch 128 --epochs 5 --fusion_type weighted \
# --experiment_name $name --pretrain 5  --sequential_image 5 \
# --train fusion  --lr 1e-5 \
# --init_state_noise 0.4 



