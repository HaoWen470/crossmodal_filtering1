#!/bin/bash


load="checkpoints/fusion_poe_blackout_0.8_1-phase_3_e2e.ckpt"
name="fusion_poe_old_seq5_0"
echo $name

python train_fusion.py --data_size 1000 --batch 128 --epochs 5 --fusion_type poe \
--experiment_name $name --pretrain 5  --sequential_image 5 \
--lr 1e-5 	--train fusion --load_checkpoint load --old_weighting

python train_fusion.py --data_size 1000 --batch 128 --epochs 5 --fusion_type poe \
--experiment_name $name --pretrain 5  --sequential_image 5 \
--lr 1e-5 	--train fusion --old_weighting

	python train_fusion.py --data_size 1000 --batch 128 --epochs 5 --fusion_type poe \
--experiment_name $name --pretrain 5  --sequential_image 5 \
--lr 1e-5 	--train fusion --old_weighting
