#!/bin/bash

#python train_fusion.py --data_size 1000 --batch 128 --epochs 5 --fusion_type weighted \
#--experiment_name fusion_weighted_final_1 --pretrain 5 \
#--train fusion --load_checkpoint checkpoints/fusion_poe_1000_hu64_1loss-phase_3_e2e.ckpt --lr 1e-5
#
#
#python train_fusion.py --data_size 1000 --batch 128 --epochs 5 --fusion_type weighted \
#--experiment_name fusion_weighted_final_1 --pretrain 5 \
#--train fusion  --lr 1e-5
#
#
#python train_fusion.py --data_size 1000 --batch 128 --epochs 5 --fusion_type weighted \
#--experiment_name fusion_weighted_final_1 --pretrain 5 \
#--train fusion  --lr 1e-5

python train_fusion.py --data_size 1000 --batch 128 --epochs 1 --fusion_type weighted \
--experiment_name fusion_weighted_truncated_0 --pretrain 5 \
--lr 1e-5 --start_timestep 200

python train_fusion.py --data_size 1000 --batch 128 --epochs 4 --fusion_type weighted \
--experiment_name fusion_weighted_truncated_0 --pretrain 5 \
--train fusion --lr 1e-5 --start_timestep 200


python train_fusion.py --data_size 1000 --batch 128 --epochs 5 --fusion_type weighted \
--experiment_name fusion_weighted_truncated_0 --pretrain 5 \
--train fusion  --lr 1e-5 --start_timestep 200


python train_fusion.py --data_size 1000 --batch 128 --epochs 5 --fusion_type weighted \
--experiment_name fusion_weighted_truncated_0 --pretrain 5 \
--train fusion  --lr 1e-5 --start_timestep 200