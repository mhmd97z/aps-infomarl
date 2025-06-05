#!/bin/sh
env="aps"
algo="fmat"
exp="partial_olp/test/ap20_ue6_sinr0_offaps0/veh_10step_50ms/fmat/localpsum1_sumcost0_conncost1"
seed=1
python train_fmat.py --env_name ${env} --algorithm_name ${algo} --gamma 0.01 \
 --experiment_name ${exp} --seed ${seed} --n_training_threads 16 --use_wandb False \
 --n_rollout_threads 16 --num_mini_batch 1 --episode_length 100 --num_env_steps 300000 \
 --lr 0.0001 --critic_lr 0.0001 --entr_lr 0.0001 --tar_en_coef 0.3 \
 --max_grad_norm 1 --clip_param 0.5 --ppo_epoch 5 \
 --use_valuenorm --use_linear_lr_decay --n_block 1 --log_interval 1
