#!/bin/sh
env="aps"
algo="gnnmappo"
exp="mrt/ap20_ue6_sinr0_lastg_8env/veh_10step_50ms/gnnmappo-pret-4strongest/localpsum1_sumcost1_conncost1_pcoef10"
seed=1
python train_gnnmappo.py --use_valuenorm --env_name ${env} --algorithm_name ${algo} \
 --experiment_name ${exp} --seed ${seed} --n_training_threads 8 --n_rollout_threads 8 \
 --num_mini_batch 1 --episode_length 100 --num_env_steps 300000 \
 --ppo_epoch 5 --use_ReLU --lr 7e-4 --critic_lr 7e-4 \
 --user_name "marl" --use_recurrent_policy False --max_grad_norm 1 \
 --gamma 0.01 --use_linear_lr_decay --log_interval 1 \
 --entropy_coef 0.1 \
 --model_dir pretrained_models/4strongest \
#  --use_eval
