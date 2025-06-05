#!/bin/sh
env="aps"
algo="gnnmappo"
exp="optimal_olp/test/ap20_ue6_sinr0_roffaps0_lastg/veh_10step_50ms/gnnmappo-pret-8strongest-eval" # /localpsum0_sumcost0_conncost1_pcoef5"
seed=1
python train_gnnmappo.py --use_valuenorm --env_name ${env} --algorithm_name ${algo} \
 --experiment_name ${exp} --seed ${seed} --n_training_threads 16 --n_rollout_threads 1 \
 --num_mini_batch 1 --episode_length 100 --num_env_steps 300000 \
 --ppo_epoch 5 --use_ReLU --lr 7e-4 --critic_lr 7e-4 \
 --user_name "marl" --use_recurrent_policy False --max_grad_norm 1 \
 --gamma 0.01 --use_linear_lr_decay --log_interval 1 \
 --entropy_coef 0.1 \
 --model_dir pretrained_models/8strongest \
 --use_eval
