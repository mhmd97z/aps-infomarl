#!/bin/sh
env="aps"
algo="mappo"
exp="updated_power/ped/gnnmappo/pcoef1_localpsum0_conncost1"
seed_max=1

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    python train_aps.py --use_valuenorm \
    --env_name ${env} --algorithm_name ${algo} \
    --experiment_name ${exp} --seed ${seed} --n_training_threads 16 --n_rollout_threads 16 \
    --num_mini_batch 1 --episode_length 100 --num_env_steps 160000 \
    --ppo_epoch 5 --use_ReLU --lr 7e-4 --critic_lr 7e-4 \
    --user_name "marl" --use_recurrent_policy False --max_grad_norm 1 \
    --gamma 0.01 --use_linear_lr_decay --log_interval 1
done
