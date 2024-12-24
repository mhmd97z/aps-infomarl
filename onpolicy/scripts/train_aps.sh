#!/bin/sh
env="aps"
algo="mappo"
exp="test"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train_aps.py --use_valuenorm \
    --use_popart --env_name ${env} --algorithm_name ${algo} \
    --experiment_name ${exp} --seed ${seed} --n_training_threads 2 --n_rollout_threads 2 \
    --num_mini_batch 1 --episode_length 25 --num_env_steps 100000 \
    --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 \
    --user_name "marl" --use_recurrent_policy False
done
