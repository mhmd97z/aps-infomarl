#!/bin/sh
env="aps"
algo="mappo"
exp="comp/scen3_gnnmappo"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    python train_aps.py --use_valuenorm \
    --env_name ${env} --algorithm_name ${algo} \
    --experiment_name ${exp} --seed ${seed} --n_training_threads 16 --n_rollout_threads 2 \
    --num_mini_batch 1 --episode_length 10 --num_env_steps 300000 \
    --ppo_epoch 5 --use_ReLU --lr 7e-4 --critic_lr 7e-4 \
    --user_name "marl" --use_recurrent_policy False --max_grad_norm 1 \
    --gamma 0.01 --use_linear_lr_decay --max_grad_norm 1
done
