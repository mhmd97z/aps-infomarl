#!/bin/sh
env="aps"
algo="kstrongest"
seed=1
values="4"
for k in $values; do
    exp="mrt/ap20_ue6_sinr0_lastg_8env/veh_10step_50ms/${k}strongest"
    python baseline.py --env_name ${env} --algorithm_name ${algo} \
    --n_rollout_threads 8 --seed ${seed} \
    --episode_length 100 --num_env_steps 300000 \
    --experiment_name ${exp} --K ${k} --largest \
    --log_interval 1
done
