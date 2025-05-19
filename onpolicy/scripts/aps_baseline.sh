#!/bin/sh
env="aps"
algo="kstrongest"
seed=1
values="5 6"
for k in $values; do
    exp="partial_olp/ap20_ue6_sinr0_roffaps0_lastg/veh_10step_50ms/${k}strongest"
    python baseline.py --env_name ${env} --algorithm_name ${algo} \
    --n_rollout_threads 16 --seed ${seed} \
    --episode_length 100 --num_env_steps 300000 \
    --experiment_name ${exp} --K ${k} --largest \
    --log_interval 1
done
