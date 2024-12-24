#!/usr/bin/env python
import argparse
from distutils.util import strtobool
import wandb
import socket
import yaml
import setproctitle
import numpy as np
from pathlib import Path
from argparse import Namespace
import torch
import torch.multiprocessing as mp

import os, sys

sys.path.append(os.path.abspath(os.getcwd()))
sys.path.append("../../")
from utils.utils import print_args, print_box, connected_to_internet
from onpolicy.config import get_config
from onpolicy.envs.aps.aps import Aps
from onpolicy.envs.env_wrappers import (
    SubprocVecEnv,
    DummyVecEnv,
    GraphSubprocVecEnv,
    GraphDummyVecEnv,
    ApsSubprocVecEnv
)

def make_train_env(all_args: argparse.Namespace):
    def get_env_fn(rank: int):
        def init_env():
            if all_args.env_name == "aps":
                env = Aps(all_args.env_args)
            else:
                print(f"Can not support the {all_args.env_name} environment")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        if all_args.env_name == "aps":
            raise
        return DummyVecEnv([get_env_fn(0)])
    else:
        if all_args.env_name == "aps":
            return ApsSubprocVecEnv(
                [get_env_fn(i) for i in range(all_args.n_rollout_threads)]
            )
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args: argparse.Namespace):
    def get_env_fn(rank: int):
        def init_env():
            if all_args.env_name == "aps":
                env = Aps(all_args.env_args)
            else:
                print(f"Can not support the {all_args.env_name} environment")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        if all_args.env_name == "aps":
            raise
        return DummyVecEnv([get_env_fn(0)])
    else:
        if all_args.env_name == "aps":
            return ApsSubprocVecEnv(
                [get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)]
            )
        return SubprocVecEnv(
            [get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)]
        )

def merge_namespaces(namespace, yaml_namespace):
    for key, value in vars(yaml_namespace).items():
        if isinstance(value, Namespace):
            if hasattr(namespace, key):
                merge_namespaces(getattr(namespace, key), value)
            else:
                setattr(namespace, key, value)
        else:
            setattr(namespace, key, value)

def parse_args(args, parser):
    parser.add_argument("--num_agents", type=int, default=3)
    all_args = parser.parse_known_args(args)[0]
    return all_args, parser


def main(args):
    parser = get_config()
    all_args = parser.parse_known_args(args)[0]
    
    yaml_path = "/home/mzi/aps-infomarl/onpolicy/aps-config.yaml"
    with open(yaml_path, 'r') as file:
        yaml_config = yaml.safe_load(file)
        def yaml_to_namespace(config):
            if isinstance(config, dict):
                return Namespace(**{key: yaml_to_namespace(value) for key, value in config.items()})
            return config
        yaml_namespace = yaml_to_namespace(yaml_config)
        merge_namespaces(all_args, yaml_namespace)

    if all_args.algorithm_name in ["rmappo"]:
        assert (
            all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy
        ), "check recurrent policy!"
    elif all_args.algorithm_name in ["mappo"]:
        assert (
            all_args.use_recurrent_policy == False
            and all_args.use_naive_recurrent_policy == False
        ), "check recurrent policy!"
    else:
        raise NotImplementedError

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print_box("Choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print_box("Choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # if all_args.verbose:
    print_args(all_args)

    # run dir
    run_dir = (
        Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")
        / all_args.env_name
        / all_args.algorithm_name
        / all_args.experiment_name
    )
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    all_args.use_wandb = False    
    # wandb
    if all_args.use_wandb:
        # for supercloud when no internet_connection
        if not connected_to_internet():
            import json

            # save a json file with your wandb api key in your
            # home folder as {'my_wandb_api_key': 'INSERT API HERE'}
            # NOTE this is only for running on systems without internet access
            # have to run `wandb sync wandb/run_name` to sync logs to wandboard
            with open(os.path.expanduser("~") + "/keys.json") as json_file:
                key = json.load(json_file)
                my_wandb_api_key = key["my_wandb_api_key"]  # NOTE change here as well
            os.environ["WANDB_API_KEY"] = my_wandb_api_key
            os.environ["WANDB_MODE"] = "dryrun"
            os.environ["WANDB_SAVE_CODE"] = "true"

        print_box("Creating wandboard...")
        run = wandb.init(
            config=all_args,
            project=all_args.project_name,
            # project=all_args.env_name,
            entity=all_args.user_name,
            notes=socket.gethostname(),
            name=str(all_args.algorithm_name)
            + "_"
            + str(all_args.experiment_name)
            + "_seed"
            + str(all_args.seed),
            # group=all_args.scenario_name,
            dir=str(run_dir),
            # job_type="training",
            reinit=True,
        )
    else:
        if not run_dir.exists():
            curr_run = "run1"
        else:
            exst_run_nums = [
                int(str(folder.name).split("run")[1])
                for folder in run_dir.iterdir()
                if str(folder.name).startswith("run")
            ]
            if len(exst_run_nums) == 0:
                curr_run = "run1"
            else:
                curr_run = "run%i" % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(all_args.algorithm_name)
        + "-"
        + str(all_args.env_name)
        + "-"
        + str(all_args.experiment_name)
        + "@"
        + str(all_args.user_name)
    )

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.env_args.simulation_scenario.number_of_aps * all_args.env_args.simulation_scenario.number_of_ues

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    # run experiments
    if all_args.share_policy:
        if all_args.env_name == "aps":
            from onpolicy.runner.shared.aps_runner import ApsRunner as Runner
        else:
            raise
    else:
        raise NotImplementedError

    runner = Runner(config)
    # if all_args.verbose:
    #     print_box("Actor Network", 80)
    #     if type(runner.policy) == list:
    #         print_box(runner.policy[0].actor, 80)
    #         print_box("Critic Network", 80)
    #         print_box(runner.policy[0].critic, 80)
    #     else:
    #         print_box(runner.policy.actor, 80)
    #         print_box("Critic Network", 80)
    #         print_box(runner.policy.critic, 80)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
        runner.writter.close()


if __name__ == "__main__":
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)


    main(sys.argv[1:])
