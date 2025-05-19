#!/usr/bin/env python
import argparse
import wandb
import socket
import yaml
import pickle
import setproctitle
import os
import sys
import random
import torch
import numpy as np
from pathlib import Path
from argparse import Namespace
import torch.multiprocessing as mp
from torch.utils.data import Subset
from torch_geometric.data import DataLoader
import torch.nn as nn

sys.path.append(os.path.abspath(os.getcwd()))
sys.path.append("../../")
from utils.utils import print_args, print_box, connected_to_internet
from onpolicy.config import get_config
from onpolicy.envs.aps.aps import Aps
from onpolicy.envs.env_wrappers import (
    SubprocVecEnv,
    DummyVecEnv,
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

    return runner.policy.actor, runner.save_dir



if __name__ == "__main__":
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)

    with open("/home/mzi/aps-infomarl/onpolicy/scripts/8strongest_20aps_6ues.pickle", 'rb') as f:
        data_list = pickle.load(f)
    print("pickled data is retireved")

    for item in data_list:
        item['channel', 'same_ap', 'channel'].edge_index = item['channel', 'same_ap', 'channel'].edge_index.T
        item['channel', 'same_ue', 'channel'].edge_index = item['channel', 'same_ue', 'channel'].edge_index.T
    print("fixed edges")

    random.seed(0)
    random.shuffle(data_list)
    train_data, test_data = \
        data_list[:int(len(data_list)*0.9)], \
        data_list[int(len(data_list)*0.9):]
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    actor, save_dir = main(sys.argv[1:])

    optimizer = torch.optim.Adam(actor.parameters(), lr=0.001, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0
    patience = 5
    trigger_times = 0
    rnn_states = None
    masks = None

    actor.eval()
    correct, size = 0, 0
    for data in test_loader:
        data = data.to(torch.device("cuda:0"))
        actions, action_log_probs, rnn_states, action_logits = actor(data, rnn_states, masks)
        correct += (actions.flatten() == data['channel'].y).sum().item()
        size += data['channel'].y.size(0)
    print("Test Accuracy: {:.4f}".format(correct/size))

    for epoch in range(1, 101):
        print("\n ----- Epoch: ", epoch)

        actor.train()

        # get a subset of the training data
        total_indices = list(range(len(train_loader.dataset)))
        sampled_indices = random.sample(total_indices, int(0.3 * len(total_indices)))
        subset_loader = DataLoader(
            Subset(train_loader.dataset, sampled_indices),
            batch_size=train_loader.batch_size,
            shuffle=True
        )

        for data in subset_loader:
            optimizer.zero_grad()
            data = data.to(torch.device("cuda:0"))
            actions, action_log_probs, rnn_states, action_logits = actor(data, rnn_states, masks)
            loss = loss_fn(action_logits, data['channel'].y.long())
            loss.backward()
            optimizer.step()
        print(f"Loss: {loss.item()}")

        actor.eval()
        correct, size = 0, 0
        for data in subset_loader:
            data = data.to(torch.device("cuda:0"))
            actions, action_log_probs, rnn_states, action_logits = actor(data, rnn_states, masks)
            correct += (actions.flatten() == data['channel'].y).sum().item()
            size += data['channel'].y.size(0)
        print("Train Accuracy: {:.4f}".format(correct / size))

        actor.eval()
        correct, size = 0, 0
        for data in test_loader:
            data = data.to(torch.device("cuda:0"))
            actions, action_log_probs, rnn_states, action_logits = actor(data, rnn_states, masks)
            correct += (actions.flatten() == data['channel'].y).sum().item()
            size += data['channel'].y.size(0)
        acc = correct / size
        print("Test Accuracy: {:.4f}".format(acc))

        # Early stopping
        if acc > best_acc:
            best_acc = acc
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Save the model
    torch.save(actor.state_dict(), str(save_dir) + "/actor.pt")

