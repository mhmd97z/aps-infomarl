#!/usr/bin/env python
import yaml
import os
import sys
import torch
import pickle
import numpy as np
from argparse import Namespace
import torch.multiprocessing as mp

sys.path.append(os.path.abspath(os.getcwd()))
sys.path.append("../../")
from onpolicy.config import get_config
from onpolicy.scripts.train_aps import make_train_env
from onpolicy.envs.aps.lib.aps_utils import get_adj
from torch_geometric.data import HeteroData


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

    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    envs = make_train_env(all_args, if_graph=False)

    return envs, all_args


if __name__ == "__main__":
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)

    envs, all_args = main(sys.argv[1:])
    n_envs, n_aps, n_ues = all_args.n_rollout_threads, all_args.env_args.simulation_scenario.number_of_aps, all_args.env_args.simulation_scenario.number_of_ues
    obs, share_obs, available_actions, _ = envs.reset()
    action = np.ones((n_envs, n_aps, n_ues))
    data_list = []
    episodes = 100
    k = 4

    same_ue_edges, same_ap_edges = get_adj(n_ues, n_aps, if_transpose=False)
    for episode in range(200):
        print("episode: ", episode)
        for step in range(100):
            obs, _, _, _, _, _ = envs.step(
                action
            )
            obs = torch.tensor(obs)

            channel_abs = obs[:, :, 0].reshape((n_envs, n_aps, n_ues))
            indices = torch.topk(channel_abs, k, dim=1, largest=True).indices.to(device=obs.device)
            mask = torch.zeros((n_envs, n_aps, n_ues)).scatter_(1, indices, 1).to(dtype=torch.int, device=obs.device).reshape((n_envs, -1))

            for iii in range(n_envs):
                data_ = HeteroData()
                data_['channel'].y = mask[iii]
                data_['channel'].x = obs[iii]
                data_['channel', 'same_ue', 'channel'].edge_index = torch.tensor(same_ue_edges)
                data_['channel', 'same_ap', 'channel'].edge_index = torch.tensor(same_ap_edges)

                data_list.append(data_)

    print(f"saving at /home/mzi/aps-infomarl/onpolicy/scripts/{k}strongest_{n_aps}aps_{n_ues}ues_dataset.pickle")
    with open(f"/home/mzi/aps-infomarl/onpolicy/scripts/pret_dataset/{k}strongest_{n_aps}aps_{n_ues}ues_dataset.pickle", 'wb') as f:
        pickle.dump(data_list, f)
