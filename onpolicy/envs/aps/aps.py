import gym
import yaml
import os
import sys
import torch
from gym import spaces
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../envs/aps/lib")))
from network_simlator import NetworkSimulator
from data_store import DataStore
from aps_utils import clip_abs, tpdv_parse, get_adj


class Aps(gym.Env):
    def __init__(self, env_args=None, args=None, if_graph=False):
        self.env_args = env_args
        self.if_graph = if_graph
        tpdv_parse(self.env_args)
        self.simulator = NetworkSimulator(env_args.simulation_scenario)
        self.history_length = self.env_args.history_length
        self.datastore = DataStore(self.history_length, ['obs'])

        if self.env_args.if_include_channel_rank:
            self.feature_length = 3
        else:
            self.feature_length = 2

        self.num_ues = self.simulator.scenario_conf.number_of_ues
        self.num_aps = self.simulator.scenario_conf.number_of_aps
        self.n_agents = self.num_ues * self.num_aps

        if self.if_graph:
            self.same_ue_edges, self.same_ap_edges = get_adj(self.num_ues, self.num_aps, if_transpose=False)

        self.action_space = [spaces.Discrete(2) for _ in range(self.n_agents)]
        self.observation_space = [
            spaces.Box(low=0, high=1, 
                       shape=(self.history_length * self.feature_length,), 
                       dtype=float)
            for _ in range(self.n_agents)]
        self.share_observation_space = [
            spaces.Box(low=0, high=1, 
                       shape=(self.n_agents * self.history_length * self.feature_length,), 
                       dtype=float)
            for _ in range(self.n_agents)]

        with open(self.env_args.simulation_scenario.data_normalization_config, 'r') as config_file:
            self.normalization_dict = yaml.safe_load(config_file)


    def step(self, actions):
        actions = torch.from_numpy(actions).to(self.env_args.simulation_scenario.device_sim)
        self.simulator.step(actions)

        obs, state, reward, mask, info = self.compute_state_reward()
        done = [False] * self.n_agents

        if self.if_graph:
            return obs, state, reward, done, info, mask, self.same_ue_edges, self.same_ap_edges
        else:
            return obs, state, reward, done, info, mask


    def compute_state_reward(self):
        # state calc
        simulator_info = self.simulator.datastore.get_last_k_elements()
        serving_mask = self.simulator.serving_mask.clone().detach().flatten().to(torch.int32)

        channel_coef = simulator_info['channel_coef']
        # TODO: aggregate over step length
        # self.datastore.add(obs=channel_coef.mean(dim=0))
        self.datastore.add(obs=channel_coef[-1])

        G = self.datastore.get_last_k_elements()['obs']
        # TODO: aggregate over history
        G = G.squeeze()
        G = clip_abs(G)
        x = torch.reshape(G, (-1, 1))
        x = torch.cat((torch.log2(torch.abs(x)), x.angle()), 1)
        x_mean = torch.tensor(self.normalization_dict['x_mean']).to(device=x.device)
        x_std = torch.tensor(self.normalization_dict['x_std']).to(device=x.device)
        x = (x - x_mean[:2]) / x_std[:2]

        obs = x.clone()
        state = obs.view(-1, obs.shape[0]*obs.shape[1]).repeat(obs.shape[0], 1).clone()

        # power cost
        mu = self.env_args.power_coef
        transmission_power_consumption = simulator_info['transmission_power_consumption'].mean(dim=0)
        ap_circuit_power_consumption = simulator_info['ap_circuit_power_consumption'].mean(dim=0)

        if self.env_args.if_use_local_power_sum and not self.env_args.if_full_cooperation:
            transmission_power_consumption_ = transmission_power_consumption.sum(dim=0, keepdim=True).expand_as(transmission_power_consumption)
        else:
            transmission_power_consumption_ = transmission_power_consumption.clone()
        ap_circuit_power_consumption_ = ap_circuit_power_consumption.unsqueeze(1).expand_as(transmission_power_consumption)

        totoal_ = ap_circuit_power_consumption_ + transmission_power_consumption_
        if self.env_args.simulation_scenario.if_power_in_db:
            totoal_ = 10 * torch.log10(totoal_)
            totoal_ = torch.clip(totoal_, min=-30) + 31
        power_coef_cost = mu * torch.reshape(totoal_ , (-1, 1))

        if self.env_args.if_connection_cost and not self.env_args.if_full_cooperation:
            power_coef_cost += mu * serving_mask.reshape(power_coef_cost.shape).to(power_coef_cost)

        if self.env_args.if_full_cooperation:
            power_coef_cost.fill_(power_coef_cost.sum())

        power_coef_cost = power_coef_cost.to(dtype=self.env_args.simulation_scenario.float_dtype_sim, 
                                             device=self.env_args.simulation_scenario.device_sim)

        # se cost
        eta = self.env_args.se_coef
        threshold = self.env_args.sinr_threshold
        if self.env_args.if_full_cooperation:
            constraints = simulator_info['sinr'].clone().mean(dim=0)
            constraints.fill_(simulator_info['sinr'].min() - threshold)
        else:
            constraints = (simulator_info['sinr'] - threshold).mean(dim=0)

        se_violation_cost = torch.clip(torch.exp(-eta * constraints), max=500)
        se_violation_cost = se_violation_cost.expand(self.num_aps, -1).clone()
        se_violation_cost = torch.reshape(se_violation_cost, (-1, 1))

        se_violation_cost = se_violation_cost.to(dtype=self.env_args.simulation_scenario.float_dtype_sim, 
                                             device=self.env_args.simulation_scenario.device_sim)

        if self.env_args.if_sum_cost:
            reward = -(se_violation_cost + power_coef_cost).clone().detach()
        else:
            reward = - se_violation_cost.clone().detach()
            reward[se_violation_cost < float(self.env_args.sec_to_pc_switch_threshold)] = \
                - power_coef_cost[se_violation_cost < float(self.env_args.sec_to_pc_switch_threshold)].clone().detach()

        mask = self.simulator.channel_manager.measurement_mask.clone().detach() \
            .flatten().to(torch.int32).unsqueeze(1)

        truncated_sinr_std = simulator_info['sinr'].std(dim=1, unbiased=False).mean()
        clean_sinr_std = simulator_info['clean_sinr'].std(dim=1, unbiased=False).mean()

        info = {
            'min_sinr': simulator_info['sinr'].mean(dim=0).min().mean(),
            'mean_sinr': simulator_info['sinr'].mean(),
            'truncated_sinr_std': truncated_sinr_std,
            'clean_sinr_std': clean_sinr_std,
            'transmission_power_consumption': transmission_power_consumption.sum(),
            'circuit_power_consumption': ap_circuit_power_consumption.sum(),
            'totoal_power_consumption': transmission_power_consumption.sum() + ap_circuit_power_consumption.sum(),
            'active_ap_count': torch.sum(serving_mask.reshape((self.num_aps, self.num_ues)), dim=1).sign().sum().float(),
            'reward': reward.mean(),
            'mean_serving_ap_count': serving_mask.reshape((self.num_aps, self.num_ues)).sum(dim=0).float().mean(),
            'mean_served_ue_count': serving_mask.reshape((self.num_aps, self.num_ues)).sum(dim=1).float().mean(),
            'se_violation_cost': se_violation_cost.mean(),
            'power_coef_cost': power_coef_cost.mean(),
        }

        return obs.to(torch.float32).cpu().numpy(), state.to(torch.float32).cpu().numpy(), reward.to(torch.float32).cpu().numpy(), mask.to(torch.float32).cpu().numpy(), info


    def get_obs_size(self):
        """ Returns the shape of the observation """
        return self.observation_space[0].shape[0]


    def get_avail_actions(self):
        return torch.ones((self.n_agents, self.get_total_actions()))


    def get_total_actions(self):
        return self.action_space[0].n


    def seed(self, seed):
        self.simulator.set_seed(seed)


    def reset(self):
        self.simulator.reset()
        obs, state, _, mask, info = self.compute_state_reward()

        if self.if_graph:
            return obs, state, mask, info, self.same_ue_edges, self.same_ap_edges
        else:
            return obs, state, mask, info


    def process_obs_graph(self, graph):
        x = graph['channel'].x[:, :2]
        if self.env_args.if_include_channel_rank:
            sorted_indices = torch.argsort(x[:, 0]).to(device=x.device)
            ranks = torch.empty_like(sorted_indices).to(device=x.device)        
            ranks[sorted_indices] = torch.arange(len(x[:, 0])).to(device=x.device)
            normalized_ranks = (ranks / (len(x[:, 0]) - 1)).unsqueeze(dim=1)
            x = torch.cat((x, normalized_ranks), dim=1)
        graph['channel'].x = x

