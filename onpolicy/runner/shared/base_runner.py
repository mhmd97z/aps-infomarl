import os
import wandb
import torch
import numpy as np
from typing import Dict
from tensorboardX import SummaryWriter  # tensorboardX to work with macos


class Runner(object):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """

    def __init__(self, config: Dict):
        self.all_args = config["all_args"]
        self.envs = config["envs"]
        self.eval_envs = config["eval_envs"]
        self.device = config["device"]
        self.num_agents = config["num_agents"]
        # total entites is agents + goals + obstacles
        if config.__contains__("render_envs"):
            self.render_envs = config["render_envs"]

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir
        self.run_dir = config["run_dir"]
        self.log_dir = str(self.run_dir / "logs")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writter = SummaryWriter(self.log_dir)
        self.save_dir = str(self.run_dir / "models")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # NOTE change variable input here
        if self.use_centralized_V:
            share_observation_space = self.envs.share_observation_space[0]
        else:
            share_observation_space = self.envs.observation_space[0]

        if self.all_args.algorithm_name == "gnnmappo":
            from onpolicy.algorithms.gnnmappo.graph_aps_mappo import GR_MAPPO
            from onpolicy.algorithms.gnnmappo.graph_aps_MAPPOPolicy import GR_MAPPOPolicy
            from onpolicy.utils.gnnmappo_graph_buffer import GnnMappoReplayBuffer
            self.policy = GR_MAPPOPolicy(
                self.all_args,
                self.envs.observation_space[0],
                share_observation_space,
                self.envs.action_space[0],
                device=self.device,
            )
            if self.model_dir is not None:
                print(f"Restoring from checkpoint stored in {self.model_dir}")
                self.restore()
            self.trainer = GR_MAPPO(self.all_args, self.policy, device=self.device)
            self.buffer = GnnMappoReplayBuffer(
                self.all_args,
                self.num_agents,
                self.envs.observation_space[0],
                share_observation_space,
                self.envs.action_space[0],
            )

        elif self.all_args.algorithm_name == "fmat":
            from onpolicy.algorithms.fmat.mat_trainer import MATTrainer
            from onpolicy.algorithms.fmat.transformer_policy import TransformerPolicy
            from onpolicy.utils.fmat_graph_buffer import FmatReplayBuffer
            self.policy = TransformerPolicy(
                self.all_args,
                self.envs.observation_space[0],
                share_observation_space,
                self.envs.action_space[0],
                self.num_agents,
                device=self.device,
            )
            if self.model_dir is not None:
                print(f"Restoring from checkpoint stored in {self.model_dir}")
                self.restore()
            self.trainer = MATTrainer(self.all_args, self.policy,  self.num_agents, device=self.device)
            self.buffer = FmatReplayBuffer(
                self.all_args,
                self.num_agents,
                self.envs.observation_space[0],
                share_observation_space,
                self.envs.action_space[0],
                self.all_args.env_name
            )

        elif self.all_args.algorithm_name == "mat":
            from onpolicy.algorithms.mat.mat_trainer import MATTrainer
            from onpolicy.algorithms.mat.transformer_policy import TransformerPolicy
            from onpolicy.utils.mat_graph_buffer import MatReplayBuffer
            self.policy = TransformerPolicy(self.all_args,
                             self.envs.observation_space[0],
                             share_observation_space,
                             self.envs.action_space[0],
                             self.num_agents,
                             device=self.device)
            if self.model_dir is not None:
                print(f"Restoring from checkpoint stored in {self.model_dir}")
                self.restore()
            self.trainer = MATTrainer(self.all_args, self.policy, self.num_agents, device=self.device)
            self.buffer = MatReplayBuffer(
                self.all_args,
                self.num_agents,
                self.envs.observation_space[0],
                share_observation_space,
                self.envs.action_space[0],
                    self.all_args.env_name
            )

        elif self.all_args.algorithm_name == "mappo":
            from onpolicy.algorithms.mappo.mappo_algo import R_MAPPO
            from onpolicy.algorithms.mappo.mappo_policy import R_MAPPOPolicy
            from onpolicy.utils.mappo_replay_buffer import MappoReplayBuffer
            self.policy = R_MAPPOPolicy(self.all_args, self.envs.observation_space[0], 
                                        share_observation_space, self.envs.action_space[0], 
                                        device=self.device)
            if self.model_dir is not None:
                print(f"Restoring from checkpoint stored in {self.model_dir}")
                self.restore(self.model_dir)
            self.trainer = R_MAPPO(self.all_args, self.policy, device = self.device)
            self.buffer = MappoReplayBuffer(
                self.all_args,
                self.num_agents,
                self.envs.observation_space[0],
                share_observation_space,
                self.envs.action_space[0]
            )

        else:
            raise NotImplementedError(
                f"Algorithm {self.all_args.algorithm_name} is not supported."
            )

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError

    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        raise NotImplementedError

    def train(self):
        """Train policies with data in buffer."""
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)
        self.buffer.after_update()
        return train_infos

    def save(self, episode=None):
        """Save policy's actor and critic networks."""
        if episode is None:
            policy_actor = self.trainer.policy.actor
            torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor.pt")
            policy_critic = self.trainer.policy.critic
            torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic.pt")
        else:
            self.policy.save(self.save_dir, episode)

    def restore(self):
        """Restore policy's networks from a saved model."""
        policy_actor_state_dict = torch.load(
            str(self.model_dir) + "/actor.pt", map_location=torch.device("cpu")
        )
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        if not self.all_args.use_render:
            try:
                policy_critic_state_dict = torch.load(
                    str(self.model_dir) + "/critic.pt", map_location=torch.device("cpu")
                )
                self.policy.critic.load_state_dict(policy_critic_state_dict)
            except FileNotFoundError:
                print("No critic model found, using default critic.")
            

    def process_infos(self, infos):
        """Process infos returned by environment."""
        env_infos = {}
        for agent_id in range(self.num_agents):
            idv_rews = []
            dist_goals, time_to_goals, min_times_to_goal = [], [], []
            idv_collisions, obst_collisions = [], []
            for info in infos:
                if "individual_reward" in info[agent_id].keys():
                    idv_rews.append(info[agent_id]["individual_reward"])
                if "Dist_to_goal" in info[agent_id].keys():
                    dist_goals.append(info[agent_id]["Dist_to_goal"])
                if "Time_req_to_goal" in info[agent_id].keys():
                    times = info[agent_id]["Time_req_to_goal"]
                    if times == -1:
                        times = (
                            self.all_args.episode_length * self.dt
                        )  # NOTE: Hardcoding `dt`
                    time_to_goals.append(times)
                if "Num_agent_collisions" in info[agent_id].keys():
                    idv_collisions.append(info[agent_id]["Num_agent_collisions"])
                if "Num_obst_collisions" in info[agent_id].keys():
                    obst_collisions.append(info[agent_id]["Num_obst_collisions"])
                if "Min_time_to_goal" in info[agent_id].keys():
                    min_times_to_goal.append(info[agent_id]["Min_time_to_goal"])

            agent_rew = f"agent{agent_id}/individual_rewards"
            times = f"agent{agent_id}/time_to_goal"
            dists = f"agent{agent_id}/dist_to_goal"
            agent_col = f"agent{agent_id}/num_agent_collisions"
            obst_col = f"agent{agent_id}/num_obstacle_collisions"
            min_times = f"agent{agent_id}/min_time_to_goal"

            env_infos[agent_rew] = idv_rews
            env_infos[times] = time_to_goals
            env_infos[min_times] = min_times_to_goal
            env_infos[dists] = dist_goals
            env_infos[agent_col] = idv_collisions
            env_infos[obst_col] = obst_collisions
        return env_infos

    def log_train(self, train_infos: Dict, total_num_steps: int):
        """
        Log training info.
        train_infos: (dict)
            information about training update.
        total_num_steps: (int)
            total number of training env steps.
        """
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    def log_env(self, env_infos: Dict, total_num_steps: int):
        """
        Log env info.
        env_infos: (dict)
            information about env state.
        total_num_steps: (int)
            total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)

    def get_collisions(self, env_infos: Dict):
        """
        Get the collisions from the env_infos
        Example: {'agent0/individual_rewards': [5],
                'agent0/time_to_goal': [0.6000000000000001],
                'agent0/min_time_to_goal': [0.23632679886748278],
                'agent0/dist_to_goal': [0.03768003822249384],
                'agent0/num_agent_collisions': [1.0],
                'agent0/num_obstacle_collisions': [0.0],
                'agent1/individual_rewards': [5],
                'agent1/time_to_goal': [0.6000000000000001],
                'agent1/min_time_to_goal': [0.3067362645187025],
                'agent1/dist_to_goal': [0.0387233764393595],
                'agent1/num_agent_collisions': [1.0],
                'agent1/num_obstacle_collisions': [0.0]}

        """
        collisions = 0
        for k, v in env_infos.items():
            if "collision" in k:
                collisions += v[0]
        return collisions

    def get_fraction_episodes(self, env_infos: Dict):
        """
        Get the fraction of episode required to get to the goals
        from env_infos
        Example: {'agent0/individual_rewards': [5],
                'agent0/time_to_goal': [0.6000000000000001],
                'agent0/min_time_to_goal': [0.23632679886748278],
                'agent0/dist_to_goal': [0.03768003822249384],
                'agent0/num_agent_collisions': [1.0],
                'agent0/num_obstacle_collisions': [0.0],
                'agent1/individual_rewards': [5],
                'agent1/time_to_goal': [0.6000000000000001],
                'agent1/min_time_to_goal': [0.3067362645187025],
                'agent1/dist_to_goal': [0.0387233764393595],
                'agent1/num_agent_collisions': [1.0],
                'agent1/num_obstacle_collisions': [0.0]}
        """
        fracs = []
        success = []
        for k, v in env_infos.items():
            if "time_to_goal" in k and "min_time_to_goal" not in k:
                fracs.append(v[0] / (self.all_args.episode_length * self.dt))
                # if didn't reach goal then time_to_goal >= episode_len * dt
                if v[0] < self.all_args.episode_length * self.dt:
                    success.append(1)
                else:
                    success.append(0)
        assert len(success) == self.all_args.num_agents
        if sum(success) == self.all_args.num_agents:
            success = True
        else:
            success = False

        return fracs, success
