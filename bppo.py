import torch
import numpy as np
from buffer import OnlineReplayBuffer
from net import GaussPolicyMLP
from critic import ValueLearner, QLearner
from ppo import ProximalPolicyOptimization
from utils import CONST_EPS, log_prob_func, orthogonal_initWeights
from melee import enums
from melee_env.myenv import MeleeEnv
from melee_env.agents.basic import *
from Bot import nnAgent

class BehaviorCloning:
    _device: torch.device
    _policy: GaussPolicyMLP
    _optimizer: torch.optim
    _policy_lr: float
    _batch_size: int
    def __init__(
        self,
        device: torch.device,
        state_dim: int,
        hidden_dim: int, 
        depth: int,
        action_dim: int,
        policy_lr: float,
        batch_size: int
    ) -> None:
        super().__init__()
        self._s_dim = state_dim
        self._a_dim = action_dim
        self._device = device
        self._policy = GaussPolicyMLP(state_dim, hidden_dim, depth, action_dim).to(device)
        orthogonal_initWeights(self._policy)
        self._optimizer = torch.optim.Adam(
            self._policy.parameters(),
            lr = policy_lr
        )
        self._lr = policy_lr
        self._batch_size = batch_size
        

    def loss(
        self, replay_buffer: OnlineReplayBuffer,
    ) -> torch.Tensor:
        s, a, _, _, _, _, _, _ = replay_buffer.sample(self._batch_size)
        dist = self._policy(s)
        log_prob = log_prob_func(dist, a) 
        loss = (-log_prob).mean()

        return loss


    def update(
        self, replay_buffer: OnlineReplayBuffer,
        ) -> float:
        policy_loss = self.loss(replay_buffer)

        self._optimizer.zero_grad()
        policy_loss.backward()
        self._optimizer.step()

        return policy_loss.item()


    def select_action(
        self, s: torch.Tensor, is_sample: bool
    ) -> torch.Tensor:
        dist = self._policy(s)
        if is_sample:
            action = dist.sample()
        else:    
            action = dist.mean
        # clip 
        action = action.clamp(-1., 1.)
        return action


    def offline_evaluate(
        self,
        env_name: str,
        seed: int,
        mean: np.ndarray,
        std: np.ndarray,
        eval_episodes: int=10
        ) -> float:
        self._policy.eval()
        
        observation_space = ObservationSpace()
        action_space = MyActionSpace()

        device = "cuda" if torch.cuda.is_available() else "cpu"

        agent = nnAgent(state_dim=self._s_dim, act_dim=self._a_dim, model=self._policy, obs_space=observation_space, device=device)

        players = [agent, CPU(enums.Character.FOX, 9)] #CPU(enums.Character.KIRBY, 5)]
        
        total_reward = 0
        for _ in range(eval_episodes):
            env = MeleeEnv("C:/Users/justi/Desktop/pk_ai/ssb/sl/ssbm-bot/ssbm.iso", players, fast_forward=True, ai_starts_game=True)
            env.start()
            now_s, done = env.reset(enums.Stage.FINAL_DESTINATION)
            action_pair = [0, 0]
            while not done:
                now_action, act_data = players[0].act(now_s)
                action_pair[0] = now_action
                action_pair[1] = players[1].act(now_s[0])
                now_s, r, done, _ = env.step(*action_pair)
                total_reward += r[0]
            env.close()
        
        self._policy.train()
        
        avg_reward = total_reward / eval_episodes
        return avg_reward

    def save(
        self, path: str
    ) -> None:
        torch.save(self._policy.state_dict(), path)
        print('Behavior policy parameters saved in {}'.format(path))
    

    def load(
        self, path: str
    ) -> None:
        self._policy.load_state_dict(torch.load(path, map_location=self._device))
        print('Behavior policy parameters loaded')



class BehaviorProximalPolicyOptimization(ProximalPolicyOptimization):

    def __init__(
        self,
        device: torch.device,
        state_dim: int,
        hidden_dim: int, 
        depth: int,
        action_dim: int,
        policy_lr: float,
        clip_ratio: float,
        entropy_weight: float,
        decay: float,
        omega: float,
        batch_size: int
    ) -> None:
        super().__init__(
            device = device,
            state_dim = state_dim,
            hidden_dim = hidden_dim,
            depth = depth,
            action_dim = action_dim,
            policy_lr = policy_lr,
            clip_ratio = clip_ratio,
            entropy_weight = entropy_weight,
            decay = decay,
            omega = omega,
            batch_size = batch_size)


    def loss(
        self, 
        replay_buffer: OnlineReplayBuffer,
        Q: QLearner,
        value: ValueLearner,
        is_clip_decay: bool,
    ) -> torch.Tensor:
        # -------------------------------------Advantage-------------------------------------
        s, _, _, _, _, _, _, _ = replay_buffer.sample(self._batch_size)
        s=s.squeeze(1)
        old_dist = self._old_policy(s)
        a = old_dist.rsample()
        advantage = Q(s, a) - value(s)
        advantage = (advantage - advantage.mean()) / (advantage.std() + CONST_EPS)
        # -------------------------------------Advantage-------------------------------------
        new_dist = self._policy(s)

        new_log_prob = log_prob_func(new_dist, a)
        old_log_prob = log_prob_func(old_dist, a)
        ratio = (new_log_prob - old_log_prob).exp()
        
        advantage = self.weighted_advantage(advantage)

        loss1 =  ratio * advantage 

        if is_clip_decay:
            self._clip_ratio = self._clip_ratio * self._decay
        else:
            self._clip_ratio = self._clip_ratio

        loss2 = torch.clamp(ratio, 1 - self._clip_ratio, 1 + self._clip_ratio) * advantage 
        
        entropy_loss = new_dist.entropy().sum(-1, keepdim=True) * self._entropy_weight
        
        loss = -(torch.min(loss1, loss2) + entropy_loss).mean()

        return loss


    def offline_evaluate(
        self,
        env_name: str,
        seed: int,
        mean: np.ndarray,
        std: np.ndarray,
        eval_episodes: int=10
        ) -> float:
        self._policy.eval()
        
        observation_space = ObservationSpace()
        action_space = MyActionSpace()

        device = "cuda" if torch.cuda.is_available() else "cpu"

        agent = nnAgent(state_dim=self._s_dim, act_dim=self._a_dim, model=self._policy, obs_space=observation_space, device=device)

        players = [agent, CPU(enums.Character.FOX, 9)] #CPU(enums.Character.KIRBY, 5)]
        
        total_reward = 0
        for _ in range(eval_episodes):
            env = MeleeEnv("C:/Users/justi/Desktop/pk_ai/ssb/sl/ssbm-bot/ssbm.iso", players, fast_forward=True, ai_starts_game=True)
            env.start()
            now_s, done = env.reset(enums.Stage.FINAL_DESTINATION)
            action_pair = [0, 0]
            while not done:
                now_action, act_data = players[0].act(now_s)
                action_pair[0] = now_action
                action_pair[1] = players[1].act(now_s[0])
                now_s, r, done, _ = env.step(*action_pair)
                total_reward += r[0]
            env.close()
        
        self._policy.train()
        
        avg_reward = total_reward / eval_episodes
        return avg_reward
