"""
The file of Ppo class.
"""

import torch.optim as optim
import numpy as np
import torch
from torch.distributions import Categorical
from parameters import (
    LR_ACTOR,
    LR_CRITIC,
    GAMMA,
    LAMBDA,
    BATCH_SIZE,
    EPSILON,
    L2_RATE,
    BUFFER_SIZE,
    BATCH_NUM,
    ENTROPY_WEIGHT,
    DELAY,
)
from math import log


class Ppo:
    """
    The class which Proximal Policy Optimization is implemented in.
    """

    def __init__(self, s_dim, a_dim, device):
        self.device = device
        self.s_dim = s_dim

    def models_to_device(self, device):
        """
        Move actor and critic to the specific device('cpu' or 'cuda:x')
        """
        self.actor_net.to(device)
        self.critic_net.to(device)

    def choose_action(self, s, agent_id):
        """
        Convert state to preprocessed tensor,
        and then return action probability by actor_net
        """
        s_ts1, s_ts2 = self.state_preprocessor(s, agent_id)
        s_ts1 = torch.from_numpy(s_ts1).unsqueeze(0).to(self.device)
        s_ts2 = torch.from_numpy(s_ts2).unsqueeze(0).to(self.device)
        return self.actor_net.choose_action((s_ts1, s_ts2))

    def push_an_episode(self, data, agent_id):
        """
        Push an episode to replay buffer.

        Args:
            data: an array of (state, action, reward, mask, action_prob)
        Returns:
            None
        """

        s_lst, a_lst, r_lst, mask_lst, prob_lst = [], [], [], [], []
        for s, a, r, mask, prob in data:
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            mask_lst.append(mask)
            prob_lst.append((torch.Tensor(prob)))

        s_lst1 = []
        s_lst2 = []
        for s in s_lst:
            s1, s2 = self.state_preprocessor(s, agent_id)
            s_lst1.append(s1)
            s_lst2.append(s2)

        s_ts1 = torch.Tensor(np.array(s_lst1, dtype=np.float32)).to(self.device)
        s_ts2 = torch.Tensor(np.array(s_lst2, dtype=np.float32)).to(self.device)
        r_ts = torch.Tensor(np.array(r_lst, dtype=np.float32))
        masks = torch.Tensor(np.array(mask_lst, dtype=np.float32))

        with torch.no_grad():
            self.critic_net.eval()
            v_lst = []
            for idx in range(0, len(s_ts1), BATCH_SIZE):
                idx_end = min(idx + BATCH_SIZE, len(s_ts1))
                v_lst.append(
                    self.critic_net((s_ts1[idx:idx_end], s_ts2[idx:idx_end])).cpu()
                )
            v_ts = torch.concatenate(v_lst, dim=0)
            ret_ts, adv_ts = self.get_gae(r_ts, masks, v_ts)

        for idx, _ in enumerate(s_ts1):
            self.buffer.push(
                (
                    s_lst[idx],
                    a_lst[idx],
                    adv_ts[idx].item(),
                    ret_ts[idx].item(),
                    prob_lst[idx],
                    agent_id,
                )
            )

    def train(self):
        """
        Train Actor network and Critic network with data in buffer.
        """
        print("buffer size: ", self.buffer.size())

        self.actor_net.train()
        self.critic_net.train()
        critic_loss_lst, actor_loss_lst = [], []
        for batch_id in range(BATCH_NUM):

            s_lst, a_lst, adv_lst, ret_lst, op_lst, id_lst = self.buffer.pull(
                BATCH_SIZE
            )

            s_lst1 = []
            s_lst2 = []
            for s, agent_id in zip(s_lst, id_lst):
                s1, s2 = self.state_preprocessor(s, agent_id)
                s_lst1.append(s1)
                s_lst2.append(s2)

            st_ts1 = torch.Tensor(np.stack(s_lst1, axis=0)).to(self.device)
            st_ts2 = torch.Tensor(np.stack(s_lst2, axis=0)).to(self.device)
            a_ts = torch.LongTensor(a_lst).to(self.device).unsqueeze(1)
            
            # adv for PPO
            adv_ts = (
                torch.Tensor(np.array(adv_lst, dtype=np.float32))
                .unsqueeze(1)
                .to(self.device)
            )
            
            ret_ts = (
                torch.Tensor(np.array(ret_lst, dtype=np.float32))
                .unsqueeze(1)
                .to(self.device)
            )
            op_ts = torch.Tensor(np.stack(op_lst, axis=0)).to(self.device)

            v_ts = self.critic_net((st_ts1, st_ts2))
            critic_loss = self.critic_loss_func(v_ts, ret_ts)
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            new_probs_ts = self.actor_net((st_ts1, st_ts2))
            np_ts = torch.softmax(new_probs_ts, dim=1)

            entropy_loss = Categorical(np_ts).entropy().mean()
            op_ts = op_ts.gather(1, a_ts)
            np_ts = np_ts.gather(1, a_ts)
            
            ############## adv for BPPO ##################
            # old_dist = np_ts
            # a = old_dist.rsample()
            # adv_ts = self.actor_net((s, a)) - self.critic_net((s,None))
            # adv_ts = (adv_ts - adv_ts.mean()) / (adv_ts.std() + 1e-10)
            ##############################################

            ratio = torch.exp(torch.log(np_ts) - torch.log(op_ts))
            surrogate_loss = ratio * adv_ts

            ratio = torch.clamp(ratio, 1.0 - EPSILON, 1.0 + EPSILON)
            clipped_loss = ratio * adv_ts

            actor_loss = -torch.min(surrogate_loss, clipped_loss).mean()
            actor_loss -= ENTROPY_WEIGHT * entropy_loss
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            actor_loss_lst.append(actor_loss.item())
            critic_loss_lst.append(critic_loss.item())
            if (batch_id + 1) % 15 == 0:
                print(
                    f"critic loss: {sum(critic_loss_lst)/len(critic_loss_lst):.8f}",
                    f"\t\tactor loss: {sum(actor_loss_lst)/len(actor_loss_lst):.8f}",
                )
                actor_loss_lst.clear()
                critic_loss_lst.clear()

    def get_gae(self, rewards, masks, values):
        """
        Calculate Generalized Advantage Estimation.
        """
        rewards = torch.Tensor(rewards)
        masks = torch.Tensor(masks)
        returns = torch.zeros_like(rewards)
        advants = torch.zeros_like(rewards)
        previous_value = 0
        previous_reward = 0
        running_advants = 0

        for t in reversed(range(0, len(rewards))):
            running_tderror = (
                rewards[t] + GAMMA * previous_value * masks[t] - values.data[t]
            )
            running_advants = (
                running_tderror + GAMMA * LAMBDA * running_advants * masks[t]
            )

            returns[t] = rewards[t] + GAMMA * previous_reward * masks[t]
            previous_value = values.data[t]
            previous_reward = rewards.data[t]
            advants[t] = running_advants
        # advants = (advants - advants.mean()) / advants.std()
        return returns, advants

    def state_preprocessor(self, s, agent_id, opponent_id=None):
        
        if opponent_id == None:
            opponent_id = 3 - agent_id

        gamestate, previous_actions = s

        p1 = gamestate.players[agent_id]
        p2 = gamestate.players[opponent_id]

        state1 = np.zeros((self.s_dim,), dtype=np.float32)

        state1[0] = p1.position.x
        state1[1] = p1.position.y
        state1[2] = p2.position.x
        state1[3] = p2.position.y
        state1[4] = p1.position.x - p2.position.x
        state1[5] = p1.position.y - p2.position.y
        state1[6] = 1.0 if p1.facing else -1.0
        state1[7] = 1.0 if p2.facing else -1.0
        state1[8] = 1.0 if (p1.position.x - p2.position.x) * state1[6] < 0 else -1.0
        state1[9] = log(abs(p1.position.x - p2.position.x) + 1)
        state1[10] = log(abs(p1.position.y - p2.position.y) + 1)
        state1[11] = p1.hitstun_frames_left
        state1[12] = p2.hitstun_frames_left
        state1[13] = p1.invulnerability_left
        state1[14] = p2.invulnerability_left
        state1[15] = p1.jumps_left
        state1[16] = p2.jumps_left
        state1[17] = p1.off_stage * 1.0
        state1[18] = p2.off_stage * 1.0
        state1[19] = p1.on_ground * 1.0
        state1[20] = p2.on_ground * 1.0
        state1[21] = p1.percent
        state1[22] = p2.percent
        state1[23] = p1.shield_strength
        state1[24] = p2.shield_strength
        state1[25] = p1.speed_air_x_self
        state1[26] = p2.speed_air_x_self
        state1[27] = p1.speed_ground_x_self
        state1[28] = p2.speed_ground_x_self
        state1[29] = p1.speed_x_attack
        state1[30] = p2.speed_x_attack
        state1[31] = p1.speed_y_attack
        state1[32] = p2.speed_y_attack
        state1[33] = p1.speed_y_self
        state1[34] = p2.speed_y_self
        state1[35] = p1.action_frame
        state1[36] = p2.action_frame
        if p1.action.value < 386:
            state1[37 + p1.action.value] = 1.0
        if p2.action.value < 386:
            state1[37 + 386 + p2.action.value] = 1.0

        state2 = np.zeros((1,), dtype=np.float32)

        return (state1, state2)

