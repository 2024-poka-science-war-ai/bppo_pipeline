import melee
import numpy as np
from abc import ABC, abstractmethod
from melee_env.agents.util import *
from melee import enums
import MovesList
from melee_env.agents.util import ObservationSpace, ActionSpace, from_action_space

import torch
import torch.nn.functional as F

from math import log
import random
from melee.stages import EDGE_POSITION
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Agent(ABC):
    def __init__(self):
        self.agent_type = "AI"
        self.controller = None
        self.port = None  # this is also in controller, maybe redundant?
        self.action = 0
        self.press_start = False
        self.self_observation = None
        self.current_frame = 0

    @abstractmethod
    def act(self):
        pass

class nnAgent(Agent):
    def __init__(self, state_dim, act_dim, model, obs_space, device):
        super().__init__()
        self.device = device
        self._s_dim = state_dim
        self._a_dim = act_dim
        self.net = model
        self.character = enums.Character.FOX

        self.action_space = MyActionSpace()
        self.observation_space = obs_space
        self.action = 0
        self.states = np.zeros(self._a_dim)
        self.actions = np.zeros(self._s_dim)
        self.action_q = []
        self.action_q_idx = 0
        self.test_mode = True
        self.agent_id = 1
    
    def act(self, s):

        act_data = None

        if self.action_q_idx >= len(self.action_q):
            # the agent should select action
            self.action_q_idx = 0

            dist = self.net(torch.tensor(self.state_preprocessor(s, 1, 2)[0]).to(self.device))
            assert isinstance(dist, torch.distributions.Normal)
            action_prob_np = dist.sample().clamp(-1., 1.)

            if self.test_mode:
                # choose the most probable action
                final_weights = self.neglect_invalid_actions(s[0], action_prob_np)
                a = torch.argmax(final_weights).item()
            else:
                # choose an action with probability weights
                # max_weight = np.max(action_prob_np)
                # exp_weights = np.exp((action_prob_np - max_weight) / TAU)
                # exp_weights = self.neglect_invalid_actions(s[0], exp_weights)
                # final_weights = exp_weights / np.sum(exp_weights)
                # a = random.choices(
                #     list(range(self.a_dim)), weights=final_weights, k=1)[0]
                final_weights = np.empty_like(action_prob_np)
                final_weights[:] = action_prob_np
                final_weights = self.neglect_invalid_actions(s[0], final_weights)
                final_weights = final_weights / np.sum(final_weights)
                a = random.choices(list(range(self.a_dim)), weights=final_weights, k=1)[
                    0
                ]
            self.action_q = self.action_space.high_action_space[a]
            act_data = (a, action_prob_np)

        now_action = self.action_q[self.action_q_idx]
        self.action_q_idx += 1

        return now_action, act_data
    
    def state_preprocessor(self, s, agent_id, opponent_id=None):
        
        if opponent_id == None:
            opponent_id = 3 - agent_id

        gamestate, previous_actions = s

        p1 = gamestate.players[agent_id]
        p2 = gamestate.players[opponent_id]

        state1 = np.zeros((self._s_dim,), dtype=np.float32)

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
    
    def neglect_invalid_actions(self, s, action_prob_np):
        """
        Prevent invalid/unsafe actions
        """
        p1 = s.players[self.agent_id]
        edge = EDGE_POSITION.get(s.stage)
        if p1.jumps_left == 0:
            # if already double-jumped, prevent jumping
            action_prob_np[2:8] = 0.0
        if p1.action in [
            enums.Action.SWORD_DANCE_1,
            enums.Action.SWORD_DANCE_1_AIR,
            enums.Action.SWORD_DANCE_2_HIGH,
            enums.Action.SWORD_DANCE_2_HIGH_AIR,
            enums.Action.SWORD_DANCE_2_MID,
            enums.Action.SWORD_DANCE_2_MID_AIR,
            enums.Action.SWORD_DANCE_3_LOW,
            enums.Action.SWORD_DANCE_3_MID,
            enums.Action.SWORD_DANCE_3_HIGH,
            enums.Action.SWORD_DANCE_3_LOW_AIR,
            enums.Action.SWORD_DANCE_3_MID_AIR,
            enums.Action.SWORD_DANCE_3_HIGH_AIR,
            enums.Action.SWORD_DANCE_4_LOW,
            enums.Action.SWORD_DANCE_4_MID,
            enums.Action.SWORD_DANCE_4_HIGH,
        ]:
            # if currently firefoxing, only tilting stick possible
            action_prob_np[:30] = 0.0
            if p1.position.x > 0:
                # prevent suicide
                action_prob_np[30] = 0.0
                action_prob_np[34] = 0.0
                action_prob_np[36] = 0.0
            else:
                # prevent suicide
                action_prob_np[31] = 0.0
                action_prob_np[35] = 0.0
                action_prob_np[37] = 0.0
        elif p1.action in [
            enums.Action.GRAB,
            enums.Action.GRAB_PULL,
            enums.Action.GRAB_PULLING,
            enums.Action.GRAB_PULLING_HIGH,
            enums.Action.GRAB_PUMMEL,
            enums.Action.GRAB_WAIT,
        ]:
            # if grabbing opponent, only hitting or throwing possible
            action_prob_np[3:8] = 0.0
            action_prob_np[9:27] = 0.0
            action_prob_np[28:] = 0.0
        else:
            # if currently not firefoxing or grabbing,
            # some tilting actions are useless
            action_prob_np[27:] = 0.0
        if p1.action in [
            enums.Action.EDGE_CATCHING,
            enums.Action.EDGE_HANGING,
            enums.Action.EDGE_TEETERING,
            enums.Action.EDGE_TEETERING_START,
        ]:
            # if grabbing edge now,
            # only jumping / rolling / moving possible
            action_prob_np[3] = 0.0
            action_prob_np[6:8] = 0.0
            action_prob_np[9:24] = 0.0
            action_prob_np[26:] = 0.0
            if p1.facing:
                # prevent suicide
                action_prob_np[1] = 0.0
                action_prob_np[25] = 0.0
            else:
                # prevent suicide
                action_prob_np[0] = 0.0
                action_prob_np[24] = 0.0
        if p1.action in [enums.Action.LYING_GROUND_DOWN, enums.Action.TECH_MISS_DOWN]:
            # when lying down,
            # only jumping / jab / rolling possible
            action_prob_np[0:2] = 0.0
            action_prob_np[3:8] = 0.0
            action_prob_np[9:24] = 0.0
            action_prob_np[26:] = 0.0
        if not p1.on_ground:
            # prevent impossible actions when in the air
            action_prob_np[3] = 0.0
            action_prob_np[6:8] = 0.0
            action_prob_np[13:17] = 0.0
            action_prob_np[22:26] = 0.0
        if p1.facing:
            # weak jab only possible in facing direction
            action_prob_np[14] = 0.0
        else:
            # weak jab only possible in facing direction
            action_prob_np[13] = 0.0
        if p1.position.x > 0:
            # prevent suicide
            action_prob_np[18] = 0.0
        if p1.position.x < 0:
            # prevent suicide
            action_prob_np[19] = 0.0
        if p1.position.x < -edge + 5:
            # prevent suicide
            action_prob_np[1] = 0.0
            action_prob_np[31] = 0.0
        if p1.position.x > edge - 5:
            # prevent suicide
            action_prob_np[0] = 0.0
            action_prob_np[30] = 0.0
        if p1.position.y < -10:
            # emergency!!! only firefoxing allowed
            if p1.action not in [
                enums.Action.EDGE_CATCHING,
                enums.Action.EDGE_HANGING,
                enums.Action.EDGE_TEETERING,
                enums.Action.EDGE_TEETERING_START,
            ]:
                action_prob_np[:20] = 0.0
                if p1.position.x < -edge - 10:
                    action_prob_np[21:34] = 0.0
                    action_prob_np[35:] = 0.0
                elif p1.position.x > edge + 10:
                    action_prob_np[21:35] = 0.0
                    action_prob_np[36:] = 0.0
                else:
                    action_prob_np[21:32] = 0.0
                    action_prob_np[33:] = 0.0

        return action_prob_np
