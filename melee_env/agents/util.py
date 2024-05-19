import numpy as np
import melee
from melee.enums import Action
from collections import deque
from parameters import ACTION_DIM


class ObservationSpace:
    def __init__(self):
        self.previous_gamestate = None
        self.current_gamestate = None
        self.previous_actions = deque(maxlen=10)
        self.previous_actions.extend(((0, 0), (0, 0)) * 5)

    def __call__(self, gamestate, actions, p_port=None, op_port=None):
        
        reward = (0, 0)
        info = None
        self.current_gamestate = gamestate

        self.previous_actions.append((actions[0], actions[1]))

        if self.previous_gamestate is not None:
            p1_dmg = (
                self.current_gamestate.players[p_port].percent
                - self.previous_gamestate.players[p_port].percent
            )
            p1_shield_dmg = (
                self.previous_gamestate.players[p_port].shield_strength
                - self.current_gamestate.players[p_port].shield_strength
            ) / (self.current_gamestate.players[p_port].shield_strength + 1)
            p1_stock_loss = int(self.previous_gamestate.players[p_port].stock) - int(
                self.current_gamestate.players[p_port].stock
            )
            p2_dmg = (
                self.current_gamestate.players[op_port].percent
                - self.previous_gamestate.players[op_port].percent
            )
            p2_shield_dmg = (
                self.previous_gamestate.players[op_port].shield_strength
                - self.current_gamestate.players[op_port].shield_strength
            ) / (self.current_gamestate.players[op_port].shield_strength + 1)
            p2_stock_loss = int(self.previous_gamestate.players[op_port].stock) - int(
                self.current_gamestate.players[op_port].stock
            )

            p1_dmg = max(p1_dmg, 0)
            p2_dmg = max(p2_dmg, 0)
            if p1_stock_loss > 1:
                p1_stock_loss = 0
            if p2_stock_loss > 1:
                p2_stock_loss = 0
            p1_stock_loss = max(p1_stock_loss, 0)
            p2_stock_loss = max(p2_stock_loss, 0)
            p1_shield_dmg = max(p1_shield_dmg, 0)
            p2_shield_dmg = max(p2_shield_dmg, 0)

            w_dmg, w_shield, w_stock = 0.1, 0.3, 5
            p1_loss = (
                w_dmg * p1_dmg + w_shield * p1_shield_dmg + w_stock * p1_stock_loss
            )
            p2_loss = (
                w_dmg * p2_dmg + w_shield * p2_shield_dmg + w_stock * p2_stock_loss
            )

            reward = (p2_loss * 2 - p1_loss, p1_loss * 2 - p2_loss)
        else:
            reward = (0, 0)

        self.previous_gamestate = self.current_gamestate

        stocks = np.array(
            [gamestate.players[i].stock for i in list(gamestate.players.keys())]
        )
        done = not np.sum(stocks[np.argsort(stocks)][::-1][1:])

        return (
            (gamestate, np.array(self.previous_actions)),
            reward,
            done,
            info,
        )

    def reset(self):
        self.__init__()
        # print("observation space got reset!")


class ActionSpace:
    def __init__(self):
        mid = np.sqrt(2) / 2

        self.stick_space_reduced = np.array(
            [
                [0.0, 0.0],  # no op
                [0.0, 1.0],
                [mid, mid],
                [1.0, 0.0],
                [mid, -mid],
                [0.0, -1.0],
                [-mid, -mid],
                [-1.0, 0.0],
                [-mid, mid],
            ]
        )

        self.button_space_reduced = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

        # Action space size is total number of possible actions. In this case,
        #    is is all possible main stick positions * all c-stick positions *
        #    all the buttons. A normal controller has ~51040 possible main stick
        #    positions. Each trigger has 255 positions. The c-stick can be
        #    reduced to ~5 positions. Finally, if all buttons can be pressed
        #    in any combination, that results in 32 combinations. Not including
        #    the dpad or start button, that is 51040 * 5 * 255 * 2 * 32 which
        #    is a staggering 4.165 billion possible control states.

        # Given this, it is reasonable to reduce this. In the above class, the
        #    main stick has been reduced to the 8 cardinal positions plus the
        #    center (no-op). Only A, B, Z, and R are used, as these correspond
        #    to major in-game functions (attack, special, grab, shield). Every
        #    action can theoretically be performed with just these buttons. A
        #    final "button" is added for no-op.
        #
        #    Action space = 9 * 5 = 45 possible actions.
        self.action_space = np.zeros(
            (self.stick_space_reduced.shape[0] * self.button_space_reduced.shape[0], 3)
        )

        for button in self.button_space_reduced:
            self.action_space[int(button) * 9 : (int(button) + 1) * 9, :2] = (
                self.stick_space_reduced
            )
            self.action_space[int(button) * 9 : (int(button) + 1) * 9, 2] = button

        # self.action_space will look like this, where the first two columns
        #   represent the control stick's position, and the final column is the
        #   currently depressed button.

        # ACT  Left/Right    Up/Down      Button
        # ---  ------        ------       ------
        # 0   [ 0.        ,  0.        ,  0. (NO-OP)] Center  ---
        # 1   [ 0.        ,  1.        ,  0.        ] Up         |
        # 2   [ 0.70710678,  0.70710678,  0.        ] Up/Right   |
        # 3   [ 1.        ,  0.        ,  0.        ] Right      |
        # 4   [ 0.70710678, -0.70710678,  0.        ] Down/Right |- these repeat
        # 5   [ 0.        , -1.        ,  0.        ] Down       |
        # 6   [-0.70710678, -0.70710678,  0.        ] Down/Left  |
        # 7   [-1.        ,  0.        ,  0.        ] Left       |
        # 8   [-0.70710678,  0.70710678,  0.        ] Up/Left  ---
        # 9   [ 0.        ,  0.        ,  1. (A)    ]
        # 10  [ 0.        ,  1.        ,  1.        ]
        # 11  [ 0.70710678,  0.70710678,  1.        ]
        # 12  [ 1.        ,  0.        ,  1.        ]
        # 13  [ 0.70710678, -0.70710678,  1.        ]
        # 14  [ 0.        , -1.        ,  1.        ]
        # 15  [-0.70710678, -0.70710678,  1.        ]
        # 16  [-1.        ,  0.        ,  1.        ]
        # 17  [-0.70710678,  0.70710678,  1.        ]
        # 18  [ 0.        ,  0.        ,  2. (B)    ]
        # 19  [ 0.        ,  1.        ,  2.        ]
        # 20  [ 0.70710678,  0.70710678,  2.        ]
        # 21  [ 1.        ,  0.        ,  2.        ]
        # 22  [ 0.70710678, -0.70710678,  2.        ]
        # 23  [ 0.        , -1.        ,  2.        ]
        # 24  [-0.70710678, -0.70710678,  2.        ]
        # 25  [-1.        ,  0.        ,  2.        ]
        # 26  [-0.70710678,  0.70710678,  2.        ]
        # 27  [ 0.        ,  0.        ,  3. (Z)    ]
        # 28  [ 0.        ,  1.        ,  3.        ]
        # 29  [ 0.70710678,  0.70710678,  3.        ]
        # 30  [ 1.        ,  0.        ,  3.        ]
        # 31  [ 0.70710678, -0.70710678,  3.        ]
        # 32  [ 0.        , -1.        ,  3.        ]
        # 33  [-0.70710678, -0.70710678,  3.        ]
        # 34  [-1.        ,  0.        ,  3.        ]
        # 35  [-0.70710678,  0.70710678,  3.        ]
        # 36  [ 0.        ,  0.        ,  4. (R)    ]
        # 37  [ 0.        ,  1.        ,  4.        ]
        # 38  [ 0.70710678,  0.70710678,  4.        ]
        # 39  [ 1.        ,  0.        ,  4.        ]
        # 40  [ 0.70710678, -0.70710678,  4.        ]
        # 41  [ 0.        , -1.        ,  4.        ]
        # 42  [-0.70710678, -0.70710678,  4.        ]
        # 43  [-1.        ,  0.        ,  4.        ]
        # 45  [-0.70710678,  0.70710678,  4.        ]

        self.size = self.action_space.shape[0]

    def sample(self):
        return np.random.choice(self.size)

    def __call__(self, action):
        if action > self.size - 1:
            print(action)
            exit("Error: invalid action!")

        return ControlState(self.action_space[action])


class MyActionSpace:
    def __init__(self):

        mid = np.sqrt(2) / 2
        self.action_space = np.array(
            [
                [0, 0, 0],  # 0
                [1, 0, 0],  # 1
                [-1, 0, 0],  # 2
                [0, 1, 0],  # 3
                [0, -1, 0],  # 4
                [mid, mid, 0],  # 5
                [-mid, mid, 0],  # 6
                [mid, -mid, 0],  # 7
                [-mid, -mid, 0],  # 8
                [0, 0, 1],  # 9
                [1, 0, 1],  # 10
                [-1, 0, 1],  # 11
                [0, 1, 1],  # 12
                [0, -1, 1],  # 13
                [0.3, 0, 1],  # 14
                [-0.3, 0, 1],  # 15
                [0, 0.3, 1],  # 16
                [0, -0.3, 1],  # 17
                [0, 0, 2],  # 18
                [1, 0, 2],  # 19
                [-1, 0, 2],  # 20
                [0, 1, 2],  # 21
                [0, -1, 2],  # 22
                [0, 0, 3],  # 23
                [0, 0, 4],  # 24
                [1, 0, 4],  # 25
                [-1, 0, 4],  # 26
                [0, -1, 4],  # 27
            ],
            dtype=np.float32,
        )
        self.size = self.action_space.shape[0]

        self.high_action_space = [
            [1, 0],  # 0
            [2, 0],  # 1
            [3, 0],  # 2
            [3, 3, 3, 0],  # 3
            [5, 0],  # 4
            [6, 0],  # 5
            [5, 5, 5, 0],  # 6
            [6, 6, 6, 0],  # 7
            [9, 0],  # 8
            [10, 0],  # 9
            [11, 0],  # 10
            [12, 0],  # 11
            [13, 0],  # 12
            [14, 0],  # 13
            [15, 0],  # 14
            [16, 0],  # 15
            [17, 0],  # 16
            [18, 0],  # 17
            [19, 0],  # 18
            [20, 0],  # 19
            [21, 0],  # 20
            [22, 0],  # 21
            [23, 0],  # 22
            [24, 0],  # 23
            [0, 25, 25, 0],  # 24
            [0, 26, 26, 0],  # 25
            [27, 0],  # 26
            [4, 0],  # 27
            [7, 0],  # 28
            [8, 0],  # 29
            [1, 1],  # 30
            [2, 2],  # 31
            [3, 3],  # 32
            [4, 4],  # 33
            [5, 5],  # 34
            [6, 6],  # 35
            [7, 7],  # 36
            [8, 8],  # 37
        ]

        self.sensor = {
            Action.DASHING: [0, 1],
            Action.TURNING: [0, 1],
            Action.JUMPING_FORWARD: [2, 3, 4, 5, 6, 7],
            Action.JUMPING_ARIAL_FORWARD: [2, 3, 4, 5, 6, 7],
            Action.JUMPING_BACKWARD: [4, 5, 6, 7],
            Action.JUMPING_ARIAL_BACKWARD: [4, 5, 6, 7],
            Action.NEUTRAL_ATTACK_1: [8],
            Action.NEUTRAL_ATTACK_2: [8],
            Action.LOOPING_ATTACK_START: [8],
            # Action.LOOPING_ATTACK_MIDDLE: [8],
            # Action.LOOPING_ATTACK_END: [8],
            Action.NAIR: [8, 13, 14, 15, 16],
            # Action.NAIR_LANDING: [8],
            Action.DASH_ATTACK: [8],
            Action.FSMASH_MID: [9, 10],
            Action.FAIR: [9, 10],
            # Action.FAIR_LANDING: [9, 10],
            Action.BAIR: [9, 10],
            # Action.BAIR_LANDING: [9, 10],
            Action.UPSMASH: [11],
            Action.UAIR: [11, 2],
            # Action.UAIR_LANDING: [11, 2],
            Action.DOWNSMASH: [12],
            Action.DAIR: [12],
            # Action.DAIR_LANDING: [12],
            Action.FTILT_MID: [13, 14],
            Action.UPTILT: [15],
            Action.DOWNTILT: [16],
            # Action.CROUCH_END: [16],
            Action.LASER_GUN_PULL: [17],
            Action.NEUTRAL_B_CHARGING: [17],
            # Action.NEUTRAL_B_ATTACKING: [17],
            Action.NEUTRAL_B_FULL_CHARGE: [17],
            # Action.WAIT_ITEM: [17],
            Action.NEUTRAL_B_CHARGING_AIR: [17],
            Action.NEUTRAL_B_ATTACKING_AIR: [18, 19],
            # Action.NEUTRAL_B_FULL_CHARGE_AIR: [18, 19],
            # Action.SWORD_DANCE_1: [18, 19],
            Action.SWORD_DANCE_2_HIGH: [18, 19],
            Action.SWORD_DANCE_2_MID: [18, 19],
            Action.SWORD_DANCE_3_HIGH: [18, 19, 20],
            # Action.LANDING_SPECIAL: [18, 19],
            # Action.SWORD_DANCE_1_AIR: [20],
            # Action.SWORD_DANCE_2_HIGH_AIR: [20],
            Action.SWORD_DANCE_3_LOW: [20],
            Action.SWORD_DANCE_3_MID: [20],
            Action.SWORD_DANCE_3_LOW_AIR: [20],
            # Action.SWORD_DANCE_3_LOW_AIR: [20, 21],
            Action.SWORD_DANCE_3_MID_AIR: [20],
            Action.SWORD_DANCE_3_HIGH_AIR: [20],
            Action.SWORD_DANCE_4_LOW: [30, 31, 32, 33, 34, 35, 36, 37],
            Action.SWORD_DANCE_4_MID: [30, 31, 32, 33, 34, 35, 36, 37],
            Action.SWORD_DANCE_4_HIGH: [30, 31, 32, 33, 34, 35, 36, 37],
            Action.DOWN_B_GROUND_START: [21],
            # Action.DOWN_B_GROUND: [21],
            Action.DOWN_B_STUN: [21],
            # Action.DOWN_B_AIR: [21],
            # Action.SHINE_RELEASE_AIR: [21],
            Action.GRAB: [22],
            # Action.GRAB_PULLING: [22],
            # Action.GRAB_WAIT: [22],
            # Action.GRAB_BREAK: [22],
            Action.GRAB_RUNNING: [22],
            Action.GRAB_PUMMEL: [8],
            Action.THROW_FORWARD: [0, 1],
            Action.THROW_BACK: [0, 1],
            Action.THROW_UP: [2],
            Action.THROW_DOWN: [27],
            # Action.SHIELD_START: [23, 24, 25],
            Action.SHIELD_START: [23],
            # Action.SHIELD_STUN: [23],
            # Action.SHIELD_RELEASE: [23],
            Action.ROLL_FORWARD: [24, 25],
            Action.ROLL_BACKWARD: [24, 25],
            Action.SPOTDODGE: [26],
            Action.EDGE_JUMP_1_QUICK: [2, 4, 5],
            Action.EDGE_JUMP_2_QUICK: [2, 4, 5],
            Action.EDGE_JUMP_1_SLOW: [2, 4, 5],
            Action.EDGE_JUMP_2_SLOW: [2, 4, 5],
            Action.EDGE_ATTACK_QUICK: [8],
            Action.EDGE_ATTACK_SLOW: [8],
            Action.EDGE_GETUP_QUICK: [0, 1],
            Action.EDGE_GETUP_SLOW: [0, 1],
            Action.EDGE_ROLL_QUICK: [24, 25],
            Action.EDGE_ROLL_SLOW: [24, 25],
            Action.GETUP_ATTACK: [8],
            Action.NEUTRAL_GETUP: [2],
            Action.GROUND_ROLL_BACKWARD_DOWN: [24, 25],
            Action.GROUND_ROLL_FORWARD_DOWN: [24, 25],
        }

    def sample(self):
        return np.random.choice(self.size)

    def __call__(self, action):
        if action > self.size - 1:
            print(action)
            exit("Error: invalid action!")

        return ControlState(self.action_space[action])


class ControlState:
    def __init__(self, state):
        self.state = state
        self.buttons = [
            False,
            melee.enums.Button.BUTTON_A,
            melee.enums.Button.BUTTON_B,
            melee.enums.Button.BUTTON_Z,
            melee.enums.Button.BUTTON_R,
        ]

    def __call__(self, controller):
        controller.release_all()
        if self.state[2]:  # only press button if not no-op
            if self.state[2] != 4.0:  # special case for r shoulder
                controller.press_button(self.buttons[int(self.state[2])])
            else:
                controller.press_shoulder(melee.enums.Button.BUTTON_R, 1)

        controller.tilt_analog_unit(
            melee.enums.Button.BUTTON_MAIN, self.state[0], self.state[1]
        )


def from_observation_space(act):
    def get_observation(self, *args):
        gamestate = args[0]
        observation = self.observation_space(gamestate)
        return act(self, observation)

    return get_observation


def from_action_space(act):
    def get_action_encoding(self, *args):
        gamestate = args[0]
        action = act(self, gamestate)
        control = self.action_space(action)
        control(self.controller)
        return

    return get_action_encoding
