from melee_env.dconfig import DolphinConfig
import melee
from melee import enums
import sys
from melee_env.agents.util import ObservationSpace
import numpy as np
import psutil


class MeleeEnv:
    def __init__(self,
                 iso_path,
                 players,
                 fast_forward=False, 
                 blocking_input=True,
                 ai_starts_game=True):

        self.d = DolphinConfig()
        # self.d.set_ff(fast_forward)

        self.iso_path = iso_path
        self.players = players
        # inform other players of other players
        # for player in self.players:
        #     player.set_player_keys(len(self.players))

        # if len(self.players) == 2:
        #     self.d.set_center_p2_hud(True)
        # else:
        #     self.d.set_center_p2_hud(False)

        self.blocking_input = blocking_input
        self.ai_starts_game = ai_starts_game
        self.observation_space = ObservationSpace()

        self.gamestate = None
        self.console = None
        self.menu_control_agent = 0
        self.ai_press_start = ai_starts_game

    def start(self):
        if sys.platform == "linux":
            dolphin_home_path = str(self.d.slippi_home)+"/"
        elif sys.platform == "win32" or sys.platform == "win64":
            dolphin_home_path = None

        self.console = melee.Console(
            path=str(self.d.slippi_bin_path),
            dolphin_home_path=dolphin_home_path,
            blocking_input=self.blocking_input,
            tmp_home_directory=True)

        # print(self.console.dolphin_home_path)  # add to logging later
        # Configure Dolphin for the correct controller setup, add controllers
        human_detected = False

        for i, _ in enumerate(self.players):
            curr_player = self.players[i]
            if curr_player.agent_type == "HMN":
                self.d.set_controller_type(
                    i+1, enums.ControllerType.GCN_ADAPTER)
                curr_player.controller = melee.Controller(
                    console=self.console,
                    port=i+1,
                    type=melee.ControllerType.GCN_ADAPTER)
                curr_player.port = i+1
                human_detected = True
            elif curr_player.agent_type in ["AI", "CPU"]:
                self.d.set_controller_type(
                    i+1, enums.ControllerType.GCN_ADAPTER)
                curr_player.controller = melee.Controller(
                    console=self.console, port=i+1)
                self.menu_control_agent = i
                curr_player.port = i+1
            else:  # no player
                self.d.set_controller_type(i+1, enums.ControllerType.UNPLUGGED)

        # self.menu_control_agent = 0 # edited part

        if self.ai_starts_game and not human_detected:
            self.ai_press_start = True

        else:
            self.ai_press_start = False
            # don't let ai press start without the human player joining in.

        if self.ai_starts_game and self.ai_press_start:
            self.players[self.menu_control_agent].press_start = True

        self.console.run(iso_path=self.iso_path)
        self.console.connect()

        for player in self.players:
            if player is not None:
                player.controller.connect()

        self.gamestate = self.console.step()

    def step(self, *actions):
        for i, player in enumerate(self.players):
            if player.agent_type == "CPU":
                continue
            action = actions[i]
            control = player.action_space(action)
            control(player.controller)

        if self.gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
            self.gamestate = self.console.step()
        return self.observation_space(self.gamestate, actions, p_port=1, op_port=2)

    def reset(self, stage):
        self.observation_space.reset()
        for player in self.players:
            player.defeated = False

        while True:
            self.gamestate = self.console.step()
            if self.gamestate.menu_state is melee.Menu.CHARACTER_SELECT:
                for i, _ in enumerate(self.players):
                    if self.players[i].agent_type == "AI":
                        melee.MenuHelper.choose_character(
                            character=self.players[i].character,
                            gamestate=self.gamestate,
                            controller=self.players[i].controller,
                            costume=i,
                            swag=False,
                            start=self.players[i].press_start)
                    if self.players[i].agent_type == "CPU":
                        melee.MenuHelper.choose_character(
                            character=self.players[i].character,
                            gamestate=self.gamestate,
                            controller=self.players[i].controller,
                            costume=i,
                            swag=False,
                            cpu_level=self.players[i].lvl,
                            start=self.players[i].press_start)
            elif self.gamestate.menu_state is melee.Menu.STAGE_SELECT:
                # time.sleep(0.1)
                melee.MenuHelper.choose_stage(
                    stage=stage,
                    gamestate=self.gamestate,
                    controller=self.players[self.menu_control_agent].controller)

            elif self.gamestate.menu_state in [melee.Menu.IN_GAME,
                                               melee.Menu.SUDDEN_DEATH]:
                previous_actions = np.zeros((10, 2), dtype=np.float32)
                return (self.gamestate, previous_actions), False
                # game is not done on start

            else:
                melee.MenuHelper.choose_versus_mode(
                    self.gamestate,
                    self.players[self.menu_control_agent].controller)

    def close(self):
        for proc in psutil.process_iter():
            # print(proc.name)
            if proc.name() == "Slippi Dolphin.exe":
                parent_pid = proc.pid
                parent = psutil.Process(parent_pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
