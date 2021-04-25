import logging
import random
import numpy as np

from loderunnerclient.internals.actions import LoderunnerAction, get_action_by_num
from loderunnerclient.internals.board import Board
from loderunnerclient.game_client import GameClient
from loderunnerclient.internals.element import index_to_char, char_to_index
from loderunnerclient.internals.constants import ElementsCount

from rnd import Agent

import traceback

# TODO: Move to util
def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
      input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
      num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((num_classes, n), dtype=dtype)
    categorical[y, np.arange(n)] = 1
    output_shape = (num_classes,) + input_shape
    categorical = np.reshape(categorical, output_shape)
    return categorical

class Environment:
    def __init__(self):
        ### Параметры нейросети ###
        ## Это можно вынести в аргументы конструктора, но к чему эти выебоны?

        self.window_size         = (15, 15)
        
        self.n_step_update       = 32 # How many steps before you update the RND. Recommended set to 128 for Discrete
        self.n_eps_update        = 5 # How many episode before you update the PPO. Recommended set to 5 for Discrete
        self.n_init_episode      = 256
        self.n_saved             = 100 # How many episode to run before saving the weights
        
        self.policy_kl_range     = 0.0008 # Recommended set to 0.0008 for Discrete
        self.policy_params       = 20 # Recommended set to 20 for Discrete
        self.value_clip          = 1.0 # How many value will be clipped. Recommended set to the highest or lowest possible reward
        self.entropy_coef        = 0.05 # How much randomness of action you will get
        self.vf_loss_coef        = 1.0 # Just set to 1
        self.minibatch           = 4 # How many batch per update. size of batch = n_update / minibatch. Recommended set to 4 for Discrete
        self.PPO_epochs          = 10 # How many epoch per update. Recommended set to 10 for Discrete
        
        self.gamma               = 0.99 # Just set to 0.99
        self.lam                 = 0.95 # Just set to 0.95
        self.learning_rate       = 2.5e-4 # Just set to 0.95
        
        self.agent_weights_path = "agent_weights" # Путь, куда сохранять модель

        ### Динамические параметры нейросети ###
        ## Они могут меняться вместе с изменением окружения
        
        self.turn_num          = 0 # Номер текущего хода
        self.start_new_episode = False # В начале нового эпизода мы корректируем поведение нейросети на основе накопленного опыта
        self.last_state = None
        self.episode_num = 0
        self.count_undone_turns = 0

        ### Параметры окружения ###

        self.memory_size = 2 # Как много предыдущих состояний сохраняется в окружении
        self.suicide_punishment = 1000 # Наказывать нейросеть за совершение суицида (чем больше значение, тем сильнее наказание)

        ### Динамические параметры окружения ###
    
        self.boards = []
        self.actions = []
        self.agent = Agent(
            14,
            len(LoderunnerAction) - 3,
            self.policy_kl_range,
            self.policy_params,
            self.value_clip,
            self.entropy_coef,
            self.vf_loss_coef,
            self.minibatch,
            self.PPO_epochs,
            self.gamma,
            self.lam,
            self.learning_rate,
            self.agent_weights_path
        )
    
        # TODO: move to game logic class
        self.current_golds_strick = {
            "YELLOW_GOLD": 0,
            "GREEN_GOLD": 0,
            "RED_GOLD": 0
        }

    def __get_reward__(self):
        if len(self.boards) < 2:
            return 0

        def calc_gold_reward(gold_pos, my_pos):
            """
            Вычисляет вознаграждение как сумму экспонент расстояний до всех
            мешочков золота.
            """
            reward = 0
            x, y = my_pos.get_x(), my_pos.get_y()
            for pos in gold_pos:
                distance = np.sqrt((x - pos.get_x())**2 + (y - pos.get_y())**2)
                exp = np.exp((10 - distance) / 2)
                reward += exp
            return reward
        
        prev_board = self.boards[-2]
        current_board = self.boards[-1]

        my_pos = current_board.get_my_position()
        my_prev_pos = prev_board.get_my_position()

        prev_gold_positions = prev_board.get_gold_positions()
        gold_positions = current_board.get_gold_positions()
        
        prev_gold_reward = calc_gold_reward(prev_gold_positions, my_prev_pos)
        gold_reward = calc_gold_reward(gold_positions, my_pos)

        reward = gold_reward - prev_gold_reward
        reward -= (reward == 0) * self.count_undone_turns
        if len(self.actions) > 0:
            reward -= 1000 * (current_board.is_game_over() or self.actions[-1] == LoderunnerAction.SUICIDE)
        return reward

    def __gold_was_taken__(self):
        if len(self.boards) < 2:
            return 0

        prev_board = self.boards[-2]
        current_board = self.boards[-1]

        my_pos = current_board.get_my_position()
        x, y = my_pos.get_x(), my_pos.get_y()
        prev_element_name = prev_board.get_at(x, y).get_name()
        if (prev_element_name == "YELLOW_GOLD" or 
            prev_element_name == "GREEN_GOLD" or 
            prev_element_name == "RED_GOLD"
        ):
            return True
        return False


    def __get_current_nn_state__(self):
        """
        Вырезает из текущего состояние небольшую подмарцу размера self.window_size
        вокруг текущего местоположения героя

        Returns
        -------
        np.array
            Двумерная матрица размера self.window_size

        """
        x_shift = self.window_size[0] // 2
        y_shift = self.window_size[1] // 2

        board = self.boards[-1]
        table = board.get_index_table((y_shift, x_shift))
        my_pos = board.get_my_position()
        x, y = my_pos.get_x(), my_pos.get_y()
        return table[y : y + self.window_size[0], x : x + self.window_size[1]]

    def __on_turn_start__(self, board):
        """
        Происходит сразу после поступление очередного запроса с сервера.
        Тут случается всякая инициализация параметров.

        Returns
        -------
        None.

        """
        self.boards.append(board)
        
    def __on_turn_end__(self, action):
        """
        Происходит непосредственно перед отправкой очередного действия на сервер.
        Тут происходит очистка всякого мусора.

        Returns
        -------
        None.

        """
        self.actions.append(action)
        if len(self.actions) > self.memory_size:
            del self.actions[0]

        if len(self.boards) > self.memory_size:
            del self.boards[0]
        
    def __on_turn__(self, reward):
        if self.turn_num % 100 == 0:
            print("\r", self.turn_num, "      ")
        # Получаем вознаграждение за совершенное на предыдущем этапе действие
        # reward = self.__get_reward__()
        # print("CALIMED REWARD: ", reward)
        done = self.__gold_was_taken__()
        # print("DONE:", done)

        # Если мы умерли - то пора учится на своих ошибках
        start_new_episode = (
            len(self.actions) == 0
            or done
            # or self.boards[-1].is_game_over()
            # or self.actions[-1] == LoderunnerAction.SUICIDE
        )

        # Вычисляем состояние, которое нейронная сможет скушать
        state = self.__get_current_nn_state__()
        state = to_categorical(state, num_classes=14)
        # Сейчас в next_state находится чанк размера (1, 53, self.window_size[0], self.window_size[1])
        # 53 - это количество возможных полей в игре

        if not self.last_state is None:
            if start_new_episode:
                # print("NEW EPISODE BEGAN!")
                self.episode_num += 1
                if self.episode_num % self.n_eps_update == 0:
                    self.agent.update_ppo()
                self.agent.save_weights()
                self.count_undone_turns = 0
    
            self.agent.save_eps(
                self.last_state.tolist(),
                float(self.actions[-1].num),
                float(reward),
                float(done),
                state.tolist()
            )
            self.agent.save_observation(state)
            
            if self.turn_num % self.n_step_update == 0:
                self.agent.update_rnd()

        self.last_state = state
        return get_action_by_num(int(self.agent.act(state)))
    
    def on_turn(self, board):
        # user_input = input()
        # if user_input == "w":
        #     return LoderunnerAction.GO_UP
        # if user_input == "s":
        #     return LoderunnerAction.GO_DOWN
        # if user_input == "a":
        #     return LoderunnerAction.GO_LEFT
        # if user_input == "d":
        #     return LoderunnerAction.GO_RIGHT
        # if user_input == "q":
        #     return LoderunnerAction.DRILL_LEFT
        # if user_input == "e":
        #     return LoderunnerAction.DRILL_RIGHT
        try:
            # Такая грануляция только лишь для того, чтобы это удобно было читать
            self.turn_num += 1
            self.count_undone_turns += 1
            self.__on_turn_start__(board)
            # action = self.__on_turn__(last_reward)
            state = self.__get_current_nn_state__()
            state = to_categorical(state, num_classes=14)
            action = get_action_by_num(int(self.agent.act(state)))
            self.__on_turn_end__(action)
            if action == LoderunnerAction.SUICIDE:
                action = LoderunnerAction.DO_NOTHING
            # if self.count_undone_turns > 1000:
            #     self.count_undone_turns = 0
            #     action = LoderunnerAction.SUICIDE
            return action
        except Exception:
            traceback.print_exc()
            raise TypeError("Shit happened")
