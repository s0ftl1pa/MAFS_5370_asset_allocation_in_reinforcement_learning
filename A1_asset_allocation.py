# -*- coding: utf-8 -*-
# @author  : Yi Jiahe, Zhang Ruiyi
# @time    : 2025/03/07
# @file    : A1_asset_allocation.py

"""This script implements the process of asset allocating.

Totally three classes are included to perform two TD algorithm: SARSA and Q-Learning. Class Framework defines the
environment and shared methods for SARSA and Q-Learning. The other two, i.e. class SARSA and Q-Learning, are responsible
for the implementation of algorithm. The description in detail can be found in the docstring of each class.

    Important methods:

    Framework.generate_space() : Discretize the state and action spaces.
    SARSA.episode() : Perform one episode and renew the q_table
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm


class Framework:
    """This class defines the framework of asset allocation process, including the environment and basic shared methods.

    It should be noted that, it is a continuous states and continuous actions discrete-time finite-horizon MDP.
    Discretization and functional approximation (e.g. deep reinforcement learning) are two efficient ways to handle this problem.
    Here, we discretize the state and action spaces by the method generate_space().

    Attributes:
        time_steps : The terminal time T. Default: 10
        wealth_init : The total wealth at t=0. Default: 1
        rf : Single-step return of riskless asset. Default: 0.05
        (p_up, p_down, up, down) : The BN distribution of return of the risky asset. Default: (0.4, 0.6, 0.3, -0.2)
        aversion : risk aversion coefficient of CARA function. Default: 1
    """

    def __init__(self, time_steps=10, wealth_init=1.0, rf=0.05, dist=(0.4, 0.6, 0.3, -0.2), aversion=1.0):
        # Initialize the environment
        self.time_steps = time_steps

        self.wealth_init = wealth_init
        self.min_wealth = 0
        self.max_wealth = 0

        self.rf = rf
        self.p_up = dist[0]
        self.p_down = dist[1]
        self.up = dist[2]
        self.down = dist[3]

        self.num_state = 0
        self.num_action = 0
        self.state_space = {}
        self.action_space = {}

        self.aversion = aversion

    def generate_space(self, num_state=101, num_action=11, min_wealth=0, max_wealth=5):
        """Discretize the state and action space.

        For the state space, we transform the numerical variable, i.e. wealth, into an interval type variable.
        The process is controlled by three parameters: num_state, min_wealth, max_wealth.
        Example: Given num_state = 3, min_wealth = 0, max_wealth = 1, the state space will be {[0, 0.5), [0.5, 1), [1, inf)}.
        State 1 means the wealth is not lower than 0.5 and smaller than 1.

        For the action space, we consider the proportion of wealth allocated to the risky asset.
        The process is controlled by the parameter num_action.
        Example: Given num_action = 3, the action space will be {0, 0.5, 1}. And action 0 means that allocation zero to risky asset.

        :param num_state: A parameter to generate the state space, and the value represent all the possible state. Default: 101
        :param num_action: A parameter to generate the action space, and the value represent all the possible action. Default: 11
        :param min_wealth: The minimal possible wealth. Default: 0
        :param max_wealth: A preset upper bound of wealth. All wealth higher than max_wealth will be considered as the last state. Default: 5
        """
        state_space = np.linspace(min_wealth, max_wealth, num_state)
        self.state_space = state_space
        self.action_space = np.linspace(0, 1, num_action)
        self.num_state = num_state
        self.num_action = num_action
        self.min_wealth = min_wealth
        self.max_wealth = max_wealth

    def wealth_to_state(self, wealth):
        """For given amount of wealth, find the state it belongs to."""
        if wealth > self.max_wealth:
            # As long as wealth is higher than max_wealth, the state is set to be the last state.
            return self.num_state - 1
        else:
            return int(np.floor((wealth - self.min_wealth) * (self.num_state-1) / (self.max_wealth - self.min_wealth)))

    def crsp(self, wealth):
        return (-np.exp(-self.aversion * wealth))/self.aversion

    def reward(self, time, wealth):
        """Calculate the reward by the CARA utility function.
        :param time: Input time.
        :param wealth: The wealth in the input time.
        :return: Return 0 if the input time is not the terminal time; otherwise the value given by CARA.
        """
        if self.is_terminal(time):
            return self.crsp(wealth)
        else:
            # Reward is 0 except the terminal time.
            return 0.0

    def is_terminal(self, t):
        """
        Check if the time is the terminal time.
        :param t: Input time.
        :return: Return true if t is the terminal time; otherwise false.
        """
        return t >= self.time_steps

    # def optimal_policy(self):
    #     const = np.log((self.p_up * (self.up - self.rf)) / ((self.p_up - 1) * (self.down - self.rf))) / (self.aversion * (self.up - self.down))
    #     risky_wealth = np.zeros(self.time_steps + 1)
    #
    #     for i in range(self.time_steps + 1):
    #         risky_wealth[i] = const / ((1 + self.rf) ** (self.time_steps - i - 1))
    #     return risky_wealth


class SARSA:
    """This class performs the SARSA algorithm.

    Note that
    1) Two policy are adopted--The ordinary epsilon greedy method and the method with epsilon decay.
    2) We intentionally distinguish the difference between wealth and state among all methods.
    In fact, wealth represents the actual value, while state represent the state-th interval that wealth is in (same meaning for action).

    Attributes:
        framework : The instance of class Framework.
        epsilon_decay : Adopt epsilon_decay greedy policy if true; otherwise adopt epsilon greedy policy. Default: False
        epsilon : A parameter to measure the extent to explore (valid when epsilon_decay=False). Default: 0.1
        alpha : Learning rate. Default: 0.5
        epsilon_begin : The initial value of epsilon (valid when epsilon_decay=True). Default: 1
        epsilon_end : The final value of epsilon (valid when epsilon_decay=True). Default: 0.01
    """
    def __init__(self, framework, epsilon_decay=False, epsilon=0.2, alpha=0.5, epsilon_begin=1, epsilon_end=0.01):
        self.framework = framework

        self.epsilon = epsilon if not epsilon_decay else epsilon_begin
        self.epsilon_decay = epsilon_decay
        self.epsilon_begin = epsilon_begin
        self.epsilon_end = epsilon_end
        self.alpha = alpha

        self.q_table = np.zeros((self.framework.time_steps+1, self.framework.num_state, self.framework.num_action))
        # self.q_table = np.random.normal(scale=0.01, size=(self.framework.time_steps+1, self.framework.num_state, self.framework.num_action))
        # self.q_table[self.framework.time_steps] = 0
        self.wealth_list = {}
        self.reward_list = {}
        self.action_list = {}

    def epsilon_greedy_policy(self, time, state):
        """Generate one single step action according to epsilon greedy policy.
        :param time: The input time.
        :param state: The current state.
        :return: The action in next time.
        """
        if np.random.binomial(1, self.epsilon):
            return np.random.randint(self.framework.num_action)
        else:
            return np.argmax(self.q_table[time][state])

    def step(self, wealth, action):
        """Perform one step and generate the wealth at next time, according to the input wealth and action."""
        if np.random.binomial(1, self.framework.p_up):
            return wealth * (1 + self.framework.rf + self.framework.action_space[action] * (self.framework.up - self.framework.rf))
        else:
            return wealth * (1 + self.framework.rf + self.framework.action_space[action] * (self.framework.down - self.framework.rf))

    def episode(self):
        """Play one episode. Then renew the q_table."""
        time = 0
        wealth = self.framework.wealth_init
        state = self.framework.wealth_to_state(wealth)
        action = self.epsilon_greedy_policy(time, state)
        action_list = np.zeros(self.framework.time_steps + 1)
        action_list[time] = self.framework.action_space[action]

        while not self.framework.is_terminal(time):
            next_wealth = self.step(wealth, action)
            next_state = self.framework.wealth_to_state(next_wealth)
            next_action = self.epsilon_greedy_policy(time+1, next_state)

            self.q_table[time, state, action] += \
                self.alpha*(self.framework.reward(time+1, next_wealth) + self.q_table[time+1, next_state, next_action] - self.q_table[time, state, action])

            # print("time:", time)
            # print("value of q:", self.q_table[time, state, action])
            # print("wealth:", wealth)
            # print("proportion investing in risky asset", self.framework.action_space[action])
            # print("next_wealth", next_wealth)
            # print("reward:", self.framework.reward(time+1, next_wealth))
            # print("")
            time += 1
            wealth = next_wealth
            state = next_state
            action = next_action
            action_list[time] = self.framework.action_space[action]
        return wealth, action_list

    def train(self, round_=10000):
        """Control the process of SARSA.
        :param round_: The number of episode will be played. Default: 10000
        """
        n = 0
        self.wealth_list = np.zeros(round_)
        self.reward_list = np.zeros(round_)
        action_lists = []
        while n < round_:
            wealth, action_list = self.episode()
            self.wealth_list[n] = wealth
            self.reward_list[n] = self.framework.crsp(wealth)
            action_lists.append(action_list)
            self.alpha = max(0.1, self.alpha * 0.9999)
            if self.epsilon_decay:
                self.epsilon -= (self.epsilon_begin - self.epsilon_end) / round_
                # self.epsilon = self.epsilon_end + (self.epsilon_begin - self.epsilon_end) * np.exp(-n/(round_ * 0.1))
            n += 1
            if n % 1000 == 0:
                print("episode", n, "ends!!!")
        print("max wealth in all episode:", np.max(self.wealth_list), "episode", np.argmax(self.wealth_list))
        self.action_list = np.array(action_lists)

    def get_optimize_policy(self):
        return np.argmax(self.q_table, axis=2)



class QLearning:
    """This class performs the Q Learning algorithm.

    Note that
    1) Two policy are adopted--The ordinary epsilon greedy method and the method with epsilon decay.
    2) We intentionally distinguish the difference between wealth and state among all methods.
    In fact, wealth represents the actual value, while state represent the state-th interval that wealth is in (same meaning for action).

    Attributes:
        framework : The instance of class Framework.
        epsilon_decay : Adopt epsilon_decay greedy policy if true; otherwise adopt epsilon greedy policy. Default: False
        epsilon : A parameter to measure the extent to explore (valid when epsilon_decay=False). Default: 0.1
        alpha : Learning rate. Default: 0.5
        epsilon_begin : The initial value of epsilon (valid when epsilon_decay=True). Default: 1
        epsilon_end : The final value of epsilon (valid when epsilon_decay=True). Default: 0.01
    """
    def __init__(self, framework, epsilon_decay=False, epsilon=0.2, alpha=0.5, epsilon_begin=1, epsilon_end=0.01):
        self.framework = framework

        self.epsilon = epsilon if not epsilon_decay else epsilon_begin
        self.epsilon_decay = epsilon_decay
        self.epsilon_begin = epsilon_begin
        self.epsilon_end = epsilon_end
        self.alpha = alpha

        self.q_table = np.zeros((self.framework.time_steps+1, self.framework.num_state, self.framework.num_action))
        # self.q_table = np.random.normal(scale=0.01, size=(self.framework.time_steps+1, self.framework.num_state, self.framework.num_action))
        # self.q_table[self.framework.time_steps] = 0
        self.wealth_list = {}
        self.reward_list = {}
        self.action_list = {}

    def epsilon_greedy_policy(self, time, state):
        """Generate one single step action according to epsilon greedy policy.
        :param time: The input time.
        :param state: The current state.
        :return: The action in next time.
        """
        if np.random.binomial(1, self.epsilon):
            return np.random.randint(self.framework.num_action)
        else:
            return np.argmax(self.q_table[time][state])

    def step(self, wealth, action):
        """Perform one step and generate the wealth at next time, according to the input wealth and action."""
        if np.random.binomial(1, self.framework.p_up):
            return wealth * (1 + self.framework.rf + self.framework.action_space[action] * (self.framework.up - self.framework.rf))
        else:
            return wealth * (1 + self.framework.rf + self.framework.action_space[action] * (self.framework.down - self.framework.rf))

    def episode(self):
        """Play one episode. Then renew the q_table."""
        time = 0
        wealth = self.framework.wealth_init
        state = self.framework.wealth_to_state(wealth)
        action_list = np.zeros(self.framework.time_steps)

        while not self.framework.is_terminal(time):
            action = self.epsilon_greedy_policy(time, state)
            action_list[time] = self.framework.action_space[action]
            next_wealth = self.step(wealth, action)
            next_state = self.framework.wealth_to_state(next_wealth)
            # Find the best action under time t+1 and next state
            q_values = self.q_table[time+1, next_state, :]
            next_action = np.argmax(q_values)

            self.q_table[time, state, action] += \
                self.alpha*(self.framework.reward(time+1, next_wealth) + self.q_table[time+1, next_state, next_action] - self.q_table[time, state, action])

            # print("time:", time)
            # print("value of q:", self.q_table[time, state, action])
            # print("wealth:", wealth)
            # print("proportion investing in risky asset", self.framework.action_space[action])
            # print("next_wealth", next_wealth)
            # print("reward:", self.framework.reward(time+1, next_wealth))
            # print("")
            time += 1
            wealth = next_wealth
            state = next_state
        return wealth, action_list

    def train(self, round_=10000):
        """Control the process of SARSA.
        :param round_: The number of episode will be played. Default: 10000
        """
        n = 0
        self.wealth_list = np.zeros(round_)
        self.reward_list = np.zeros(round_)
        action_lists = []
        while n < round_:
            wealth, action_list = self.episode()
            self.wealth_list[n] = wealth
            self.reward_list[n] = self.framework.crsp(wealth)
            action_lists.append(action_list)
            # self.alpha = max(0.1, self.alpha * 0.9999)
            if self.epsilon_decay:
                 self.epsilon -= (self.epsilon_begin - self.epsilon_end) / round_
                # self.epsilon = self.epsilon_end + (self.epsilon_begin - self.epsilon_end) * np.exp(-n/(round_ * 0.1))
            n += 1
            if n % 1000 == 0:
                print("episode", n, "ends!!!")
                # print(self.q_table[self.framework.time_steps-1])
        print("max wealth in all episode:", np.max(self.wealth_list), "episode", np.argmax(self.wealth_list))
        self.action_list = np.array(action_lists)

    def get_optimize_policy(self):
        return np.argmax(self.q_table, axis=2)


class Auxiliary:
    @staticmethod
    def plot_convergence(wealth_list, title, window=200):
        plt.figure(figsize=(12, 8))

        plt.plot(wealth_list, alpha=0.3, label='wealth for single episode')
        wealth_mv = np.convolve(wealth_list, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(wealth_list)), wealth_mv, linewidth=2, color='#FF4676', label=f'wealth of {window}th moving average')

        max_idx = np.argmax(wealth_list)
        max_wealth = wealth_list[max_idx]
        plt.scatter(max_idx, max_wealth, color='#00C0F9', label='max wealth')
        plt.text(max_idx-1000, max_wealth+0.1, f'({max_idx}, {max_wealth:.2f})')

        plt.xlabel('episode')
        plt.ylabel('final wealth')
        plt.title(f'Convergence of {title}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.show()

    @staticmethod
    def plot_comparison_diagram(wl_1, wl_2, window=1000, label_1='Q-Learning', label_2='SARSA'):
        wealth_mv1 = np.convolve(wl_1, np.ones(window) / window, mode='valid')
        plt.plot(range(window - 1, len(wl_1)), wealth_mv1, linewidth=1, color='#00C0F9', label=label_1)

        wealth_mv2 = np.convolve(wl_2, np.ones(window) / window, mode='valid')
        plt.plot(range(window - 1, len(wl_2)), wealth_mv2, linewidth=1, color='#FF4676', label=label_2)

        plt.xlabel('episode')
        plt.ylabel('score')
        plt.title('Algorithm Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.show()

    @staticmethod
    def plot_action_dist(action_list, label):
        data = pd.DataFrame(action_list[-1000:])
        fig, ax = plt.subplots(4, 3)
        counter = 0
        for i in range(3):
            for j in range(3):
                ax[i, j].hist(data[counter], color='#00C0F9', bins=10, edgecolor='black')
                ax[i, j].set_title(f'Time {counter}')
                counter += 1
        ax[3, 0].hist(data[counter], color='#00C0F9', bins=10, edgecolor='black')
        ax[3, 0].set_title(f'Time {counter}')
        fig.delaxes(ax[3, 1])
        fig.delaxes(ax[3, 2])
        plt.suptitle(f'{label}: Action Distribution at different time steps')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_optimal_policy(policy, action_space, num_action, label):
        action_proportions = action_space[policy]

        # plt.figure(figsize=(15, 8))

        cmap = plt.get_cmap('Reds')
        levels = np.linspace(0, 1, num_action)
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        plt.imshow(action_proportions.T, aspect='auto', origin='lower', cmap=cmap, norm=norm)


        plt.xlabel('Time Step')
        plt.ylabel('State')

        cbar = plt.colorbar(ticks=np.linspace(0, 1, 11))
        cbar.set_label('Risky Asset Allocation Ratio')

        plt.title(f'{label}: Optimal Policy Visualization\n(Proportion Invested in Risky Asset)')
        plt.show()




