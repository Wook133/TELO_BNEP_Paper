import numpy
import time
import random
from copy import deepcopy
import math
from functools import partial
import multiprocessing
from collections import namedtuple
import argparse
import deap
import gym
import pybulletgym
import pybullet_envs

from other.es import CMAES, OpenES
from pytorch_utils import Actor, listToString, argument_settings_single, get_game, save_model, policy_middle
import snn
import utils

try:
    import torch
    import torch.utils.data
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import optim
    from torch.autograd import Variable
except:
    raise ImportError("For this example you need to install pytorch.")

# Settings to Store for Later Use
argument_settings = namedtuple('Arguments',
                               ['env_name', 'shared_directory', 'num_processors', 'num_evaluations', 'cycle_iterations',
                                'population_size', 'max_num_episodes', 'layer_1', 'layer_2', 'rank_fitness',
                                'use_max_action', 'probability_recording', 'path', 'bne_initial_sigma',
                                'bne_weight_decay', 'min_bound', 'max_bound'])
actor_settings = namedtuple('Actor_Arguments',
                            ['state_dim', 'action_dim', 'max_action', 'use_max_action', 'num_parameters', 'layer_1',
                             'layer_2', 'initial_mu', 'initial_sigma'])
snn_settings = namedtuple('Simulator_Arguments',
                          ['input_size', 'output_size', 'hidden_layer_size', 'dropout_rate', 'bne_cycles',
                           'simulator_train', 'reward_pos', 'max_episode', 'reward_threshold'])
statistics = namedtuple('Settings_Statistics',
                        ['dynamics', 'normalization', 'normalization_type', 'x_min', 'x_mean', 'x_max', 'x_std',
                         'y_min', 'y_mean', 'y_max', 'y_std'])
settings = {}


# Helper functions to write to files
def write_settings(filename, settings):
    with open(filename, "w") as text_file:
        text_file.write(settings)


def generate_row(action, observation, subsequent_observation, reward=None):
    """
    :param path:
    :param action:
    :param observation:
    :param subsequent_observation:
    :param reward:
    :return:
    """
    action = action * 1.0
    probability = numpy.random.uniform(0.0, 1.0)
    if reward is None:
        row = to_pybullet_string(action, observation, subsequent_observation)
    else:
        row = to_pybullet_string(action, observation, subsequent_observation, reward)
    return row


def to_pybullet_string(action, observation, subsequent_observation, reward=None):
    """""
    Method to convert inputs to string
    :param action: the action performed in the environment
    :param observation: the observation made by the agent which was used to select an action
    :param subsequent_observation: the resultant observation after the action was applied
    :param reward: the reward achieved which is associated with the subsequent observation
    :return: the concatenated string of the input arguments deliminited by a "," or comma
    """
    action_list = isinstance(action, numpy.ndarray)
    line = ""
    if action_list:
        # act = action[0]
        line += str(action[0])
        for index in range(len(action) - 1):
            line += "," + str(action[index + 1])

        line += "," + str(observation[0])
    else:
        line = str(action) + ","
        line += str(observation[0])

    for index in range(len(observation) - 1):
        line += "," + str(observation[index + 1])
    line += "," + str(subsequent_observation[0])
    for index in range(len(subsequent_observation) - 1):
        line += "," + str(subsequent_observation[index + 1])
    if reward is None:
        return line
    else:
        line += "," + str(reward)
    return line


def write_to_files(file_name_history=None, file_name_rw_history=None, file_name_sim_history=None,
                   file_name_rw_time=None, file_name_sim_time=None, fitness=None, rw_fitness=None, sim_fitness=None,
                   rw_time=None, sim_time=None):
    if file_name_history is not None:
        file_history = open(file_name_history, "a+")
        file_history.write(listToString(fitness))
        file_history.write("\n")
        file_history.close()

    if file_name_rw_history is not None:
        file_rw_history = open(file_name_rw_history, "a+")
        file_rw_history.write(listToString(rw_fitness))
        file_rw_history.write("\n")
        file_rw_history.close()

    if file_name_sim_history is not None:
        file_sim_history = open(file_name_sim_history, "a+")
        file_sim_history.write(listToString(sim_fitness))
        file_sim_history.write("\n")
        file_sim_history.close()

    if file_name_rw_time is not None:
        file_rw_time = open(file_name_rw_time, "a+")
        file_rw_time.write(listToString(rw_time))
        file_rw_time.write("\n")
        file_rw_time.close()

    if file_name_sim_time is not None:
        file_sim_time = open(file_name_sim_time, "a+")
        file_sim_time.write(listToString(sim_time))
        file_sim_time.write("\n")
        file_sim_time.close()
    print("Writing to historical files")


def write_file(file_name, file_contents):
    file_history = open(file_name, "a+")
    file_history.write(listToString(file_contents))
    file_history.write("\n")
    file_history.close()


def write_history(file_name, file_contents):
    file_history = open(file_name, "a+")
    file_history.write(file_contents)
    file_history.write("\n")
    file_history.close()


# Evaluation Functions

def generate_evaluate(individual, state_dim, action_dim, max_action, use_max_action, layer_1, layer_2, env_name,
                      num_evaluations, seed=-1):
    policy = Actor(state_dim=state_dim, action_dim=action_dim, max_action=max_action, use_max_action=use_max_action,
                   layer_1=layer_1, layer_2=layer_2)
    policy.set_params(individual)
    env = gym.make(env_name)
    reward_list = []
    t_list = []
    list_rows = []
    mean_reward = 0.0

    if seed >= 0:
        random.seed(seed)
        numpy.random.seed(seed)
        env.seed(seed)
        env.action_space.seed(seed)

    env.reset()
    for i in range(num_evaluations):
        obs = deepcopy(env.reset())
        total_reward = 0.0
        done = False
        cur_episode = 0
        while not done:
            batch = torch.from_numpy(obs[numpy.newaxis, ...]).float()
            prediction = policy(Variable(batch))
            action = prediction.data.numpy()
            prev_obs = obs
            obs, reward, done, info = env.step(action[0])
            list_rows.append(
                generate_row(action=action[0], observation=prev_obs, subsequent_observation=obs, reward=reward))
            total_reward += reward
            cur_episode = cur_episode + 1
        reward_list.append(total_reward)
        t_list.append(cur_episode)
    mean_reward = numpy.mean(reward_list)
    del policy, reward_list, t_list
    return (mean_reward, list_rows)


def real_evaluate(policy, seed, env_name, num_evaluations, shared_directory, probability_recording, max_steps,
                  max_reward):
    env = gym.make(env_name)
    reward_list = []
    t_list = []
    list_rows = []
    if seed >= 0:
        random.seed(seed)
        numpy.random.seed(seed)
        env.seed(seed)
        env.action_space.seed(seed)
    env.reset()
    for i in range(num_evaluations):
        obs = deepcopy(env.reset())
        total_reward = 0.0
        done = False
        cur_episode = 0
        p = numpy.random.uniform(0.0, 1.0)
        while (cur_episode <= max_steps) and (total_reward <= max_reward) and not done:
            batch = torch.from_numpy(obs[numpy.newaxis, ...]).float()
            prediction = policy(Variable(batch))
            action = prediction.data.numpy()
            prev_obs = obs
            obs, reward, done, info = env.step(action[0])
            if shared_directory is not None and p <= probability_recording:
                list_rows.append(
                    generate_row(action=action[0], observation=prev_obs, subsequent_observation=obs, reward=reward))
            total_reward += reward
            cur_episode = cur_episode + 1
        reward_list.append(total_reward)
        t_list.append(cur_episode)
    del policy
    return numpy.mean(reward_list), numpy.mean(t_list), list_rows


def evaluate_simulator(policy, simulator, env_name, num_evaluations, max_episode, reward_threshold, normalization_type,
                       x_min=0.0, x_mean=0.0, x_max=0.0, x_std=0.0, y_min=0.0, y_mean=0.0, y_max=0.0, y_std=0.0):
    reward_list = []
    t_list = []
    env = gym.make(env_name)
    seed = int(time.time())
    if seed >= 0:
        random.seed(seed)
        numpy.random.seed(seed)
        env.seed(seed)
    env.reset()
    for i in range(num_evaluations):
        obs = env.reset()
        total_reward = 0.0
        cur_episode = 0
        while (cur_episode < max_episode) and (total_reward < reward_threshold):
            batch = torch.from_numpy(obs[numpy.newaxis, ...]).float()
            prediction = policy(Variable(batch))
            action = prediction.data.numpy()
            prev_obs = obs
            obs, reward = simulator.step_state(action[0], prev_obs, normalization_type=normalization_type, x_min=x_min,
                                               x_mean=x_mean, x_max=x_max, x_std=x_std, y_min=y_min, y_mean=y_mean,
                                               y_max=y_max, y_std=y_std)
            total_reward = total_reward + reward
            cur_episode = cur_episode + 1
        reward_list.append(total_reward)
        t_list.append(cur_episode)
    env.close()
    del env, cur_episode, total_reward, obs, reward, action, prev_obs
    return numpy.mean(reward_list), numpy.mean(t_list)


def evaluate_outer_wrapper(individual, env_name, state_dim, action_dim, max_action, max_steps, max_reward,
                           num_evaluations, shared_directory, use_max_action, layer_1=64, layer_2=64, seed=-1):
    solution_individual = numpy.array(individual)
    env = gym.make(env_name)

    def policy(state):
        return env.action_space.sample()

    policy = Actor(state_dim=state_dim, action_dim=action_dim, max_action=max_action, use_max_action=use_max_action,
                   layer_1=layer_1, layer_2=layer_2)
    policy.set_params(solution_individual)
    real_reward, real_time, list_rows = real_evaluate(policy=policy, seed=seed, env_name=env_name,
                                                      num_evaluations=num_evaluations,
                                                      shared_directory=shared_directory, probability_recording=1.0,
                                                      max_steps=max_steps, max_reward=max_reward)
    env.close()
    return real_reward, real_time, list_rows


def middle_evaluate(cur_policy, simulator, state_dim, action_dim, max_action,
                    use_max_action, env_name, shared_directory, normalization_type=None,
                    x_min=0.0, x_mean=0.0, x_max=0.0, x_std=0.0, y_min=0.0,
                    y_mean=0.0, y_max=0.0, y_std=0.0, layer_1=64, layer_2=64,
                    H=5, max_steps=10, max_reward=10):
    print("Middle Evaluate")
    policy = Actor(state_dim=state_dim, action_dim=action_dim, max_action=max_action,
                   use_max_action=use_max_action, layer_1=layer_1, layer_2=layer_2)
    print("cur_policy_params")
    cur_policy_params = cur_policy.getPolicyParameters()
    print("cur_policy_params = ", cur_policy_params)
    policy.set_params(cur_policy_params)
    env = gym.make(env_name)
    if cur_policy.getSeed() >= 0:
        random.seed(cur_policy.getSeed())
        numpy.random.seed(cur_policy.getSeed())
        env.seed(cur_policy.getSeed())
        env.action_space.seed(cur_policy.getSeed())
    list_rows = []
    obs = env.reset()
    done = False
    reward_list = []
    t_list = []
    cur_episode = 0
    total_reward = 0.0
    # Real World Evaluate
    # while (cur_episode <= max_steps) and (total_reward <= max_reward) and not done:
    while (cur_episode <= max_steps) and not done:
        batch = torch.from_numpy(obs[numpy.newaxis, ...]).float()
        prediction = policy(Variable(batch))
        action = prediction.data.numpy()
        prev_obs = obs
        obs, reward, done, info = env.step(action[0])
        if shared_directory is not None:
            list_rows.append(
                generate_row(action=action[0], observation=prev_obs, subsequent_observation=obs, reward=reward))
        total_reward += reward
        cur_episode = cur_episode + 1
    t_list.append(cur_episode)

    # Simulate Evaluate
    i = 0
    cur_reward = 0.0
    while i < H:
        batch = torch.from_numpy(obs[numpy.newaxis, ...]).float()
        prediction = policy(Variable(batch))
        action = prediction.data.numpy()
        prev_obs = obs
        obs, reward = simulator.step_state(action[0], prev_obs, normalization_type=normalization_type, x_min=x_min,
                                           x_mean=x_mean, x_max=x_max, x_std=x_std, y_min=y_min, y_mean=y_mean,
                                           y_max=y_max, y_std=y_std)
        cur_reward = cur_reward + reward
        i = i + 1
    final_reward = total_reward + cur_reward
    del reward, policy
    return total_reward, final_reward, t_list, list_rows


# max_episode=int(cur_max_timestep_outer * 1.5),
# reward_threshold=int(cur_max_reward_outer * 1.5),
def inner_evaluate(individual, simulator, env_name, state_dim, action_dim, max_action, use_max_action, layer_1, layer_2,
                   num_evaluations, max_episode, reward_threshold, normalization_type, x_min=0.0, x_mean=0.0, x_max=0.0,
                   x_std=0.0, y_min=0.0, y_mean=0.0, y_max=0.0, y_std=0.0):
    print("Inner Evaluate: max_episode = ", max_episode, " reward_threshold = ", reward_threshold)

    def policy(state):
        return env.action_space.sample()

    policy = Actor(state_dim=state_dim, action_dim=action_dim, max_action=max_action, use_max_action=use_max_action,
                   layer_1=layer_1, layer_2=layer_2)
    policy.set_params(individual)
    reward_list = []
    t_list = []
    env = gym.make(env_name)
    seed = int(time.time())
    if seed >= 0:
        random.seed(seed)
        numpy.random.seed(seed)
        env.seed(seed)
        env.action_space.seed(seed)
    env.reset()
    for i in range(num_evaluations):
        obs = env.reset()
        total_reward = 0.0
        cur_episode = 0
        while (cur_episode < max_episode) and (total_reward < reward_threshold):
            batch = torch.from_numpy(obs[numpy.newaxis, ...]).float()
            prediction = policy(Variable(batch))
            action = prediction.data.numpy()
            prev_obs = obs
            obs, reward = simulator.step_state(action[0], prev_obs, normalization_type=normalization_type, x_min=x_min,
                                               x_mean=x_mean, x_max=x_max, x_std=x_std, y_min=y_min, y_mean=y_mean,
                                               y_max=y_max, y_std=y_std)
            total_reward = total_reward + reward
            cur_episode = cur_episode + 1
        reward_list.append(total_reward)
        t_list.append(cur_episode)
    env.close()
    del env, cur_episode, total_reward
    return numpy.mean(reward_list), numpy.mean(t_list)


def train_simulator(net, settings, cycle=0):
    print("Training Simulator = ", settings['statistic_settings'].normalization)
    optimizer = optim.AdamW(net.parameters())
    criterion = torch.nn.MSELoss()
    if not settings['statistic_settings'].normalization:
        net = utils.train_simulator(simulator=net, path=settings['settings'].shared_directory,
                                    epochs=settings['snn_settings'].simulator_train, criterion=criterion,
                                    optimizer=optimizer, batch_size=int(64 * (1 + cycle)), shuffle=True,
                                    checkpoint=False, num_workers=0, reward_pos=settings['snn_settings'].reward_pos,
                                    env_name=settings['settings'].env_name, cycle=cycle, round_to=5, use_cuda=False,
                                    evaluate_fitness=True,
                                    normalization_type=settings['statistic_settings'].normalization_type,
                                    dynamics=settings['statistic_settings'].dynamics)
        x_min, x_mean, x_max, x_std, y_min, y_mean, y_max, y_std = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        return net, x_min, x_mean, x_max, x_std, y_min, y_mean, y_max, y_std
    else:
        print("Hello There, train_simulator = ", settings['statistic_settings'].normalization)
        net, x_min, x_mean, x_max, x_std, y_min, y_mean, y_max, y_std = utils.train_simulator(simulator=net,
                                                                                              path=settings[
                                                                                                  'settings'].shared_directory,
                                                                                              epochs=settings[
                                                                                                  'snn_settings'].simulator_train,
                                                                                              criterion=criterion,
                                                                                              optimizer=optimizer,
                                                                                              batch_size=256,
                                                                                              shuffle=True,
                                                                                              checkpoint=False,
                                                                                              num_workers=0,
                                                                                              reward_pos=settings[
                                                                                                  'snn_settings'].reward_pos,
                                                                                              env_name=settings[
                                                                                                  'settings'].env_name,
                                                                                              cycle=cycle, round_to=5,
                                                                                              use_cuda=False,
                                                                                              evaluate_fitness=True,
                                                                                              normalization_type=
                                                                                              settings[
                                                                                                  'statistic_settings'].normalization_type,
                                                                                              dynamics=settings[
                                                                                                  'statistic_settings'].dynamics)
        return net, x_min, x_mean, x_max, x_std, y_min, y_mean, y_max, y_std


# Initialize Simulator
def initialize_simulator(net, settings):
    print("Initializing Simulator")
    number_of_evaluations = settings['settings'].num_evaluations
    oes = OpenES(num_params=settings['actor_settings'].num_parameters,
                 sigma_init=settings['actor_settings'].initial_sigma, sigma_decay=0.999, learning_rate=0.1,
                 learning_rate_decay=1.0, popsize=4, weight_decay=0.01, rank_fitness=False, forget_best=True)

    fitness_list = []
    list_rows = []
    solutions, seeding = oes.ask()
    pool = multiprocessing.Pool(settings['settings'].num_processors)
    partial_evaluate = partial(generate_evaluate, env_name=settings['settings'].env_name,
                               state_dim=settings['actor_settings'].state_dim,
                               action_dim=settings['actor_settings'].action_dim,
                               max_action=settings['actor_settings'].max_action,
                               use_max_action=settings['actor_settings'].use_max_action,
                               layer_1=settings['actor_settings'].layer_1, layer_2=settings['actor_settings'].layer_2,
                               num_evaluations=number_of_evaluations)
    temp_list = list(zip(pool.map(partial_evaluate, solutions)))
    pool.close()
    for item in temp_list:
        print("Item type = ", type(item), " length tuple = ", len(item))
        print("Item[0] type = ", type(item[0]), " length tuple = ", len(item[0]))
        fitness = item[0][0]
        print("Fitness = ", fitness)
        row = item[0][1]
        fitness_list.append(fitness)
        list_rows.append(row)
        # mean_reward, list_rows
    reward_history = []
    reward_history.append(fitness_list)

    utils.rows_to_save(list_rows, path=settings['settings'].shared_directory, validation_probability=0.3)
    print("Saved Training and Validation Data")
    utils.generate_config_file(path=settings['settings'].shared_directory, env_name=settings['settings'].env_name,
                               save_reward=True)
    optimizer = optim.AdamW(net.parameters())
    criterion = torch.nn.MSELoss()

    print("Normalization Type in mp_BNS = ", settings['statistic_settings'].normalization_type)
    if not settings['statistic_settings'].normalization:
        print("Keemstar")
        net = utils.train_simulator(simulator=net, path=settings['settings'].shared_directory,
                                    epochs=settings['snn_settings'].simulator_train, criterion=criterion,
                                    optimizer=optimizer, batch_size=256, shuffle=True, checkpoint=False, num_workers=0,
                                    reward_pos=settings['snn_settings'].reward_pos,
                                    env_name=settings['settings'].env_name, cycle=0, round_to=5, use_cuda=False,
                                    evaluate_fitness=True,
                                    normalization_type=settings['statistic_settings'].normalization_type,
                                    dynamics=settings['statistic_settings'].dynamics)
        x_min, x_mean, x_max, x_std, y_min, y_mean, y_max, y_std = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        return net, x_min, x_mean, x_max, x_std, y_min, y_mean, y_max, y_std, reward_history  # , solutions, fitness_list
    else:
        print("Vaush")
        net, x_min, x_mean, x_max, x_std, y_min, y_mean, y_max, y_std = utils.train_simulator(simulator=net,
                                                                                              path=settings[
                                                                                                  'settings'].shared_directory,
                                                                                              epochs=settings[
                                                                                                  'snn_settings'].simulator_train,
                                                                                              criterion=criterion,
                                                                                              optimizer=optimizer,
                                                                                              batch_size=256,
                                                                                              shuffle=True,
                                                                                              checkpoint=False,
                                                                                              num_workers=0,
                                                                                              reward_pos=settings[
                                                                                                  'snn_settings'].reward_pos,
                                                                                              env_name=settings[
                                                                                                  'settings'].env_name,
                                                                                              cycle=0, round_to=5,
                                                                                              use_cuda=False,
                                                                                              evaluate_fitness=True,
                                                                                              normalization_type=
                                                                                              settings[
                                                                                                  'statistic_settings'].normalization_type,
                                                                                              dynamics=settings[
                                                                                                  'statistic_settings'].dynamics)
        return net, x_min, x_mean, x_max, x_std, y_min, y_mean, y_max, y_std, reward_history  # , solutions, fitness_list


def make_simulator_env(env_name, simulator, dynamics):
    real_env = gym.make(env_name)
    real_start_state = real_env.reset()
    size_obs = len(real_start_state)
    action_space = real_env.action_space
    real_env.close()

    # Get Action Space Type
    if isinstance(action_space, gym.spaces.discrete.Discrete):
        sim_env = snn.wrapper_SNN_discrete_action(starting_state=real_start_state, action_space=action_space,
                                                  size_observation=size_obs, SNN=simulator, round_to=5,
                                                  dynamics=dynamics, env_name=env_name)
    elif isinstance(action_space, gym.spaces.box.Box):
        sim_env = snn.wrapper_SNN_box_action(starting_state=real_start_state, action_space=action_space,
                                             size_observation=size_obs, SNN=simulator, round_to=5, dynamics=dynamics,
                                             env_name=env_name)
    # Get Observation Space Type?
    print("Done Initializing Simulator Environment")
    return sim_env


def main(env_name, shared_directory, num_processors, num_evaluations, population_size, max_num_episodes, initial_sigma,
         weight_decay, layer_1, layer_2, use_max_action, probability_recording, dynamics_model,
         standardization, bne_cycles, simulator_train, hidden_layer_size, dropout_rate, path, rank_fitness, initial_mu,
         min_bound, max_bound):
    global_start = time.time()
    cycle_iterations = int((max_num_episodes / population_size) / bne_cycles)
    cycle_iterations = int((max_num_episodes) / bne_cycles)

    cur_settings = argument_settings(env_name=env_name, shared_directory=shared_directory,
                                     num_processors=num_processors, num_evaluations=num_evaluations,
                                     population_size=population_size, cycle_iterations=cycle_iterations,
                                     max_num_episodes=max_num_episodes, layer_1=layer_1, layer_2=layer_2,
                                     rank_fitness=rank_fitness, use_max_action=use_max_action,
                                     probability_recording=probability_recording, path=path,
                                     bne_initial_sigma=initial_sigma, bne_weight_decay=weight_decay,
                                     min_bound=min_bound, max_bound=max_bound)
    settings['settings'] = cur_settings
    env = gym.make(env_name)
    env.reset()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = int(env.action_space.high[0])
    max_episode = env.spec.max_episode_steps
    reward_threshold = env.spec.reward_threshold

    sampled_action = env.action_space.sample()
    observation, reward, done, info = env.step(sampled_action)
    snn_input_size = utils.switch_type(sampled_action) + utils.switch_type(observation)
    snn_output_size = utils.switch_type(reward) + utils.switch_type(observation)
    reward_pos = utils.get_reward_position(env_name)

    cur_snn_settings = snn_settings(input_size=snn_input_size, output_size=snn_output_size, bne_cycles=bne_cycles,
                                    simulator_train=simulator_train, hidden_layer_size=hidden_layer_size,
                                    reward_pos=reward_pos, dropout_rate=dropout_rate, max_episode=max_episode,
                                    reward_threshold=reward_threshold)
    settings['snn_settings'] = cur_snn_settings

    net = snn.Simulator_NN(nn_in=settings['snn_settings'].input_size, nn_out=settings['snn_settings'].output_size,
                           dropout=settings['snn_settings'].dropout_rate,
                           nn_hidden=settings['snn_settings'].hidden_layer_size, environment=env_name, default=True)

    actor = Actor(state_dim=state_dim, action_dim=action_dim, max_action=max_action, layer_1=layer_1, layer_2=layer_2,
                  use_max_action=use_max_action)
    print(actor)
    num_parameters = actor.get_size()
    print("num_parameters = ", num_parameters)
    cur_actor_settings = actor_settings(state_dim=state_dim, action_dim=action_dim, initial_sigma=initial_sigma,
                                        max_action=max_action, layer_1=layer_1, layer_2=layer_2,
                                        num_parameters=num_parameters, use_max_action=use_max_action,
                                        initial_mu=initial_mu)
    settings['actor_settings'] = cur_actor_settings
    del actor

    if standardization:
        normalize = "standardization"
    else:
        normalize = "none"

    statistic_settings = statistics(dynamics=dynamics_model, normalization_type=normalize, x_min=0, x_mean=0, x_max=0,
                                    x_std=0, normalization=standardization, y_min=0, y_mean=0, y_max=0, y_std=0)
    settings['statistic_settings'] = statistic_settings

    print(settings['settings'])
    write_settings(shared_directory + "settings_settings.txt", str(settings['settings']))
    print(settings['actor_settings'])
    write_settings(shared_directory + "settings_actor.txt", str(settings['actor_settings']))
    print(settings['statistic_settings'])
    write_settings(shared_directory + "settings_statistics.txt", str(settings['statistic_settings']))
    print(settings['snn_settings'])
    write_settings(shared_directory + "settings_snn.txt", str(settings['snn_settings']))
    print("Initializing Simulator")
    net, x_min, x_mean, x_max, x_std, y_min, y_mean, y_max, y_std, reward_history = initialize_simulator(net=net,
                                                                                                         settings=settings)

    print("type rw = ", type(reward_history))
    for row in reward_history:
        print("row = ", row)

    file_name_middle_history = str(shared_directory) + "/history_rw_middle_fitness.csv"
    file_name_middle_plan_history = str(shared_directory) + "/history_rw_middle_plan_fitness.csv"

    file_name_history = str(shared_directory) + "/history_fitness.csv"
    file_name_history_time = str(shared_directory) + "/history_time.csv"

    file_name_summary = str(shared_directory) + "/summary.csv"
    file_name_phenotype = str(shared_directory) + "/history_phenotype_"

    generation = 0
    best_fitness = -10000.0
    best_solution = []

    current_middle_population_size = 64
    H = 15
    reward_factor_outer = 0.9
    reward_factor_outer = 1.0
    cur_max_timestep_outer = (settings['snn_settings'].max_episode * 0.01 * 1.0) + 10
    cur_max_reward_outer = settings['snn_settings'].reward_threshold * 0.01 * reward_factor_outer
    k_factor = 1.15
    counter = 0
    num_steps = 0
    num_episodes = 0
    middle_counter = 0
    reset = False
    initial_mu = settings['actor_settings'].initial_mu
    write_history(file_name=file_name_summary,
                  file_contents="counter;num_episodes;num_steps;min_fitness;mean_fitness;max_fitness;cur_max_reward_outer;std_fitness;plan_min_fitness;plan_mean_fitness;plan_max_fitness;plan_std_fitness;sum_timesteps;cur_max_timestep_outer;pop_size;cycle_iterations;max_cycle_iterations;time_taken")

    reset_schedule = [5000, 15000, 25000, 30000, 50000, 75000, 100000, 300000, 600000, 800000, 8000000]#, 9000000]
    #reset_schedule = [10000, 30000, 50000, 60000, 100000, 75000, 100000, 300000, 600000, 9000000, 20000000, 40000000, 80000000]#, 9000000]

    schedule_counter = -1
    cur_sigma = []
    cur_sigma = settings['actor_settings'].initial_sigma
    middle_cmaes = CMAES(num_params=settings['actor_settings'].num_parameters,
                         sigma_init=settings['actor_settings'].initial_sigma,
                         mu_init=initial_mu, reset=reset,
                         weight_decay=settings['settings'].bne_weight_decay,
                         rank_fitness=settings['settings'].rank_fitness,
                         upper_bound=settings['settings'].max_bound + numpy.max(initial_mu),
                         lower_bound=settings['settings'].min_bound + numpy.min(initial_mu),
                         popsize=current_middle_population_size, ftarget=-1.0*settings['snn_settings'].reward_threshold)
    middle_solutions = None
    middle_planning_fitness = None
    for i in range(settings['snn_settings'].bne_cycles):
        net, x_min, x_mean, x_max, x_std, y_min, y_mean, y_max, y_std = train_simulator(net=net, settings=settings,
                                                                                        cycle=i + 1)
        simulator = make_simulator_env(env_name=settings['settings'].env_name, simulator=net,
                                       dynamics=settings['statistic_settings'].dynamics)
        print("Restarting Population")
        current_middle_population_size = int(current_middle_population_size * 2.0)
        if current_middle_population_size > settings['settings'].population_size:
            current_middle_population_size = settings['settings'].population_size
        schedule_counter = schedule_counter + 1

        max_bondage = numpy.max(initial_mu)
        if max_bondage > settings['settings'].max_bound:
            max_bondage = settings['settings'].max_bound + max_bondage
        else:
            max_bondage = settings['settings'].max_bound
        min_bondage = numpy.min(initial_mu)
        if min_bondage < settings['settings'].min_bound:
            min_bondage = settings['settings'].min_bound + min_bondage
        else:
            min_bondage = settings['settings'].min_bound
        if current_middle_population_size >= 256:
            current_middle_population_size = 32
        if i >= 11:
            current_middle_population_size = 256
        elif i < 9:
            current_middle_population_size = 64
        elif i >= 10:
            current_middle_population_size = 256


        if 6 > 8:
            cur_max_timestep_outer = cur_max_timestep_outer + (0.01 * settings['snn_settings'].max_episode)
            cur_max_reward_outer = cur_max_reward_outer + (0.01 * settings['snn_settings'].reward_threshold)
            if cur_max_timestep_outer > settings['snn_settings'].max_episode:
                cur_max_timestep_outer = settings['snn_settings'].max_episode
            if cur_max_reward_outer > settings['snn_settings'].reward_threshold:
                cur_max_reward_outer = settings['snn_settings'].reward_threshold

        j = 0
        while j < reset_schedule[schedule_counter]:
            if (j == 0) and ((i == 9) or (i == 4)):
                middle_cmaes = CMAES(num_params=settings['actor_settings'].num_parameters,
                                     sigma_init=settings['actor_settings'].initial_sigma,
                                     mu_init=initial_mu, reset=True,
                                     weight_decay=settings['settings'].bne_weight_decay,
                                     rank_fitness=settings['settings'].rank_fitness,
                                     upper_bound=max_bondage,
                                     lower_bound=min_bondage,
                                     popsize=current_middle_population_size)
                print("Do the reset with solutions")
                #middle_cmaes.tell(reward_table_result=middle_planning_fitness, solutions=middle_solutions)
            print("J = ", j, " cycle iterations = ", settings['settings'].cycle_iterations)
            start = time.time()
            # Middle Loop

            if middle_counter == 0:
                middle_solutions, seeder = middle_cmaes.seeder_ask(pop_size=current_middle_population_size)
            else:
                middle_solutions, seeder = middle_cmaes.seeder_ask(pop_size=current_middle_population_size)

            cur_pop = zip(middle_solutions, seeder)
            middle_population = []
            for a, b in cur_pop:
                middle_population.append(policy_middle(policy_parameters=a, seed=b))
            pool = multiprocessing.Pool(num_processors)
            partial_middle_evaluate = partial(middle_evaluate, simulator=simulator,
                                              state_dim=settings['actor_settings'].state_dim,
                                              action_dim=settings['actor_settings'].action_dim,
                                              max_action=settings['actor_settings'].max_action,
                                              use_max_action=settings['settings'].use_max_action,
                                              env_name=settings['settings'].env_name,
                                              normalization_type=settings['statistic_settings'].normalization_type,
                                              x_min=x_min, x_mean=x_mean, x_max=x_max, x_std=x_std, y_min=y_min,
                                              y_mean=y_mean, y_max=y_max, y_std=y_std,
                                              layer_1=settings['actor_settings'].layer_1,
                                              layer_2=settings['actor_settings'].layer_2, H=H,
                                              shared_directory=shared_directory,
                                              max_steps=cur_max_timestep_outer, max_reward=cur_max_reward_outer)
            middle_temp_list = list(zip(pool.map(partial_middle_evaluate, middle_population)))
            pool.close()
            middle_fitness = []
            middle_planning_fitness = []
            middle_rw_time = []
            data_list = []
            for item in middle_temp_list:
                cur_middle_fitness = item[0][0]
                middle_fitness.append(cur_middle_fitness)
                cur_middle_planning_fitness = item[0][1]
                middle_planning_fitness.append(cur_middle_planning_fitness[0])
                timer = item[0][2]
                middle_rw_time.append(timer[0])
                p = numpy.random.uniform(0.0, 1.0)
                if p < settings['settings'].probability_recording:
                    data_list.append(item[0][3])
                # return total_reward, final_reward, t_list, list_rows
            reward_history.append(middle_planning_fitness)
            print("middle_planning_fitness Size = ", len(middle_planning_fitness), "_____", " number of solutions = ",
                  len(middle_cmaes.solutions), "___", len(middle_solutions))
            print("middle_planning_fitness = ", middle_planning_fitness)
            # CHANGE TO PLANNED************
            middle_cmaes.tell(reward_table_result=middle_planning_fitness, solutions=middle_solutions)
            end = time.time()
            sum_timesteps = numpy.sum(middle_rw_time)
            num_steps = num_steps + sum_timesteps
            num_episodes = num_episodes + (num_evaluations * current_middle_population_size)
            min_fitness = numpy.min(middle_fitness)
            mean_fitness = round(numpy.mean(middle_fitness), 3)
            max_fitness = numpy.max(middle_fitness)
            std_fitness = round(numpy.std(middle_fitness), 3)

            plan_min_fitness = numpy.min(middle_planning_fitness)
            plan_mean_fitness = round(numpy.mean(middle_planning_fitness), 3)
            plan_max_fitness = numpy.max(middle_planning_fitness)
            plan_std_fitness = round(numpy.std(middle_planning_fitness), 3)
            max_timer = 0
            for ti in middle_rw_time:
                max_timer = max_timer + ti
            j = j + max_timer
            print_string = '{0};{1};{2};{3};{4};{5};{6};{7};{8};{9};{10};{11};{12};{13};{14};{15};{16};{17} seconds'.format(
                counter,
                num_episodes, num_steps, min_fitness,
                mean_fitness, max_fitness, cur_max_reward_outer,
                std_fitness, plan_min_fitness, plan_mean_fitness,
                plan_max_fitness, plan_std_fitness, sum_timesteps,
                cur_max_timestep_outer, len(middle_solutions),
                str(j), str(reset_schedule[schedule_counter]), end - start)

            # Save Collected Data
            utils.rows_to_save(data_list, path=settings['settings'].shared_directory, validation_probability=0.3)
            # Write fitness and time to files
            write_file(file_name=file_name_middle_history, file_contents=middle_fitness)
            write_file(file_name=file_name_middle_plan_history, file_contents=middle_planning_fitness)
            # Over-arching Telemetry
            write_file(file_name=file_name_history, file_contents=middle_fitness)
            write_file(file_name=file_name_history_time, file_contents=middle_rw_time)
            middle_counter = middle_counter + 1

            # Dynamic Expansion
            cur_max_timestep_outer, cur_max_reward_outer, string_out = utils.calculate_maximum_timestep_count_1(
                threshold_timestep=settings['snn_settings'].max_episode,
                threshold_reward=settings['snn_settings'].reward_threshold,
                reward_history=reward_history,
                cur_timestep=cur_max_timestep_outer,
                cur_reward=cur_max_reward_outer,
                threshold_A=0.80, threshold_B=0.65, threshold_C=0.50,
                rate_of_change_reward=1.1, rate_of_change_timestep=1.09)
            #rate_of_change_reward=1.1, rate_of_change_timestep=1.9)

            '''current_middle_population_size = utils.increase_population_size(current_middle_population_size,
                                                                            max_pop_size=256,
                                                                            k=k_factor,
                                                                            generation_counter=middle_counter,
                                                                            g=30)'''
            print_string = print_string + ";" + string_out
            print(print_string)
            write_history(file_name=file_name_summary, file_contents=print_string)

            initial_mu = middle_cmaes.get_mu()
            cur_sigma = middle_cmaes.get_sigma()

            reset = True
        counter = counter + 1

    # save_model(policy_weights=numpy.asarray(best_solution), state_dim=state_dim, action_dim=action_dim, max_action=max_action, use_max_action=use_max_action, layer_1=layer_1, layer_2=layer_2, path=str(shared_directory + "/best_policy.model"))
    print("Best Fitness = ", best_fitness)
    pheno_mean, pheno_std = middle_cmaes.current_params()
    mean_mu = []
    mean_mu.append(numpy.mean(pheno_mean))
    print("Mean Mu = ", mean_mu)
    std_mu = []
    std_mu.append(numpy.std(pheno_mean))
    print("Std Mu = ", std_mu)
    numpy.savetxt(file_name_phenotype + ".csv", pheno_mean, delimiter=',')
    numpy.savetxt(file_name_phenotype + "mean.csv", mean_mu, delimiter=',')
    numpy.savetxt(file_name_phenotype + "std.csv", std_mu, delimiter=',')
    global_end = time.time()
    final_time = global_end - global_start
    print("Took ", final_time, " seconds or ", final_time / (3600.0))
    return 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark BNE MO-CMA-ES.')
    parser.add_argument('--env_name', type=str, help='The name of the Environment.', default="MinitaurBulletEnv-v0")

    parser.add_argument('--shared_directory', type=str,
                        help='A directory that is accessible for all processes, e.g. a NFS share.',
                        default="D:/SACAIR Data/BNE_nov_v5")
    parser.add_argument('--probability_recording', type=float, help='Probability of Recording Evaluation',
                        default=0.01)
    parser.add_argument('--num_evaluations', type=int, help='Number of Evaluations.', default=1)
    parser.add_argument('--num_processors', type=int, help='Number of processors to use.', default=5)

    parser.add_argument('--population_size', type=int, help='Size of the ES population', default=256)
    parser.add_argument('--initial_range', type=float, help='Symmetric range for initial population', default=0.25)

    parser.add_argument('--initial_mu', type=float, help='Initial Mu starting point', default=-0.00869837283010777)

    parser.add_argument('--initial_sigma', type=float, help='Initial Sigma starting point', default=0.0617566507403939)

    parser.add_argument('--rank_fitness', type=int, help='Whether to use Rank Fitness in ES.', default=0)

    parser.add_argument("--dynamics_model", type=int, help='Whether to use a Dynamics Model or Not', default=1)
    parser.add_argument("--standardization", type=int, help='Whether to Standardize/Normalize x and y', default=1)

    parser.add_argument('--layer_1', type=int, help='Layer 1 size', default=64)
    parser.add_argument('--layer_2', type=int, help='Layer 1 size', default=32)

    parser.add_argument('--use_max_action', type=int, help='Whether Policy uses Max Action or Not', default=0)

    parser.add_argument('--weight_decay', type=float, help='Initial Value of Sigma for CMA-ES for BNE',
                        default=0.238433123453988)

    parser.add_argument('--min_bound', type=float, help='Minimum bounds', default=-1.5)
    parser.add_argument('--max_bound', type=float, help='Maximum bounds', default=1.5)

    parser.add_argument('--bne_cycles', type=int, help='Maximum Number of Episodes to use', default=11)
    parser.add_argument('--simulator_train', type=int, help='How long to train a Simulator Neural Network',
                        default=50000)
    parser.add_argument('--hidden_layer_size', type=int, help='Size of hidden layers in SNN', default=256)
    parser.add_argument('--dropout_rate', type=float, help='SNN Dropout Rate', default=0.1)

    parser.add_argument('--max_num_episodes', type=float, help='Maximum Number of Episodes to use', default=50000000)

    args = parser.parse_args()
    env_name = args.__getattribute__("env_name")
    path = args.__getattribute__("shared_directory")
    shared_directory = utils.path_maker(env_name, path)
    num_processors = args.__getattribute__("num_processors")
    num_evaluations = args.__getattribute__("num_evaluations")
    population_size = args.__getattribute__("population_size")
    probability_recording = args.__getattribute__("probability_recording")
    max_num_episodes = args.__getattribute__("max_num_episodes")
    layer_1 = args.__getattribute__("layer_1")
    layer_2 = args.__getattribute__("layer_2")

    use_max_action = args.__getattribute__("use_max_action")
    if use_max_action == 0:
        use_max_action = False
    else:
        use_max_action = True

    dynamics_model = args.__getattribute__("dynamics_model")
    if dynamics_model == 0:
        dynamics_model = False
    else:
        dynamics_model = True

    standardization = args.__getattribute__("standardization")
    if standardization == 0:
        standardization = False
    else:
        standardization = True

    rank_fitness = args.__getattribute__("rank_fitness")
    if rank_fitness == 0:
        rank_fitness = False
    else:
        rank_fitness = True

    bne_cycles = args.__getattribute__("bne_cycles")
    simulator_train = args.__getattribute__("simulator_train")
    hidden_layer_size = args.__getattribute__("hidden_layer_size")
    dropout_rate = args.__getattribute__("dropout_rate")
    initial_sigma = args.__getattribute__("initial_sigma")
    initial_mu = args.__getattribute__("initial_mu")
    weight_decay = args.__getattribute__("weight_decay")

    min_bound = args.__getattribute__("min_bound")
    max_bound = args.__getattribute__("max_bound")

    solutions = main(env_name=env_name, shared_directory=shared_directory, num_processors=num_processors,
                     num_evaluations=num_evaluations, population_size=population_size, initial_sigma=initial_sigma,
                     use_max_action=use_max_action, probability_recording=probability_recording,
                     max_num_episodes=max_num_episodes, layer_1=layer_1, layer_2=layer_2, dynamics_model=dynamics_model,
                     standardization=standardization, bne_cycles=bne_cycles, simulator_train=simulator_train, path=path,
                     hidden_layer_size=hidden_layer_size, dropout_rate=dropout_rate, rank_fitness=rank_fitness,
                     weight_decay=weight_decay, initial_mu=initial_mu,
                     min_bound=min_bound, max_bound=max_bound)

