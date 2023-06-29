try:
    import torch
    import torch.utils.data
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.autograd import Variable
except:
    raise ImportError("For this example you need to install pytorch.")

try:
    import torchvision
    import torchvision.transforms as transforms
except:
    raise ImportError("For this example you need to install pytorch-vision.")

from es import CMAES
from pytorch_utils import Actor, listToString, argument_settings_single, get_game, save_model
import numpy, random
from copy import deepcopy
from functools import partial
import argparse
import time
import multiprocessing
import gym
import pybulletgym
import pybullet_envs
import highway_env


def write_history(file_name, file_contents):
    file_history = open(file_name, "a+")
    file_history.write(file_contents)
    file_history.write("\n")
    file_history.close()


def evaluate(policy_weights, env_name, state_dim, action_dim, max_action, num_evaluations=1, stochastic=True, seed=-1, USE_CUDA=False, use_max_action=True, layer_1=64, layer_2=64):
    def policy(state):
        return env.action_space.sample()
    policy = Actor(state_dim=state_dim, action_dim=action_dim, max_action=max_action, use_max_action=use_max_action, layer_1=layer_1, layer_2=layer_2)
    print("Policy Weights = ", policy_weights)
    print("length Policy Weights = ", len(policy_weights))
    policy.set_params(policy_weights)
    if USE_CUDA:
        policy.cuda()
    env = gym.make(env_name)
    reward_list = []
    t_list = []
    if seed >= 0:
        random.seed(seed)
        numpy.random.seed(seed)
        env.seed(seed)
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
            total_reward += reward
            cur_episode = cur_episode + 1
        reward_list.append(total_reward)
        t_list.append(cur_episode)
    del policy
    return numpy.mean(reward_list), numpy.mean(t_list)


def main(env_name, USE_CUDA, rank_fitness, num_processors, population_size, num_evaluations, sigma_init, weight_decay, shared_directory, budget, use_max_action, layer_1, layer_2, use_default_layers, mu_init):
    print("Number of Processors = ", num_processors)
    env = gym.make(env_name)
    env.reset()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = int(env.action_space.high[0])
    actor = Actor(state_dim, action_dim, max_action, layer_1=layer_1, layer_2=layer_2)
    print(actor)
    num_parameters = actor.get_size()
    file_name_summary = str(shared_directory) + "/summary.csv"
    write_history(file_name=file_name_summary,
                  file_contents="num_episodes;num_steps;min_fitness;mean_fitness;max_fitness;std_fitness;sum_timesteps;time_taken")
    open_es = CMAES(num_params=num_parameters, popsize=population_size, sigma_init=sigma_init,
                    weight_decay=weight_decay, rank_fitness=rank_fitness, mu_init=mu_init)
    num_steps = 0
    num_episodes = 0
    num_generations = 0

    total_time_seconds = 0.0

    min_fitness = 0.0
    mean_fitness = 0.0
    max_fitness = 0.0
    std_fitness = 0.0

    min_time = 0.0
    mean_time = 0.0
    max_time = 0.0
    std_time = 0.0
    file_name_history = str(shared_directory) + "/history_fitness.csv"
    file_name_time = str(shared_directory) + "/history_time.csv"

    file_name_pheno_mean = str(shared_directory)+"/history_pheno_mean_"
    file_name_pheno_std = str(shared_directory)+"/history_pheno_std_"


    history_fitness = []
    history_time = []
    generation_counter = 0
    # while num_steps < 100000:
    while num_episodes < (int(budget)):
        print("Generation = ", generation_counter)
        start = time.time()
        pool = multiprocessing.Pool(num_processors)
        solutions = open_es.ask()
        partial_fitness_function = partial(evaluate, env_name=env_name, state_dim=state_dim, USE_CUDA=USE_CUDA, action_dim=action_dim, max_action=max_action, num_evaluations=1, use_max_action=use_max_action, layer_1=layer_1, layer_2=layer_2)
        temp_list = []
        temp_list = list(zip(pool.map(partial_fitness_function, solutions)))
        fitness_list = []
        time_list = []
        for i in range(len(temp_list)):
            fitness_list.append(temp_list[i][0][0])
            time_list.append(temp_list[i][0][1])
        open_es.tell(fitness_list)
        result = open_es.result()

        end = time.time()
        sum_timesteps = numpy.sum(time_list)

        min_fitness = numpy.min(fitness_list)
        mean_fitness = round(numpy.mean(fitness_list), 3)
        max_fitness = numpy.max(fitness_list)
        std_fitness = round(numpy.std(fitness_list), 3)

        min_time = numpy.min(time_list)
        mean_time = round(numpy.mean(time_list), 3)
        max_time = numpy.max(time_list)
        std_time = round(numpy.std(time_list), 3)
        num_steps = num_steps + sum_timesteps
        num_episodes = num_episodes + (num_evaluations * population_size)


        num_generations = num_generations + 1
        file_history = open(file_name_history, "a+")
        file_history.write(listToString(fitness_list))
        file_history.write("\n")
        file_history.close()
        file_time = open(file_name_time, "a+")
        file_time.write(listToString(time_list))
        file_time.write("\n")
        file_time.close()
        pool.close()
        generation_counter = generation_counter + 1
        total_time_seconds = total_time_seconds + (end - start)
        del pool, temp_list, solutions, partial_fitness_function
        print_string = '{0};{1};{2};{3};{4};{5};{6};{7}_seconds'.format(
            num_episodes,
            num_steps,
            min_fitness,
            mean_fitness,
            max_fitness,
            std_fitness,
            sum_timesteps,
            total_time_seconds)
        print(print_string)
        write_history(file_name=file_name_summary, file_contents=print_string)

    pheno_mean, pheno_std = open_es.get_mu(), open_es.get_sigma()
    mean_mu = []
    mean_mu.append(numpy.mean(pheno_mean))
    print("Mean Mu = ", mean_mu)
    std_mu = []
    std_mu.append(numpy.std(pheno_mean))
    print("Std Mu = ", std_mu)
    numpy.savetxt(file_name_pheno_mean + '.csv', pheno_mean, delimiter=',')
    numpy.savetxt(file_name_pheno_mean + "mean_" + '.csv', mean_mu, delimiter=',')
    numpy.savetxt(file_name_pheno_mean + "std_" + '.csv', std_mu, delimiter=',')


if __name__ == "__main__":
    '''
def main(env_name, USE_CUDA, rank_fitness, num_processors, population_size, num_evaluations, sigma_init,
         weight_decay, shared_directory, budget):
    '''
    parser = argparse.ArgumentParser(description='Benchmark CMA-ES.')
    #Ant
    #use_default_layers 0 = [64, 32]
    #use_default_layers 1 = [64, 64]

    #HalfCheetah
    #use_default_layers 0 = [64, 32]
    #use_default_layers 1 = [64, 64]

    #Hopper
    #use_default_layers 0 = [75, 15]
    #use_default_layers 1 = [64, 64]

    #MinitaurBulletEnv
    #use_default_layers 0 = [64, 32]
    #use_default_layers 1 = [64, 64]

    parser.add_argument('--env_name', type=str, help='The name of the Environment.', default="InvertedPendulumPyBulletEnv-v0")
    parser.add_argument('--USE_CUDA', type=int, help='Whether to use CUDA devices if available.', default=0)
    parser.add_argument('--rank_fitness', type=int, help='Whether to use Rank Fitness in ES.', default=1)
    parser.add_argument('--num_processors', type=int, help='Number of processors to use.', default=4)
    parser.add_argument('--population_size', type=int, help='Size of the ES population', default=16)
    parser.add_argument('--num_evaluations', type=int, help='Number of Evaluations.', default=1)
    parser.add_argument('--shared_directory', type=str, help='A directory that is accessible for all processes, e.g. a NFS share.', default="es_output/")
    parser.add_argument('--max_num_episodes', type=float, help='Maximum Number of Episodes to use', default=4800000)
    parser.add_argument('--mu_init', type=float, help='Starting Mu Value.', default=0.0)
    parser.add_argument('--sigma_init', type=float, help='Starting Sigma Value.', default=0.10)
    parser.add_argument('--weight_decay', type=float, help='Minimum budget used during the optimization.', default=0.01)
    parser.add_argument('--use_default_layers', type=int, help='Use Default hidden layers size of 64', default=0)
    parser.add_argument('--use_max_action', type=int, help='Whether Policy uses Max Action or Not', default=1)
    args = parser.parse_args()
    env_name = args.__getattribute__("env_name")
    USE_CUDA = args.__getattribute__("USE_CUDA")
    if USE_CUDA == 0:
        USE_CUDA = False
    else:
        USE_CUDA = True
    num_processors = args.__getattribute__("num_processors")
    population_size = args.__getattribute__("population_size")
    num_evaluations = args.__getattribute__("num_evaluations")
    shared_directory = args.__getattribute__("shared_directory")
    shared_directory = shared_directory + env_name + "/" + str(time.time())
    max_num_episodes = args.__getattribute__("max_num_episodes")
    mu_init = args.__getattribute__("mu_init")
    sigma_init = args.__getattribute__("sigma_init")
    weight_decay = args.__getattribute__("weight_decay")
    rank_fitness = args.__getattribute__("rank_fitness")
    if rank_fitness == 0:
        rank_fitness = False
    else:
        rank_fitness = True
    use_default_layers = args.__getattribute__("use_default_layers")
    if use_default_layers == 0:
        use_default_layers = False
    else:
        use_default_layers = True
    use_max_action = args.__getattribute__("use_max_action")
    if use_max_action == 0:
        use_max_action = False
    else:
        use_max_action = True
    es = "CMA-ES"
    game = get_game(env_name=env_name, default=use_default_layers)
    layer_1 = game.layers[0]
    layer_2 = game.layers[1]

    argument_settings_single(env_name=env_name, USE_CUDA=USE_CUDA, rank_fitness=rank_fitness, num_processors=num_processors, population_size=population_size, num_evaluations=num_evaluations, sigma_init=sigma_init, weight_decay=weight_decay, shared_directory=shared_directory, budget=max_num_episodes, es=es, use_default_layers=use_default_layers, layer_1=layer_1, layer_2=layer_2, use_max_action=use_max_action, mu_init=mu_init)

    print(args)
    main(env_name=env_name, USE_CUDA=USE_CUDA, rank_fitness=rank_fitness, num_processors=num_processors, population_size=population_size, num_evaluations=num_evaluations, sigma_init=sigma_init, weight_decay=weight_decay, shared_directory=shared_directory, budget=max_num_episodes, use_default_layers=use_default_layers, layer_1=layer_1, layer_2=layer_2, use_max_action=use_max_action, mu_init=mu_init)
