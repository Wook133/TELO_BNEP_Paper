import numpy
import os
import time
import config
import csv
import gym
import pybulletgym
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import snn_dataset, snn
import pybullet_envs


def to_string(action, observation, subsequent_observation, reward=None):
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
        act = action[0]
        line += str(act[0])
        for index in range(len(act) - 1):
            line += "," + str(act[index + 1])
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
        #act = action[0]
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


def snn_training_data_generate(path, action, observation, subsequent_observation, reward=None, validation_probability=0.1):
    """
    :param path:
    :param action:
    :param observation:
    :param subsequent_observation:
    :param reward:
    :return:
    """
    s = path + "train"
    fileDataSNN = open(s + ".csv", "a+")
    s = path + "validation"
    validation_set = open(s + ".csv", "a+")
    action = action * 1.0
    #print("action = ", action)
    probability = numpy.random.uniform(0.0, 1.0)
    if reward is None:
        row = to_string(action, observation, subsequent_observation)
    else:
        row = to_string(action, observation, subsequent_observation, reward)
    if len(row) > 50:
        if probability > validation_probability:
            fileDataSNN.write(row)
            fileDataSNN.write("\n")  # write new line
        else:
            validation_set.write(row)
            validation_set.write("\n")  # write new line
    else:
        print("Caught problem which leads to NAN tensor?")
    fileDataSNN.close()
    validation_set.close()


def generate_training_data(path, action, observation, subsequent_observation, reward=None, validation_probability=0.1):
    """
    :param path:
    :param action:
    :param observation:
    :param subsequent_observation:
    :param reward:
    :return:
    """
    s = path + "train"
    fileDataSNN = open(s + ".csv", "a+")
    s = path + "validation"
    validation_set = open(s + ".csv", "a+")
    action = action * 1.0
    #print("action = ", action)
    probability = numpy.random.uniform(0.0, 1.0)
    if reward is None:
        row = to_pybullet_string(action, observation, subsequent_observation)
    else:
        row = to_pybullet_string(action, observation, subsequent_observation, reward)
    print("Row = ", row)
    if len(row) > 50:
        if probability > validation_probability:
            fileDataSNN.write(row)
            fileDataSNN.write("\n")  # write new line
        else:
            validation_set.write(row)
            validation_set.write("\n")  # write new line
    else:
        print("Caught problem which leads to NAN tensor?")
    fileDataSNN.close()
    validation_set.close()


def generate_row(action, observation, subsequent_observation, reward=None):
    """
    :param path:
    :param action:
    :param observation:
    :param subsequent_observation:
    :param reward:
    :return:
    """
    print("General Kenobi")
    action = action * 1.0
    probability = numpy.random.uniform(0.0, 1.0)
    if reward is None:
        row = to_pybullet_string(action, observation, subsequent_observation)
    else:
        row = to_pybullet_string(action, observation, subsequent_observation, reward)
    return row


def rows_to_save(list_rows, path, validation_probability):
    start = time.time()
    print("Rows to Save")
    s = path + "train"
    fileDataSNN = open(s + ".csv", "a+")
    s = path + "validation"
    validation_set = open(s + ".csv", "a+")
    is_string = False
    if len(list_rows) > 0:
        is_string = isinstance(list_rows[0], str)
        print("is list_rows[0] a string? ", is_string, "____ ")
    if not is_string:
        for list in list_rows:
            for row in list:
                probability = numpy.random.uniform(0.0, 1.0)
                if len(row) > 50:
                    if probability > validation_probability:
                        fileDataSNN.write(row)
                        fileDataSNN.write("\n")  # write new line
                    else:
                        validation_set.write(row)
                        validation_set.write("\n")  # write new line
                else:
                    print("Caught problem which leads to NAN tensor? ", row)
    else:
        for row in list_rows:
            probability = numpy.random.uniform(0.0, 1.0)
            if len(row) > 50:
                if probability > validation_probability:
                    fileDataSNN.write(row)
                    fileDataSNN.write("\n")  # write new line
                else:
                    validation_set.write(row)
                    validation_set.write("\n")  # write new line
            else:
                print("Caught problem which leads to NAN tensor? ", row)
    fileDataSNN.close()
    validation_set.close()
    end = time.time()
    print("Rows Saved. Took ", end-start, " seconds.")


def save_policy(path, policy_parameters, game=None):
    file_policy = open(path + "policy.txt", "a+")
    if game is not None:
        file_policy.write(str(game))
        file_policy.write("\n")
    for i in policy_parameters:
        temp = str(i) + ","
        file_policy.write(temp)
    file_policy.close()


def path_maker(env_name="CartPole-v1", experiment_name=None):
    time_now = time.time()
    path = env_name
    if experiment_name is not None:
        path = experiment_name+'/'
        if not os.path.exists(path):
            os.mkdir(path)
    path = path+env_name
    if not os.path.exists(path):
        os.mkdir(path)
    path = path+'/'+str(time_now)+'/'
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def get_game(env_name):
    game = None
    if "Gravitar-ramDeterministic-v4" in env_name:
        game = config.gravitar
    elif "Walker2DPyBulletEnv-v0" in env_name:
        game = config.twodwalker
    elif "AntPyBulletEnv-v0" in env_name:
        game = config.ant
    elif "BipedalWalker-v2" in env_name:
        game = config.biped
    elif "SpaceInvaders-ram-v4" in env_name:
        game = config.space_invaders_ram
    elif "LunarLander-v2" in env_name:
        game = config.lunar
    elif "Walker2DBulletEnv-v0" in env_name:
        game = config.bullet_walker
    elif "AntBulletEnv-v0" in env_name:
        game = config.bullet_ant
    elif "HopperBulletEnv-v0" in env_name:
        game = config.bullet_hopper
    elif "RacecarBulletEnv-v0" in env_name:
        game = config.bullet_racecar
    elif "MinitaurBulletEnv-v0" in env_name:
        game = config.bullet_minitaur
    elif "MinitaurBulletDuckEnv-v0" in env_name:
        game = config.bullet_minitaur_duck
    elif "HalfCheetahBulletEnv-v0" in env_name:
        game = config.bullet_cheetah
    elif "CarRacing-v0" in env_name:
        game = config.twod_car
    else:
        game = config.cartpole
    return game


#Update Parameters
def argument_settings(full_path, env_name, processes, generations, sim_generations, sim_train, mu, sim_mu, es_lambda, sim_lambda, save_training_data, hof_size, mate_crossover_alpha, mutate_c, mutate_indpb, selection_tournsize, cxpb, mutpb, exp_name, path, number_of_trials, size_of_hidden_layer, min_value, max_value, min_strategy_value, max_strategy_value, save_reward, done_limit, bns_cycles, use_cuda, snn_dropout, preprocessing=None, net=None, optim=None):
    with open(full_path + 'parameter_settings.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["env_name", "processes", "generations", "sim_generations", "sim_train", "mu", "sim_mu", "es_lambda", "sim_lambda", "save_training_data", "hof_size", "mate_crossover_alpha", "mutate_c", "mutate_indpb", "selection_tournsize", "cxpb", "mutpb", "exp_name", "path", "number_of_trials", "size_of_hidden_layer", "min_value", "max_value", "min_strategy_value", "max_strategy_value", "save_reward", "done_limit", "bns_cycles", "use_cuda", "snn_dropout", "preprocessing", "snn", "optimizer"])
        writer.writerow([str(env_name), str(processes), str(generations), str(sim_generations), str(sim_train), str(mu), str(sim_mu), str(es_lambda), str(sim_lambda), str(save_training_data), str(hof_size), str(mate_crossover_alpha), str(mutate_c), str(mutate_indpb), str(selection_tournsize), str(cxpb), str(mutpb), str(exp_name), str(path), str(number_of_trials), str(size_of_hidden_layer), str(min_value), str(max_value), str(min_strategy_value), str(max_strategy_value), str(save_reward), str(done_limit), str(bns_cycles), str(use_cuda), str(snn_dropout), str(preprocessing), str(net), str(optim)])


def track_simulator_fitness(path, cycle, r2_train, mse_train, r2_val, mse_val):
    file_snn_fitness = open(path + '/simulator_fitness.csv', "a+")
    file_snn_fitness.write(str(cycle) + "," + str(r2_train) + "," + str(mse_train) + "," + str(r2_val) + "," + str(mse_val))
    file_snn_fitness.write("\n")
    file_snn_fitness.close()


def switch_type(argument):
    type_action = str(type(argument))
    if "int" in type_action:
        return 1
    elif "tuple" in type_action:
        return len(argument)
    elif "ndarray" in type_action:
        b = numpy.ravel(argument)
        return b.shape[0]
    elif "float" in type_action:
        return 1


def generate_config_file(path, env_name, save_reward=False):
    """
    Assumes that action size, x and y size remain constant
    Create Config file to help generate SNN input and output shape
    :param path: path to folder to save experiment
    :param env_name: Name of the gym environment
    :return:
    """
    env = gym.make(env_name)
    file_snn_data_config = open(path + '/config.csv', "a+")
    action_sample = env.action_space.sample()
    action_length = switch_type(action_sample)
    env.reset()
    obs, reward, done, info = env.step(action_sample)
    # get in and out dimensions
    x_size = int(switch_type(obs) + switch_type(action_sample))  #
    if save_reward:
        y_size = int(switch_type(obs) + switch_type(reward))
    else:
        y_size = int(switch_type(obs))
    # write to SNN config file
    file_snn_data_config.write(env_name + "," + str(x_size) + "," + str(y_size))
    file_snn_data_config.write("\n")
    file_snn_data_config.close()
    # Close environment and file
    env.close()
    file_snn_data_config.close()
    print("Generated Config File")


def train_simulator(simulator, path, epochs, criterion, optimizer, batch_size, shuffle, checkpoint=False, env_name=None, reward_pos=5, num_workers=0, cycle=None, round_to=0, use_cuda=True, evaluate_fitness=True, normalization_type="none", dynamics=False, reward=True):
    print("env_name= ", env_name)
    if dynamics:
        print("Using Dynamics Model")
        #return train_dynamic_snn(simulator, path, epochs, criterion, optimizer, batch_size, shuffle, env_name, checkpoint=checkpoint, reward_pos=reward_pos, num_workers=num_workers, cycle=cycle, round_to=round_to, use_cuda=use_cuda, evaluate_fitness=evaluate_fitness, normalization_type=normalization_type)
        #return train_dynamic_snn(simulator, path, epochs, criterion, optimizer, batch_size, shuffle, env_name,
        #                         checkpoint=checkpoint, reward_pos=reward_pos, num_workers=num_workers, cycle=cycle,
        #                         round_to=round_to, use_cuda=use_cuda, evaluate_fitness=evaluate_fitness,
        #                         normalization_type=normalization_type, reward=reward)
        return train_dynamic_snn_till_validation(simulator, path, epochs, criterion, optimizer, batch_size, shuffle, env_name,
                                 checkpoint=checkpoint, reward_pos=reward_pos, num_workers=num_workers, cycle=cycle,
                                 round_to=round_to, use_cuda=use_cuda, evaluate_fitness=evaluate_fitness,
                                 normalization_type=normalization_type, reward=reward, training_cycles=100)
    else:
        print("Using Non-Dynamics Model")
        return train_snn(simulator, path, epochs, criterion, optimizer, batch_size, shuffle, checkpoint=False, env_name=env_name, reward_pos=reward_pos, num_workers=num_workers, cycle=cycle, round_to=round_to, use_cuda=use_cuda, evaluate_fitness=evaluate_fitness, normalization_type=normalization_type)


def train_snn(simulator, path, epochs, criterion, optimizer, batch_size, shuffle, checkpoint=False, env_name=None, reward_pos=5, num_workers=0, cycle=None, round_to=0, use_cuda=True, evaluate_fitness=True, normalization_type="none"):
    """
    Train a Neural Network
    :param simulator: the NN to train
    :param path: the path to the training data
    :param epochs: the number of epochs to run for
    :param criterion: the criterion, e.g. criterion = torch.nn.MSELoss()
    :param optimizer: the optimizer to use, e.g. optimizer = optim.Adam(net.parameters(), weight_decay=0.0001)
    :param batch_size: size of batches
    :param shuffle: whether to shuffle
    :param checkpoint: whether to checkpoint
    :return: trained simulator
    """

    if env_name is not None:
        env = gym.make(env_name)
        ob = env.reset()
        sample_action = env.action_space.sample()

        obs_high = env.observation_space.high
        obs_low = env.observation_space.low
        if 2 > 84:
            action_low = env.action_space.low
            action_high = env.action_space.high
        reward_pos = len(ob) + 1
        print("Position of Reward = ", reward_pos)
        env.close()

    start = time.time()
    simulator = simulator.train()
    if "PyBullet" in env_name:
        train_loader = snn_dataset.SNN_Dataset_PyBullet(directory=path, type='train')
    else:
        train_loader = snn_dataset.SNN_Dataset(directory=path, type_set='train')
    # Use Batching and Shuffle
    # https://conorsdatablog.wordpress.com/2018/05/03/up-and-running-with-pytorch-minibatching-dataloading-and-model-building/
    if "none" not in normalization_type:
        x_min, x_mean, x_max, x_std, y_min, y_mean, y_max, y_std = train_loader.get_stats()
        print("X Min = ", x_min)
        print("X Mean = ", x_mean)
        print("X Max = ", x_max)
        print("X Std = ", x_std)
        print("Y Min = ", y_min)
        print("Y Mean = ", y_mean)
        print("Y Max = ", y_max)
        print("Y Std = ", y_std)
    else:
        print("Not normalizing")

    train_loader = DataLoader(train_loader, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    print("Loaded Training Data")
    # Pass SNN to GPU
    if use_cuda:
        use_cuda = torch.cuda.is_available()
    print("use Cuda = ", use_cuda)

    dev = torch.device("cuda:0" if use_cuda else "cpu")
    print("Available device = ", dev)
    device = torch.device(dev)
    simulator = simulator.to(device)
    print("Batch Size = ", batch_size)
    print("Optimizer = ", optimizer)


    # https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    for epoch in range(1, epochs):  ## run the model for 10 epochs
        train_loss, valid_loss = [], []
        epoch_time = time.time()
        for i, (x, y) in enumerate(train_loader, 0):
            # print("vanilla x = ", type(x))
            if "none" not in normalization_type:
                #print("Doing normalization")
                x = x.numpy()
                x = snn_dataset.normalization(x=x, normalization_type=normalization_type, x_min=x_min, x_mean=x_mean, x_max=x_max, x_std=x_std)
                x = torch.from_numpy(x)
                y = y.numpy()
                y = snn_dataset.normalization(x=y, normalization_type=normalization_type, x_min=y_min, x_mean=y_mean, x_max=y_max, x_std=y_std)
                y = torch.from_numpy(y)
            if use_cuda:
                x = x.to(device)
                # print("type x to device = ", type(x))
                y = y.to(device)
            # x, y = Variable(x.to(device)), Variable(y.to(device))
            x, y = Variable(x), Variable(y)
            # print("Variable x = ", type(x))
            # forward pass: compute predicted y by passing x to the model
            x = x.float()

            #print("Type of x = ", type(x))
            # print("Float x = ", type(x))
            y_pred = simulator(x)
            # compute and print loss
            loss = criterion(y_pred, y.float())
            # print(epoch, i, loss.data)

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 1 == 0:
            print("Epoch:", epoch, " took ", time.time() - epoch_time, "seconds to run, with loss = ", str(loss))

    # Reset Trainloader
    end = time.time()
    print("Simulator training took ", end - start, " seconds")
    simulator = simulator.eval()
    start = time.time()
    if evaluate_fitness:
        if "PyBullet" in env_name:
            train_loader = snn_dataset.SNN_Dataset_PyBullet(directory=path, type='train')
        else:
            train_loader = snn_dataset.SNN_Dataset(directory=path, type_set='train')
        r2_train, mse_train = snn.evaluate_r2_mse(simulator, train_loader, round_reward=False, reward=False, position_reward=reward_pos, device=device, round_to=0, normalization_type=normalization_type)

        end = time.time()
        print("r2 = ", r2_train, " took ", end - start, " seconds to evaluate")
        print("mse = ", mse_train, " took ", end - start, " seconds to evaluate")
        r2_train, mse_train = snn.evaluate_r2_mse_reverse_normalization(simulator, train_loader, round_reward=False, reward=False, position_reward=reward_pos, device=device, round_to=0, normalization_type=normalization_type)
        end = time.time()
        print("reverse r2 = ", r2_train, " took ", end - start, " seconds to evaluate")
        print("reverse mse = ", mse_train, " took ", end - start, " seconds to evaluate")
        if "PyBullet" in env_name:
            validation_loader = snn_dataset.SNN_Dataset_PyBullet(directory=path, type='validation')
        else:
            validation_loader = snn_dataset.SNN_Dataset(directory=path, type_set='validation')
        r2_val, mse_val = snn.evaluate_r2_mse(simulator, validation_loader, reward=False, round_reward=False, position_reward=reward_pos, device=device, round_to=0, normalization_type=normalization_type)
        print("Validation r2 = ", r2_val, " took ", end - start, " seconds to evaluate")
        print("Validation mse = ", mse_val, " took ", end - start, " seconds to evaluate")

        r2_val, mse_val = snn.evaluate_r2_mse_reverse_normalization(simulator, validation_loader, reward=False, round_reward=False, position_reward=reward_pos, device=device, round_to=0, normalization_type=normalization_type)
        print("reverse Validation r2 = ", r2_val, " took ", end - start, " seconds to evaluate")
        print("reverse Validation mse = ", mse_val, " took ", end - start, " seconds to evaluate")
        track_simulator_fitness(path, cycle, r2_train, mse_train, r2_val, mse_val)
    # Pass SNN to CPU
    cpu_device = torch.device("cpu")
    simulator = simulator.to(cpu_device)
    if "none" in normalization_type:
        print("You're shorter")
        return simulator
    else:
        print("Than I expected")
        return simulator, x_min, x_mean, x_max, x_std, y_min, y_mean, y_max, y_std


def get_lengths(env_name):
    print("Get Lengths = ", env_name)
    env = gym.make(env_name)
    action_sample = env.action_space.sample()
    env.reset()
    obs, reward, done, info = env.step(action_sample)
    action_length = switch_type(action_sample)
    observation_length = switch_type(obs)
    reward_length = switch_type(reward)
    return action_length, observation_length, reward_length


def train_dynamic_snn(simulator, path, epochs, criterion, optimizer, batch_size, shuffle, env_name, checkpoint=False, reward_pos=5, num_workers=0, cycle=None, round_to=0, use_cuda=True, evaluate_fitness=True, normalization_type="none", reward=False):
    """
    Train a Neural Network
    :param simulator: the NN to train
    :param path: the path to the training data
    :param epochs: the number of epochs to run for
    :param criterion: the criterion, e.g. criterion = torch.nn.MSELoss()
    :param optimizer: the optimizer to use, e.g. optimizer = optim.Adam(net.parameters(), weight_decay=0.0001)
    :param batch_size: size of batches
    :param shuffle: whether to shuffle
    :param checkpoint: whether to checkpoint
    :return: trained simulator
    """
    print("Training: train_dynamic_snn")
    env = gym.make(env_name)
    ob = env.reset()
    #sample_action = env.action_space.sample()
    #obs_high = env.observation_space.high
    #obs_low = env.observation_space.low
    #action_low = env.action_space.low
    #action_high = env.action_space.high
    reward_pos = len(ob) + 1
    print("Position of Reward = ", reward_pos)
    env.close()

    action_length, observation_length, reward_length = get_lengths(env_name)

    start = time.time()
    simulator = simulator.train()
    if "PyBullet" in env_name:
        train_loader = snn_dataset.SNN_Dataset_PyBullet(directory=path, type='train')
    else:
        train_loader = snn_dataset.SNN_Dataset(directory=path, type_set='train')
    # Use Batching and Shuffle
    # https://conorsdatablog.wordpress.com/2018/05/03/up-and-running-with-pytorch-minibatching-dataloading-and-model-building/
    if "none" not in normalization_type:
        x_min, x_mean, x_max, x_std, y_min, y_mean, y_max, y_std = train_loader.get_stats()
        print("X Min = ", x_min)
        print("X Mean = ", x_mean)
        print("X Max = ", x_max)
        print("X Std = ", x_std)
        print("Y Min = ", y_min)
        print("Y Mean = ", y_mean)
        print("Y Max = ", y_max)
        print("Y Std = ", y_std)

    train_loader = DataLoader(train_loader, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    print("Loaded Training Data")
    # Pass SNN to GPU
    if use_cuda:
        use_cuda = torch.cuda.is_available()
    print("use Cuda = ", use_cuda)

    dev = torch.device("cuda:0" if use_cuda else "cpu")
    print("Available device = ", dev)
    device = torch.device(dev)
    simulator = simulator.to(device)
    print("Batch Size = ", batch_size)
    print("Optimizer = ", optimizer)


    # https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    for epoch in range(1, epochs):  ## run the model for 10 epochs
        train_loss, valid_loss = [], []
        epoch_time = time.time()
        for i, (x, y) in enumerate(train_loader, 0):
            # print("vanilla x = ", type(x))
            if "none" not in normalization_type:
                #print("Doing normalization")
                x = x.numpy()
                x = snn_dataset.normalization(x=x, normalization_type=normalization_type,
                                              x_min=x_min, x_mean=x_mean, x_max=x_max,
                                              x_std=x_std)
                #cur_y = x[:, action_length:action_length + observation_length]
                x = torch.from_numpy(x)
                y = y.numpy()
                y = snn_dataset.normalization(x=y, normalization_type=normalization_type,
                                              x_min=y_min, x_mean=y_mean, x_max=y_max,
                                              x_std=y_std)
                y = torch.from_numpy(y)
            if use_cuda:
                x = x.to(device)
                # print("type x to device = ", type(x))
                y = y.to(device)
            cur_y = x[:, action_length:action_length + observation_length]
            # x, y = Variable(x.to(device)), Variable(y.to(device))
            x, y = Variable(x), Variable(y)
            # print("Variable x = ", type(x))
            # forward pass: compute predicted y by passing x to the model
            x = x.float()
            #add column of 0's for reward so prediction isn't skewed
            num_rows = cur_y.shape[0]
            zeros = numpy.zeros((num_rows, 1))
            cur_y = numpy.c_[cur_y, zeros]
            cur_y = torch.from_numpy(cur_y)
            y_pred = torch.add(cur_y, simulator(x))
            # compute and print loss
            #probability = numpy.random.uniform(0.0, 1.0)
            #if probability > 0.999:
            #    print("Difference between y_pred and y_double = ", (y_pred - y.double()))
            loss = criterion(y_pred, y.double())
            # print(epoch, i, loss.data)
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 1 == 0:
            print("Epoch:", epoch, " took ", time.time() - epoch_time, "seconds to run, with loss = ", str(loss))

    # Reset Trainloader
    end = time.time()
    print("Simulator training took ", end - start, " seconds")
    simulator = simulator.eval()
    start = time.time()
    if evaluate_fitness:
        if "PyBullet" in env_name:
            train_loader = snn_dataset.SNN_Dataset_PyBullet(directory=path, type='train')
        else:
            train_loader = snn_dataset.SNN_Dataset(directory=path, type_set='train')
        r2_train, mse_train = snn.evaluate_dynamic_r2_mse(env_name, simulator, train_loader, round_reward=False, reward=reward, position_reward=reward_pos, device=device, round_to=0, normalization_type=normalization_type)
        end = time.time()
        print("reverse r2 = ", r2_train, " took ", end - start, " seconds to evaluate")
        print("reverse mse = ", mse_train, " took ", end - start, " seconds to evaluate")
        if "PyBullet" in env_name:
            validation_loader = snn_dataset.SNN_Dataset_PyBullet(directory=path, type='validation')
        else:
            validation_loader = snn_dataset.SNN_Dataset(directory=path, type_set='validation')
        r2_val, mse_val = snn.evaluate_dynamic_r2_mse(env_name, simulator, validation_loader, reward=False, round_reward=reward, position_reward=reward_pos, device=device, round_to=0, normalization_type=normalization_type)
        print("Validation r2 = ", r2_val, " took ", end - start, " seconds to evaluate")
        print("Validation mse = ", mse_val, " took ", end - start, " seconds to evaluate")
        track_simulator_fitness(path, cycle, r2_train, mse_train, r2_val, mse_val)
    # Pass SNN to CPU
    cpu_device = torch.device("cpu")
    simulator = simulator.to(cpu_device)
    print("After training normalization_type = ", normalization_type)
    if "none" not in normalization_type:
        print("hello there")
        return simulator, x_min, x_mean, x_max, x_std, y_min, y_mean, y_max, y_std
    else:
        print("general kenobi")
        return simulator


def train_dynamic_snn_till_validation(simulator, path, epochs, criterion, optimizer, batch_size, shuffle, env_name, training_cycles, checkpoint=False, reward_pos=5, num_workers=0, cycle=None, round_to=0, use_cuda=True, evaluate_fitness=True, normalization_type="none", reward=True):
    """
    Train a Neural Network
    :param simulator: the NN to train
    :param path: the path to the training data
    :param epochs: the number of epochs to run for
    :param criterion: the criterion, e.g. criterion = torch.nn.MSELoss()
    :param optimizer: the optimizer to use, e.g. optimizer = optim.Adam(net.parameters(), weight_decay=0.0001)
    :param batch_size: size of batches
    :param shuffle: whether to shuffle
    :param checkpoint: whether to checkpoint
    :return: trained simulator
    """
    epochs_per_training_cycle = int(epochs / training_cycles)
    print("Training: train_dynamic_snn_till_validation,  epochs_per_training_cycle = ", epochs_per_training_cycle)
    env = gym.make(env_name)
    ob = env.reset()
    #sample_action = env.action_space.sample()
    #obs_high = env.observation_space.high
    #obs_low = env.observation_space.low
    #action_low = env.action_space.low
    #action_high = env.action_space.high
    reward_pos = len(ob) + 1
    print("Position of Reward = ", reward_pos)
    env.close()

    action_length, observation_length, reward_length = get_lengths(env_name)

    start = time.time()
    simulator = simulator.train()
    if "PyBullet" in env_name:
        train_loader = snn_dataset.SNN_Dataset_PyBullet(directory=path, type='train')
    else:
        train_loader = snn_dataset.SNN_Dataset(directory=path, type_set='train')
    # Use Batching and Shuffle
    # https://conorsdatablog.wordpress.com/2018/05/03/up-and-running-with-pytorch-minibatching-dataloading-and-model-building/
    if "none" not in normalization_type:
        x_min, x_mean, x_max, x_std, y_min, y_mean, y_max, y_std = train_loader.get_stats()
        print("X Min = ", x_min)
        print("X Mean = ", x_mean)
        print("X Max = ", x_max)
        print("X Std = ", x_std)
        print("Y Min = ", y_min)
        print("Y Mean = ", y_mean)
        print("Y Max = ", y_max)
        print("Y Std = ", y_std)

    print("Loaded Training Data")
    # Pass SNN to GPU
    if use_cuda:
        use_cuda = torch.cuda.is_available()
    print("use Cuda = ", use_cuda)

    dev = torch.device("cuda:0" if use_cuda else "cpu")
    print("Available device = ", dev)
    device = torch.device(dev)
    simulator = simulator.to(device)
    print("Batch Size = ", batch_size)
    print("Optimizer = ", optimizer)

    #training_cycles
    #epochs

    j = 0

    list_train_r2 = []
    list_validation_r2 = []

    previous_train_r_2 = -100.0
    previous_validation_r_2 = -100.0

    current_train_r_2 = -10.0
    current_validation_r_2 = -10.0

    j = 0
    while (j < training_cycles) and (current_validation_r_2 > previous_validation_r_2):# and (current_train_r_2 > previous_train_r_2):
        previous_train_r_2 = current_train_r_2
        previous_validation_r_2 = current_validation_r_2

        train_loader = DataLoader(train_loader, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        k = 0
        simulator = simulator.train()
        while k < epochs_per_training_cycle:
            epoch_time = time.time()
            for i, (x, y) in enumerate(train_loader, 0):
                # print("vanilla x = ", type(x))
                if "none" not in normalization_type:
                    x = x.numpy()
                    x = snn_dataset.normalization(x=x, normalization_type=normalization_type,
                                                  x_min=x_min, x_mean=x_mean, x_max=x_max,
                                                  x_std=x_std)
                    x = torch.from_numpy(x)
                    y = y.numpy()
                    y = snn_dataset.normalization(x=y, normalization_type=normalization_type,
                                                  x_min=y_min, x_mean=y_mean, x_max=y_max,
                                                  x_std=y_std)
                    y = torch.from_numpy(y)
                if use_cuda:
                    x = x.to(device)
                    y = y.to(device)
                #print("|x| = ", len(x), " size = ", list(x.size()))
                cur_y = x[:, action_length:action_length + observation_length]

                x, y = Variable(x), Variable(y)
                # forward pass: compute predicted y by passing x to the model
                x = x.float()
                # add column of 0's for reward so prediction isn't skewed
                num_rows = cur_y.shape[0]
                zeros = numpy.zeros((num_rows, 1))
                cur_y = numpy.c_[cur_y, zeros]
                cur_y = torch.from_numpy(cur_y)
                y_pred = torch.add(cur_y, simulator(x))
                # compute and print loss
                loss = criterion(y_pred, y.double())
                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if k % int(epochs_per_training_cycle*0.1) == 0:
                print("Training Cycle = ", j," Epoch:", k, " took ", time.time() - epoch_time, "seconds to run, with loss = ", str(loss))
            k = k + 1
        simulator = simulator.eval()
        if evaluate_fitness:
            if "PyBullet" in env_name:
                train_loader = snn_dataset.SNN_Dataset_PyBullet(directory=path, type='train')
            else:
                train_loader = snn_dataset.SNN_Dataset(directory=path, type_set='train')
            current_train_r_2, mse_train = snn.evaluate_dynamic_r2_mse(env_name, simulator, train_loader, round_reward=False,
                                                              reward=reward, position_reward=reward_pos, device=device,
                                                              round_to=5, normalization_type=normalization_type)
            end = time.time()
            if "PyBullet" in env_name:
                validation_loader = snn_dataset.SNN_Dataset_PyBullet(directory=path, type='validation')
            else:
                validation_loader = snn_dataset.SNN_Dataset(directory=path, type_set='validation')
            current_validation_r_2, mse_val = snn.evaluate_dynamic_r2_mse(env_name, simulator, validation_loader, reward=reward,
                                                          round_reward=False, position_reward=reward_pos, device=device,
                                                          round_to=5, normalization_type=normalization_type)
        list_train_r2.append(current_train_r_2)
        list_validation_r2.append(current_validation_r_2)
        j = j + 1
        print("J = ", j, " Training_cycles = ", training_cycles, " cur_train = ", current_train_r_2, " prev_cur_train = ", previous_train_r_2,
              " cur_val = ", current_validation_r_2, " prev_cur_val = ", previous_validation_r_2)
    save_accuracy(path=path, cycle=cycle, train=list_train_r2, val=list_validation_r2)
    # Reset Trainloader
    end = time.time()
    print("Simulator training took ", end - start, " seconds")
    simulator = simulator.eval()
    start = time.time()
    if evaluate_fitness:
        if "PyBullet" in env_name:
            train_loader = snn_dataset.SNN_Dataset_PyBullet(directory=path, type='train')
        else:
            train_loader = snn_dataset.SNN_Dataset(directory=path, type_set='train')
        r2_train, mse_train = snn.evaluate_dynamic_r2_mse(env_name, simulator, train_loader, round_reward=False, reward=reward, position_reward=reward_pos, device=device, round_to=0, normalization_type=normalization_type)
        end = time.time()
        print("reverse r2 = ", r2_train, " took ", end - start, " seconds to evaluate")
        print("reverse mse = ", mse_train, " took ", end - start, " seconds to evaluate")
        if "PyBullet" in env_name:
            validation_loader = snn_dataset.SNN_Dataset_PyBullet(directory=path, type='validation')
        else:
            validation_loader = snn_dataset.SNN_Dataset(directory=path, type_set='validation')
        r2_val, mse_val = snn.evaluate_dynamic_r2_mse(env_name, simulator, validation_loader, reward=reward, round_reward=False, position_reward=reward_pos, device=device, round_to=0, normalization_type=normalization_type)
        print("Validation r2 = ", r2_val, " took ", end - start, " seconds to evaluate")
        print("Validation mse = ", mse_val, " took ", end - start, " seconds to evaluate")
        track_simulator_fitness(path, cycle, r2_train, mse_train, r2_val, mse_val)
    # Pass SNN to CPU
    cpu_device = torch.device("cpu")
    simulator = simulator.to(cpu_device)
    print("After training normalization_type = ", normalization_type)
    if "none" not in normalization_type:
        print("hello there")
        return simulator, x_min, x_mean, x_max, x_std, y_min, y_mean, y_max, y_std
    else:
        print("general kenobi")
        return simulator


def save_accuracy(path, cycle, train, val):
    acc = zip(train, val)
    file_snn_fitness = open(path + '/'+str(cycle)+'_simulator_fitness.csv', "a+")
    for t, v in acc:
        file_snn_fitness.write(
            str(t) + "," + str(v))
        file_snn_fitness.write("\n")
    file_snn_fitness.close()


def get_reward_position(env_name):
    env = gym.make(env_name)
    env.reset()
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
    env.close()
    return len(obs) + 1


def get_max_episode(env_name):
    env = gym.make(env_name)
    env.reset()
    max_episode = env.spec.max_episode_steps
    env.close()
    del env
    return max_episode


def get_discrete_action(env_name):
    env = gym.make(env_name)
    env.reset()
    action_space = env.action_space
    discrete_action = isinstance(action_space, gym.spaces.discrete.Discrete)
    env.close()
    del env
    return discrete_action


def get_action_max_reward(env_name):
    env = gym.make(env_name)
    env.reset()
    action = env.action_space.sample()

    obs, reward, done, _ = env.step(action)
    reward_pos = len(obs) + 1

    action_space = env.action_space
    discrete_action = isinstance(action_space, gym.spaces.discrete.Discrete)

    max_episode = env.spec.max_episode_steps

    reward_threshold = env.spec.reward_threshold

    snn_input_size = switch_type(action) + switch_type(obs)
    snn_output_size = switch_type(reward) + switch_type(obs)

    env.close()
    del env
    return reward_pos, discrete_action, max_episode, reward_threshold, snn_input_size, snn_output_size



def generate_config_file(path, env_name, save_reward=False):
    """
    Assumes that action size, x and y size remain constant
    Create Config file to help generate SNN input and output shape
    :param path: path to folder to save experiment
    :param env_name: Name of the gym environment
    :return:
    """
    env = gym.make(env_name)
    file_snn_data_config = open(path + '/config.csv', "a+")
    action_sample = env.action_space.sample()
    action_length = switch_type(action_sample)
    env.reset()
    obs, reward, done, info = env.step(action_sample)
    # get in and out dimensions
    x_size = int(switch_type(obs) + switch_type(action_sample))  #
    if save_reward:
        y_size = int(switch_type(obs) + switch_type(reward))
    else:
        y_size = int(switch_type(obs))
    # write to SNN config file
    file_snn_data_config.write(env_name + "," + str(x_size) + "," + str(y_size))
    file_snn_data_config.write("\n")
    file_snn_data_config.close()
    # Close environment and file
    env.close()
    file_snn_data_config.close()
    print("Generated Config File")

def write_parameter_settings(time, env_name, num_processors, path, population_size, generations, use_cuda, bns_cycles, simulator_train, simulator_evaluations, real_evaluations, dynamics_model, standardization, multiobjective, recording_probability, parent_population_size):
    parameter_settings = open(path + '/parameter_settings.csv', "a+")
    parameter_settings.write("time" + "," + str(time))
    parameter_settings.write("\n")
    parameter_settings.write("env_name" + "," + str(env_name))
    parameter_settings.write("\n")
    parameter_settings.write("num_processors" + "," + str(num_processors))
    parameter_settings.write("\n")
    parameter_settings.write("path" + "," + str(path))
    parameter_settings.write("\n")
    parameter_settings.write("population_size" + "," + str(population_size))
    parameter_settings.write("\n")
    parameter_settings.write("generations" + "'," + str(generations))
    parameter_settings.write("\n")
    parameter_settings.write("use_cuda" + "," + str(use_cuda))
    parameter_settings.write("\n")
    parameter_settings.write("bns_cycles" + "," + str(bns_cycles))
    parameter_settings.write("\n")
    parameter_settings.write("simulator_train" + "," + str(simulator_train))
    parameter_settings.write("\n")
    parameter_settings.write("simulator_evaluations" + "," + str(simulator_evaluations))
    parameter_settings.write("\n")
    parameter_settings.write("real_evaluations" + "," + str(real_evaluations))
    parameter_settings.write("\n")
    parameter_settings.write("dynamics_model" + "," + str(dynamics_model))
    parameter_settings.write("\n")
    parameter_settings.write("standardization" + "," + str(standardization))
    parameter_settings.write("\n")
    parameter_settings.write("multiobjective" + "," + str(multiobjective))
    parameter_settings.write("\n")
    parameter_settings.write("recording_probability" + "," + str(recording_probability))
    parameter_settings.write("\n")
    parameter_settings.write("parent_population_size" + "," + str(parent_population_size))
    parameter_settings.close()




def initialize_simulator(env_name, simulator, dynamics, default=True):
    print("Initialize Simulator for ", env_name)
    real_env = gym.make(env_name)
    real_start_state = real_env.reset()
    size_obs = len(real_start_state)
    action_space = real_env.action_space
    real_env.close()

    #Get Action Space Type
    if isinstance(action_space, gym.spaces.discrete.Discrete):
        sim_env = snn.wrapper_SNN_discrete_action(starting_state=real_start_state, action_space=action_space, size_observation=size_obs, SNN=simulator, round_to=5, dynamics=dynamics, env_name=env_name)
    elif isinstance(action_space, gym.spaces.box.Box):
        sim_env = snn.wrapper_SNN_box_action(starting_state=real_start_state, action_space=action_space, size_observation=size_obs, SNN=simulator, round_to=5, dynamics=dynamics, env_name=env_name)
    #Get Observation Space Type?
    print("Done Initializing Simulator")
    return sim_env


def write_fitness_to_file(filename_path, fitness):
    try:
        str_write = ""
        for fit in fitness:
            str_write = str_write + str(fit) + ","
        str_write = str_write[:-1]
        fileData = open(filename_path, "a+")
        fileData.write(str_write)
        fileData.write("\n")  # write new line
    except:
        print("Writing fitness to file failed.")
    finally:
        fileData.close()

class mdp_row:
    def __init__(self, prior_observation, action, observation, reward):
        self.prior_observation = prior_observation
        self.action = action*1.0
        self.observation = observation
        self.reward = reward
    def get_print(self):
        action_list = isinstance(self.action, numpy.ndarray)
        line = ""
        if action_list:
            act = self.action[0]
            line += str(act[0])
            for index in range(len(act) - 1):
                line += "," + str(act[index + 1])
            line += "," + str(self.prior_observation[0])
        else:
            line = str(self.action) + ","
            line += str(self.prior_observation[0])

        for index in range(len(self.prior_observation) - 1):
            line += "," + str(self.prior_observation[index + 1])
        line += "," + str(self.observation[0])
        for index in range(len(self.observation) - 1):
            line += "," + str(self.observation[index + 1])
        if self.reward is None:
            return line
        else:
            line += "," + str(self.reward)
        return line


def mdp_list_writer(path, mdp_list, validation_probability=0.1):
    """
    :param path:
    :param action:
    :param observation:
    :param subsequent_observation:
    :param reward:
    :return:
    """
    s = path + "train"
    fileDataSNN = open(s + ".csv", "a+")
    s = path + "validation"
    validation_set = open(s + ".csv", "a+")
    probability = numpy.random.uniform(0.0, 1.0)
    for row in mdp_list:
        writable_row = row.get_print()
        if probability > validation_probability:
            fileDataSNN.write(writable_row)
            fileDataSNN.write("\n")  # write new line
        else:
            validation_set.write(writable_row)
            validation_set.write("\n")  # write new line
    fileDataSNN.close()
    validation_set.close()

def list_to_string(lister):
    str_out = ""
    for i in lister:
        str_out = str_out + str(i) + ","
    str_out = str_out[:-1]
    return str_out


def increase_population_size(pop_size, max_pop_size, generation_counter, k=1.5, g=50):
    '''
    Increase the Population Size if Condition is met, very simple continuation of IPOP-CMA-ES idea without restart
    %E:\Google Drive folder\Masters\Machine Learning Books\Evolutionary Computation\CMA-ES\A restart CMA evolution strategy with increasing population size.pdf
@inproceedings{auger2005restart,
  title={A restart CMA evolution strategy with increasing population size},
  author={Auger, Anne and Hansen, Nikolaus},
  booktitle={2005 IEEE congress on evolutionary computation},
  volume={2},
  pages={1769--1776},
  year={2005},
  organization={IEEE}
}
    :param pop_size: current population size
    :param max_pop_size: the maximum possible population size
    :param k: factor to multiply current population size
    :param generation_counter: current generation counter
    :param g: when to increase population size
    :return: new population size
    '''
    cur_pop_size = pop_size
    if generation_counter % g == 0:
        cur_pop_size = cur_pop_size * k
        if cur_pop_size > max_pop_size:
            cur_pop_size = max_pop_size
    return int(cur_pop_size)


def calculate_maximum_timestep_count(threshold_timestep, threshold_reward, reward_history, cur_timestep,
                                    cur_reward, check_A, check_B, threshold_A=0.9, threshold_B=0.75, rate_of_change_reward=1.5, rate_of_change_timestep=1.5):
    '''
    An algorithm to increase the maximum time-steps allowed per given episode. Typical Policy Optimisation algorithms
    can update their policies \theta multiple times per episode and exploit the presence of frequent reward signals.
    Prior BNE algorithms calculated the fitness over the entire episode, this may have "wasted" information.
    :param threshold_timestep: The maximum number of time-steps for a given environment
    :param threshold_reward: The maximum total reward signal offered by the environment
    :param reward_history: The history of fitness rewards
    :param cur_timestep: the current threshold number of time-steps to evaluate a policy
    :param cur_reward:  the current threshold total reward to evaluate a policy
    :param check_A: the primary function that checks whether cur_timestep and cur_reward should be updated
    :param threshold_A: the threshold check A has to beat, multiply this number by
     reward per timestep with current time-step
    :param check_B: the secondary function that checks whether cur_timestep and cur_reward should be updated
    :param threshold_B: the threshold check B has to beat, multiply this number by
     reward per timestep with current time-step
    :param rate_of_change: rate to change the reward and timestep
    :return: updated time-step and reward
    '''
    print("Threshold Time steps = ", threshold_timestep, " cur_reward= ", threshold_reward)
    cur_max = numpy.max(reward_history[len(reward_history)-1])
    print("cur_max = ", cur_max, " > cur_reward = ", cur_reward)
    if cur_max > (cur_reward * threshold_A):
        new_reward = cur_reward * rate_of_change_reward
        if new_reward > threshold_reward:
            new_reward = threshold_reward
        new_timestep = cur_timestep * rate_of_change_timestep
        if new_timestep > threshold_timestep:
            new_timestep = threshold_timestep
        print("0) Timesteps: ", cur_timestep, " => ", new_timestep)
        print("0) Reward: ", cur_reward, " => ", new_reward)
        return new_timestep, new_reward
    else:
        a = (threshold_reward / threshold_timestep)
        new_reward = cur_reward
        new_timestep = cur_timestep
        print("A = ", a)
        base_A = a * threshold_A * cur_timestep
        metric_A = check_A(reward_history=reward_history[len(reward_history)-1], base_A=base_A, window_size=1, percentile=95.0)

        if metric_A:
            new_reward = cur_reward * rate_of_change_reward
            if new_reward > threshold_reward:
                new_reward = threshold_reward
            new_timestep = cur_timestep * rate_of_change_timestep
            if new_timestep > threshold_timestep:
                new_timestep = threshold_timestep
        else:
            base_B = a * threshold_B * cur_timestep
            metric_B = check_B(reward_history=reward_history, base_B=base_B, window_size=5, percentile=75.0)
            if metric_B:
                new_reward = cur_reward * rate_of_change_timestep
                if new_reward > threshold_reward:
                    new_reward = threshold_reward
                new_timestep = cur_timestep * rate_of_change_timestep
                if new_timestep > threshold_timestep:
                    new_timestep = threshold_timestep
        print("Timesteps: ", cur_timestep, " => ", new_timestep)
        print("Reward: ", cur_reward, " => ", new_reward)

        return new_timestep, new_reward


def calculate_maximum_timestep_count_0(threshold_timestep, threshold_reward, reward_history, cur_timestep,
                                    cur_reward, threshold_A=0.9, threshold_B=0.75, threshold_C=0.5,
                                       rate_of_change_reward=1.5, rate_of_change_timestep=1.5,
                                       window_size=5):
    '''
    An algorithm to increase the maximum time-steps allowed per given episode. Typical Policy Optimisation algorithms
    can update their policies \theta multiple times per episode and exploit the presence of frequent reward signals.
    Prior BNE algorithms calculated the fitness over the entire episode, this may have "wasted" information.
    :param threshold_timestep: The maximum number of time-steps for a given environment
    :param threshold_reward: The maximum total reward signal offered by the environment
    :param reward_history: The history of fitness rewards
    :param cur_timestep: the current threshold number of time-steps to evaluate a policy
    :param cur_reward:  the current threshold total reward to evaluate a policy
    :param check_A: the primary function that checks whether cur_timestep and cur_reward should be updated
    :param threshold_A: the threshold check A has to beat, multiply this number by
     reward per timestep with current time-step
    :param check_B: the secondary function that checks whether cur_timestep and cur_reward should be updated
    :param threshold_B: the threshold check B has to beat, multiply this number by
     reward per timestep with current time-step
    :param rate_of_change: rate to change the reward and timestep
    :return: updated time-step and reward

    '''
    a = (threshold_reward / threshold_timestep)
    print("Threshold Time steps = ", threshold_timestep, " cur_reward= ", threshold_reward, " ideal reward for this timestep = ", a*cur_timestep)
    cur_max = numpy.max(reward_history[len(reward_history)-1])
    cur_median = numpy.percentile(reward_history[len(reward_history) - 1], 50.0)
    cur_mean = numpy.mean(reward_history[len(reward_history) - 1])
    print("cur_max = ", cur_max, " > cur_reward = ", cur_reward)
    if (cur_median > cur_reward) or (cur_mean > cur_reward):
        new_reward = cur_reward * (rate_of_change_reward*2.0)
        if new_reward > threshold_reward:
            new_reward = threshold_reward
        new_timestep = cur_timestep * (rate_of_change_timestep*1.5)
        if new_timestep > threshold_timestep:
            new_timestep = threshold_timestep
        print("Threshold Median or Threshold: Reward = ", cur_reward, "=> ", new_reward, " time-steps = ", cur_timestep, "=> ",
              new_timestep)
        return new_timestep, new_reward
    if cur_max > cur_reward:
        new_reward = cur_reward * rate_of_change_reward
        if new_reward > threshold_reward:
            new_reward = threshold_reward
        new_timestep = cur_timestep * rate_of_change_timestep
        if new_timestep > threshold_timestep:
            new_timestep = threshold_timestep
        print("Threshold 0: Reward = ", cur_reward, "=> ", new_reward, " time-steps = ", cur_timestep, "=> ", new_timestep)
        return new_timestep, new_reward

    print("Threshold A:  cur_max =", cur_max, " > ", (a*cur_timestep*threshold_A))
    if cur_max > (a*cur_timestep*threshold_A):
        new_reward = cur_reward * rate_of_change_reward
        if new_reward > threshold_reward:
            new_reward = threshold_reward
        new_timestep = cur_timestep * rate_of_change_timestep
        if new_timestep > threshold_timestep:
            new_timestep = threshold_timestep
        print("Threshold A: Reward = ", cur_reward, "=> ", new_reward, " time-steps = ", cur_timestep, "=> ", new_timestep)
        return new_timestep, new_reward
    else:
        len_history = len(reward_history)
        if len_history >= window_size:
            threshold_met_counter = 0
            window = reward_history[-window_size:]
            beat_b = a * cur_timestep * threshold_B
            beat_b_0 = cur_max * threshold_B
            print("Threshold B: ", beat_b)
            for cur_frame in window:
                cur_percentile = numpy.percentile(a=cur_frame, q=90.0)
                print(cur_percentile)
                if (cur_percentile > beat_b) or (cur_percentile > beat_b_0):
                    threshold_met_counter = threshold_met_counter + 1
            if threshold_met_counter == (window_size-1):
                new_reward = cur_reward * rate_of_change_reward
                if new_reward > threshold_reward:
                    new_reward = threshold_reward
                new_timestep = cur_timestep * rate_of_change_timestep
                if new_timestep > threshold_timestep:
                    new_timestep = threshold_timestep
                print("Threshold B: Reward = ", cur_reward, "=> ", new_reward, " time-steps = ", cur_timestep, "=> ",
                      new_timestep)
                print("Timesteps: ", cur_timestep, " => ", new_timestep)
                print("Reward: ", cur_reward, " => ", new_reward)
                return new_timestep, new_reward
        if len_history >= (2*window_size):
            threshold_met_counter = 0
            window = reward_history[-(window_size*2):]
            beat_c = a * cur_timestep * threshold_C
            beat_c_0 = cur_max * threshold_C
            print("Threshold C: ", beat_c)
            for cur_frame in window:
                cur_percentile = numpy.percentile(a=cur_frame, q=75.0)
                print(cur_percentile)
                if (cur_percentile > beat_c) or (cur_percentile > beat_c_0):
                    threshold_met_counter = threshold_met_counter + 1
            if threshold_met_counter == ((2*window_size)-1):
                new_reward = cur_reward * rate_of_change_reward
                if new_reward > threshold_reward:
                    new_reward = threshold_reward
                new_timestep = cur_timestep * rate_of_change_timestep
                if new_timestep > threshold_timestep:
                    new_timestep = threshold_timestep
                print("Threshold C: Reward = ", cur_reward, "=> ", new_reward, " time-steps = ", cur_timestep, "=> ",
                      new_timestep)
                print("Timesteps: ", cur_timestep, " => ", new_timestep)
                print("Reward: ", cur_reward, " => ", new_reward)
                return new_timestep, new_reward
        else:
            return cur_timestep, cur_reward
    new_reward = cur_reward
    new_timestep = cur_timestep
    print("Timesteps: ", cur_timestep, " => ", new_timestep)
    print("Reward: ", cur_reward, " => ", new_reward)
    return new_timestep, new_reward


def calculate_maximum_timestep_count_1(threshold_timestep, threshold_reward, reward_history, cur_timestep,
                                    cur_reward, threshold_A=0.9, threshold_B=0.75, threshold_C=0.5,
                                       rate_of_change_reward=1.5, rate_of_change_timestep=1.5,
                                       window_size=5):
    '''
    An algorithm to increase the maximum time-steps allowed per given episode. Typical Policy Optimisation algorithms
    can update their policies \theta multiple times per episode and exploit the presence of frequent reward signals.
    Prior BNE algorithms calculated the fitness over the entire episode, this may have "wasted" information.
    :param threshold_timestep: The maximum number of time-steps for a given environment
    :param threshold_reward: The maximum total reward signal offered by the environment
    :param reward_history: The history of fitness rewards
    :param cur_timestep: the current threshold number of time-steps to evaluate a policy
    :param cur_reward:  the current threshold total reward to evaluate a policy
    :param check_A: the primary function that checks whether cur_timestep and cur_reward should be updated
    :param threshold_A: the threshold check A has to beat, multiply this number by
     reward per timestep with current time-step
    :param check_B: the secondary function that checks whether cur_timestep and cur_reward should be updated
    :param threshold_B: the threshold check B has to beat, multiply this number by
     reward per timestep with current time-step
    :param rate_of_change: rate to change the reward and timestep
    :return: updated time-step and reward

    '''
    a = (threshold_reward / threshold_timestep)
    string_output = ""
    print("Threshold Time steps = ", threshold_timestep, " cur_reward= ", threshold_reward, " ideal reward for this timestep = ", a*cur_timestep)
    cur_max = numpy.max(reward_history[len(reward_history)-1])
    cur_median = numpy.percentile(reward_history[len(reward_history) - 1], 50.0)
    cur_mean = numpy.mean(reward_history[len(reward_history) - 1])
    print("cur_max = ", cur_max, " > cur_reward = ", cur_reward)
    if (cur_median > cur_reward) or (cur_mean > cur_reward):
        new_reward = cur_reward * (rate_of_change_reward*2.0)
        if new_reward > threshold_reward:
            new_reward = threshold_reward
        new_timestep = cur_timestep * (rate_of_change_timestep*1.5)
        if new_timestep > threshold_timestep:
            new_timestep = threshold_timestep
        print("Threshold Median or Threshold: Reward = ", cur_reward, "=> ", new_reward, " time-steps = ", cur_timestep, "=> ",
              new_timestep)
        string_output = "median or mean greater than cur_reward"
        return new_timestep, new_reward, string_output
    if cur_max > cur_reward:
        new_reward = cur_reward * rate_of_change_reward
        if new_reward > threshold_reward:
            new_reward = threshold_reward
        new_timestep = cur_timestep * rate_of_change_timestep
        if new_timestep > threshold_timestep:
            new_timestep = threshold_timestep
        print("Threshold 0: Reward = ", cur_reward, "=> ", new_reward, " time-steps = ", cur_timestep, "=> ", new_timestep)
        string_output = "cur_max greater than cur_reward"
        return new_timestep, new_reward, string_output

    print("Threshold A:  cur_max =", cur_max, " > ", (a*cur_timestep*threshold_A))
    if cur_max > (a*cur_timestep*threshold_A):
        new_reward = cur_reward * rate_of_change_reward
        if new_reward > threshold_reward:
            new_reward = threshold_reward
        new_timestep = cur_timestep * rate_of_change_timestep
        if new_timestep > threshold_timestep:
            new_timestep = threshold_timestep
        print("Threshold A: Reward = ", cur_reward, "=> ", new_reward, " time-steps = ", cur_timestep, "=> ", new_timestep)
        string_output = "cur_max greater than (a*cur_timestep*threshold_A)"
        return new_timestep, new_reward, string_output
    else:
        len_history = len(reward_history)
        if len_history >= window_size:
            threshold_met_counter = 0
            window = reward_history[-window_size:]
            beat_b = a * cur_timestep * threshold_B
            beat_b_0 = cur_max * threshold_B
            beat_b_0 = beat_b
            print("Threshold B: ", beat_b, " or ", beat_b_0)
            for cur_frame in window:
                cur_percentile = numpy.percentile(a=cur_frame, q=75.0)
                print(cur_percentile)
                if (cur_percentile > beat_b) or (cur_percentile > beat_b_0):
                    threshold_met_counter = threshold_met_counter + 1
            if threshold_met_counter == (window_size-1):
                new_reward = cur_reward * rate_of_change_reward
                if new_reward > threshold_reward:
                    new_reward = threshold_reward
                new_timestep = cur_timestep * rate_of_change_timestep
                if new_timestep > threshold_timestep:
                    new_timestep = threshold_timestep
                print("Threshold B: Reward = ", cur_reward, "=> ", new_reward, " time-steps = ", cur_timestep, "=> ",
                      new_timestep)
                print("Timesteps: ", cur_timestep, " => ", new_timestep)
                print("Reward: ", cur_reward, " => ", new_reward)
                string_output = "cur_percentile greater than (beat_b or beat_b_0)"
                return new_timestep, new_reward, string_output
        if len_history >= (2*window_size):
            threshold_met_counter = 0
            window = reward_history[-(window_size*2):]
            beat_c = a * cur_timestep * threshold_C
            beat_c_0 = cur_max * threshold_C
            beat_c_0 = beat_c
            print("Threshold C: ", beat_c, " or ", beat_c_0)
            for cur_frame in window:
                cur_percentile = numpy.percentile(a=cur_frame, q=50.0)
                print(cur_percentile)
                if (cur_percentile > beat_c) or (cur_percentile > beat_c_0):
                    threshold_met_counter = threshold_met_counter + 1
            if threshold_met_counter == ((2*window_size)-1):
                new_reward = cur_reward * rate_of_change_reward
                if new_reward > threshold_reward:
                    new_reward = threshold_reward
                new_timestep = cur_timestep * rate_of_change_timestep
                if new_timestep > threshold_timestep:
                    new_timestep = threshold_timestep
                print("Threshold C: Reward = ", cur_reward, "=> ", new_reward, " time-steps = ", cur_timestep, "=> ",
                      new_timestep)
                print("Timesteps: ", cur_timestep, " => ", new_timestep)
                print("Reward: ", cur_reward, " => ", new_reward)
                string_output = "cur_percentile (median) greater than (beat_c or beat_c_0)"

                return new_timestep, new_reward, string_output
        else:
            string_output = "no change"
            return cur_timestep, cur_reward, string_output
    new_reward = cur_reward
    new_timestep = cur_timestep
    print("Timesteps: ", cur_timestep, " => ", new_timestep)
    print("Reward: ", cur_reward, " => ", new_reward)
    string_output = "no change"
    return new_timestep, new_reward, string_output


def default_check_a(reward_history, base_A, window_size=1, percentile=100.0):
    '''
    Check whether the reward history for the given window size meets the First Perntile Condition
    :param reward_history: the fitness history to be used
    :param base_A: the metric the reward history should beat
    :param window_size: the size of the window to examine
    :param percentile: the percentile to calculate the reward history metric
    :return: True if the reward history window beats the metric, False if it doesn't
    '''
    len_history = len(reward_history)
    print("Base A = ", base_A, " : len_history = ", len_history)

    window = []
    for row in reward_history:
        window.append(row)
    #if len_history >= window_size:
    #    window = reward_history[-window_size:]
    #else:
    #    window = reward_history[-len_history:]
    window_percentiles = []
    print("Base A = ", base_A, " : len_history = ", len_history, ": Window Size = ", len(window))
    print("Window = ", window)
    cur_percentile = numpy.percentile(a=window, q=percentile)
    print("Cur percentile = ", cur_percentile, " base_A = ", base_A)
    if cur_percentile < base_A:
        return False
    return True


def default_check_b(reward_history, base_B, window_size=5, percentile=75.0):
    '''
    Check whether the reward history for the given window size meets the Second Percentile Condition
    :param reward_history: the fitness history to be used
    :param base_B: the metric the reward history should beat
    :param window_size: the size of the window to examine
    :param percentile: the percentile to calculate the reward history metric
    :return: True if the reward history window beats the metric, False if it doesn't
    '''
    result = False
    len_history = len(reward_history)

    window = []
    if len_history >= window_size:
        window = reward_history[-window_size:]
    else:
        return False

    print("WindowB = ", window)
    window_percentiles = []
    print("Base B = ", base_B, " : len_history = ", len_history, ": Window Size = ", len(window))

    for i in window:
        cur_percentile = numpy.percentile(a=i, q=percentile)
        window_percentiles.append(cur_percentile)
        print("Cur percentile = ", cur_percentile, " base_B = ", base_B)
    for j in window_percentiles:
        if j < base_B:
            return False
    return True


