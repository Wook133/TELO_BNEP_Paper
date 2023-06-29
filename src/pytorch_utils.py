import copy
from copy import deepcopy
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from pathlib import Path
import config
if torch.cuda.is_available():
    FloatTensor = torch.cuda.FloatTensor
else:
    FloatTensor = torch.FloatTensor

USE_CUDA = torch.cuda.is_available()


def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()


class RLNN(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(RLNN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

    def set_params(self, params):
        """
        Set the params of the network to the given parameters
        """
        cpt = 0
        for param in self.parameters():
            tmp = np.product(param.size())

            if torch.cuda.is_available():
                param.data.copy_(torch.from_numpy(
                    params[cpt:cpt + tmp]).view(param.size()).cuda())
            else:
                param.data.copy_(torch.from_numpy(
                    params[cpt:cpt + tmp]).view(param.size()))
            cpt += tmp

    def get_params(self):
        """
        Returns parameters of the actor
        """
        return deepcopy(np.hstack([to_numpy(v).flatten() for v in
                                   self.parameters()]))

    def get_grads(self):
        """
        Returns the current gradient
        """
        return deepcopy(np.hstack([to_numpy(v.grad).flatten() for v in self.parameters()]))

    def get_size(self):
        """
        Returns the number of parameters of the network
        """
        return self.get_params().shape[0]

    def load_model(self, filename, net_name):
        """
        Loads the model
        """
        if filename is None:
            return

        self.load_state_dict(torch.load('{}/{}.pkl'.format(filename, net_name), map_location=lambda storage, loc: storage))

    def save_model(self, output, net_name):
        """
        Saves the model
        """
        torch.save(self.state_dict(), '{}/{}.pkl'.format(output, net_name))


class Actor(RLNN):
    def __init__(self, state_dim, action_dim, max_action, layer_norm=False, init=True, layer_1=64, layer_2=64, use_max_action=True):
        #Add SoftMax for Discrete Action Space
        #Max Action Might Be Causing A Problem? Consider Removing/Deactivating?
        super(Actor, self).__init__(state_dim, action_dim, max_action)
        self.l1 = nn.Linear(state_dim, layer_1)
        self.l2 = nn.Linear(layer_1, layer_2)
        self.l3 = nn.Linear(layer_2, action_dim)
        if layer_norm:
            self.n1 = nn.LayerNorm(layer_1)
            self.n2 = nn.LayerNorm(layer_2)
        self.layer_norm = layer_norm
        self.use_max_action = use_max_action

    def forward(self, x):
        if self.use_max_action:
            if not self.layer_norm:
                x = torch.tanh(self.l1(x))
                x = torch.tanh(self.l2(x))
                x = self.max_action * torch.tanh(self.l3(x))
            else:
                x = torch.tanh(self.n1(self.l1(x)))
                x = torch.tanh(self.n2(self.l2(x)))
                x = self.max_action * torch.tanh(self.l3(x))
        else:
            if not self.layer_norm:
                x = torch.tanh(self.l1(x))
                x = torch.tanh(self.l2(x))
                x = torch.tanh(self.l3(x))
            else:
                x = torch.tanh(self.n1(self.l1(x)))
                x = torch.tanh(self.n2(self.l2(x)))
                x = torch.tanh(self.l3(x))
        return x


class Policy(RLNN):
    def __init__(self, state_dim, action_dim, max_action, layer_norm=False, init=True, layer_1=64, layer_2=64, use_max_action=True):
        super(Policy, self).__init__(state_dim, action_dim, max_action)
        self.policy_model = nn.Sequential(
            nn.Linear(state_dim, layer_1),
            nn.Tanh(),
            nn.Linear(layer_1, layer_2),
            nn.Tanh(),
            nn.Linear(layer_2, action_dim)
        )

    def forward(self, x):
        return self.policy_model.forward(x)



def clone_model(model, weights):
    cloned_model = copy.deepcopy(model)
    for i, param in enumerate(cloned_model.parameters()):
        try:
            param.data.copy_(weights[i])
        except:
            param.data.copy_(weights[i].data)
    return cloned_model

#mother_parameters = list(cloned_model.parameters())
#actor.set_params(es_params[i])
#es_params[i] = actor.get_params()


def argument_settings(full_path, env_name, min_budget, max_budget, population_size, n_iterations, num_evaluations, n_workers, n_processors, worker, nic_name, shared_directory, run_id, es):
    Path(full_path).mkdir(parents=True, exist_ok=True)
    with open(full_path + '/parameter_settings.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["env_name", "min_budget", "max_budget", "population_size", "n_iterations", "num_evaluations", "n_workers", "n_processors", "worker", "nic_name", "shared_directory", "run_id", "es"])
        writer.writerow([str(env_name), str(min_budget), str(max_budget), str(population_size), str(n_iterations), str(num_evaluations), str(n_workers), str(n_processors), str(worker), str(nic_name), str(shared_directory), str(run_id), str(es)])
        file.close()


def ddpg_argument_settings(full_path, env_name, min_budget, max_budget, n_iterations, n_workers, n_processors, worker, nic_name, shared_directory, run_id, es, population_size=1, num_evaluations=1, number_to_average=1):
    Path(full_path).mkdir(parents=True, exist_ok=True)
    with open(full_path + '/parameter_settings.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["env_name", "min_budget", "max_budget", "population_size", "n_iterations", "num_evaluations", "n_workers", "n_processors", "worker", "nic_name", "shared_directory", "run_id", "number_to_average", "es"])
        writer.writerow([str(env_name), str(min_budget), str(max_budget), str(population_size), str(n_iterations), str(num_evaluations), str(n_workers), str(n_processors), str(worker), str(nic_name), str(shared_directory), str(run_id), str(number_to_average), str(es)])
        file.close()


def ppo_argument_settings(full_path, env_name, min_budget, max_budget, n_iterations, num_evaluations, n_workers, n_processors, worker, nic_name, shared_directory, layer_1, layer_2, run_id, seed, es):
    Path(full_path).mkdir(parents=True, exist_ok=True)
    with open(full_path + '/parameter_settings.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["env_name", "min_budget", "max_budget", "n_iterations", "num_evaluations", "n_workers", "n_processors", "worker", "nic_name", "shared_directory", "run_id", "layer_1", "layer_2",  "seed", "es"])
        writer.writerow([str(env_name), str(min_budget), str(max_budget), str(n_iterations), str(num_evaluations), str(n_workers), str(n_processors), str(worker), str(nic_name), str(shared_directory), str(run_id),str(layer_1), str(layer_2), str(seed), str(es)])
        file.close()

def ppo_parameters(full_path, env_name, shared_directory, budget, batch_size, n_steps, gamma, learning_rate, ent_coef, clip_range, n_epochs, gae_lambda, max_grad_norm, vf_coef, sde_sample_freq, seed, num_processors, layer_1, layer_2):
    Path(full_path).mkdir(parents=True, exist_ok=True)
    with open(full_path + '/ppo_parameter_settings.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["env_name", "shared_directory", "budget", "batch_size", "n_steps", "gamma", "learning_rate", "ent_coef", "clip_range", "n_epochs", "gae_lambda", "max_grad_norm", "vf_coef", "sde_sample_freq", "seed", "num_processors", "layer_1", "layer_2"])
        writer.writerow([str(env_name), shared_directory, str(budget), str(batch_size), str(n_steps), str(gamma), str(learning_rate), str(ent_coef), str(clip_range), str(n_epochs), str(gae_lambda), str(max_grad_norm), str(vf_coef), str(sde_sample_freq), str(seed), str(num_processors), str(layer_1), str(layer_2)])
        file.close()


def argument_settings_single(env_name, USE_CUDA, rank_fitness, num_processors, population_size, num_evaluations, sigma_init, weight_decay, shared_directory, budget, es, use_default_layers, layer_1, layer_2, use_max_action, mu_init):
    Path(shared_directory).mkdir(parents=True, exist_ok=True)
    with open(shared_directory + '/parameter_settings.csv', 'x', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["env_name", "USE_CUDA", "rank_fitness", "num_processors", "population_size", "num_evaluations", "mu_init", "sigma_init", "weight_decay", "shared_directory", "budget", "use_default_layers", "layer_1", "layer_2", "use_max_action", "es"])
        writer.writerow([str(env_name), str(USE_CUDA), str(rank_fitness), str(num_processors), str(population_size), str(num_evaluations), str(mu_init), str(sigma_init), str(weight_decay), str(shared_directory), str(budget), str(use_default_layers), str(layer_1), str(layer_2), str(use_max_action), str(es)])
        file.close()

def bne_parameters():
    print()


#ToDo: Add more Arguments
def arguments_ddpg(shared_directory):
    Path(shared_directory).mkdir(parents=True, exist_ok=True)


def listToString(s):
    # initialize an empty string
    str1 = str(s[0])
    # traverse in the string
    for i in range(1, len(s)):
        str1 = str1 + ";" + str(s[i])
    #for ele in s:
    #    str1 = str1 + ";" + str(ele)
        # return string
    return str1


def get_game(env_name, default=True):
    if default:
        return config.default
    game = None
    if "BipedalWalkerHardcore-v3" in env_name:
        game = config.bipedhard_stoc
    elif "BipedalWalker-v2" in env_name:
        game = config.biped
    elif "LunarLander-v2" in env_name:
        game = config.lunar
    elif "AntPyBulletEnv-v0" in env_name:
        game = config.pybullet_ant
    elif "AntMuJoCoEnv-v0" in env_name:
        game = config.mujoco_ant
    elif "AntBulletEnv-v0" in env_name:
        game = config.bullet_ant
    elif "Walker2DBulletEnv-v0" in env_name:
        game = config.bullet_twodwalker
    elif "Walker2DPyBulletEnv-v0" in env_name:
        game = config.pybullet_twodwalker
    elif "Walker2DMuJoCoEnv-v0" in env_name:
        game = config.mujoco_twodwalker
    elif "CartPole-v1" in env_name:
        game = config.cartpole
    elif "SpaceInvaders-ram-v4" in env_name:
        game = config.space_invaders_ram
    elif "Gravitar-ramDeterministic-v4" in env_name:
        game = config.gravitar
    elif "MiniGrid-DoorKey-5x5-v0" in env_name:
        game = config.minigrid
    elif "HopperBulletEnv-v0" in env_name:
        game = config.bullet_hopper
    elif "HopperPyBulletEnv-v0" in env_name:
        game = config.pybullet_hopper
    elif "HopperMuJoCoEnv-v0" in env_name:
        game = config.mujoco_hopper
    elif "RacecarBulletEnv-v0" in env_name:
        game = config.bullet_racecar
    elif "MinitaurBulletEnv-v0" in env_name:
        game = config.bullet_minitaur
    elif "MinitaurBulletDuckEnv-v0" in env_name:
        game = config.bullet_minitaur_duck
    elif "CarRacing-v0" in env_name:
        game = config.twod_car
    elif "HalfCheetahBulletEnv-v0" in env_name:
        game = config.bullet_half_cheetah
    elif "HalfCheetahPyBulletEnv-v0" in env_name:
        game = config.pybullet_half_cheetah
    elif "HalfCheetahMuJoCoEnv-v0" in env_name:
        game = config.mujoco_half_cheetah
    elif "HumanoidBulletEnv-v0" in env_name:
        game = config.bullet_humanoid
    elif "InvertedPendulumSwingupBulletEnv-v0" in env_name:
        game = config.bullet_pendulum
    elif "InvertedDoublePendulumBulletEnv-v0" in env_name:
        game = config.bullet_double_pendulum
    else:
        game = config.default
    return game


def save_model(policy_weights, state_dim, action_dim, max_action, use_max_action, layer_1, layer_2, path):
    policy = Actor(state_dim=state_dim, action_dim=action_dim, max_action=max_action, use_max_action=use_max_action, layer_1=layer_1, layer_2=layer_2)
    policy.set_params(policy_weights)
    torch.save(policy.state_dict(), path)


class policy_middle:
    def __init__(self, policy_parameters, seed, number=-1):
        self.policy_parameters = policy_parameters
        self.seed = seed
        self.fitness = False
        self.timesteps = 0
        self.number = number

    def getSeed(self):
        return self.seed

    def getPolicyParameters(self):
        return self.policy_parameters

    def getFitness(self):
        return self.fitness

    def setFitness(self, fitness):
        self.fitness = fitness

    def getTimestep(self):
        return self.timesteps

    def setNumber(self, number):
        self.number = number

    def getNumber(self):
        return self.number

    def setTimestep(self, timestep):
        self.timesteps = timestep