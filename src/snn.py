import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from gym.utils import seeding
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import utils
import gym
import pybulletgym
#Why set eval and train explicitly: https://jamesmccaffrey.wordpress.com/2019/01/23/pytorch-train-vs-eval-mode/
import snn_dataset


def evaluate_r2_mse(model, data_loader, round_reward=False, position_reward=4, device=None, reward=True, round_to=0, normalization_type="none"):
    if "none" not in normalization_type:
        x_min, x_mean, x_max, x_std, y_min, y_mean, y_max, y_std = data_loader.get_stats()
    model.eval()
    y_actual = []
    y_pred = []
    if "none" not in normalization_type:
        print("Doing normalization in evaluation")
    with torch.no_grad():
        for i, (x, y) in enumerate(data_loader, 0):
            if "none" not in normalization_type:
                y = y.numpy()
                y = snn_dataset.normalization(x=y, normalization_type=normalization_type, x_min=y_min, x_mean=y_mean, x_max=y_max, x_std=y_std)
                y_actual.append(y)
            else:
                y_actual.append(y.numpy())
            y_cur_pred = 0
            if device is None:
                if "none" not in normalization_type:
                    x = x.numpy()
                    x = snn_dataset.normalization(x=x, normalization_type=normalization_type, x_min=x_min, x_mean=x_mean, x_max=x_max, x_std=x_std)
                    x = torch.from_numpy(x)
                    y_cur_pred = model(x.float()).numpy()
                else:
                    y_cur_pred = model(x.float()).numpy()
            else:
                if "none" not in normalization_type:
                    x = x.numpy()
                    x = snn_dataset.normalization(x=x, normalization_type=normalization_type, x_min=x_min, x_mean=x_mean, x_max=x_max, x_std=x_std)
                    x = torch.from_numpy(x)
                    x = x.to(device)
                    x = Variable(x)
                    x = x.float()
                    y_cur_pred = model(x)
                    y_cur_pred = torch.Tensor.cpu(y_cur_pred)
                    y_cur_pred = y_cur_pred.numpy()
                else:
                    x = x.to(device)
                    x = Variable(x)
                    x = x.float()
                    y_cur_pred = model(x)
                    y_cur_pred = torch.Tensor.cpu(y_cur_pred)
                    y_cur_pred = y_cur_pred.numpy()

            if reward is True:
                if round_reward:
                    y_cur_pred[position_reward] = np.round(y_cur_pred[position_reward], round_to)
            y_pred.append(y_cur_pred)
    r2 = r2_score(y_actual, y_pred)
    mse = mean_squared_error(y_actual, y_pred)
    return r2, mse


def evaluate_dynamic_r2_mse(env_name, model, data_loader, round_reward=False, position_reward=4, device=None, reward=True, round_to=0, normalization_type="none"):
    if "none" not in normalization_type:
        x_min, x_mean, x_max, x_std, y_min, y_mean, y_max, y_std = data_loader.get_stats()
    action_length, observation_length, reward_length = utils.get_lengths(env_name)
    model.eval()
    y_actual = []
    y_pred = []
    if "none" not in normalization_type:
        print("Doing normalization in evaluation")
    with torch.no_grad():
        for i, (x, y) in enumerate(data_loader, 0):
            if "none" not in normalization_type:
                y = y.numpy()
                y = snn_dataset.normalization(x=y, normalization_type=normalization_type, x_min=y_min, x_mean=y_mean, x_max=y_max, x_std=y_std)
                y_actual.append(y)
            else:
                y_actual.append(y.numpy())
            y_cur_pred = 0
            if device is None:
                if "none" not in normalization_type:
                    x = x.numpy()
                    x = snn_dataset.normalization(x=x, normalization_type=normalization_type, x_min=x_min, x_mean=x_mean, x_max=x_max, x_std=x_std)
                    cur_y = x[action_length:action_length + observation_length]
                    zeros = np.zeros(1)
                    cur_y = np.append(cur_y, zeros)
                    cur_y = torch.from_numpy(cur_y)
                    x = torch.from_numpy(x)
                    y_cur_pred = model(x.float())
                    y_cur_pred = torch.add(cur_y, y_cur_pred)
                    y_cur_pred = y_cur_pred.numpy()
                else:
                    y_cur_pred = model(x.float()).numpy()
            else:
                if "none" not in normalization_type:
                    x = x.numpy()
                    x = snn_dataset.normalization(x=x, normalization_type=normalization_type, x_min=x_min, x_mean=x_mean, x_max=x_max, x_std=x_std)
                    cur_y = x[action_length:action_length + observation_length]
                    zeros = np.zeros(1)
                    cur_y = np.append(cur_y, zeros)
                    cur_y = torch.from_numpy(cur_y)
                    x = torch.from_numpy(x)
                    x = x.to(device)
                    x = Variable(x)
                    y_cur_pred = model(x.float())
                    y_cur_pred = torch.add(cur_y, y_cur_pred)
                    y_cur_pred = y_cur_pred.numpy()
                else:
                    cur_y = x[action_length:action_length + observation_length]
                    zeros = np.zeros(1)
                    cur_y = np.append(cur_y, zeros)
                    cur_y = torch.from_numpy(cur_y)
                    x = x.to(device)
                    x = Variable(x)
                    x = x.float()
                    y_cur_pred = model(x)
                    y_cur_pred = torch.Tensor.cpu(y_cur_pred)
                    y_cur_pred = torch.add(cur_y, y_cur_pred)
                    y_cur_pred = y_cur_pred.numpy()

            '''
            if reward is True:
                if round_reward:
                    y_cur_pred[position_reward] = np.round(y_cur_pred[position_reward], round_to)
            '''
            y_pred.append(y_cur_pred)
    r2 = r2_score(y_actual, y_pred)
    mse = mean_squared_error(y_actual, y_pred)
    return r2, mse


def evaluate_r2_mse_reverse_normalization(model, data_loader, round_reward=False, position_reward=4, device=None, reward=True, round_to=0, normalization_type="none"):
    if "none" not in normalization_type:
        x_min, x_mean, x_max, x_std, y_min, y_mean, y_max, y_std = data_loader.get_stats()
    model.eval()
    y_actual = []
    y_pred = []
    if "none" not in normalization_type:
        print("Doing normalization in evaluation with reverse")
    with torch.no_grad():
        for i, (x, y) in enumerate(data_loader, 0):
            y_actual.append(y.numpy())
            y_cur_pred = 0
            if device is None:
                if "none" not in normalization_type:
                    x = x.numpy()
                    x = snn_dataset.normalization(x=x, normalization_type=normalization_type, x_min=x_min, x_mean=x_mean, x_max=x_max, x_std=x_std)
                    x = torch.from_numpy(x)
                    y_cur_pred = model(x.float()).numpy()
                    y_cur_pred = snn_dataset.reverse_normalization(x=y_cur_pred, normalization_type=normalization_type, x_min=y_min, x_mean=y_mean, x_max=y_max, x_std=y_std)
                else:
                    y_cur_pred = model(x.float()).numpy()
            else:
                if "none" not in normalization_type:
                    x = x.numpy()
                    x = snn_dataset.normalization(x=x, normalization_type=normalization_type, x_min=x_min, x_mean=x_mean, x_max=x_max, x_std=x_std)
                    x = torch.from_numpy(x)
                    x = x.to(device)
                    x = Variable(x)
                    x = x.float()
                    y_cur_pred = model(x)
                    y_cur_pred = torch.Tensor.cpu(y_cur_pred)
                    y_cur_pred = y_cur_pred.numpy()
                    y_cur_pred = snn_dataset.reverse_normalization(x=y_cur_pred, normalization_type=normalization_type, x_min=y_min, x_mean=y_mean, x_max=y_max, x_std=y_std)
                else:
                    x = x.to(device)
                    x = Variable(x)
                    x = x.float()
                    y_cur_pred = model(x)
                    y_cur_pred = torch.Tensor.cpu(y_cur_pred)
                    y_cur_pred = y_cur_pred.numpy()

            if reward is True:
                if round_reward:
                    y_cur_pred[position_reward] = np.round(y_cur_pred[position_reward], round_to)
            y_pred.append(y_cur_pred)
    r2 = r2_score(y_actual, y_pred)
    mse = mean_squared_error(y_actual, y_pred)
    return r2, mse


class Simulator_NN(nn.Module):
    def __init__(self, nn_in=9, nn_out=9, nn_hidden=128, use_cuda=False, cuda_device='cpu', dropout=0.0, environment="CartPole", default=True):
        '''
        :param nn_in: Size of action and observation
        :param nn_out: Size of subsequent observation and reward
        :param nn_hidden:
        '''
        super(Simulator_NN, self).__init__()
        self.use_cuda = use_cuda
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        print("Simulator using Device: ", str(self.device))
        print("SNN In = ", nn_in)
        self.environment = environment
        if default:
            self.snn = default_snn(nn_in, nn_out, nn_hidden, dropout)
        else:
            if "CartPole" in self.environment:
                self.snn = cartpole_snn(nn_in, nn_out, nn_hidden, dropout)
            elif "LunarLander" in self.environment:
                self.snn = lunarlander_snn(nn_in, nn_out, nn_hidden, dropout)
            elif "Minitaur" in self.environment:
                self.snn = minitaur_snn_new(nn_in, nn_out, nn_hidden, dropout)
            else:
                self.snn = default_snn(nn_in, nn_out, nn_hidden, dropout)

    def forward(self, x):
        x = self.snn(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def cartpole_snn(nn_in, nn_out, nn_hidden, dropout):
    return nn.Sequential(nn.Linear(nn_in, nn_hidden), nn.ReLU(), nn.Linear(nn_hidden, nn_hidden), nn.Dropout(dropout), nn.ReLU(), nn.Linear(nn_hidden, nn_hidden), nn.Dropout(dropout), nn.ReLU(), nn.Linear(nn_hidden, nn_hidden), nn.Dropout(dropout), nn.ReLU(), nn.Linear(nn_hidden, nn_out))


def default_snn(nn_in, nn_out, nn_hidden, dropout):
    print("Did the Default Simulator")
    return nn.Sequential(nn.Linear(nn_in, nn_hidden),
                         nn.ReLU(),
                         nn.Linear(nn_hidden, nn_hidden),
                         nn.ReLU(),
                         nn.Dropout(dropout),
                         nn.Linear(nn_hidden, nn_out))


def lunarlander_snn(nn_in, nn_out, nn_hidden, dropout):
    return nn.Sequential(nn.Linear(nn_in, nn_hidden), nn.Dropout(dropout), nn.Tanh(), nn.Linear(nn_hidden, nn_out))

def minitaur_snn(nn_in, nn_out, nn_hidden, dropout):
    snn = nn.Sequential(nn.Linear(nn_in, nn_hidden), nn.Dropout(dropout), nn.ELU(), nn.Linear(nn_hidden, nn_out))
    return snn

def minitaur_snn_0(nn_in, nn_out, nn_hidden, dropout):
    snn = nn.Sequential(nn.Linear(nn_in, nn_hidden), nn.Dropout(dropout), nn.Sigmoid(), nn.Linear(nn_hidden, nn_hidden), nn.Dropout(dropout), nn.Sigmoid(), nn.Linear(nn_hidden, nn_out))
    return snn

def minitaur_snn_new(nn_in, nn_out, nn_hidden, dropout):
    snn = nn.Sequential(nn.Linear(nn_in, nn_hidden), nn.Dropout(dropout), nn.ELU(), nn.Linear(nn_hidden, nn_hidden), nn.Dropout(dropout), nn.ELU(), nn.Linear(nn_hidden, nn_out))
    return snn

class wrapper_SNN_discrete_action(Simulator_NN):
    def __init__(self, SNN, env_name, starting_state=None, action_space=None, size_observation=None, round_to=0, dynamics=False):
        super(Simulator_NN, self).__init__()
        self.SNN = SNN
        self.env_name = env_name
        self.starting_state = starting_state
        self.current_state = starting_state
        self.size_observation = size_observation
        self.action_space = action_space
        self.round_to = round_to
        self.SNN.eval()
        self.dynamics = dynamics
        if dynamics:
            self.action_length, self.observation_length, self.reward_length = utils.get_lengths(self.env_name)


    def restart(self, state):
        self.starting_state = state
        return self.starting_state, 1.0

    def reset(self):
        return self.starting_state

    def step(self, action):
        # print(action, self.current_state)
        input = np.concatenate(([action], self.current_state), axis=0)
        # print("Input: ", input)
        tensor_x = torch.from_numpy(input)
        tensor_prediction = self.SNN(tensor_x.float())
        np_prediction = np.array(tensor_prediction.clone().detach())
        self.current_state = np_prediction[:self.size_observation]
        reward = np.round(np_prediction[self.size_observation:self.size_observation + 1], self.round_to)
        return self.current_state, reward

    def step_state(self, action, state, normalization_type=None, x_min=0.0, x_mean=0.0, x_max=0.0, x_std=0.0, y_min=0.0, y_mean=0.0, y_max=0.0, y_std=0.0):
        print("This step state")
        act = np.array([action])
        input = np.concatenate((act, state), axis=0)
        if normalization_type is not None:
            input = snn_dataset.normalization(x=input, normalization_type=normalization_type, x_min=x_min, x_mean=x_mean, x_max=x_max, x_std=x_std)
        if self.dynamics:
            cur_y = input[:, self.action_length:self.action_length + self.observation_length]
            zeros = np.zeros(1)
            cur_y = np.append(cur_y, zeros)
            tensor_x = torch.from_numpy(input)
            tensor_prediction = self.SNN(tensor_x.float())
            tensor_prediction = torch.add(cur_y, tensor_prediction)
        else:
            tensor_x = torch.from_numpy(input)
            tensor_prediction = self.SNN(tensor_x.float())
        np_prediction = np.array(tensor_prediction.clone().detach())
        if normalization_type is not None:
            np_prediction = snn_dataset.reverse_normalization(x=np_prediction, normalization_type=normalization_type, x_min=y_min, x_mean=y_mean, x_max=y_max, x_std=y_std)
        return np_prediction[:self.size_observation], np.round(np_prediction[self.size_observation:self.size_observation + 1], self.round_to)
    #        return np_prediction[:4], np.round(np_prediction[4:5], 0)


class wrapper_SNN_box_action(Simulator_NN):
    def __init__(self, SNN, env_name, starting_state=None, action_space=None, size_observation=None, round_to=0, dynamics=False):
        super(Simulator_NN, self).__init__()
        self.SNN = SNN
        self.env_name = env_name
        self.starting_state = starting_state
        self.current_state = starting_state
        self.size_observation = size_observation
        self.action_space = action_space
        self.round_to = round_to
        self.SNN.eval()
        self.dynamics = dynamics
        if dynamics:
            self.action_length, self.observation_length, self.reward_length = utils.get_lengths(self.env_name)
        print("SNN Dimensions")
        print(self.SNN)


    def restart(self, state):
        self.starting_state = state
        return self.starting_state, 1.0

    def reset(self):
        return self.starting_state

    def step(self, action):
        # print(action, self.current_state)
        input = np.concatenate(([action], self.current_state), axis=0)
        # print("Input: ", input)
        tensor_x = torch.from_numpy(input)
        tensor_prediction = self.SNN(tensor_x.float())
        np_prediction = np.array(tensor_prediction.clone().detach())
        self.current_state = np_prediction[:self.size_observation]
        reward = np.round(np_prediction[self.size_observation:self.size_observation + 1], self.round_to)
        return self.current_state, reward

    def step_state(self, action, state, normalization_type="none", x_min=0.0, x_mean=0.0, x_max=0.0, x_std=0.0, y_min=0.0, y_mean=0.0, y_max=0.0, y_std=0.0):
        input = np.concatenate((action, state), axis=0)
        if "none" not in normalization_type:
            input = snn_dataset.normalization(x=input, normalization_type=normalization_type, x_min=x_min, x_mean=x_mean, x_max=x_max, x_std=x_std)
        if self.dynamics:
            cur_y = input[self.action_length:self.action_length + self.observation_length]
            zeros = np.zeros(1)
            cur_y = np.append(cur_y, zeros)
            tensor_x = torch.from_numpy(input)
            tensor_prediction = self.SNN(tensor_x.float())
            tensor_prediction = torch.add(torch.from_numpy(cur_y), tensor_prediction)
        else:
            tensor_x = torch.from_numpy(input)
            tensor_prediction = self.SNN(tensor_x.float())
        np_prediction = np.array(tensor_prediction.clone().detach())
        if "none" not in normalization_type:
            np_prediction = snn_dataset.reverse_normalization(x=np_prediction, normalization_type=normalization_type, x_min=y_min, x_mean=y_mean, x_max=y_max, x_std=y_std)
        observation = np_prediction[:self.size_observation]
        reward = np.round(np_prediction[self.size_observation:self.size_observation + 1], self.round_to)
        if (math.isnan(reward)):
            return observation, np.array([0.0])
        elif -1000.0 < reward < 1000.0:
            return observation, reward
        else:
            return observation, np.array([0.0])

        def step_state_wo_normalization(self, action, state):
            input = np.concatenate((action, state), axis=0)
            tensor_x = torch.from_numpy(input)
            tensor_prediction = self.SNN(tensor_x.float())
            np_prediction = np.array(tensor_prediction.clone().detach())
            # contains_nan = np.isnan(np_prediction)
            observation = np_prediction[:self.size_observation]
            reward = np.round(np_prediction[self.size_observation:self.size_observation + 1], self.round_to)
            if -1000.0 < reward < 1000.0:
                return observation, reward
            else:
                return observation, 0.0


class cartpole_SNN(Simulator_NN):
    def __init__(self, SNN, size_observation=None):
        super(Simulator_NN, self).__init__()
        self.SNN = SNN
        self.SNN.eval()
        self.steps_beyond_done = None
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        self.size_observation = size_observation
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def step(self, action):
        state = self.state
        act = np.array([action])
        input = np.concatenate((act, state), axis=0)
        tensor_x = torch.from_numpy(input)
        tensor_prediction = self.SNN(tensor_x.float())
        np_prediction = np.array(tensor_prediction.clone().detach())
        state = np_prediction[:self.size_observation]
        #print("Step Observation =", state)
        #Num	Observation                 Min         Max
        #0	Cart Position             -4.8            4.8
        #1	Cart Velocity             -Inf            Inf
        #2	Pole Angle                 -24 deg        24 deg
        #3	Pole Velocity At Tip      -Inf            Inf
        x, x_dot, theta, theta_dot = state[0], state[1], state[2], state[3]

        self.state = state
        done = x < -self.x_threshold or x > self.x_threshold  or theta < -self.theta_threshold_radians or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                print("You are calling 'step()' even though this Simulated environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0
        return np.array(self.state), reward, done, {}


class lunar_lander_SNN(Simulator_NN):
    def __init__(self, SNN, size_observation=None):
        super(Simulator_NN, self).__init__()
        self.SNN = SNN
        self.SNN.eval()
        self.prev_shaping = None
        self.state = self.reset()
        self.size_observation = size_observation
        self.seed()


    def reset(self):
        env = gym.make("LunarLander-v2")
        state, reward, done, _ = env.reset()
        env.close()
        shaping = - 100*np.sqrt(state[0]*state[0] + state[1]*state[1]) - 100*np.sqrt(state[2]*state[2] + state[3]*state[3]) - 100*abs(state[4]) + 10*state[6] + 10*state[7]
        # And ten points for legs contact, the idea is if you
        # lose contact again after landing, you get negative reward
        self.prev_shaping = shaping
        self.game_over = False
        return state

    def step(self, action):
        state = self.state
        act = action[0]
        input = np.concatenate((act, state), axis=0)
        tensor_x = torch.from_numpy(input)
        tensor_prediction = self.SNN(tensor_x.float())
        np_prediction = np.array(tensor_prediction.clone().detach())
        state = np_prediction[:self.size_observation]
        reward = 0
        shaping = - 100*np.sqrt(state[0]*state[0] + state[1]*state[1]) - 100*np.sqrt(state[2]*state[2] + state[3]*state[3]) - 100*abs(state[4]) + 10*state[6] + 10*state[7]
        # And ten points for legs contact, the idea is if you
        # lose contact again after landing, you get negative reward
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping
        m_power = 1.0
        s_power = 1.0
        reward -= m_power*0.30  # less fuel spent is better, about -30 for heurisic landing
        reward -= s_power*0.03

        done = False
        if self.game_over or abs(state[0]) >= 1.0:
            done = True
            reward = -100
        if not self.lander.awake:
            done = True
            reward = +100
        return np.array(state, dtype=np.float64), reward, done, {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


