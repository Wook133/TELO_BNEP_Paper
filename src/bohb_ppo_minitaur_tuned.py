import os
import gym, pybulletgym, pybullet_envs
import time
import torch
import argparse
import pickle
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy

from hpbandster.core.worker import Worker
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB as BOHB

from pytorch_utils import ppo_argument_settings, ppo_parameters


def make_env(env_id, rank=0, seed=0, log_dir=None, wrapper_class=None, env_kwargs=None):
    """
    Helper function to multiprocess training
    and log the progress.
    :param env_id: (str)
    :param rank: (int)
    :param seed: (int)
    :param log_dir: (str)
    :param wrapper_class: (Type[gym.Wrapper]) a subclass of gym.Wrapper
        to wrap the original env with
    :param env_kwargs: (Dict[str, Any]) Optional keyword argument to pass to the env constructor
    """
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)

    if env_kwargs is None:
        env_kwargs = {}

    def _init():
        set_random_seed(seed + rank)
        env = gym.make(env_id, **env_kwargs)

        # Wrap first with a monitor (e.g. for Atari env where reward clipping is used)
        log_file = os.path.join(log_dir, str(rank)) if log_dir is not None else None
        # Monitor success rate too for the real robot
        info_keywords = ("is_success",) if "NeckEnv" in env_id else ()
        env = Monitor(env, log_file, info_keywords=info_keywords)

        # Dict observation space is currently not supported.
        # https://github.com/hill-a/stable-baselines/issues/321
        # We allow a Gym env wrapper (a subclass of gym.Wrapper)
        if wrapper_class:
            env = wrapper_class(env)

        env.seed(seed + rank)
        return env

    return _init


class stablebaselinesWorkerPPO(Worker):
    def __init__(self, env_name, shared_directory, sleep_interval=0, seed=0, num_processors=2, layer_1=64, layer_2=64, **kwargs):
        super().__init__(**kwargs)
        self.env_name = env_name
        self.num_processors = num_processors
        self.BO_iterations = 0
        self.shared_directory = shared_directory
        self.seed = seed
        self.layer_1 = layer_1
        self.layer_2 = layer_2
        Path(self.shared_directory).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_configspace():
        """
        It builds the configuration space with the needed hyperparameters.
        It is easily possible to implement different types of hyperparameters.
        Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
        :return: ConfigurationsSpace-Object
[[11, 0, 0], 2000000.0, {"submitted": 1599556733.984276, "started": 1599556733.984736, "finished": 1599559070.6112943}, {
"loss": -0.432396219849634, "info": {
"optimizer": "PPO", "mean_fitness": 0.432396219849634, "std_reward": 0.3653421510603804
, "env_name": "MinitaurBulletEnv-v0", "total_time": 2328.4759516716003}}, null]

[[11, 0, 0], {
"batch_size": 16, "clip_range": 0.2, "ent_coef": 0.0090296575629347,
 "gae_lambda": 0.98, "gamma": 0.98, "learning_rate": 0.0005966541125225859,
 "max_grad_norm": 1, "n_epochs": 1, "n_steps": 1024,
 "sde_sample_freq": -1, "vf_coef": 0.9578955610542378
 }, {"model_based_pick": false}]

        """
        cs = CS.ConfigurationSpace()
        batch_size = CSH.CategoricalHyperparameter('batch_size', choices=[32])
        n_steps = CSH.CategoricalHyperparameter('n_steps', choices=[2048])
        gamma = CSH.CategoricalHyperparameter('gamma', choices=[0.995])
        learning_rate = CSH.CategoricalHyperparameter('learning_rate', choices=[1.820243158147613e-05])
        ent_coef = CSH.CategoricalHyperparameter('ent_coef', choices=[1.574077329055763e-05])
        clip_range = CSH.CategoricalHyperparameter('clip_range', choices=[0.1])
        n_epochs = CSH.CategoricalHyperparameter('n_epochs', choices=[10])
        gae_lambda = CSH.CategoricalHyperparameter('gae_lambda', choices=[0.98])
        max_grad_norm = CSH.CategoricalHyperparameter('max_grad_norm', choices=[0.9])
        vf_coef = CSH.CategoricalHyperparameter('vf_coef', choices=[0.896142640634874])
        sde_sample_freq = CSH.CategoricalHyperparameter('sde_sample_freq', choices=[256])
        meaningless_choice = CSH.CategoricalHyperparameter('meaningless_choice', choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        cs.add_hyperparameters([batch_size, n_steps, gamma, learning_rate, ent_coef, clip_range, n_epochs, gae_lambda, max_grad_norm, vf_coef, sde_sample_freq, meaningless_choice])
        return cs

    def compute(self, config, budget, working_directory, *args, **kwargs):
        print("Start Compute = ", self.BO_iterations)
        #log_dir = working_directory + "/" + str(self.BO_iterations)
        log_dir = self.shared_directory + "/" + str(self.BO_iterations)
        policy_name = log_dir + "/ppo_model"
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        env = gym.make(self.env_name)
        env.reset()

        ppo_parameters(full_path=log_dir, env_name=self.env_name, shared_directory=log_dir, budget=budget, batch_size=config['batch_size'], n_steps=config['n_steps'], gamma=config['gamma'], learning_rate=config['learning_rate'], ent_coef=config['ent_coef'], clip_range=config['clip_range'], n_epochs=config['n_epochs'], gae_lambda=config['gae_lambda'], max_grad_norm=config['max_grad_norm'], vf_coef=config['vf_coef'], sde_sample_freq=config['sde_sample_freq'], seed=self.seed, num_processors=self.num_processors, layer_1=self.layer_1, layer_2=self.layer_2)

        env.close()
        env = SubprocVecEnv([make_env(env_id=self.env_name, seed=self.seed, rank=i, log_dir=log_dir) for i in range(self.num_processors)])

        # Logs will be saved in log_dir/monitor.csv
        # Custom MLP policy of two layers of size 64 each with Tanh activation function
        policy_kwargs = dict(activation_fn=torch.nn.Tanh, net_arch=[layer_1, layer_2])
        model = PPO('MlpPolicy', env=env, policy_kwargs=policy_kwargs, verbose=1, batch_size=config['batch_size'], n_steps=config['n_steps'], gamma=config['gamma'], learning_rate=config['learning_rate'], ent_coef=config['ent_coef'], clip_range=config['clip_range'], n_epochs=config['n_epochs'], gae_lambda=config['gae_lambda'], max_grad_norm=config['max_grad_norm'], vf_coef=config['vf_coef'], sde_sample_freq=config['sde_sample_freq'])

        # We create a separate environment for evaluation
        eval_env = gym.make(self.env_name)
        start_time = time.time()
        model.learn(total_timesteps=budget)
        model.save(policy_name)
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
        end_time = time.time()

        self.BO_iterations = self.BO_iterations + 1
        return ({'loss': mean_reward * -1.0, 'info': {'optimizer': "PPO", 'mean_fitness': mean_reward, 'std_reward': std_reward, 'env_name': self.env_name, 'total_time': (end_time-start_time),}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HP Tuned BOHB PPO.')
    parser.add_argument('--min_budget', type=float, help='Minimum budget used during the optimization.', default=10000000)
    parser.add_argument('--max_budget', type=float, help='Maximum budget used during the optimization.', default=10500000)
    parser.add_argument('--env_name', type=str, help='Size of the Population', default="MinitaurBulletEnv-v0")
    parser.add_argument('--n_iterations', type=int, help='Number of iterations performed by the optimizer', default=2)
    parser.add_argument('--num_evaluations', type=int, help='Number of Evaluations', default=1)
    parser.add_argument('--n_workers', type=int, help='Number of workers to run in parallel.', default=1)
    parser.add_argument('--n_processors', type=int, help='Number of workers to run in parallel.', default=4)
    parser.add_argument('--layer_1', type=int, help='Number of Neurons in Layer 1.', default=64)
    parser.add_argument('--layer_2', type=int, help='Number of Neurons in Layer 2.', default=32)
    parser.add_argument('--worker', help='Flag to turn this into a worker process', action='store_true')
    parser.add_argument('--nic_name', type=str, help='Which network interface to use for communication.', default='lo')
    parser.add_argument('--shared_directory', type=str, help='A directory that is accessible for all processes, e.g. a NFS share.', default="ppo_output/")
    parser.add_argument('--run_id', type=str, help='A unique run id for this optimization run. An easy option is to use the job id of the clusters scheduler.')
    args = parser.parse_args()
    min_budget = args.__getattribute__("min_budget")
    max_budget = args.__getattribute__("max_budget")
    env_name = args.__getattribute__("env_name")
    n_iterations = args.__getattribute__("n_iterations")
    num_evaluations = args.__getattribute__("num_evaluations")
    n_workers = args.__getattribute__("n_workers")
    n_processors = args.__getattribute__("n_processors")
    worker = args.__getattribute__("worker")
    nic_name = args.__getattribute__("nic_name")
    shared_directory = args.__getattribute__("shared_directory")
    run_id = args.__getattribute__("run_id")
    layer_1 = args.__getattribute__("layer_1")
    layer_2 = args.__getattribute__("layer_2")

    es = "PPO"
    seed = time.time()
    shared_directory = shared_directory + "/" + env_name + "/" + str(seed)
    seed = int(seed)
    ppo_argument_settings(full_path=shared_directory, env_name=env_name, min_budget=min_budget, max_budget=max_budget, n_iterations=n_iterations, num_evaluations=num_evaluations, n_workers=n_workers, n_processors=n_processors, worker=worker, nic_name=nic_name, shared_directory=shared_directory, layer_1=layer_1, layer_2=layer_2, run_id=run_id, seed=seed, es=es)

    print(args)

    # Every process has to lookup the hostname
    print(str(args.nic_name))
    if args.nic_name == 'lo':
        host = '127.0.0.1'
    else:
        host = hpns.nic_name_to_host(args.nic_name)

    if args.worker:
        import time

        time.sleep(30)  # short artificial delay to make sure the nameserver is already running
        w = stablebaselinesWorkerPPO(run_id=args.run_id, host=host, timeout=20, sleep_interval=0.5, seed=seed)
        w.load_nameserver_credentials(working_directory=shared_directory)
        w.run(background=False)
        exit(0)

    # This example shows how to log live results. This is most useful
    # for really long runs, where intermediate results could already be
    # interesting. The core.result submodule contains the functionality to
    # read the two generated files (results.json and configs.json) and
    # create a Result object.
    result_logger = hpres.json_result_logger(directory=shared_directory, overwrite=False)

    # Start a nameserver:
    NS = hpns.NameServer(run_id=args.run_id, host=host, port=0, working_directory=shared_directory)
    ns_host, ns_port = NS.start()

    # Start local worker
    w = stablebaselinesWorkerPPO(run_id=args.run_id, host=host, nameserver=ns_host, nameserver_port=ns_port, timeout=120, env_name=env_name, num_processors=n_processors, shared_directory=shared_directory, seed=seed)
    w.run(background=True)

    # Run an optimizer
    bohb = BOHB(configspace=stablebaselinesWorkerPPO.get_configspace(), run_id=args.run_id, host=host, nameserver=ns_host, nameserver_port=ns_port, result_logger=result_logger, min_budget=args.min_budget, max_budget=args.max_budget, )
    result = bohb.run(n_iterations=args.n_iterations, min_n_workers=args.n_workers)

    # store results
    with open(os.path.join(shared_directory, 'results.pkl'), 'wb') as fh:
        pickle.dump(result, fh)

    # shutdown
    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()

