from collections import namedtuple

Game = namedtuple('Game', ['env_name', 'layers', 'activation'])

games = {}

bipedhard_stoc = Game(env_name='BipedalWalkerHardcore-v3',
                      layers=[120, 20],
                      activation='tanh',)
games['bipedhard_stoc'] = bipedhard_stoc

bipedhard = Game(env_name='BipedalWalkerHardcore-v3',
                 layers=[40, 40],
                 activation='tanh',)
games['bipedhard'] = bipedhard

biped = Game(env_name='BipedalWalker-v2',
             layers=[64, 64],
             activation='tanh',)
games['biped'] = biped

lunar = Game(env_name='LunarLander-v2',
             layers=[64, 64],
             activation='tanh',)
games['lunar'] = lunar

#Ant
pybullet_ant = Game(env_name='AntPyBulletEnv-v0',
           layers=[64, 32],
           activation='tanh',)
games['pybullet_ant'] = pybullet_ant

mujoco_ant = Game(env_name='AntMuJoCoEnv-v0',
           layers=[64, 32],
           activation='tanh',)
games['mujoco_ant'] = mujoco_ant

bullet_ant = Game(env_name='AntBulletEnv-v0',
           layers=[64, 32],
           activation='tanh',)
games['bullet_ant'] = bullet_ant

#Walker2D
bullet_twodwalker = Game(env_name='Walker2DBulletEnv-v0',
                  layers=[110, 30],
                  activation='tanh',)
games['bullet_twodwalker'] = bullet_twodwalker

pybullet_twodwalker = Game(env_name='Walker2DPyBulletEnv-v0',
                  layers=[110, 30],
                  activation='tanh',)
games['pybullet_twodwalker'] = pybullet_twodwalker

mujoco_twodwalker = Game(env_name='Walker2DMuJoCoEnv-v0',
                  layers=[110, 30],
                  activation='tanh',)
games['mujoco_twodwalker'] = mujoco_twodwalker

cartpole = Game(env_name='CartPole-v1',
                layers=[32, 32],
                activation='tanh',)
games['cartpole'] = cartpole

#Atari
space_invaders_ram = Game(env_name='SpaceInvaders-ram-v4',
                          layers=[64, 64],
                          activation='tanh',)
games['space_invaders_ram'] = space_invaders_ram

gravitar = Game(env_name='Gravitar-ramDeterministic-v4',
                layers=[64, 64],
                activation='tanh',)
games['gravitar'] = gravitar

minigrid = Game(env_name='MiniGrid-DoorKey-5x5-v0',
                layers=[64, 64],
                activation='tanh',)
games['minigrid'] = minigrid

#Hopper
bullet_hopper = Game(env_name='HopperBulletEnv-v0',
                     layers=[75, 15],
                     activation='tanh',)
games['bullet_hopper'] = bullet_hopper

pybullet_hopper = Game(env_name='HopperPyBulletEnv-v0',
                     layers=[75, 15],
                     activation='tanh',)
games['pybullet_hopper'] = pybullet_hopper

mujoco_hopper = Game(env_name='HopperMuJoCoEnv-v0',
                     layers=[75, 15],
                     activation='tanh',)
games['mujoco_hopper'] = mujoco_hopper


bullet_racecar = Game(env_name='RacecarBulletEnv-v0',
                      layers=[20, 20],
                      activation='tanh',)
games['bullet_racecar'] = bullet_racecar

bullet_minitaur = Game(env_name='MinitaurBulletEnv-v0',
                       layers=[64, 32],
                       activation='tanh',)
games['bullet_minitaur'] = bullet_minitaur

bullet_minitaur_duck = Game(env_name='MinitaurBulletDuckEnv-v0',
                            layers=[64, 32],
                            activation='tanh',)
games['bullet_minitaur_duck'] = bullet_minitaur_duck

twod_car = Game(env_name='CarRacing-v0',
                layers=[32, 32],
                activation='tanh',)
games['twod_car'] = twod_car
#Half-Cheetah
bullet_half_cheetah = Game(env_name='HalfCheetahBulletEnv-v0',
                           layers=[64, 32],
                           activation='tanh',)
games['bullet_half_cheetah'] = bullet_half_cheetah

pybullet_half_cheetah = Game(env_name='HalfCheetahPyBulletEnv-v0',
                           layers=[64, 32],
                           activation='tanh',)
games['pybullet_half_cheetah'] = pybullet_half_cheetah

mujoco_half_cheetah = Game(env_name='HalfCheetahMuJoCoEnv-v0',
                           layers=[64, 32],
                           activation='tanh',)
games['mujoco_half_cheetah'] = mujoco_half_cheetah

bullet_humanoid = Game(env_name='HumanoidBulletEnv-v0',
                       layers=[220, 85],
                       activation='tanh',)
games['bullet_humanoid'] = bullet_humanoid

bullet_pendulum = Game(env_name='InvertedPendulumSwingupBulletEnv-v0',
                       layers=[25, 5],
                       activation='tanh',)
games['bullet_pendulum'] = bullet_pendulum

bullet_double_pendulum = Game(env_name='InvertedDoublePendulumBulletEnv-v0',
                              layers=[45, 5],
                              activation='tanh',)
games['bullet_double_pendulum'] = bullet_double_pendulum

bullet_pendulum = Game(env_name='InvertedPendulumSwingupBulletEnv-v0',
                       layers=[25, 5],
                       activation='tanh',)
games['bullet_pendulum'] = bullet_pendulum

bullet_double_pendulum = Game(env_name='InvertedDoublePendulumBulletEnv-v0',
                              layers=[45, 5],
                              activation='tanh',)
games['bullet_double_pendulum'] = bullet_double_pendulum

bullet_minitaur_ball = Game(env_name='MinitaurBallGymEnv-v0',
                       layers=[64, 32],
                       activation='tanh',)
games['bullet_minitaur_ball'] = bullet_minitaur_ball

bullet_minitaur_trotting = Game(env_name='MinitaurTrottingEnv-v0',
                       layers=[64, 32],
                       activation='tanh',)
games['bullet_minitaur_trotting'] = bullet_minitaur_trotting

bullet_minitaur_extended = Game(env_name='MMinitaurExtendedEnv-v0',
                       layers=[64, 32],
                       activation='tanh',)
games['bullet_minitaur_extended'] = bullet_minitaur_extended


default = Game(env_name='None',
               layers=[64, 64],
               activation='tanh',)
games['default'] = default
