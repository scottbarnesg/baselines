import argparse
import gym
import numpy as np
import os
import tensorflow as tf
import time
import pickle

# import maddpg.common.tf_util as U
# sfrom maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers

# baselines libraries
from baselines import logger
from baselines.common import set_global_seeds
from baselines import bench
from baselines.common.vec_env import VecEnv
#from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.a2c.a2c import learn, Model
from baselines.a2c.policies import CnnPolicy, LstmPolicy, LnLstmPolicy, MlpPolicy
from baselines.a2c.utils import fc, conv, conv_to_fc, sample


class EnvVec(VecEnv):
    def __init__(self, env_fns, particleEnv):
        # print(env_fns)
        # self.envs = [fn() for fn in env_fns]
        self.envs = env_fns
        env = self.envs[0]
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.ts = np.zeros(len(self.envs), dtype='int')
        self.remotes = [0]*len(env_fns)
        self.n = env.n
        self.particleEnv=particleEnv
        print('Successfully Loaded Particle_Env Environments')
        print('----------------------------------------------')

    def step(self, action_n, ind=0):
        obs = []
        rews = []
        dones = []
        infos = []
        imgs = []
        # print('action_n = ' + str(action_n))
        # print(action_n)
        if self.particleEnv == False:
            for (a,env) in zip(action_n, self.envs): # REPLACE
                ob, rew, done, info = env.step(a, ind) # MAY NOT BE CORRECT
        else:
            # print(self.num_envs)
            for i in range(self.num_envs):
                # print(i)
                env = self.envs[i]
                a = [[action_n[0][0][0][i], action_n[0][0][1][i]], [action_n[1][0][0][i], action_n[1][0][1][i]]]
                # print('a: ', a)
                ob, rew, done, info = env.step(a)
            # plt.imshow(ob)
            # plt.draw()
            # plt.pause(0.000001)
            # Need to fix below. What is the difference between obs and imgs?
                obs.append(ob)
                rews.append(rew)
                dones.append(done)
                infos.append(info)
                imgs.append(ob)
        self.ts += 1
        for (i, done) in enumerate(dones):
            # print ('Debug')
            # print('dones ' + str(dones))
            # print('envs' + str(self.envs))
            # print()
            if done:
                # print (i)
                # obs[i] = self.envs[i].reset()
                self.ts[i] = 0
        return np.array(imgs), np.array(rews), np.array(dones), infos

    def reset(self):
        results = []
        for env in self.envs:
            obs = env.reset()
            results.append(obs)
        #results = [env.reset() for env in self.envs]
        return np.array(results)

    @property
    def num_envs(self):
        return len(self.envs)



def make_env(scenario_name, benchmark=False, rank=-1, seed=0):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    # print(world.agents)
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    env.seed = (seed+rank)
    env.ID = rank
    return env

def train(env_id, num_timesteps, seed, policy, lrschedule, num_cpu, continuous_actions=False, numAgents=2, benchmark=False):
    # Create environment
    env = EnvVec([make_env(env_id, benchmark=benchmark, rank=idx, seed=seed) for idx in range(num_cpu)], particleEnv=True)
    # env = make_env(env_id, benchmark)
    # print('action space: ', env.action_space)
    # env = GymVecEnv([make_env(idx) for idx in range(num_cpu)])
    policy_fn = policy_fn_name(policy)
    learn(policy_fn, env, seed, nsteps=128, nstack=1, total_timesteps=int(num_timesteps * 1.1), lr=1e-2, lrschedule=lrschedule, continuous_actions=continuous_actions, numAgents=numAgents, continueTraining=False, debug=False, particleEnv=True, model_name='partEnv_model_')


def policy_fn_name(policy_name):
    if policy_name == 'cnn':
        policy_fn = CnnPolicy
    elif policy_name == 'lstm':
        policy_fn = LstmPolicy
    elif policy_name == 'lnlstm':
        policy_fn = LnLstmPolicy
    elif policy_name == 'mlp':
        policy_fn = MlpPolicy
    return policy_fn


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='simple_reference')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm', 'mlp'], default='mlp')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')
    parser.add_argument('--num-timesteps', type=int, default=int(1e7))
    parser.add_argument('-c', '--continuous_actions', default=False, action='store_true')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--interactive', default=False, action='store_true')
    # parser.add_argument('--multiAgent', default=False, action='store_true')
    args = parser.parse_args()
    logger.configure()

    '''
    if args.env == 'Pendulum-v0':
        continuous_actions = True
    else:
    '''
    continuous_actions = False
    numAgents = 2

    if args.test:
        test(args.env, args.policy, args.seed, nstack=1)
    else:
        train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
              policy=args.policy, lrschedule=args.lrschedule, num_cpu=4, continuous_actions=continuous_actions, numAgents=numAgents)
