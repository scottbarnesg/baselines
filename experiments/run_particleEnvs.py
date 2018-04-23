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
import logging


class EnvVec(VecEnv):
    def __init__(self, env_fns, particleEnv, test=False):
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
        self.test = test
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
                if self.test == False:
                    a = [[action_n[0][0][0][i], action_n[0][0][1][i]], [action_n[1][0][0][i], action_n[1][0][1][i]]]
                else:
                    a = [[action_n[0][0][i], action_n[0][1][i]], [action_n[1][0][i], action_n[1][1][i]]]
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
    learn(policy_fn, env, seed, nsteps=16, nstack=1, total_timesteps=int(num_timesteps * 1.1), lr=1e-4, lrschedule=lrschedule, continuous_actions=continuous_actions, numAgents=numAgents, continueTraining=False, debug=False, particleEnv=True, model_name='partEnv_model_', log_interval=100)

def test(env_id, policy_name, seed, nstack=1, numAgents=2, benchmark=False):
    iters = 100
    rwd = []
    percent_exp = []
    env = EnvVec([make_env(env_id, benchmark=benchmark, rank=idx, seed=seed) for idx in range(1)], particleEnv=True, test=True)
    # print(env_id)
    # print("logger dir: ", logger.get_dir())
    # env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir()))
    if env_id == 'Pendulum-v0':
        if continuous_actions:
            env.action_space.n = env.action_space.shape[0]
        else:
            env.action_space.n = 10
    gym.logger.setLevel(logging.WARN)
    ob_space = env.observation_space
    ac_space = env.action_space

    # def get_img(env):
    #     ax, img = env.get_img()
    #    return ax, img

    # def process_img(img):
    #     img = rgb2grey(copy.deepcopy(img))
    #    img = resize(img, img_shape)
    #    return img

    policy_fn = policy_fn_name(policy_name)

    nsteps=5
    total_timesteps=int(80e6)
    vf_coef=0.5
    ent_coef=0.01
    max_grad_norm=0.5
    lr=7e-4
    lrschedule='linear'
    epsilon=1e-5
    alpha=0.99
    continuous_actions=False
    debug=False
    if numAgents == 1:
        model = Model(policy=policy_fn, ob_space=ob_space, ac_space=ac_space, nenvs=1, nsteps=nsteps, nstack=nstack, num_procs=1, ent_coef=ent_coef, vf_coef=vf_coef,
                  max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule, continuous_actions=continuous_actions, debug=debug)
        m_name = 'test_model_Mar7_1mil.pkl'
        model.load(m_name)
    else:
        model = []
        for i in range(numAgents):
            model.append(Model(policy=policy_fn, ob_space=ob_space, ac_space=ac_space, nenvs=1, nsteps=nsteps, nstack=nstack, num_procs=1, ent_coef=ent_coef, vf_coef=vf_coef,
                      max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule, continuous_actions=continuous_actions, debug=debug, itr=i, particleEnv=True))
        for i in range(numAgents):
            m_name = 'partEnv_model_' + str(i) + '.pkl'
            model[i].load(m_name)
            print('---------------------------------------------')
            print("Successfully Loaded: ", m_name)
            print('---------------------------------------------')

    obs = env.reset()
    states = [[], []]
    dones = [False, False]
    rwd = [[], []]
    percent_exp = [[], []]
    for i in range(1, iters+1):
        if i % 1 == 0:
            for j in range(numAgents):
                print('-----------------------------------')
                print('Agent ' + str(j))
                print('Iteration: ', i)
                avg_rwd = sum(rwd[j])/i
                # avg_pct_exp = sum(percent_exp[j])/i
                # med_pct_exp = statistics.median(percent_exp[j])
                print('Average Reward: ', avg_rwd)
                # print('Average Percent Explored: ', avg_pct_exp, '%')
                # print('Median Percent Explored: ', med_pct_exp)
                print('-----------------------------------')
        actions = [[], []]
        values = [[], []]
        total_rewards = [[0], [0]]
        nstack = 1
        for tidx in range(1000):
            # if tidx % nstack == 0:
            for j in range(numAgents):
                # if tidx > 0:
                    # input_imgs = np.expand_dims(np.squeeze(np.stack(img_hist, -1)), 0)
                    # print(np.shape(input_imgs))
                    # plt.imshow(input_imgs[0, :, :, 0])
                    # plt.imshow(input_imgs[0, :, :, 1])
                    # plt.draw()
                    # plt.pause(0.000001)
                # print(obs[:, j])
                # print(states[j])
                # print(dones)
                actions[j], values[j], states[j] = model[j].step(obs[:, j].reshape(1, 21), states[j], dones[j])
                    # action = actions[0]
                    # value = values[0]

            obs, rewards, dones, _ = env.step(actions)
            dones = dones.flatten()
            total_rewards += rewards
            print(total_rewards)
            # print(dones)
                # img = get_img(env)
                # obs_hist[j].append(img[j])
                # imsave(os.path.join(frames_dir[j], 'frame_{:04d}.png'.format(tidx)), resize(img[j], (img_shape[0], img_shape[1], 3)))
            # print(tidx, '\tAction: ', action, '\tValues: ', value, '\tRewards: ', reward, '\tTotal rewards: ', total_rewards)#, flush=True)
            if True in dones:
                # print('Faultered at tidx: ', tidx)
                for j in range(numAgents):
                    rwd[j].append(total_rewards[j])
                    # percent_exp[j].append(env.env.percent_explored[j])
                obs = env.reset()
                break
    for i in range(numAgents):
        print('-----------------------------------')
        print('Agent ' + str(i))
        print('Iteration: ', iters)
        avg_rwd = sum(rwd[i])/iters
        # avg_pct_exp = sum(percent_exp[i])/iters
        # med_pct_exp = statistics.median(percent_exp[i])
        print('Average Reward: ', avg_rwd)
        # print('Average Percent Explored: ', avg_pct_exp, '%')
        # print('Median Percent Explored: ', med_pct_exp)
        print('-----------------------------------')

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
    parser.add_argument('--num-timesteps', type=int, default=int(1e9))
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
              policy=args.policy, lrschedule=args.lrschedule, num_cpu=32, continuous_actions=continuous_actions, numAgents=numAgents)
