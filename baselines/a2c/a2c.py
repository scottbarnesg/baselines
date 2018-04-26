import os.path as osp
import gym
import time
import joblib
import logging
import numpy as np
import tensorflow as tf
from baselines import logger

from baselines.common import set_global_seeds, explained_variance
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

from baselines.a2c.utils import discount_with_dones
from baselines.a2c.utils import Scheduler, make_path, find_trainable_variables
from baselines.a2c.policies import CnnPolicy
from baselines.a2c.utils import cat_entropy, mse

from matplotlib import pyplot as plt
from itertools import chain

class Model(object):

    def __init__(self, policy, ob_space, ac_space, nenvs, nsteps, nstack, num_procs,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
                 alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear', continuous_actions=False, debug=False, numAgents=2, itr=1, particleEnv=False, communication=False):
        self.continuous_actions = continuous_actions
        self.nenvs = nenvs
        # print('numAgents = ' + str(numAgents))
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=num_procs,
                                inter_op_parallelism_threads=num_procs)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        # print('action space: ', ac_space)
        if particleEnv == False:
            nact = ac_space.n
        elif communication == False:
            nact = ac_space[itr].n
        else:
            nact = ac_space[itr].high - ac_space[itr].low # modified
        self.nact = nact
            # print('nact: ', nact)
        # print(nact)
        nbatch = nenvs*nsteps
        # print(nbatch)
        # print('batch size: ', nbatch)
        if self.continuous_actions:
            A = tf.placeholder(tf.float32, [nbatch])
        elif particleEnv == False or communication == False:
            A = tf.placeholder(tf.int32, [nbatch])
        else:
            actions_per_agent = 2
            A = tf.placeholder(tf.int32, [actions_per_agent, nbatch])
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])
        if particleEnv == False:
            step_model = policy(sess, ob_space, ac_space, nenvs, 1, nstack, reuse=tf.AUTO_REUSE, continuous_actions=continuous_actions) #, itr=itr)
            train_model = policy(sess, ob_space, ac_space, nenvs, nsteps, nstack, reuse=tf.AUTO_REUSE, continuous_actions=continuous_actions) #, itr=itr)
        elif communication == False:
            # print('step model')
            step_model = policy(sess, ob_space, ac_space, nenvs, 1, nstack, reuse=False, continuous_actions=continuous_actions, itr=itr, communication=communication)
            # print('train model')
            train_model = policy(sess, ob_space, ac_space, nenvs, nsteps, nstack, reuse=tf.AUTO_REUSE, continuous_actions=continuous_actions, itr=itr, communication=communication)
        else:
            step_model = policy(sess, ob_space, ac_space, nenvs, 1, nstack, reuse=tf.AUTO_REUSE, continuous_actions=continuous_actions, itr=itr, communication=communication)
            train_model = policy(sess, ob_space, ac_space, nenvs, nsteps, nstack, reuse=tf.AUTO_REUSE, continuous_actions=continuous_actions, itr=itr, communication=communication)
        # else:
        # else:
        #     step_model = []
        #     train_model = []
        #     for i in range(numAgents):
        #         step_model.append(policy(sess, ob_space, ac_space, nenvs, 1, nstack, reuse=tf.AUTO_REUSE, continuous_actions=continuous_actions))
        #         train_model.append(policy(sess, ob_space, ac_space, nenvs, nsteps, nstack, reuse=True, continuous_actions=continuous_actions))

        # print(train_model)
        if self.continuous_actions:
            neglogpac = tf.log(mse(train_model.mu, A))
        elif particleEnv == False or communication == False:
            # print('A: ', A)
            neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi, labels=A)
            vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), R))
            entropy = tf.reduce_mean(cat_entropy(train_model.pi))
            pg_loss = tf.reduce_mean(ADV * neglogpac)
            loss = pg_loss - entropy*ent_coef + vf_loss * vf_coef
        else:
            neglogpac = []
            entropy = []
            pg_loss = []
            loss = []
            vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), R))
            neglogpac_ = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi_c, labels=A[0])
            entropy_ = tf.reduce_mean(cat_entropy(train_model.pi_c))
            pg_loss_ = tf.reduce_mean(ADV * neglogpac_)
            entropy.append(entropy_)
            pg_loss.append(pg_loss_)
            loss.append(pg_loss_ - entropy_*ent_coef + vf_loss * vf_coef)
            neglogpac_ = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi_u, labels=A[1])
            entropy_ = tf.reduce_mean(cat_entropy(train_model.pi_u))
            pg_loss_ = tf.reduce_mean(ADV * neglogpac_)
            entropy.append(entropy_)
            pg_loss.append(pg_loss_)
            loss.append(pg_loss_ - entropy_*ent_coef + vf_loss * vf_coef)

        params = find_trainable_variables("model")
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        # f itr == 0:
        # trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
        _train = tf.train.AdamOptimizer(learning_rate=LR, name=str(itr)).apply_gradients(grads)  # , decay=alpha, epsilon=epsilon, name=str(itr)).apply_gradients(grads)
        # _train = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon, name=str(itr)).apply_gradients(grads) # Error here

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, states, rewards, masks, actions, values, debug=False, numAgents=2):
            # print('train rewards and values')
            # print(actions[0])
            # print(actions[1])
            # print(rewards)
            # print(values)
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = lr.value()
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards, LR:cur_lr}
            # if states != []:
            if train_model.initial_state != []:
                # print(states)
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            if debug==True:
                policy_loss, value_loss, policy_entropy, all_grad_vals, _ = sess.run(
                    [pg_loss, vf_loss, entropy, grads, _train],
                    td_map
                )
                grad_vals = [(np.min(grad_vals), np.max(grad_vals), np.sum(grad_vals)) for grad_vals in all_grad_vals]
                print('Policy Gradients: ')
                print(all_grad_vals[9])
                print('Value Gradients: ')
                print(all_grad_vals[11])
            else:
                policy_loss, value_loss, policy_entropy, _ = sess.run(
                    [pg_loss, vf_loss, entropy, _train],
                    td_map
                )
            # else:
                # td_map = []
            #     print('Train Model in train')
            #     print(train_model)
            #     for i in range(numAgents):
            #         td_map = {train_model[i].X:obs, A:actions, ADV:advs, R:rewards, LR:cur_lr}
            #         if train_model[i].initial_state != []:
            #             print('states')
            #            print(states)
            #            td_map[train_model[i].S] = states
            #            td_map[train_model[i].M] = masks
            #        if debug:
            #            print('point1')
            #            policy_loss, value_loss, policy_entropy, all_grad_vals, _ = sess.run(
            #                [pg_loss, vf_loss, entropy, grads, _train],
            #                td_map
            #            )
            #            print('point2')
            #            grad_vals = [(np.min(grad_vals), np.max(grad_vals), np.sum(grad_vals)) for grad_vals in all_grad_vals]
            #            print('Policy Gradients: ')
            #            print(all_grad_vals[9])
            #            print('Value Gradients: ')
            #            print(all_grad_vals[11])
            #        else:
            #            policy_loss, value_loss, policy_entropy, _ = sess.run(
            #                [pg_loss, vf_loss, entropy, _train],
            #                td_map
            #            )
            # print('Policy Loss: ')
            # print(policy_loss)
            # print('Value Loss: ')
            # print(value_loss)

            return policy_loss, value_loss, policy_entropy

        def save(save_path):
            ps = sess.run(params)
            #make_path(save_path)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            ps = sess.run(restores)

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        # if numAgents == 1:
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        # else:
        #     self.step = []
        #     self.value = []
        #     self.initial_state = []
        #     for i in range(numAgents):
        #         self.step.append(step_model[i].step)
        #         self.value.append(step_model[i].value)
        #         self.initial_state.append(step_model[i].initial_state)
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess)

class Runner(object):

    def __init__(self, env, model, nsteps=5, nstack=1, gamma=0.99, ind=-1, init_obs=[], particleEnv=False, numAgents=2):
        self.ind = ind
        self.env = env
        self.model = model
        self.particleEnv = particleEnv
        self.numAgents = numAgents
        if particleEnv == False:
            nh, nw, nc = env.observation_space.shape
            nenv = env.num_envs
            self.batch_ob_shape = (nenv*nsteps, nh, nw, nc*nstack)
            self.obs = np.zeros((nenv, nh, nw, nc*nstack), dtype=np.uint8)
            self.nc = nc
        else:
            # ob_shape = model[0].step_model.ob_shape
            ob_shape = [env.observation_space[i].shape for i in range(env.n)]
            self.batch_obs_shape = ob_shape
            self.nenv = env.num_envs

            # print('ob_shape: ', ob_shape)
            self.obs = [np.zeros(self.batch_obs_shape[1], dtype=np.uint8) for i in range(env.num_envs)]
            # nenv = 1
        # print('nsteps', nsteps)
        if particleEnv == True: # and self.ind == 0:
            obs = env.reset()
            # print('env reset obs: ', obs)
            self.init_obs = obs
            self.obs = self.init_obs
            # print('obs: ', obs)
            # print('obs')
            # print(obs[0])
        elif self.ind == -1:
            env, obs = env.reset() # obs = env.reset()
            self.update_obs(obs) # error here
        elif self.ind == 0:
            env, obs = env.reset() # obs = env.reset()
            self.init_obs = obs
            self.update_obs(obs[:, ind, :, :, :])
        elif particleEnv == False:
            self.update_obs(init_obs[:, ind, :, :, :])
        else:
            self.init_obs = init_obs
            self.obs = self.init_obs
            # print(self.obs)
        self.gamma = gamma
        self.nsteps = nsteps
        if particleEnv == True:
            self.states = model[0].initial_state
        else:
            self.states = model.initial_state
        # print('self.states: ', self.states)
        # print(model.initial_state)
        self.dones = [False for _ in range(self.nenv)]

    def update_obs(self, obs):  # need to fix update_obs
        # Do frame-stacking here instead of the FrameStack wrapper to reduce
        # IPC overhead
        self.obs = np.roll(self.obs, shift=-self.nc, axis=3)
        self.obs[:, :, :, -self.nc:] = obs

    def run(self): # need to fix obs
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [],[],[],[],[]
        mb_states = self.states
        if self.particleEnv == False:
            actions, values, states = self.model.step(self.obs, self.states, self.dones)
        else:
            actions = [[], []]
            values = [[], []]
            states = [[], []]
            # print(self.model)
            for i in range(self.numAgents):
                # print(i)
                # print(self.model[i])
                # print('model.step obs: ', (self.obs[:, i]).flatten())
                # print(self.states)
                # print(self.dones)
                if self.env.name == 'simple_reference':
                    actions_, values_, states_ = self.model[i].step(self.obs[: ,i].reshape(self.nenv, 21), self.states, self.dones)
                else:
                    obs_ = []
                    # print('ob_space: ', self.batch_obs_shape)
                    # print(self.obs[:, i].shape)
                    for j in range(self.obs[:, i].shape[0]):
                        for k in range(self.batch_obs_shape[i][0]):
                            obs_.append(self.obs[:, i][j][k])
                    obs_ = np.asarray(obs_).reshape(self.nenv, self.batch_obs_shape[i][0])
                    # print(obs_)
                    actions_, values_, states_ = self.model[i].step(obs_, self.states, self.dones) # changed reshape from 21
                # print('modelstep action: ', actions_)
                # print('obs[i]: ', self.obs[i])
                # print('actions_: ', actions_)
                # print('model_step values: ', values_.shape)
                actions[i].append(actions_)
                values[i].append(values_)
                states[i].append(states_)
        # print('Agent: ', self.ind)
        # print('Action: ', actions)
        mb_obs.append(np.copy(self.obs))
        mb_actions.append(actions)
        mb_values.append(values)
        if self.particleEnv == False:
            mb_dones.append(self.dones)
        if self.particleEnv == False:
            obs, rewards, dones, _ = self.env.step(np.asarray(actions), self.ind)
        else:
            obs, rewards, dones, _ = self.env.step(actions) # steps through all envs
            # print('env step obs: ', obs.shape)
            # print(obs)
            # assert 1==0
            # print('a2c obs: ', obs)
            # print('env rewards: ', rewards)
        # for i in range (6): #nenvs
        #     print('Agent ', i) # DEBUG
        #     print('Obs: ', obs.shape)
        #     plt.imshow(obs[i, self.ind])
        #     plt.draw()
        #     print('observation ' + str(i) + ', ' +str(self.ind))
        #     plt.pause(0.01)
        self.states = states
        self.dones = dones
        if self.particleEnv==False:
            mb_dones.append(self.dones)
        # print('mb_dones: ', mb_dones)
        # print('obs: ', obs)
        if self.particleEnv == True:
            dones_ = dones[0]
            # print('dones_: ', dones_)
        else:
            dones_ = dones
        for n, done in enumerate(dones_):
            # print(done)
            if done:
                self.obs[n] = self.obs[n]*0

        if self.particleEnv==True:
            self.obs = obs
        elif self.ind == -1:
            self.update_obs(obs) # error here
        else:
            self.update_obs(obs[:, self.ind]) # removes unwanted obs
        # print('Obs: ', self.obs.shape)
        # print('mb_obs: ', np.asarray(mb_obs).shape)
        # for i in range (self.model.nenvs): #nenvs
        #     print('Agent ', self.ind) # DEBUG
        #     plt.imshow(self.obs[i])
        #     plt.draw()
        #    print('observation ' + str(i) + ', ' +str(self.ind))
        #     plt.pause(0.1)
        mb_rewards.append(rewards)
        mb_dones.append(self.dones)
        #batch of steps to batch of rollouts
        # mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_ob_shape)
        if self.particleEnv==False:
            mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0).reshape(self.model.nenvs, 84, 84, 3)
            mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
            mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
            mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
            mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
            # print(mb_dones)
            # mb_masks = mb_dones[:-1]
            mb_masks = mb_dones[:, :-1]
            # mb_dones = mb_dones[1:]
            mb_dones = mb_dones[:, 1:]
            # print(mb_masks)
            last_values = self.model.value(self.obs, self.states, self.dones).tolist()
        else:
            # print('obs shape: ', np.asarray(mb_obs).shape)
            mb_obs = np.asarray(mb_obs) #, dtype=np.float32)
            mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
            mb_actions = np.asarray(mb_actions, dtype=np.int32)
            mb_values = np.asarray(mb_values, dtype=np.float32)
            mb_dones = np.asarray(mb_dones, dtype=np.bool)
            # mb_masks = mb_dones[:, :-1]
            mb_masks = mb_dones
            # mb_dones = mb_dones[:, 1:]
            last_values = []
            # print('self.obs: ', self.obs[:, i])
            for i in range(self.numAgents):
                if self.env.name == 'simple_reference':
                    last_values.append(self.model[i].value(self.obs[:, i].reshape(self.nenv, 21), self.states, self.dones).tolist())
                else:
                    obs_ = []
                    for j in range(self.obs[:, i].shape[0]):
                        for k in range(self.batch_obs_shape[i][0]):
                            obs_.append(self.obs[:, i][j][k])
                    # self.obs[:, i].reshape(self.nenv, self.batch_obs_shape[i][0])
                    last_values.append(self.model[i].value(np.asarray(obs_).reshape(self.nenv, self.batch_obs_shape[i][0]), self.states, self.dones).tolist())
        # else:
        # last_values = self.model.value[self.ind](self.obs, self.states, self.dones).tolist()
        #discount/bootstrap off value fn
        # print(mb_dones)
        # print(mb_obs)
        # print('runner actions: ', mb_actions.shape)
        # print('runner obs: ', mb_obs.shape)
        # print('runner values: ', mb_values.shape)
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            # print(dones)
            rewards = rewards.tolist()
            dones = dones.tolist()
            # print(np.asarray(rewards))
            # print(dones)
            if type(dones) == list:
                # print(dones)
                if dones[-1] == 0:
                    rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma, particleEnv=self.particleEnv)[:-1] # ERROR
                else:
                    rewards = discount_with_dones(rewards, dones, self.gamma, particleEnv=self.particleEnv)
            else:
                if dones == 0:
                    rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma, particleEnv=self.particleEnv)[:-1] # ERROR
                else:
                    rewards = discount_with_dones(rewards, dones, self.gamma, particleEnv=self.particleEnv)
            # print('rewards:', rewards)
            # print('mb_rewards: ', mb_rewards[n])
            mb_rewards[n] = rewards
        if self.particleEnv == False:
            mb_rewards = mb_rewards.flatten()
        # print(mb_actions)
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()
        # print('mb_action ', mb_actions)
        # if self.particleEnv == True:
        if self.env.name == 'simple_reference':
            action_per_agent = 2
            mb_actions = mb_actions.reshape(self.numAgents, action_per_agent, self.nenv)
            mb_values = mb_values.reshape(self.numAgents, self.nenv)
            mb_masks = mb_masks.reshape(self.numAgents, self.nenv)
        elif self.particleEnv == True:
            mb_actions = mb_actions.reshape(self.nenv, self.numAgents)
            mb_values = mb_values.reshape(self.nenv, self.numAgents)
            mb_masks = mb_masks.reshape(self.nenv, self.numAgents)

        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values

def learn(policy, env, seed, nsteps=5, nstack=1, total_timesteps=int(80e6), vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, lr=7e-4, lrschedule='linear', epsilon=1e-5, alpha=0.99, gamma=0.99, log_interval=25, continuous_actions=True, debug=False, numAgents=2, continueTraining=True, particleEnv=False, model_name='Apr3_test_model_', communication=False):
    timesteps = 100000
    tf.reset_default_graph()
    if particleEnv == False:
        set_global_seeds(seed)

    nenvs = env.num_envs
    print('Number of Environments: ', nenvs)
    print('Number of Steps', nsteps)
    nbatch = nenvs*nsteps
    print('Batch Size: ', nbatch)
    print('Learning Rate: ', lr)
    print('---------------------------------------------')
    ob_space = env.observation_space
    ac_space = env.action_space
    # print(ac_space)
    # print('action space: ', ac_space)
    num_procs = len(env.remotes) # HACK
    if numAgents == 1:
        model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, nstack=nstack, num_procs=num_procs, ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule, continuous_actions=continuous_actions, debug=debug)
    else:
        model = []
        for i in range(numAgents):
            if particleEnv == True:
                model.append(Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, nstack=nstack, num_procs=num_procs, ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule, continuous_actions=continuous_actions, debug=debug, itr=i, particleEnv=particleEnv, communication=communication))
            else:
                model.append(Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, nstack=nstack, num_procs=num_procs, ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule, continuous_actions=continuous_actions, debug=debug, itr=i, particleEnv=particleEnv))
    # print('learn models')
    # print(model)
    # print(model[0])
    if continueTraining == True:
        for i in range(numAgents):
            m_name = model_name +str(i) +'_600k.pkl'
            model[i].load(m_name)
            print('---------------------------------------------')
            print('Successfully Loaded ', m_name)
            print('---------------------------------------------')

    if numAgents == 1 or particleEnv==True:
        # print('Model: ', model)
        runner = Runner(env, model, nsteps=nsteps, nstack=nstack, gamma=gamma, particleEnv=particleEnv)
    else:
        runner = []
        for i in range(numAgents):
            if i == 0:
                # print('env: ', env)
                runner.append(Runner(env, model[i], nsteps=nsteps, nstack=nstack, gamma=gamma, ind=i, particleEnv=particleEnv))
                # print('runner model values')
                # print(model[i].value)
            else:
                runner.append(Runner(env, model[i], nsteps=nsteps, nstack=nstack, gamma=gamma, ind=i, init_obs=runner[0].init_obs, particleEnv=particleEnv))

    tstart = time.time()
    for update in range(1, total_timesteps//nbatch+1):
        if numAgents == 1:
            obs, states, rewards, masks, actions, values = runner.run()
        elif particleEnv == True: # rework to pre-sort data by agent
            obs =  [[], []]
            states =  [[], []]
            rewards =  [[], []]
            masks =  [[], []]
            actions =  [[], []]
            values =  [[], []]
            policy_loss = []
            value_loss = []
            policy_entropy = []
            ev = []
            for i in range(nsteps):
                    obs_, states_, rewards_, masks_, actions_, values_ = runner.run()
                    # print('Runner Shapes')
                    # print(obs_.shape)
                    # print(rewards_.shape)
                    # print('states: ', states_)
                    # print('masks: ', masks_)
                    # print('actions_ ', actions_)
                    # print('values: ', values_[:, 0])
                    # assert 1==0
                    # print('masks: ', masks_)
                    for j in range(numAgents):
                        if runner.env.name == 'simple_reference':
                            obs[j].append(obs_[:,:,j])
                            states[j].append(states[0])
                            rewards[j].append(rewards_[:,:,j])
                            actions[j].append(actions_[:, j])
                            values[j].append(values_[j])
                            masks[j].append(masks_[j])
                        else:
                            obs[j].append(obs_[:,:,j])
                            states[j].append(states[0])
                            rewards[j].append(rewards_[:,:,j])
                            actions[j].append(actions_[:, j])
                            values[j].append(values_[:, j])
                            masks[j].append(masks_[:, j])

        else:
            obs = [[], []]
            states = [[], []]
            rewards = [[], []]
            masks = [[], []]
            actions = [[], []]
            values = [[], []]
            policy_loss = []
            value_loss = []
            policy_entropy = []
            ev = []
            # print('nsteps: ', runner[0].nsteps)
            for j in range(nsteps):
                for i in range(numAgents): # Need to rewrite so that agents take turns stepping
                    # obs[i], states[i], rewards[i], masks[i], actions[i], values[i] = runner[i].run()
                    obs_, states_, rewards_, masks_, actions_, values_ = runner[i].run()
                    obs[i].append(obs_) # (obs_.shape = (6, 3, 3, 84))
                    states[i].append(states_)
                    rewards[i].append(rewards_)
                    masks[i].append(masks_)
                    actions[i].append(actions_)
                    values[i].append(values_)

            # print(np.asarray(values).shape)
        if numAgents == 1:
            policy_loss, value_loss, policy_entropy = model.train(obs, states, rewards, masks, actions, values)
        elif particleEnv == False:
            for i in range(numAgents):
                # print(masks[i])
                policy_loss_, value_loss_, policy_entropy_ = model[i].train(np.asarray(obs[i]).reshape(nbatch, 84, 84, 3), states[i], np.asarray(rewards[i]).reshape(nbatch), np.asarray(masks[i]).reshape(nbatch), np.asarray(actions[i]).reshape(nbatch), np.asarray(values[i]).reshape(nbatch))
                policy_loss.append(policy_loss_)
                value_loss.append(value_loss_)
                policy_entropy.append(policy_entropy_)
                ev.append(explained_variance(np.asarray(values[i]).reshape(nbatch), np.asarray(rewards[i]).reshape(nbatch)))
        else:
            # masks = np.asarray(masks).reshape(2, nbatch)
            # print(masks)
            if runner.env.name == 'simple_reference':
                actions_per_agent = 2
                actions = np.asarray(actions).swapaxes(0, 2)
            for i in range(numAgents):
                # print(obs)
                # print(obs[i])
                # print('values')
                # print(values[i])
                # print(rewards)
                # print(masks)
                # print('obs shape: ', np.asarray(obs).shape)
                # np.asarray(rewards[i]).reshape(nbatch)
                # np.asarray(masks[i]).reshape(nbatch)
                # np.asarray(actions[i]).reshape(nbatch)
                # np.asarray(values[i]).reshape(nbatch)
                # print('all actions', actions)
                # print('all values: ', values)
                # print('values shape: ', np.asarray(values).shape)
                # actions_i = np.asarray(actions[not i])
                if runner.env.name == 'simple_reference':
                    # print(np.asarray(actions).shape)
                    actions_i = np.asarray(actions[i])
                    actions_i = actions_i.swapaxes(0, 1).reshape(actions_per_agent, nbatch)
                    action_n = [actions_i[1], actions_i[0]]
                # print(action_n)
                # print(np.asarray(actions).shape)
                # assert 1 == 0
                # policy_loss_, value_loss_, policy_entropy_ = model[i].train(np.asarray(obs[i]).reshape(nbatch, 21), states[i], np.asarray(rewards[i]).reshape(nbatch), np.asarray(masks[i]).reshape(nbatch), np.asarray(actions[i]).reshape(nbatch, actions_per_agent), np.asarray(values[i]).reshape(nbatch))
                    policy_loss_, value_loss_, policy_entropy_ = model[i].train(np.asarray(obs[i]).reshape(nbatch, 21), states[i], np.asarray(rewards[i]).reshape(nbatch), np.asarray(masks[i]).reshape(nbatch), action_n, np.asarray(values[i]).reshape(nbatch))
                else:
                    # print('action shape: ', np.asarray(actions).shape)
                    actions_ = np.asarray(actions[:][:][:])
                    actions_i = actions_[:, :, i]
                    # print('action_i: ', actions_i.shape)
                    # print('obs shape: ', np.asarray(obs).shape)
                    obs_ = np.asarray(obs).swapaxes(1, 2)
                    obs_i = obs_[i, 0].flatten()
                    # print('obs_i: ', obs_i)
                    obs_n = []
                    for n in range(obs_i.shape[0]):
                        for m in range(obs_i[0].shape[0]):
                            obs_n.append(obs_i[n][m])
                    # print('obs_n: ', np.asarray(obs_n).shape)
                    # for n in range()
                    # print('obs: ', obs[i])
                    # print('shape', np.asarray(obs).shape)
                    # for j in range(np.asarray(obs[i]).shape[0]):
                    #     for k in range(runner.batch_obs_shape[i][0]):
                    #         obs_.append(obs[i][j][k])
                    np.asarray(rewards[i]).reshape(nbatch)
                    np.asarray(masks[i]).reshape(nbatch)
                    # np.asarray(actions_i).reshape(nbatch)
                    np.asarray(values[i]).reshape(nbatch)
                    # print('all actions: ', actions)
                    # print('actions_i: ',actions_i)
                    # assert 1==0
                    policy_loss_, value_loss_, policy_entropy_ = model[i].train(np.asarray(obs_n).reshape(nbatch, runner.batch_obs_shape[i][0]), states[i], np.asarray(rewards[i]).reshape(nbatch), np.asarray(masks[i]).reshape(nbatch), np.asarray(actions_i).reshape(nbatch), np.asarray(values[i]).reshape(nbatch))
                policy_loss.append(policy_loss_)
                value_loss.append(value_loss_)
                policy_entropy.append(policy_entropy_)
                ev.append(explained_variance(np.asarray(values[i]).reshape(nbatch), np.asarray(rewards[i]).reshape(nbatch)))


        # model.step_model.summarize_weights()

        nseconds = time.time()-tstart
        fps = float((update*nbatch)/nseconds)
        if update % log_interval == 0 or update == 1:
            for i in range(numAgents):
                logger.record_tabular("*Model Number*", i)
                logger.record_tabular("nupdates", update)
                logger.record_tabular("total_timesteps", update*nbatch)
                logger.record_tabular("fps", float(fps))
                if runner.env.name != 'simple_reference':
                    logger.record_tabular("policy_entropy", float(policy_entropy[i]))
                    logger.record_tabular("value_loss",float(value_loss[i]))
                    logger.record_tabular("policy_loss", float(policy_loss[i]))
                    logger.record_tabular("explained_variance", ev[i])
                    logger.dump_tabular()
                else:
                    logger.record_tabular("comm_policy_entropy", float(policy_entropy[i][0]))
                    logger.record_tabular("force_policy_entropy", float(policy_entropy[i][1]))
                    logger.record_tabular("value_loss",float(value_loss[i]))
                    logger.record_tabular("comm_policy_loss", float(policy_loss[i][0]))
                    logger.record_tabular("force_policy_loss", float(policy_loss[i][1]))
                    logger.record_tabular("explained_variance", ev[i])
                    logger.dump_tabular()
                m_name = model_name + str(i) + '.pkl'
                model[i].save(m_name)
                # print('Saving model as ', m_name)
        if particleEnv == False:
            if update*nbatch > timesteps:
                print('Saving ')
                for i in range(numAgents):
                    m_name = model_name + str(i) + '_' + str((timesteps+600000.0)/1000.0) + 'k.pkl'
                    model[i].save(m_name)
                timesteps += 100000
        else:
            update_interval = 10000
            if update % update_interval == 0:
                print('Saving for update ', str(update))
                for i in range(numAgents):
                    m_name = model_name + str(i) + '_' + str(update/1000.0) + 'k.pkl'
                    model[i].save(m_name)
    env.close()

if __name__ == '__main__':
    main()
