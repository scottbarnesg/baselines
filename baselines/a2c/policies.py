import numpy as np
import tensorflow as tf
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm, sample, sample_normal

import time

class LnLstmPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, nlstm=256, reuse=False):
        nbatch = nenv*nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc*nstack)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        with tf.variable_scope("model", reuse=reuse):
            h = conv(tf.cast(X, tf.float32)/255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
            h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h3)
            h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
            xs = batch_to_seq(h4, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lnlstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            pi = fc(h5, 'pi', nact, act=lambda x:x)
            vf = fc(h5, 'v', 1, act=lambda x:x)

        v0 = vf[:, 0]
        a0 = sample(pi)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            a, v, s = sess.run([a0, v0, snew], {X:ob, S:state, M:mask})
            return a, v, s

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class LstmPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, nlstm=256, reuse=False):
        nbatch = nenv*nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc*nstack)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        with tf.variable_scope("model", reuse=reuse):
            h = conv(tf.cast(X, tf.float32)/255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
            h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h3)
            h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
            xs = batch_to_seq(h4, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            pi = fc(h5, 'pi', nact, act=lambda x:x)
            vf = fc(h5, 'v', 1, act=lambda x:x)

        v0 = vf[:, 0]
        a0 = sample(pi)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            a, v, s = sess.run([a0, v0, snew], {X:ob, S:state, M:mask})
            return a, v, s

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class CnnPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=False, continuous_actions=False):
        self.sess = sess
        self.continuous_actions = continuous_actions
        # print('reuse= ', reuse)
        nbatch = nenv*nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc*nstack)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        with tf.variable_scope("model", reuse=reuse):
            h = conv(tf.cast(X, tf.float32)/255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
            h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h3)
            h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
            if self.continuous_actions:
                pi = fc(h4, 'pi', 2*nact, act=lambda x:x)
                self.mu = fc(pi, 'mu', nact, act=lambda x:x)
                self.sigma = fc(pi, 'sigma', nact, act=lambda x:x)
            else:
                pi = fc(h4, 'pi', nact, act=lambda x:x)
            vf = fc(h4, 'v', 1, act=lambda x:x)

        v0 = vf[:, 0]
        if self.continuous_actions:
            a0 = sample_normal(self.mu, self.sigma)
        else:
            a0 = sample(pi)
        self.initial_state = [] #not stateful

        def step(ob, *_args, **_kwargs):
            a, v = sess.run([a0, v0], {X:ob})
            #import ipdb; ipdb.set_trace()
            #print('a: ', a)
            if np.isnan(a[0]):
                import ipdb; ipdb.set_trace()
            return a, v, [] #dummy state

        def value(ob, *_args, **_kwargs):
            return sess.run(v0, {X:ob})

        def summarize_weights(*_args, **_kwargs):
            all_layer_vars = sess.run(tf.all_variables())
            import numpy as np
            layer_sums = [np.sum(layer_vars) for layer_vars in all_layer_vars]
            res = np.any(np.isnan(np.sum(layer_sums)))
            print('Layer weight sums: ', layer_sums)
            print('NaN weights: ', res)
            if res:
                import ipdb; ipdb.set_trace()

        self.summarize_weights = summarize_weights

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class MlpPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=False, continuous_actions=False, itr=0, particleEnvs=False, communication=False):
        self.sess = sess
        self.continuous_actions = continuous_actions
        # print('reuse= ', reuse)
        nbatch = nenv*nsteps
        # print('obs space: ', ob_space)
        ob_shape = np.asarray([nbatch, ob_space[itr].shape[0]])
        self.ob_space = ob_space
        # print('model ob shape: ', ob_shape)
        # nh, nw, nc = ob_space.shape
        # ob_shape = (nbatch, nh, nw, nc*nstack)
        # print('observation shape: ', ob_shape)
        # print('ac_space: ', ac_space)
        if communication == False:
            nact = ac_space[itr].n
            print('nact: ', nact)
        else:
            nact = ac_space[itr].high - ac_space[itr].low # + [1, 1]
        # print('nact: ', nact)
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        # X = tf.transpose(tf.expand_dims(X, nbatch))
        # print('Input Shape: ', X.get_shape())
        with tf.variable_scope("model", reuse=reuse):
            # f = fc(X, 'fc1', nh=64)
            f = fc(tf.cast(X, tf.float32), 'fc1_'+str(itr), nh=64, init_scale=np.sqrt(2))
            f2 = fc(f, 'fc2_'+str(itr),  nh=64, init_scale=np.sqrt(2))
            # f3 = fc(f2, 'fc3', nh=64, init_scale=np.sqrt(2))
            # h3 = conv_to_fc(h3)
            # h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
            if self.continuous_actions:
                pi = fc(f3, 'pi', 2*nact, act=lambda x:x)
                self.mu = fc(pi, 'mu', nact, act=lambda x:x)
                self.sigma = fc(pi, 'sigma', nact, act=lambda x:x)
                vf = fc(f2, 'v', 1, act=lambda x:x)
            elif communication == True:
                pi_c = fc(f2, 'pi_c', nact[1], act=lambda x:x)
                pi_u = fc(f2, 'pi_u', nact[0], act=lambda x:x)
                vf = fc(f2, 'v', 1, act=lambda x:x)
                self.pi_c = pi_c
                self.pi_u = pi_u
            else:
                pi = fc(f2, 'pi_'+str(itr), nact, act=lambda x:x)
                vf = fc(f2, 'v_'+str(itr), 1, act=lambda x:x)
                self.pi = pi

            # print('action output size:')
            # print(pi_c.get_shape())
            # print(pi_u.get_shape())
            # vf = fc(f2, 'v', 1, act=lambda x:x)

        v0 = vf[:, 0]
        if self.continuous_actions:
            a0 = sample_normal(self.mu, self.sigma)
        elif communication == True:
            a0 = [sample(pi_u), sample(pi_c)]
        else:
            a0 = sample(pi)
        self.initial_state = [] #not stateful

        def step(ob, *_args, **_kwargs):
            a, v = sess.run([a0, v0], {X:ob})
            #import ipdb; ipdb.set_trace()
            # print('a: ', a)
            # time.sleep(1)
            # if np.isnan(a[0]):
            #     import ipdb; ipdb.set_trace()
            return a, v, [] #dummy state

        def value(ob, *_args, **_kwargs):
            return sess.run(v0, {X:ob})

        def summarize_weights(*_args, **_kwargs):
            all_layer_vars = sess.run(tf.all_variables())
            import numpy as np
            layer_sums = [np.sum(layer_vars) for layer_vars in all_layer_vars]
            res = np.any(np.isnan(np.sum(layer_sums)))
            print('Layer weight sums: ', layer_sums)
            print('NaN weights: ', res)
            if res:
                import ipdb; ipdb.set_trace()

        self.summarize_weights = summarize_weights

        self.X = X
        self.vf = vf
        self.step = step
        self.value = value
