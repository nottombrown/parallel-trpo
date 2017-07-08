import multiprocessing

import numpy as np
import tensorflow as tf

import utils
from value_function import LinearVF


class TRPO(multiprocessing.Process):
    def __init__(self, args, observation_space, action_space, task_q, result_q):
        multiprocessing.Process.__init__(self)
        self.task_q = task_q
        self.result_q = result_q
        self.observation_space = observation_space
        self.action_space = action_space
        self.args = args

    def make_model(self):
        self.observation_size = self.observation_space.shape[0]
        self.action_size = np.prod(self.action_space.shape)
        self.hidden_size = 64

        weight_init = tf.random_uniform_initializer(-0.05, 0.05)
        bias_init = tf.constant_initializer(0)

        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
        self.session = tf.Session(config=config)

        self.obs = tf.placeholder(tf.float32, [None, self.observation_size])
        self.action = tf.placeholder(tf.float32, [None, self.action_size])
        self.advantage = tf.placeholder(tf.float32, [None])
        self.oldaction_dist_mu = tf.placeholder(tf.float32, [None, self.action_size])
        self.oldaction_dist_logstd = tf.placeholder(tf.float32, [None, self.action_size])

        with tf.variable_scope("policy"):
            h1 = utils.fully_connected(self.obs, self.observation_size, self.hidden_size, weight_init, bias_init, "policy_h1")
            h1 = tf.nn.relu(h1)
            h2 = utils.fully_connected(h1, self.hidden_size, self.hidden_size, weight_init, bias_init, "policy_h2")
            h2 = tf.nn.relu(h2)
            h3 = utils.fully_connected(h2, self.hidden_size, self.action_size, weight_init, bias_init, "policy_h3")
            action_dist_logstd_param = tf.Variable(
                (.01 * np.random.randn(1, self.action_size)).astype(np.float32), name="policy_logstd")
        # means for each action
        self.action_dist_mu = h3
        # log standard deviations for each actions
        self.action_dist_logstd = tf.tile(action_dist_logstd_param, tf.stack((tf.shape(self.action_dist_mu)[0], 1)))

        batch_size = tf.shape(self.obs)[0]
        # what are the probabilities of taking self.action, given new and old distributions
        log_p_n = utils.gauss_log_prob(self.action_dist_mu, self.action_dist_logstd, self.action)
        log_oldp_n = utils.gauss_log_prob(self.oldaction_dist_mu, self.oldaction_dist_logstd, self.action)

        # tf.exp(log_p_n) / tf.exp(log_oldp_n)
        ratio = tf.exp(log_p_n - log_oldp_n)

        # importance sampling of surrogate loss (L in paper)
        surr = -tf.reduce_mean(ratio * self.advantage)
        var_list = tf.trainable_variables()

        batch_size_float = tf.cast(batch_size, tf.float32)
        # kl divergence and shannon entropy
        kl = utils.gauss_KL(
            self.oldaction_dist_mu, self.oldaction_dist_logstd, self.action_dist_mu,
            self.action_dist_logstd) / batch_size_float
        ent = utils.gauss_ent(self.action_dist_mu, self.action_dist_logstd) / batch_size_float

        self.losses = [surr, kl, ent]
        # policy gradient
        self.pg = utils.flatgrad(surr, var_list)

        # KL divergence w/ itself, with first argument kept constant.
        kl_firstfixed = utils.gauss_selfKL_firstfixed(self.action_dist_mu, self.action_dist_logstd) / batch_size_float
        # gradient of KL w/ itself
        grads = tf.gradients(kl_firstfixed, var_list)
        # what vector we're multiplying by
        self.flat_tangent = tf.placeholder(tf.float32, [None])
        shapes = map(utils.var_shape, var_list)
        start = 0
        tangents = []
        for shape in shapes:
            size = np.prod(shape)
            param = tf.reshape(self.flat_tangent[start:(start + size)], shape)
            tangents.append(param)
            start += size

        # gradient of KL w/ itself * tangent
        gvp = [tf.reduce_sum(g * t) for (g, t) in zip(grads, tangents)]
        # 2nd gradient of KL w/ itself * tangent
        self.fvp = utils.flatgrad(gvp, var_list)
        # the actual parameter values
        self.gf = utils.GetFlat(self.session, var_list)
        # call this to set parameter values
        self.sff = utils.SetFromFlat(self.session, var_list)
        self.session.run(tf.global_variables_initializer())
        # value function
        # self.vf = VF(self.session)
        self.vf = LinearVF()

        self.get_policy = utils.GetPolicyWeights(self.session, var_list)

    def run(self):
        self.make_model()
        while True:
            paths = self.task_q.get()
            if paths is None:
                # kill the learner
                self.task_q.task_done()
                break
            elif paths == 1:
                # just get params, no learn
                self.task_q.task_done()
                self.result_q.put(self.get_policy())
            elif paths[0] == 2:
                # adjusting the max KL.
                self.args.max_kl = paths[1]
                self.task_q.task_done()
            else:
                mean_reward = self.learn(paths)
                self.task_q.task_done()
                self.result_q.put((self.get_policy(), mean_reward))
        return

    def learn(self, paths):
        # is it possible to replace A(s,a) with Q(s,a)?
        for path in paths:
            path["baseline"] = self.vf.predict(path)
            path["returns"] = utils.discount(path["rewards"], self.args.gamma)
            path["advantage"] = path["returns"] - path["baseline"]
            # path["advantage"] = path["returns"]

        # puts all the experiences in a matrix: total_timesteps x options
        action_dist_mu = np.concatenate([path["action_dists_mu"] for path in paths])
        action_dist_logstd = np.concatenate([path["action_dists_logstd"] for path in paths])
        obs_n = np.concatenate([path["obs"] for path in paths])
        action_n = np.concatenate([path["actions"] for path in paths])

        # standardize to mean 0 stddev 1
        advant_n = np.concatenate([path["advantage"] for path in paths])
        advant_n -= advant_n.mean()
        advant_n /= (advant_n.std() + 1e-8)

        # train value function / baseline on rollout paths
        self.vf.fit(paths)

        feed_dict = {
            self.obs: obs_n, self.action: action_n, self.advantage: advant_n,
            self.oldaction_dist_mu: action_dist_mu, self.oldaction_dist_logstd: action_dist_logstd}

        # parameters
        thprev = self.gf()

        # computes fisher vector product: F * [self.pg]
        def fisher_vector_product(p):
            feed_dict[self.flat_tangent] = p
            return self.session.run(self.fvp, feed_dict) + p * self.args.cg_damping

        g = self.session.run(self.pg, feed_dict)

        # solve Ax = g, where A is Fisher information metrix and g is gradient of parameters
        # stepdir = A_inverse * g = x
        stepdir = utils.conjugate_gradient(fisher_vector_product, -g)

        # let stepdir =  change in theta / direction that theta changes in
        # KL divergence approximated by 0.5 x stepdir_transpose * [Fisher Information Matrix] * stepdir
        # where the [Fisher Information Matrix] acts like a metric
        # ([Fisher Information Matrix] * stepdir) is computed using the function,
        # and then stepdir * [above] is computed manually.
        shs = 0.5 * stepdir.dot(fisher_vector_product(stepdir))

        lm = np.sqrt(shs / self.args.max_kl)
        # if self.args.max_kl > 0.001:
        #     self.args.max_kl *= self.args.kl_anneal

        fullstep = stepdir / lm
        negative_g_dot_steppdir = -g.dot(stepdir)

        def loss(th):
            self.sff(th)
            # surrogate loss: policy gradient loss
            return self.session.run(self.losses[0], feed_dict)

        # finds best parameter by starting with a big step and working backwards
        theta = utils.linesearch(loss, thprev, fullstep, negative_g_dot_steppdir / lm)
        # i guess we just take a fullstep no matter what
        theta = thprev + fullstep
        self.sff(theta)

        surrogate_after, kl_after, entropy_after = self.session.run(self.losses, feed_dict)

        episoderewards = np.array(
            [path["rewards"].sum() for path in paths])
        stats = {}
        stats["Average sum of rewards per episode"] = episoderewards.mean()
        stats["Entropy"] = entropy_after
        stats["max KL"] = self.args.max_kl
        stats["Timesteps"] = sum([len(path["rewards"]) for path in paths])
        # stats["Time elapsed"] = "%.2f mins" % ((time.time() - start_time) / 60.0)
        stats["KL between old and new distribution"] = kl_after
        stats["Surrogate loss"] = surrogate_after
        # print(("\n********** Iteration {} ************".format(i)))
        for k, v in stats.items():
            print(k + ": " + " " * (40 - len(k)) + str(v))

        return stats["Average sum of rewards per episode"]
