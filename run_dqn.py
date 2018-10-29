import gym
from gym import wrappers
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import itertools
import random
from collections import namedtuple
import cv2
OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])


def _process_frame84(frame):
    img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
    img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
    resized_screen = cv2.resize(img, (84, 110),  interpolation=cv2.INTER_LINEAR)
    x_t = resized_screen[18:102, :]
    x_t = np.reshape(x_t, [84, 84, 1])
    return x_t.astype(np.uint8)

def initialize_interdependent_variables(session, vars_list, feed_dict):
    """Initialize a list of variables one at a time, which is useful if
    initialization of some variables depends on initialization of the others.
    """
    vars_left = vars_list
    while len(vars_left) > 0:
        new_vars_left = []
        for v in vars_left:
            try:
                # If using an older version of TensorFlow, uncomment the line
                # below and comment out the line after it.
		#session.run(tf.initialize_variables([v]), feed_dict)
                session.run(tf.variables_initializer([v]), feed_dict)
            except tf.errors.FailedPreconditionError:
                new_vars_left.append(v)
        if len(new_vars_left) >= len(vars_left):
            # This can happend if the variables all depend on each other, or more likely if there's
            # another variable outside of the list, that still needs to be initialized. This could be
            # detected here, but life's finite.
            raise Exception("Cycle in variable dependencies, or extenrnal precondition unsatisfied.")
        else:
            vars_left = new_vars_left

def sample_n_unique(sampling_f, n):
    """Helper function. Given a function `sampling_f` that returns
    comparable objects, sample n such unique objects.
    """
    res = []
    while len(res) < n:
        candidate = sampling_f()
        if candidate not in res:
            res.append(candidate)
    return res

class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p            = final_p
        self.initial_p          = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction  = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)

class ReplayBuffer(object):
    def __init__(self, size, frame_history_len):
        """This is a memory efficient implementation of the replay buffer.

        The sepecific memory optimizations use here are:
            - only store each frame once rather than k times
              even if every observation normally consists of k last frames
            - store frames as np.uint8 (actually it is most time-performance
              to cast them back to float32 on GPU to minimize memory transfer
              time)
            - store frame_t and frame_(t+1) in the same buffer.

        For the tipical use case in Atari Deep RL buffer with 1M frames the total
        memory footprint of this buffer is 10^6 * 84 * 84 bytes ~= 7 gigabytes

        Warning! Assumes that returning frame of zeros at the beginning
        of the episode, when there is less frames than `frame_history_len`,
        is acceptable.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        frame_history_len: int
            Number of memories to be retried for each observation.
        """
        self.size = size
        self.frame_history_len = frame_history_len

        self.next_idx      = 0
        self.num_in_buffer = 0

        self.obs      = None
        self.action   = None
        self.reward   = None
        self.done     = None

    def can_sample(self, batch_size):
        """Returns true if `batch_size` different transitions can be sampled from the buffer."""
        return batch_size + 1 <= self.num_in_buffer

    def _encode_sample(self, idxes):
        obs_batch      = np.concatenate([self._encode_observation(idx)[None] for idx in idxes], 0)
        act_batch      = self.action[idxes]
        rew_batch      = self.reward[idxes]
        next_obs_batch = np.concatenate([self._encode_observation(idx + 1)[None] for idx in idxes], 0)
        done_mask      = np.array([1.0 if self.done[idx] else 0.0 for idx in idxes], dtype=np.float32)

        return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask


    def sample(self, batch_size):
        """Sample `batch_size` different transitions.

        i-th sample transition is the following:

        when observing `obs_batch[i]`, action `act_batch[i]` was taken,
        after which reward `rew_batch[i]` was received and subsequent
        observation  next_obs_batch[i] was observed, unless the epsiode
        was done which is represented by `done_mask[i]` which is equal
        to 1 if episode has ended as a result of that action.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8
        act_batch: np.array
            Array of shape (batch_size,) and dtype np.int32
        rew_batch: np.array
            Array of shape (batch_size,) and dtype np.float32
        next_obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8
        done_mask: np.array
            Array of shape (batch_size,) and dtype np.float32
        """
        assert self.can_sample(batch_size)
        idxes = sample_n_unique(lambda: random.randint(0, self.num_in_buffer - 2), batch_size)
        return self._encode_sample(idxes)

    def encode_recent_observation(self):
        """Return the most recent `frame_history_len` frames.

        Returns
        -------
        observation: np.array
            Array of shape (img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8, where observation[:, :, i*img_c:(i+1)*img_c]
            encodes frame at time `t - frame_history_len + i`
        """
        assert self.num_in_buffer > 0
        return self._encode_observation((self.next_idx - 1) % self.size)

    def _encode_observation(self, idx):
        end_idx   = idx + 1 # make noninclusive
        start_idx = end_idx - self.frame_history_len
        # this checks if we are using low-dimensional observations, such as RAM
        # state, in which case we just directly return the latest RAM.
        if len(self.obs.shape) == 2:
            return self.obs[end_idx-1]
        # if there weren't enough frames ever in the buffer for context
        if start_idx < 0 and self.num_in_buffer != self.size:
            start_idx = 0
        for idx in range(start_idx, end_idx - 1):
            if self.done[idx % self.size]:
                start_idx = idx + 1
        missing_context = self.frame_history_len - (end_idx - start_idx)
        # if zero padding is needed for missing context
        # or we are on the boundry of the buffer
        if start_idx < 0 or missing_context > 0:
            frames = [np.zeros_like(self.obs[0]) for _ in range(missing_context)]
            for idx in range(start_idx, end_idx):
                frames.append(self.obs[idx % self.size])
            return np.concatenate(frames, 2)
        else:
            # this optimization has potential to saves about 30% compute time \o/
            img_h, img_w = self.obs.shape[1], self.obs.shape[2]
            return self.obs[start_idx:end_idx].transpose(1, 2, 0, 3).reshape(img_h, img_w, -1)

    def store_frame(self, frame):
        """Store a single frame in the buffer at the next available index, overwriting
        old frames if necessary.

        Parameters
        ----------
        frame: np.array
            Array of shape (img_h, img_w, img_c) and dtype np.uint8
            the frame to be stored

        Returns
        -------
        idx: int
            Index at which the frame is stored. To be used for `store_effect` later.
        """
        if self.obs is None:
            print("------")
            print(frame.shape)
            print([self.size])
            print(list(frame.shape))
            #print([self.size] + list(frame.shape))
            self.obs      = np.empty([self.size] + list(frame.shape), dtype=np.uint8)
            self.action   = np.empty([self.size],                     dtype=np.int32)
            self.reward   = np.empty([self.size],                     dtype=np.float32)
            self.done     = np.empty([self.size],                     dtype=np.bool)
        self.obs[self.next_idx] = frame

        ret = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

        return ret

    def store_effect(self, idx, action, reward, done):
        """Store effects of action taken after obeserving frame stored
        at index idx. The reason `store_frame` and `store_effect` is broken
        up into two functions is so that once can call `encode_recent_observation`
        in between.

        Paramters
        ---------
        idx: int
            Index in buffer of recently observed frame (returned by `store_frame`).
        action: int
            Action that was performed upon observing this frame.
        reward: float
            Reward that was received when the actions was performed.
        done: bool
            True if episode was finished after performing that action.
        """
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx]   = done

def linear_interpolation(l, r, alpha):
    return l + alpha * (r - l)

class PiecewiseSchedule(object):
    def __init__(self, endpoints, interpolation=linear_interpolation, outside_value=None):
        """Piecewise schedule.
        endpoints: [(int, int)]
            list of pairs `(time, value)` meanining that schedule should output
            `value` when `t==time`. All the values for time must be sorted in
            an increasing order. When t is between two times, e.g. `(time_a, value_a)`
            and `(time_b, value_b)`, such that `time_a <= t < time_b` then value outputs
            `interpolation(value_a, value_b, alpha)` where alpha is a fraction of
            time passed between `time_a` and `time_b` for time `t`.
        interpolation: lambda float, float, float: float
            a function that takes value to the left and to the right of t according
            to the `endpoints`. Alpha is the fraction of distance from left endpoint to
            right endpoint that t has covered. See linear_interpolation for example.
        outside_value: float
            if the value is requested outside of all the intervals sepecified in
            `endpoints` this value is returned. If None then AssertionError is
            raised when outside value is requested.
        """
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._interpolation = interpolation
        self._outside_value = outside_value
        self._endpoints      = endpoints

    def value(self, t):
        """See Schedule.value"""
        for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= t and t < r_t:
                alpha = float(t - l_t) / (r_t - l_t)
                return self._interpolation(l, r, alpha)

        # t does not belong to any of the pieces, so doom.
        assert self._outside_value is not None
        return self._outside_value


def minimize_and_clip(optimizer, objective, var_list, clip_val=10):
    """Minimized `objective` using `optimizer` w.r.t. variables in
    `var_list` while ensure the norm of the gradients for each
    variable is clipped to `clip_val`
    """
    gradients = optimizer.compute_gradients(objective, var_list=var_list)
    for i, (grad, var) in enumerate(gradients):
        if grad is not None:
            gradients[i] = (tf.clip_by_norm(grad, clip_val), var)
    return optimizer.apply_gradients(gradients)






def atari_model(img_in, num_actions, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope("convnet"):
            # 8*8 || stride = 4 || 32 filters
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            # 4*4 || stride = 2 || 64 filters
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            # 3*3 || stride = 1 || 64 filters
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)

        out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            out = layers.fully_connected(out, num_outputs=512,            activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_actions,    activation_fn=None)

        return out



def main():
    # Get Atari game environment.
    env = gym.make('Breakout-v0')

    # Get session
    session = tf.Session()


    # get the dimension of observation and action space
    print("image resolution:", env.observation_space.shape)
    print(env.action_space.n)

    # discount factor
    discount_factor = 0.99
    grad_norm_clipping = 10   #
    replay_buffer_size = 1000000   #
    frame_history_len = 4
    max_iteration_step = 10000000  #
    num_iterations = max_iteration_step  #
    exploration = LinearSchedule(1000000, 0.1)  #
    learning_starts = 50000    #
    learning_freq = 4     #
    num_actions = env.action_space.n
    batch_size = 32
    target_update_freq = 10000

    img_h, img_w, img_c = env.observation_space.shape
    input_shape = (img_h, img_w, frame_history_len * img_c)
    print(input_shape)




    # set up placeholders
    # placeholder for current observation (or state)
    observation_t_placeholder = tf.placeholder(tf.uint8, [None, 84, 84, 4])
    # placeholder for current action
    action_t_placeholder = tf.placeholder(tf.int32, [None])
    # placeholder for t + 1 reward
    reward_tp1_placeholder = tf.placeholder(tf.float32, [None])
    # placeholder for t + 1 observation
    observation_tp1_placeholder = tf.placeholder(tf.float32, [None, 84, 84, 4])

    # placeholder to indicate end of the episode
    done_indicator_placeholder = tf.placeholder(tf.float32, [None])

    # cast to float, which can lower data transfer time
    observation_t_float   = tf.cast(observation_t_placeholder,   tf.float32) / 255.0
    observation_tp1_float = tf.cast(observation_tp1_placeholder, tf.float32) / 255.0


    q = atari_model(observation_t_float, num_actions, scope="q_func", reuse=False)
    target_q = atari_model(observation_tp1_float, num_actions, scope="target_q_func", reuse=False)
    Q_samp = reward_tp1_placeholder + (1 - done_indicator_placeholder) * discount_factor * tf.reduce_max(target_q, axis=1)
    Q_s = tf.reduce_sum(q * tf.one_hot(action_t_placeholder, num_actions), axis=1)
    total_error = tf.reduce_mean(tf.square(Q_samp - Q_s))
    q_function_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')
    target_q_function_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_q_func')



    # constract optimization op (with gradient clipping)
    learning_rate = tf.placeholder(tf.float32, (), name="learning_rate")
    lr_multiplier = 1.0
    lr_schedule = PiecewiseSchedule([
        (0, 1e-4 * lr_multiplier),
        (num_iterations / 10, 1e-4 * lr_multiplier),
        (num_iterations / 2, 5e-5 * lr_multiplier),
    ],
        outside_value=5e-5 * lr_multiplier)

    optimizer_spec = OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        kwargs=dict(epsilon=1e-4),
        lr_schedule=lr_schedule
    )
    optimizer = optimizer_spec.constructor(learning_rate=learning_rate, **optimizer_spec.kwargs)
    train_fn = minimize_and_clip(optimizer, total_error,
                                 var_list=q_function_variables, clip_val=grad_norm_clipping)


    # copy Q network to target Q network
    update_target_fn = []
    for var, var_target in zip(sorted(q_function_variables,        key=lambda v: v.name),
                               sorted(target_q_function_variables, key=lambda v:v.name)):
        update_target_fn.append(var_target.assign(var))
    update_target_fn = tf.group(*update_target_fn)





    # construct the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)



    ####################
    ### RUN ENVIRONMENT
    ####################
    model_initialized = False
    num_param_updates = 0
    mean_episode_reward      = -float('nan')
    best_mean_episode_reward = -float('inf')
    reward_list = []
    last_obs = _process_frame84(env.reset())


    for t in itertools.count():
        if t > max_iteration_step:
            break

        # replay memory stuff
        idx = replay_buffer.store_frame(last_obs)
        q_input = replay_buffer.encode_recent_observation()

        if (np.random.random() < exploration.value(t)) or not model_initialized:
            action = env.action_space.sample()
        else:
            action_values = session.run(q, feed_dict={observation_t_placeholder: [q_input]})[0]
            action = np.argmax(action_values)

        # perform action in env
        new_state, reward, done, info = env.step(action)
        new_state = _process_frame84(new_state)
        reward_list.append(reward)
        env.render()

        # store the transition
        replay_buffer.store_effect(idx, action, reward, done)
        last_obs = new_state

        if done:
            last_obs = _process_frame84(env.reset())
            print("total rewrad = %f ; ave reward = %f" %(np.sum(reward_list), np.mean(reward_list)))
            reward_list = []


        if (t > learning_starts and t % learning_freq == 0 and replay_buffer.can_sample(batch_size)):
            s_batch, a_batch, r_batch, sp_batch, done_mask_batch = replay_buffer.sample(batch_size)

            if not model_initialized:
                initialize_interdependent_variables(session, tf.global_variables(),
                                                    {observation_t_placeholder: s_batch, observation_tp1_placeholder: sp_batch, })
                model_initialized = True

            feed_dict = {observation_t_placeholder:      s_batch,
                         action_t_placeholder:           a_batch,
                         reward_tp1_placeholder:         r_batch,
                         observation_tp1_placeholder:    sp_batch,
                         done_indicator_placeholder:     done_mask_batch,
                         learning_rate:                  lr_schedule.value(t)}
            session.run(train_fn, feed_dict=feed_dict)

            num_param_updates += 1
            if num_param_updates % target_update_freq == 0:
                session.run(update_target_fn)
                num_param_updates = 0



def test():
    env = gym.make('Breakout-v0')
    env.reset()
    for i in range(10000):
        env.render()
        action = env.action_space.sample()
        _, _, done, _ = env.step(action)
        if done:
            env.reset()


if __name__ == "__main__":
    main()
    #test()


