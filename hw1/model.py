import os
import pickle
import tf_util
import time

import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras as k

ENVS = [
	'Ant-v2',
	'HalfCheetah-v2',
	'Hopper-v2',
	'Humanoid-v2',
	'Reacher-v2',
	'Walker2d-v2'
]


def get_model(input_shape, output_len):
    def _normalize(x):
        mean = k.backend.mean(x)
        std = k.backend.sqrt(k.backend.var(x))
        return (x - mean) / (std + 1e6)

    model = k.Sequential()
    model.add(k.layers.Lambda(_normalize, input_shape=input_shape))
    model.add(k.layers.Dense(64, activation='tanh'))
    model.add(k.layers.Dense(64, activation='tanh'))
    model.add(k.layers.Dense(output_len))
    return model


def get_data(envname):
    with open(os.path.join('expert_data', envname + '.pkl'), 'rb') as f:
        expert_data = pickle.load(f)
    return expert_data


def train(envname, data, labels, batch_size, epochs):
    env = gym.make(envname)
    output_len = env.action_space.shape[0]
    input_shape = env.observation_space.shape
    model = get_model(input_shape, output_len)
    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='mse',       # mean squared error
              metrics=['mse'])
    model.fit(data, labels, batch_size, epochs)
    model.save_weights('{}{}.h5'.format(envname, time.time()))
    print(model)


def test(envname):
    with tf.Session():
        tf_util.initialize()

        env = gym.make(envname)
        max_steps = env.spec.timestep_limit

        output_len = env.action_space.shape[0]
        input_shape = env.observation_space.shape
        model = get_model(input_shape, output_len)
        model.load_weights('./Ant-v21541609309.82227.h5')
        # model = k.models.load_model()

        returns = []
        observations = []
        actions = []
        for i in range(20):# range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = model.predict(np.expand_dims(obs, axis=0))
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if True:# args.render:
                    env.render(mode='human')
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}


def main(envname):
    raw = get_data(envname)
    data = raw['observations']
    labels = np.squeeze(raw['actions'])
    print(data.shape)
    print(labels.shape)
    # train(envname, data, labels, batch_size=32, epochs=1000)
    test(envname)


if __name__ == '__main__':
    main('Ant-v2')
