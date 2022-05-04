#!/usr/bin/env python3

import gym
import gym.wrappers

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import time
import os
import sys

def main(argv):
    model_filename = argv[1]
    if os.path.exists(model_filename):
        model = keras.models.load_model(model_filename)
    else:
        print("{} does not exist.".format(model_filename))
        sys.exit(1)

    n_epoch = 1
    n_stack = 4
    env = gym.make('MountainCar-v0')
    # env = gym.wrappers.AtariPreprocessing(env,terminal_on_life_loss=True,scale_obs=False)
    env = gym.wrappers.FrameStack(env,n_stack)
    
    for i in range(n_epoch):
        state = env.reset()
        state = np.asarray(state)
        state = state.reshape((1,)+state.shape+(1,))

        done = False
        total_reward = 0
        count = 0
        while not done:
            env.render()
            action = np.argmax(model.predict(state))
            
            state1, current_reward, done, info = env.step(action)
            state1 = np.asarray(state1)
            state1 = state1.reshape((1,)+state1.shape+(1,))
            total_reward += current_reward
            state = state1
            count += 1
        print( "reward:", total_reward, "count:", count )
        env.render()
    env.close( )

    return

if __name__ == "__main__":
    main(sys.argv)