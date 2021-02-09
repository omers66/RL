import gym
import numpy as np
import math
np.random.seed(5)

def sig(w, x):
    return 1/(1+math.exp(-np.dot(w, x)))

env = gym.make('CartPole-v0')
THETA = np.random.normal(0, 1, 4)   # initial sigmoid weights
observation_start = env.reset()
done_array = []

for e in range(5000):       # mini batch
    observation = observation_start
    obs_hist = []
    action_hist = []
    reward_hist = []
    obs_hist.append(observation)
    for t in range(10000):              # run over time steps of episode
        env.render()
        p = sig(THETA, observation)     # calc p(a=1|s) - sigmoid using current weights and observation
        action = np.random.binomial(1, p)    # take random action according to p from sigmoid
        observation, reward, done, info = env.step(action)
        obs_hist.append(observation)
        action_hist.append(action)
        reward_hist.append(reward)
        if done:    # if episode is over
            grad = 0
            print("Episode finished after {} timesteps".format(t + 1))
            if t+1 >= 200:
                done_array.append(True)
                # check if all previous 10 games reached 200 then its great and stop updating
                if np.all(done_array[len(done_array)-3:len(done_array)]):
                    print('policy converged in {} episodes'.format(e))
                    break
            else:
                done_array.append(False)
            for i in range(0, len(obs_hist)-1):
                obs = obs_hist[i]
                action = action_hist[i]
                Gt = sum(reward_hist[i:len(reward_hist)])
                if action == 1:    # if Right
                    grad = grad + Gt*(1-sig(THETA, obs))*obs
                else:
                    grad = grad - Gt*sig(THETA, obs)*obs
            THETA = THETA + 0.1 / (1 + e) * grad
            del obs_hist
            del action_hist
            del reward_hist
            env.reset()
            #env.env.state = observation_start
            break


print('done playing')
observation = env.reset()
for t in range(10000):
    env.render()
    # print(observation)
    action = env.action_space.sample()
    p = sig(W, observation)  # calc sigmoid using current weights and observation
    action = np.random.binomial(1, p)  # take random action according to p from sigmoid
    observation_help = observation
    observation, reward, done, info = env.step(action)
    #hist.append((observation_help, action, reward))
    if done:
        print("Episode finished after {} timesteps".format(t + 1))
        break