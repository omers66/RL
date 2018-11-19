import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from channel_envs import Trapdoor
from replay_buffer import ReplayBuffer
import numpy as np
import gym

episodeNum = 10000
episodeLength = 1000
batchSize = 64
gamma = 0.99
bufferSize = 1000000
rndSeed = 123
tau = 0.001
critic_lr = 0.001
actor_lr = 0.0001

env = gym.make('Pendulum-v0')
stateDim = env.observation_space.shape[0]
actionDim = env.action_space.shape[0]
action_bound = env.action_space.high        # assuming symetric action bound


class ActorNN(nn.Module):

    def __init__(self):
        super(ActorNN, self).__init__()
        self.fc1 = nn.Linear(stateDim, 400)
        #self.fc1_bn = nn.BatchNorm1d(400)
        self.fc2 = nn.Linear(400, 300)
        #self.fc2_bn = nn.BatchNorm1d(300)
        self.fc3 = nn.Linear(300, actionDim)
        self.fc3.weight.data.uniform_(-0.003, 0.003)
        self.action_bound = Variable(torch.from_numpy(action_bound)).cuda().double()

    def forward(self, s):
        x = self.fc1(s)
        #x = self.fc1_bn(x)
        x = F.relu(x)
        x = self.fc2(x)
        #x = self.fc2_bn(x)
        x = F.relu(x)
        x = F.tanh(self.fc3(x))
        #b = s.data.numpy()
        #b = Variable(torch.DoubleTensor([1 - b, b]).transpose(0, 1))
        #t = Variable(torch.DoubleTensor(np.array([1-b, b]).transpose()))
        #x = x*t
        return x*self.action_bound


class CriticNN(nn.Module):

    def __init__(self):
        super(CriticNN, self).__init__()
        self.fc1 = nn.Linear(stateDim, 400)
        self.fc1BN = nn.BatchNorm1d(400)
        self.fc2s = nn.Linear(400, 300)
        self.fc2a = nn.Linear(actionDim, 300)
        self.fc2BN = nn.BatchNorm1d(300)
        self.fc3 = nn.Linear(300, 1)
        self.fc3.weight.data.uniform_(-0.003, 0.003)

    def forward(self, s, a):
        x = self.fc1(s)
        x = self.fc1BN(x)
        x = F.relu(x)
        x = self.fc2s(x) + self.fc2a(a)
        x = self.fc2BN(x)
        x = F.relu(x)     # combine actions here
        x = self.fc3(x)
        return x

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

# Soft update parameters, drifting by tau
def update_target(target_model, model):
    for target_param, param in zip(target_model.parameters(), model.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

# Function to identically copy parameters from one network model to another
def init_target(target_model, model):
    for target_param, param in zip(target_model.parameters(), model.parameters()):
        target_param.data.copy_(param.data)


# initialize networks
actor = ActorNN().cuda().double()
actorTarget = ActorNN().cuda().double()
critic = CriticNN().cuda().double()
criticTarget = CriticNN().cuda().double()


# initialize target networks with same weights as in "original"
init_target(actorTarget, actor)
init_target(criticTarget, critic)

replay_buffer = ReplayBuffer(bufferSize, rndSeed)
actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(actionDim))

critic_optimizer = optim.Adam(lr=critic_lr, params=critic.parameters())
actor_optimizer = optim.Adam(lr=actor_lr, params=actor.parameters())

for i in range(1, episodeNum):
    s = env.reset()
    episode_reward = 0
    for j in range(1, episodeLength):
        if (i % 50 == 0):
            env.render()
        a = actor.forward(Variable(torch.from_numpy(s)).cuda().double())
        noise = actor_noise()
        s2, r, _, _ = env.step(a.cpu().data.numpy()+noise)    # add exploration noise to action
        episode_reward += r
        replay_buffer.add(s, a.cpu().data.numpy(), r, False, s2)    # maybe add fix to a+noise would enter the buffer and not just a
        s = s2
        if (replay_buffer.count >= 64):
            s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(batchSize)

            #----------------- Get y = r + Q(s',a') using target network! ---------------------
            s2_batch = Variable(torch.from_numpy(s2_batch)).cuda().double()
            a_tar = actorTarget.forward(s2_batch)
            Q_tar = criticTarget.forward(s2_batch, a_tar)
            y = torch.from_numpy(np.reshape(r_batch, [r_batch.shape[0], 1]) + gamma*Q_tar.cpu().data.numpy()).cuda().double()

            # ----------------- Get Q(s,a) -----------------------------------------------
            s_batch = Variable(torch.from_numpy(s_batch)).cuda().double()
            a_batch = Variable(torch.from_numpy(a_batch)).cuda().double()
            Q = critic.forward(s_batch, a_batch)

            # ----------------- Update critic parameters by minimizing ||Yi-Q(s,a)||^2 -------------------
            critic_optimizer.zero_grad()
            critic_loss = nn.MSELoss()
            loss = critic_loss(Q, Variable(y))
            loss.backward()
            critic_optimizer.step()

            # ------------------ Update actor ------------------------------------
            '''a_batch.requires_grad = True
            a_batch.grad = torch.zeros_like(a_batch)
            Q = critic.forward(s_batch, a_batch)
            # create a ones tensor of size batchSize, because the Q output is of batchSize size, and we want gradient
            #  for each output w.r.t inputs
            ones_tensor = torch.DoubleTensor(np.ones([batchSize, 1]))
            # get derivatives from critic w.r.t action, this give a vector of size [batchSize,actionDim], row n is:
            # [dq(n)da(1), dq(n)da(2)]
            dqda = torch.autograd.grad(Q, a_batch, retain_graph=True,
                                       grad_outputs=ones_tensor)
            actor.zero_grad()
            actor_out = actor.forward(s_batch)  # pass s_batch with actor so we can take gradient
            # calc actor gradient w.r.t nn parameters, we use ones_tensor again, and this sums the gradient of all 64
            # output, because actor_out is of a batch of 64, then this sums the gradient of all 64
            actor_out.backward(ones_tensor)

            # debug this shit -  without the [0,0] it works
            #Q = critic.forward(s_batch[0], a_batch[0, 0])
            #dqda = torch.autograd.grad(Q, a_batch[0,0], retain_graph=True)
            '''
            actor_optimizer.zero_grad()
            a = actor.forward(s_batch)
            actor_loss = -torch.mean(critic.forward(s_batch, a)) # minus sign becuase optimizers by defualt preform (w <- w -alpha dj/dw)
            actor_loss.backward()
            actor_optimizer.step()

            # ------------------ Soft update target networks --------------------
            update_target(actorTarget, actor)
            update_target(criticTarget, critic)


    ep_avg_reward = episode_reward
    print('episode ' , i, ' avg reward:' , ep_avg_reward)

    if (i % 50 == 0):
        s = env.reset()
        episode_reward = 0
        for j in range(1, episodeLength):
            a = actorTarget.forward(Variable(torch.from_numpy(s)).cuda().double())
            s2, r, _, _ = env.step(a.cpu().data.numpy())  # add exploration noise to action
            episode_reward += r
            s = s2
        print('target NN episode ', i, ' avg reward:', episode_reward)

















