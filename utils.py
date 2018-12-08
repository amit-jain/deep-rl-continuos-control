import torch
from agent import Agent
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import spline

def plot_scores(scores):
    # plot the scores
    fig = plt.figure(figsize=(20, 10))

    x = np.arange(len(scores))
    y = scores

    plt.plot(x, y)

    x_sm = np.array(x)
    y_sm = np.array(y)

    x_smooth = np.linspace(x_sm.min(), x_sm.max(), 20)
    y_smooth = spline(x, y, x_smooth)
    plt.plot(x_smooth, y_smooth, 'orange', linewidth=4)

    plt.ylabel('Score')
    plt.xlabel('Episode #')

    plt.ylim(ymin=0)

    plt.show()

def create_agent(env, brain_name, device, actor_file=None, critic_file=None, random_seed=39, fc1=128, fc2=128,
                 lr_actor=1e-04, lr_critic=1e-04, weight_decay=0, buffer_size=100000,
                 batch_size=64, gamma=0.99, tau=1e-3):
    brain = env.brains[brain_name]

    # size of each action
    action_size = brain.vector_action_space_size
    state_size = brain.vector_observation_space_size

    agent = Agent(device, state_size=state_size, action_size=action_size, random_seed=random_seed, fc1=fc1, fc2=fc2,
                  lr_actor=lr_actor, lr_critic=lr_critic, weight_decay=weight_decay, buffer_size=buffer_size,
                  batch_size=batch_size, gamma=gamma, tau=tau)

    if actor_file and critic_file:
        agent.actor_local.load_state_dict(torch.load(actor_file))
        agent.critic_local.load_state_dict(torch.load(critic_file))

    return agent
