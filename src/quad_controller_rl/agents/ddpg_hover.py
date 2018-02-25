"""Policy search agent."""

import numpy as np
from quad_controller_rl.agents.base_agent import BaseAgent
from quad_controller_rl.agents.noise import OUNoise
from quad_controller_rl.agents.actor import Actor
from quad_controller_rl.agents.critic import Critic
from quad_controller_rl.agents.replay import ReplayBuffer
from quad_controller_rl import util
import pandas as pd
import os

############
# import tensorflow as tf
# from keras import backend as K
# from ActorNetwork import ActorNetwork
# from CriticNetwork import CriticNetworks
############

class DDPG_H(BaseAgent):
    """Sample agent that searches for optimal policy randomly."""

    def __init__(self, task):
        # Task (environment) information
        self.task = task  # should contain observation_space and action_space
        self.load_weights = False
        ##############OVERRRIDE###############################
        self.state_size = 6
        self.action_size = 3
        # self.state_size = self.task.observation_space.shape[0]
        # self.action_size = self.task.action_space.shape[0]
        ######################################################


        # Weights Saver
        self.model_ext = ".h5"
        self.save_weights_every = 100
        if self.load_weights:
            self.actor_filename = os.path.join(util.get_param('out'),
                "model_hover_2018-02-20_18-11-35_actor{}".format(self.model_ext))
            self.critic_filename = os.path.join(util.get_param('out'),
                "model_hover_2018-02-20_18-11-35_critic{}".format(self.model_ext))
            print("Actor filename :", self.actor_filename)  # [debug]
            print("Critic filename:", self.critic_filename)  # [debug]
        elif self.save_weights_every:
            self.actor_filename = os.path.join(util.get_param('out'),
                "model_{}_actor{}".format(util.get_timestamp(), self.model_ext))
            self.critic_filename = os.path.join(util.get_param('out'),
                "model_{}_critic{}".format(util.get_timestamp(), self.model_ext))
            print("Actor filename :", self.actor_filename)  # [debug]
            print("Critic filename:", self.critic_filename)  # [debug]


        print('State Size : {}, Action Size : {}'.format(self.state_size, self.action_size))
        self.action_high = self.task.action_space.high[0:self.action_size]
        self.action_low = self.task.action_space.low[0:self.action_size]
        print('Action LOW : {}, Action HIGH : {}'.format(self.action_low, self.action_high))
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high, 0.001)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high, 0.001)

        self.critic_local = Critic(self.state_size, self.action_size, 0.001)
        self.critic_target = Critic(self.state_size, self.action_size, 0.001)


        # Load pre-trained model weights, if available
        if self.load_weights and os.path.isfile(self.actor_filename):
            try:
                self.actor_local.model.load_weights(self.actor_filename)
                self.critic_local.model.load_weights(self.critic_filename)
                print("Model weights loaded from file!")  # [debug]
            except Exception as e:
                print("Unable to load model weights from file!")
                print("{}: {}".format(e.__class__.__name__, str(e)))

        # Policy parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        self.buffer_size = 10000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size)

        # Algorithm parameters
        self.gamma = 0.99  # discount factor
        self.tau = 0.001  # for soft update of target parameters

        self.noise = OUNoise(self.action_size)
        # Stats Writer
        self.stats_filename = os.path.join(
            util.get_param('out'),
            "stats_{}.csv".format(util.get_timestamp()))  # path to CSV file
        self.stats_columns = ['episode', 'avg_reward']  # specify columns to save

        print("Saving stats {} to {}".format(self.stats_columns, self.stats_filename))  # [debug]


        self.episode_num = 0
        self.reset_episode_vars()



    def run(self):
        np.random.seed(1337)
        max_episode = 400
        max_step = 1000
        max_explore_eps = 100.0
        env = self.task
        self.noise = OUNoise(self.action_size)
        while(self.episode_num < max_episode):

            state = env.reset().reshape(self.state_size)
            action = self.act(state)
            while(self.count < max_step):
                env.render()
                next_state, reward, done, info = env.step(action)
                next_state = next_state.reshape(self.state_size)
                reward = reward
                action = self.step(state, reward, done)
                state = next_state
                if done:
                    break


    def reset_episode_vars(self):
        self.last_state = None
        self.last_action = None
        self.total_reward = 0.0
        self.count = 0
        self.actor_loss = 0.0
        self.critic_loss = 0.0
        self.episode_num += 1

    def step(self, state, reward, done):
        self.count += 1
        # Save exp
        if(self.last_state is not None):
            self.memory.add(self.last_state, self.last_action, reward, state, done)
            if(self.batch_size < self.memory.length()):
                # Learn exp
                experinces = self.memory.sample(self.batch_size)
                self.learn(experinces)
                self.total_reward += reward

        self.last_state = state
        if(done):
            self.actor_loss /= float(self.count)
            self.critic_loss /= float(self.count)
            print('Loss, Actor {:4f}, Critic {:4f}'.format(self.actor_loss, self.critic_loss))
            print('Episode {}, Score {:8f}, Steps {}, Normalize Score {:4f}'.format(self.episode_num, self.total_reward, self.count, self.total_reward/float(self.count)))
            # Write episode stats
            self.write_stats([self.episode_num, (self.total_reward/float(self.count))])
            # Save model weights at regular intervals
            if(self.episode_num % self.save_weights_every == 0):
                self.actor_local.model.save_weights(self.actor_filename)
                self.critic_local.model.save_weights(self.critic_filename)
                print("Model weights saved at episode", self.episode_num)  # [debug]
            self.reset_episode_vars()
            return

        action = self.act(state)
        self.last_action = action
        action = self.postprocess_action(action)
        return action

    def act(self, states):
        states = states.reshape(1, self.state_size)
        action = self.actor_local.model.predict(states)[0]
        # action = self.actor.model.predict(states)
        return (action + self.noise.sample())

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.array([e.state for e in experiences])
        actions = np.array([e.action for e in experiences])
        rewards = np.array([e.reward for e in experiences])
        dones = np.array([e.done for e in experiences])
        next_states = np.array([e.next_state for e in experiences])

        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = (rewards + self.gamma * Q_targets_next.reshape(len(experiences)) * (1 - dones))
        self.critic_loss += self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)
        actions_for_grads = self.actor_local.model.predict(states)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions_for_grads, 0]), (-1, self.action_size))
        self.actor_loss += self.actor_local.train_fn([states, action_gradients, 1])[0]  # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())
        new_weights = self.tau * local_weights + ((1 - self.tau) * target_weights)
        target_model.set_weights(new_weights)

    def write_stats(self, stats):
        """Write single episode stats to CSV file."""
        df_stats = pd.DataFrame([stats], columns=self.stats_columns)  # single-row dataframe
        df_stats.to_csv(self.stats_filename, mode='a', index=False,
            header=not os.path.isfile(self.stats_filename))  # write header first time only


    def preprocess_state(self, state):
        """Reduce state vector to relevant dimensions."""
        return state[0:3]  # position only

    def postprocess_action(self, action):
        """Return complete action vector."""
        complete_action = np.zeros(self.task.action_space.shape)  # shape: (6,)
        complete_action[0:3] = action  # linear force only
        return complete_action
