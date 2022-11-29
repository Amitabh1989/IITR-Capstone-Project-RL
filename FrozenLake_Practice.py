#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from gym.envs.registration import register

try:
    register(
        id='Amitabh-FrozenLakeNotSlippery-v0',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name' : '4x4', 'is_slippery': False},
        max_episode_steps= 100,
        reward_threshold= 0.78
    )
except:
    print("Environment already created!")


# In[3]:


env = gym.make("Amitabh-FrozenLakeNotSlippery-v0", render_mode="rgb_array")


# In[4]:


env.action_space


# In[5]:


env.action_space.n


# In[6]:


env.observation_space


# In[7]:


state = env.reset()


# In[8]:


q_table = np.zeros([env.observation_space.n, env.action_space.n])


# In[9]:


q_table.shape


# In[10]:


q_table


# In[11]:


state


# In[12]:


# Below function is a very raw, novice way to randomly select any action and perform on the environment. It has nothing to do with 
# machine learning or any such thing. 
for episode in range(5):
    done = False
    state = env.reset()
    
    while not done:
        env.render()
        action = env.action_space.sample()
        state, reward, done, truncate, info = env.step(action)
        
env.close()


# In[13]:


# Now lets implement it via code to see how machine learns to play this game and how can we teach the machine to do it
# Starting with the Q learning


# In[14]:


# Things we need
# 1. EPSILON GREEDY METHOD
# 2. Function to compute optimal q value
# 3. Few necessary variables : EPSILON, discount factor GAMMA, lerning rate ALPHA and other for loop control and Epsilon decay


# In[15]:


# PARAMETERS:
NUM_EPISODES = 20000
ALPHA = 0.01
GAMMA = 0.99
EPSILON = 1.0
MIN_EPSILON = 0.0
MAX_EPSILON = 1.0
EPSILON_DECAY = 0.001


# In[16]:


def epsilon_greedy_action(q_table, state):
    random_val = np.random.random()
    if random_val > EPSILON:
        action = np.argmax(q_table[state, :]) # argmax gets me the action
    
    else:
        action = env.action_space.sample()
    return action


# In[17]:


def compute_next_q_val(old_q_val, reward, next_optimal_q_val):
    return (old_q_val + ALPHA * (reward + (GAMMA * next_optimal_q_val) - old_q_val))


# In[18]:


def reduce_epsilon(epsilon, episode):
    return (MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON)*np.exp(-EPSILON_DECAY*episode))


# In[20]:


log_interval = 1000
rewards = []
for episode in range(NUM_EPISODES):
    done = False
    state = env.reset()
    state = state[0] if isinstance(state, tuple) else state
    total_rewards = 0
   
    
    while not done:
        env.render()
        
        # Choose an action and perform
        action = epsilon_greedy_action(q_table, state)
        next_state, reward, done, truncate, info = env.step(action)
        
        # Get old q val
        old_q_val = q_table[state, action]
        
        # Get next optimal q val
        next_optimal_q_val = np.max(q_table[next_state, :]) # whole row as we dont know what action is to be take.
        
        # Compute next q val
        new_q_val = compute_next_q_val(old_q_val, reward, next_optimal_q_val)
        
        # Update the q table
        q_table[state, action] = new_q_val
        
        
        # Update current state
        state = next_state
        
        # accumulate total reward
        total_rewards += reward

    # Decay EPSILON
    episode += episode
    EPSILON = reduce_epsilon(EPSILON, episode)
    rewards.append(total_rewards)
    
    if episode % log_interval == 0:
        print("EPISODE : ", episode, "  Reward : ", np.sum(rewards))
env.close()


# In[21]:


q_table


# In[22]:


env_human = gym.make("Amitabh-FrozenLakeNotSlippery-v0", render_mode="human")


# In[ ]:


import time
i = 1
for episode in range(200):
    done = False
    state = env_human.reset()
    state = state[0] if isinstance(state, tuple) else state
    while not done:
        env_human.render()
        action = np.argmax(q_table[state, :])
        state, reward, done, truncate, info = env_human.step(action)
        if i == 1:
            time.sleep(10)
            i += 1
        
        time.sleep(0.5)
        if done:
            print("Woohoo, you won!")
            break
env.close()


# In[ ]:


env.close()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




