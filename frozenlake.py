#q-learning, Frozen Lake
#2022.01.05 AM
#based on tutorial deeplizard.com

import numpy as np
import gym
import random
import time
#from IPython.display import clear_output ipython-notebook

env = gym.make("FrozenLake-v1")

action_space_size = env.action_space.n
state_space_size = env.observation_space.n

q_table = np.zeros((state_space_size, action_space_size))

num_episodes = 10000
max_steps_per_episode = 100

learning_rate = 0.1 #alpha
discount_rate = 0.99 #gamma

exploration_rate = 1 #epislon, initialize to 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

print(q_table)
rewards_all_episodes = []
# Q-learning algorithm
for episode in range(num_episodes):
    # initialize new episode params
    state = env.reset()
    done = False #initialize to False, keeps track if episode is done
    rewards_current_episode = 0 #initialize to 0 for the episode

    for step in range(max_steps_per_episode):
        # Exploration-exploitation trade-off
        #epsilon greedy algorithm if random num is greater than
        #epsilon then exploit else explore state-action space
        #exploration rate is epsilon
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            #exploit
            action = np.argmax(q_table[state,:])
        else:
            #explore
            action = env.action_space.sample()

        # Take new action
        #take the chosen action (explore or exploit from above)
        new_state, reward, done, info = env.step(action)

        # Update Q-table for Q(s,a)
        #new q value = (1-alpha)*old q value + alpha*(new reward plus discount)
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

        # Set new state
        state = new_state #update state

        # Add new reward
        #update reward counter for this episode
        rewards_current_episode += reward

        if done == True:
            break


    # Exploration rate decay
    exploration_rate = min_exploration_rate + \
    (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)

    # Add current episode reward to total rewards list
    rewards_all_episodes.append(rewards_current_episode)

# Calculate and print the average reward per thousand episodes
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes),num_episodes/1000)
count = 1000

print("********Average reward per thousand episodes********\n")
for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r/1000)))
    count += 1000


# Watch our agent play Frozen Lake by playing the best action
# from each state according to the Q-table
for episode in range(1):
    # initialize new episode params
    state = env.reset()
    done = False
    print("*****EPISODE ", episode+1, "*****\n\n\n\n")
    time.sleep(1)

    for step in range(max_steps_per_episode):
        #clear_output(wait=True) #ipython

        # Show current state of environment on screen
        env.render()
        time.sleep(1)

        # Choose action with highest Q-value for current state
        action = np.argmax(q_table[state,:])

        # Take new action
        new_state, reward, done, info = env.step(action)

        if done:
            #clear_output(wait=True) #ipython
            env.render()
            if reward == 1:
                # Agent reached the goal and won episode
                print("****You reached the goal!****")
                time.sleep(3)
            else:
                # Agent stepped in a hole and lost episode
                print("****You fell through a hole!****")
                time.sleep(3)
                #clear_output(wait=True) #ipython
            break

        # Set new state
        state = new_state

#close after all episodes watched
env.close()
