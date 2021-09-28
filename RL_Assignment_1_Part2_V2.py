import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import random

class environment_q_learning:
    def __init__(self,type_of_env:str,gamma_disc: float, epsilon:float,epsilon_decay:float,learning_rate:float,no_of_episodes:int,max_time_steps:int):
        
        self.chain_of_states = [[0,0]]
        self.chain_of_actions = [0]
        if max_time_steps < 10:
            raise ValueError("Timesteps should be greater than of equal to 10")
        
        if (epsilon_decay > 1) or (epsilon_decay < 0):
            raise ValueError("Epsilon decay should be less than 1 and greater than 0")
        
        if no_of_episodes < 1:
            raise ValueError("No of Episodes should be atleast equal to 1")

        # No of number of states = 25 - Requirement 1
        self.environment = np.zeros((5,5))

        # No of actions an agent can take:
        self.action_set_size = 4

        self.gamma = gamma_disc
        # Q Value Learning Table
        # self. = np.zeros((len(self.environment.flatten())),self.action_set_size)
        self.qvalue_table = {}
        for i1 in range(5):
            for i2 in range(5):
                for i in np.zeros((25,4)):
                    self.qvalue_table[(i1,i2)] = i
        
        self.max_timesteps = max_time_steps
        self.current_time_steps = 0
        
        # This determines the exploitation vs exploration phase.
        self.epsilon = epsilon

        # this determines the reduction in epsilon
        self.epsilon_decay = epsilon_decay

        # this determines how quickly the q values for a state are updated
        self.learning_rate = learning_rate

        # this tells us the no_of_epsiodes during which we will determine the optimal q value
        self.no_of_episodes = no_of_episodes
        
        self.goal_pos = [4,4]
        self.agent_current_pos = [0,0]

        self.environment[tuple(self.agent_current_pos)] = 1
        self.environment[tuple(self.goal_pos)] = 0.5

        # Collection of Rewards (the keys) and associated values (the states). -> Total No of Rewards = 4 -> Requirement 3
        self.rewards = [{-1:[0,3]},{-1:[3,0]},{3:[2,3]},{3:[3,2]}]
        self.reward_states = [(0,3),(3,0),(2,3),(3,2)]
        # Setting the colors for the reward states in the environment.
        for reward_state in self.rewards:
            for reward, position in reward_state.items():
                self.environment[tuple(position)] = reward

        # Either Deterministic or stochastic.
        self.environment_type = type_of_env

        # This tracks the reward for the agent.
        self.cumulative_reward = 0

    def reset(self):
        # Here we are essentially resetting all the values.
        self.current_time_steps = 0
        self.cumulative_reward = 0

        self.goal_pos = [4,4]
        self.agent_current_pos = [0,0]
        
        self.environment = np.zeros((5,5))
        
        self.rewards = [{-1:[0,3]},{-1:[3,0]},{3:[2,3]},{3:[3,2]},{10:[4,4]}]
        self.reward_states = [(0,3),(3,0),(2,3),(3,2),[4,4]]
        
        for reward in self.rewards:
            for reward, position in reward.items():
                self.environment[tuple(position)] = reward

        self.environment[self.agent_current_pos] = 1
        self.environment[self.goal_pos] = 0.5
    
    def step(self):
        print("Steps starting agent pos is", self.agent_current_pos)
        # We are checking wether the environment is deterministic or stochastic
        if self.environment_type == 'deterministic':
            # In Deterministic environments, there is no use for epsilon as all the actions are deterministic / greedy / pre-determined.
            
            self.epsilon = 0
            self.current_time_steps +=1

            all_possible_actions = self.get_all_possible_actions(self.agent_current_pos)
            states_the_actions_lead_to = self.get_states_for_actions(all_possible_actions,self.agent_current_pos)
            # all_possible_actions -> [1,3] like that a single list
            # states_the_actions_lead_to -> [[1,0],[0,1]] -> states which the all_possible_actions lead to
            
            dict_temp = {}
            breaker = False
            for action, state in zip(all_possible_actions, states_the_actions_lead_to):
                dict_temp[action] = state
            
            if any([self.check_and_get_reward(state)>0 for state in list(dict_temp.values())]):
                maxi = max([self.check_and_get_reward(state) for state in list(dict_temp.values())])
                for reward_counter in range(len(self.rewards)):
                    for key, value in self.rewards[reward_counter].items():
                        if key == maxi:
                            self.chain_of_states.append(value)
                            self.agent_current_pos = value
                            breaker = True
                    if breaker == True:
                        break

                for key, value in dict_temp.items():
                    if self.agent_current_pos == value:
                        self.chain_of_actions.append( key)

            
            elif all([self.check_and_get_reward(state) == 0 for state in list(dict_temp.values())]):
                random_state = random.choice(list(dict_temp.values()))
                
                self.chain_of_states.append(random_state)
                self.agent_current_pos = random_state

                for key,value in dict_temp.items():
                    if value == random_state:
                        self.chain_of_actions.append(key)

            breaker = False
            for reward_state_counter in range(len(self.rewards)):
                for reward, state in self.rewards[reward_state_counter].items():
                    # if the reward state matches the agents, sum the cum reward and delete that particular reward state space.

                    if state == self.agent_current_pos:
                        self.cumulative_reward += reward
                        # del self.rewards[reward_state_counter]
                        breaker = True
                        break

                if breaker:
                    break
            
            # We are now re-visualizing the environment
            self.environment = np.zeros((5,5)) 

            for reward_state_counter in range(len(self.rewards)):
                for reward, position in self.rewards[reward_state_counter].items():
                    self.environment[tuple(position)] = reward
                
            self.environment[tuple(self.goal_pos)] = 0.5
            self.environment[tuple(self.agent_current_pos)] = 1
            
            # if the agent has reached the final state then done
            if (self.agent_current_pos == self.goal_pos) or (self.current_time_steps == self.max_timesteps):
                done_or_not = True
            
            else:
                done_or_not = False

            return self.environment.flatten, self.cumulative_reward, done_or_not, self.current_time_steps
    
    def get_state_based_on_action(self, action, state):
        state_copy = state.copy()
        if action == 0:
            # print('Up')
            state_copy[0] -=1
            return state_copy

        elif action == 1:
            # print('Down')
            state_copy[0] +=1
            return state_copy

        elif action == 2:
            # print('Our Left or the Agents Right')
            state_copy[1] -=1
            return state_copy

        elif action == 3:
            # print('Our Right or the Agents Left')
            state_copy[1] +=1
            return state_copy

    def check_and_get_reward(self, state_result):
        # print('checking rewards for, ', state_result)
        for i in range(len(self.rewards)):
            for key, value in self.rewards[i].items():
                if value == state_result:
                    # self.rewards.pop(i)
                    # print('found reward for state_result', state_result, value)
                    return key

        return 0

    # def get_all_q_values_for_states(self,all_actions,agent_state):
    def get_states_for_actions(self, all_actions, agent_state):
    
        temp_states = agent_state.copy() 
        # vals_to_be_returned = []
        states_considered = []

        for action_to_be_taken in all_actions:

            action_taken = self.get_state_based_on_action(action_to_be_taken,agent_state)
            
            states_considered.append(action_taken)

            agent_state = temp_states.copy()

        return states_considered

    def get_all_possible_actions(self,agent_current_pos):
        
        x_pos = agent_current_pos[0]
        y_pos = agent_current_pos[1]

        if (x_pos == 0) and (y_pos == 0):
            return [1,3]
        
        elif (x_pos == 4) and (y_pos == 0):
            return [0,3]
        
        elif (x_pos == 0) and (y_pos == 4):
            return [1,2]
        
        elif (x_pos == 0) and (y_pos <= 3):
            return [1,2,3]
        
        elif (x_pos <= 3) and (y_pos == 0 ):
            return [0,1,3]
        
        elif (x_pos == 4) and (y_pos <= 3 ):
            return [0,2,3]

        elif (x_pos < 4) and (y_pos == 4 ):
            return [0,2,1]

        elif (x_pos >=1) and (x_pos<4) and (y_pos >= 1) and (y_pos <4):
            return [0,1,2,3]

        elif (x_pos==4) and (y_pos == 4):
            return 'Yay youve won'
        else:
            return None

    def render(self):
        plt.imshow(self.environment)
        plt.show()

    def train(self):
        # done = False
        self.current_time_steps =0
        for i in range(self.no_of_episodes):
            print(i)
            done = False
            while not done:
                # self.render()
                observation, reward, done, _ = self.step()
                
                if done:
                    print(reward)
            self.reset()


# class environment_q_learning:
#     def __init__(self,type_of_env:str,gamma_disc: float, epsilon:float,epsilon_decay:float,learning_rate:float,no_of_episodes:int,max_time_steps:int):
environment_q_learning_obj = environment_q_learning(type_of_env = 'deterministic',gamma_disc=0.9,epsilon=0.2,epsilon_decay=0.001,learning_rate=0.1,no_of_episodes=1,max_time_steps=50)
environment_q_learning_obj.train()