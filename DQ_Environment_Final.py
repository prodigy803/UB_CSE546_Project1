import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import random

class environment_dq_learning:
    def __init__(self,type_of_env:str,gamma_disc: float, epsilon:float,epsilon_decay:float,learning_rate:float,no_of_episodes:int,max_time_steps:int):
        
        self.chain_of_states = []
        self.chain_of_actions = []
        self.chain_of_rewards = []
        self.sarsa_action = None
        
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
        self.qvalue_table_a = {}
        for i1 in range(5):
            for i2 in range(5):
                for i in np.zeros((25,4)):
                    self.qvalue_table_a[(i1,i2)] = i

        self.qvalue_table_b = {}
        for i1 in range(5):
            for i2 in range(5):
                for i in np.zeros((25,4)):
                    self.qvalue_table_b[(i1,i2)] = i
        
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
        self.current_episode = 1
        
        self.goal_pos = [4,4]
        self.agent_current_pos = [0,0]
        self.done_or_not = False

        self.environment[tuple(self.agent_current_pos)] = 1
        self.environment[tuple(self.goal_pos)] = 0.5

        # Collection of Rewards (the keys) and associated values (the states). -> Total No of Rewards = 4 -> Requirement 3
        self.rewards = [{-1:[0,3]},{-1:[3,0]},{3:[2,3]},{3:[3,2]},{5:[4,4]}]
        self.reward_states = [(0,3),(3,0),(2,3),(3,2),(4,4)]
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
        self.chain_of_states = []
        self.chain_of_actions = []
        self.chain_of_rewards = []
        self.sarsa_state = None
        self.sarsa_action = None
        self.goal_pos = [4,4]
        self.agent_current_pos = [0,0]
        
        self.environment = np.zeros((5,5))
        
        self.rewards = [{-1:[0,3]},{-1:[3,0]},{3:[2,3]},{3:[3,2]},{5:[4,4]}]
        self.reward_states = [(0,3),(3,0),(2,3),(3,2),[4,4]]
        
        for reward in self.rewards:
            for reward, position in reward.items():
                self.environment[tuple(position)] = reward

        self.environment[self.agent_current_pos] = 1
        self.environment[self.goal_pos] = 0.5
    
    def step(self):
        
        if self.environment_type == 'deterministic':
            # In Deterministic environments, there is no use for epsilon as all the actions are deterministic / greedy / pre-determined.
            
            self.current_time_steps +=1

            all_possible_actions = self.get_all_possible_actions(self.agent_current_pos)
            states_the_actions_lead_to = self.get_states_for_actions(all_possible_actions,self.agent_current_pos)

            selected_action, selected_state = self.return_state_action_pair(all_possible_actions,states_the_actions_lead_to)
            
            # self.agent_current_pos = selected_state
            selected_reward = self.check_and_get_reward(selected_state)
            
            breaker = False
            for reward_state_counter in range(len(self.rewards)):
                for reward, state in self.rewards[reward_state_counter].items():
                    # if the reward state matches the agents, sum the cum reward and delete that particular reward state space.

                    if state == self.agent_current_pos:
                        self.cumulative_reward += reward
                        del self.rewards[reward_state_counter]
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
                self.done_or_not = True
            
            else:
                self.done_or_not = False

            if not self.done_or_not:
                if random.uniform(0, 1) > 0.5:
                    old_q_value = self.qvalue_table_a[tuple(self.agent_current_pos)][selected_action]
                    
                    max_temp = self.qvalue_table_a[tuple(selected_state)].max()
                    index = self.qvalue_table_a[tuple(selected_state)].tolist().index(max_temp)
                    
                    self.qvalue_table_a[tuple(self.agent_current_pos)][selected_action] = old_q_value + self.learning_rate * (selected_reward + self.gamma * self.qvalue_table_b[tuple(selected_state)][index] - old_q_value)
                else:
                    old_q_value = self.qvalue_table_b[tuple(self.agent_current_pos)][selected_action]
                    
                    max_temp = self.qvalue_table_b[tuple(selected_state)].max()
                    index = self.qvalue_table_b[tuple(selected_state)].tolist().index(max_temp)

                    self.qvalue_table_b[tuple(self.agent_current_pos)][selected_action] = old_q_value + self.learning_rate * (selected_reward + self.gamma * self.qvalue_table_a[tuple(selected_state)][index] - old_q_value)
               

                self.agent_current_pos = selected_state
            
            return self.environment.flatten, self.cumulative_reward, self.done_or_not, self.current_time_steps

        elif self.environment_type == 'stochastic':
            self.current_time_steps +=1

            all_possible_actions = self.get_all_possible_actions(self.agent_current_pos)
            states_the_actions_lead_to = self.get_states_for_actions(all_possible_actions,self.agent_current_pos)

            selected_action, selected_state = self.return_state_action_pair(all_possible_actions,states_the_actions_lead_to)
            
            selected_action, selected_state = self.return_final_stochastic(selected_action, selected_state,all_possible_actions,states_the_actions_lead_to)

            # self.agent_current_pos = selected_state
            selected_reward = self.check_and_get_reward(selected_state)
            
            breaker = False
            for reward_state_counter in range(len(self.rewards)):
                for reward, state in self.rewards[reward_state_counter].items():
                    # if the reward state matches the agents, sum the cum reward and delete that particular reward state space.

                    if state == self.agent_current_pos:
                        self.cumulative_reward += reward
                        del self.rewards[reward_state_counter]
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
                self.done_or_not = True
            
            else:
                self.done_or_not = False
            
            if not self.done_or_not:
                if random.uniform(0, 1) > 0.5:
                    old_q_value = self.qvalue_table_a[tuple(self.agent_current_pos)][selected_action]
                    
                    max_temp = self.qvalue_table_a[tuple(selected_state)].max()
                    index = self.qvalue_table_a[tuple(selected_state)].tolist().index(max_temp)
                    
                    self.qvalue_table_a[tuple(self.agent_current_pos)][selected_action] = old_q_value + self.learning_rate * (selected_reward + self.gamma * self.qvalue_table_b[tuple(selected_state)][index] - old_q_value)
                else:
                    old_q_value = self.qvalue_table_b[tuple(self.agent_current_pos)][selected_action]
                    
                    max_temp = self.qvalue_table_b[tuple(selected_state)].max()
                    index = self.qvalue_table_b[tuple(selected_state)].tolist().index(max_temp)

                    self.qvalue_table_b[tuple(self.agent_current_pos)][selected_action] = old_q_value + self.learning_rate * (selected_reward + self.gamma * self.qvalue_table_a[tuple(selected_state)][index] - old_q_value)
               

                self.agent_current_pos = selected_state
        return self.environment.flatten, self.cumulative_reward, self.done_or_not, self.current_time_steps

    def act_on_greedy(self):
        self.epsilon = 0
        self.current_time_steps +=1

        all_possible_actions = self.get_all_possible_actions(self.agent_current_pos)
        states_the_actions_lead_to = self.get_states_for_actions(all_possible_actions,self.agent_current_pos)

        selected_action, selected_state = self.return_state_action_pair_greedy(all_possible_actions,states_the_actions_lead_to)
#         print("Agent was at {}, he evaluated {} via actions {} and chose state {} via action {}".format(self.agent_current_pos,all_possible_actions,all_possible_actions,selected_state, selected_action))
        breaker = False
        for reward_state_counter in range(len(self.rewards)):
            for reward, state in self.rewards[reward_state_counter].items():
                # if the reward state matches the agents, sum the cum reward and delete that particular reward state space.

                if state == self.agent_current_pos:
                    self.cumulative_reward += reward
                    del self.rewards[reward_state_counter]
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
            self.done_or_not = True
        
        else:
            self.done_or_not = False

        if not self.done_or_not:
            self.agent_current_pos = selected_state
        return self.environment.flatten, self.cumulative_reward, self.done_or_not, self.current_time_steps

    def return_final_stochastic(self,selected_action, selected_state,all_possible_actions,states_the_actions_lead_to):
        
        remaining_actions = [x for x in all_possible_actions if x!=selected_action]
        remaining_states = [x for x in states_the_actions_lead_to if x!=selected_state]
        
        # print('filterd actions are {} and filtered states are {}'.format(remaining_actions,remaining_states))
        
        list_of_probabs = []
        probabs = (.20) / len(remaining_states)
        for i in range(len(remaining_states)):
            list_of_probabs.append(probabs)
        list_of_probabs.append(.80)
        
        remaining_states.append(selected_state)
        remaining_actions.append(selected_action)
        
        # print(remaining_actions,list_of_probabs)
        chosen_action = random.choices(remaining_actions,list_of_probabs)
        chosen_state = remaining_states[remaining_actions.index(chosen_action[0])]
        return chosen_action, chosen_state

    def return_state_action_pair_greedy(self,all_possible_actions,states_the_actions_lead_to):
        action_to_be_returned, state_to_be_returned, _, _ = self.get_best_state_on_q_value(self.agent_current_pos,all_possible_actions,states_the_actions_lead_to)
        return action_to_be_returned, state_to_be_returned,
        
    def return_state_action_pair(self,all_possible_actions,states_the_actions_lead_to):
        
        random_n_number = random.uniform(0, 1)
        random_action_proba = self.epsilon
        action_to_be_returned, state_to_be_returned, _ = self.get_best_state_on_q_value(self.agent_current_pos,all_possible_actions,states_the_actions_lead_to)
        
        if random_n_number > random_action_proba:
            return action_to_be_returned, state_to_be_returned
            
        else:
            remaining_actions = [x for x in all_possible_actions if x!=action_to_be_returned]
            remaining_states = [x for x in states_the_actions_lead_to if x!= state_to_be_returned]
            
            action_to_be_returned = random.choice(remaining_actions)
            state_to_be_returned = remaining_states[remaining_actions.index(action_to_be_returned)]

            return action_to_be_returned, state_to_be_returned

    def get_best_state_on_q_value(self,current_state, all_possible_actions,states_the_actions_lead_to):
        current_max = None
        state_to_be_returned = None
        action_to_be_returned = None
        

        
        for action,state in zip(all_possible_actions,states_the_actions_lead_to):
            # print(action, state, ' being evaluated at ',current_state)
            
            if current_max == None:
                # print(self.qvalue_table[tuple(state)][action])
                
                current_max = (self.qvalue_table_a[tuple(current_state)][action] + self.qvalue_table_b[tuple(current_state)][action])/2

                state_to_be_returned = state
                action_to_be_returned = action
                

            elif current_max != None:
                max1 = (self.qvalue_table_a[tuple(current_state)][action] + self.qvalue_table_b[tuple(current_state)][action])/2
                if max1 > current_max:
                    current_max = max1
                    state_to_be_returned = state
                    action_to_be_returned = action
        
        return action_to_be_returned, state_to_be_returned, current_max

    def get_best_state_on_reward(self,all_possible_actions, states_the_actions_lead_to):
        
        for action,state in zip(all_possible_actions,states_the_actions_lead_to):
            
            for i in range(len(self.rewards)):
                for key, value in self.rewards[i].items():
                    if value == state:
                        return action, state
        return None, None


    def get_state_based_on_action(self, action, state):
        state_copy = list(state).copy()
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
        
        temp_states = list(agent_state).copy() 
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
        
        elif (x_pos <= 3) and (y_pos == 0):
            return [0,1,3]
        
        elif (x_pos == 4) and (y_pos <= 3 ):
            return [0,2,3]

        elif (x_pos < 4) and (y_pos == 4 ):
            return [0,2,1]

        elif (x_pos >=1) and (x_pos<4) and (y_pos >= 1) and (y_pos <4):
            return [0,1,2,3]

        elif (x_pos==4) and (y_pos == 4):
            return [0,2]
        else:
            return None

    def render(self):
        
        plt.imshow(self.environment)
        plt.show()

    def train(self):
        # done = False
        self.reward_per_episode = []
        self.epsilons = []
        self.time_steps = []
        self.final_goal_reached = 0
        self.epsilons.append(self.epsilon)
        for i in range(self.no_of_episodes):
            self.current_time_steps = 0
            self.done_or_not = False
            self.reset()
            while not self.done_or_not:
                observation, reward, self.done_or_not, _ = self.step()
                
                if self.done_or_not:
                    self.reward_per_episode.append(reward)

            self.current_episode+=1
            if self.epsilon > 0:
                self.epsilon -= self.epsilon*self.epsilon_decay
            
            if self.agent_current_pos == [4,4]:
                self.final_goal_reached+=1
            
            self.epsilons.append(self.epsilon)
            self.time_steps.append(self.current_time_steps)
            
    def take_greedy_only(self):
        # done = False
        self.reward_per_episode = []
        self.epsilons = []
        self.time_steps = []
        self.final_goal_reached = 0
        self.epsilons.append(self.epsilon)
        for i in range(self.no_of_episodes):
            self.current_time_steps = 0
            self.done_or_not = False
            self.reset()
            while not self.done_or_not:
                observation, reward, self.done_or_not, _ = self.act_on_greedy()
                
                if self.done_or_not:
                    self.reward_per_episode.append(reward)

            self.current_episode+=1
            if self.epsilon > 0:
                self.epsilon -= self.epsilon*self.epsilon_decay
            
            if self.agent_current_pos == [4,4]:
                self.final_goal_reached+=1
            
            self.epsilons.append(self.epsilon)
            self.time_steps.append(self.current_time_steps)