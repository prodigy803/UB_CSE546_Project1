import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import random

class environment_dq_learning:
    def __init__(self,type_of_env:str,gamma_disc: float, epsilon:float,epsilon_decay:float,learning_rate:float,no_of_episodes:int,max_time_steps:int):
        
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
        # Q Value Learning Table a
        # self. = np.zeros((len(self.environment.flatten())),self.action_set_size)
        self.qvalue_table_a = {}
        for i1 in range(5):
            for i2 in range(5):
                for i in np.zeros((25,4)):
                    self.qvalue_table_a[(i1,i2)] = i

        # Q Value Learning Table b
        self.qvalue_table_b = {}
        for i1 in range(5):
            for i2 in range(5):
                for i in np.zeros((25,4)):
                    self.qvalue_table_b[(i1,i2)] = i
        
        # Setting the permissible time-steps
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
        
        # Defining the initial Goal Position, the agent start position and the done-or-not flag (that sort of indicates end of an episode)
        self.goal_pos = [4,4]
        self.agent_current_pos = [0,0]
        self.done_or_not = False

        self.environment[tuple(self.agent_current_pos)] = 1
        self.environment[tuple(self.goal_pos)] = 0.5

        # Collection of Rewards (the keys) and associated values (the states). -> Total No of Rewards = 4 -> Requirement 3
        # Change 1 - Initialized a reward function for the goal state as the agent wasnt converging towards it in the initial tries.
        self.rewards = [{-1:[0,3]},{-1:[3,0]},{3:[2,3]},{3:[3,2]},{15:[4,4]}]
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
        self.goal_pos = [4,4]
        self.agent_current_pos = [0,0]
        
        self.environment = np.zeros((5,5))
        
        self.rewards = [{-1:[0,3]},{-1:[3,0]},{3:[2,3]},{3:[3,2]},{15:[4,4]}]
        self.reward_states = [(0,3),(3,0),(2,3),(3,2),[4,4]]
        
        # resetting the environment values for visualization.
        for reward in self.rewards:
            for reward, position in reward.items():
                self.environment[tuple(position)] = reward

        self.environment[self.agent_current_pos] = 1
        self.environment[self.goal_pos] = 0.5
    
    def step(self):

        # Amendment: We have introduced an epsilon greedy policy for both the determinant and the stochastic environments.

        self.current_time_steps +=1

        # Given a state (agents current position), func get_all_possible_actions() gets all the possible the actions the agent can take.
        # For example, if the agent is in state (0,0), the agent can only take right or go below.
        # Another example, if the agent is in state (1,1), the agent can take any action (right, left, up or down)
        
        all_possible_actions = self.get_all_possible_actions(self.agent_current_pos)
        
        # get_states_for_actions() returns the states that the agent can go into, given the possible actions and the agents current position.
        # For example, if agent is in [1,1] and the possible actions are [0,1,2,3] ~ [up, down, left, right]
        # get_states_for_actions([0,1,2,3],[1,1]) -> [(0,1),(1,0),(2,1),(1,2)] -> its a list of all available states.
        states_the_actions_lead_to = self.get_states_for_actions(all_possible_actions,self.agent_current_pos)

        # this returns the selected action and state via the epsilon greedy policy.
        # self.return_state_action_pair([0,1,2,3],[(0,1),(1,0),(2,1),(1,2)]) -> 1, [2,1]
        selected_action, selected_state = self.return_state_action_pair(all_possible_actions,states_the_actions_lead_to)
        
        # if the environment is stochastic, make the probability of the selected action as 80% and spread the remaining 20% probability
        # over the remaining actions.
        if self.environment_type == 'stochastic':
            selected_action, selected_state = self.return_final_stochastic(selected_action, selected_state,all_possible_actions,states_the_actions_lead_to)

        # This returns the rewards available at selected_state, if any:
        # check_and_get_reward([2,1]) -> 0
        selected_reward = self.check_and_get_reward(selected_state)

        # if the agent has reached the final state then done or if it has taken the maximum time-steps
        if (self.agent_current_pos == self.goal_pos) or (self.current_time_steps == self.max_timesteps):
            self.done_or_not = True
        
        else:
            self.done_or_not = False

        # if the agent hasnt reached the final time-step or the goal state, update the Q-A or Q-B Tables.
        if not self.done_or_not:
            # random.uniform(0, 1) helps us to do the following:
            # 1. If the the value of random() is greater than 0.5, then we will update the Q-A table
            # 2. If the value of random() is less than or equal to 0.5, then we will update the Q-B table.
            if random.uniform(0, 1) > 0.5:
                old_q_value = self.qvalue_table_a[tuple(self.agent_current_pos)][selected_action]
                
                max_temp = self.qvalue_table_a[tuple(selected_state)].max()
                index = self.qvalue_table_a[tuple(selected_state)].tolist().index(max_temp)
                
                # Here we are updating the q-value of table_a for current agent position for the chosen/selected action(as per the epsilon greedy approach)
                # self.qvalue_table_b[tuple(selected_state)][index] - Qvalue in table_b for action which has got max qvalue in table a
                self.qvalue_table_a[tuple(self.agent_current_pos)][selected_action] = old_q_value + self.learning_rate * (selected_reward + self.gamma * self.qvalue_table_b[tuple(selected_state)][index] - old_q_value)
            else:
                old_q_value = self.qvalue_table_b[tuple(self.agent_current_pos)][selected_action]
                
                max_temp = self.qvalue_table_b[tuple(selected_state)].max()
                index = self.qvalue_table_b[tuple(selected_state)].tolist().index(max_temp)

                # Here we are updating the q-value of table_b for current agent position for the chosen/selected action(as per the epsilon greedy approach)
                # self.qvalue_table_a[tuple(selected_state)][index] - Qvalue in table_a for action which has got max qvalue in table b
                self.qvalue_table_b[tuple(self.agent_current_pos)][selected_action] = old_q_value + self.learning_rate * (selected_reward + self.gamma * self.qvalue_table_a[tuple(selected_state)][index] - old_q_value)
            
            # change the agent position to the selected state.
            self.agent_current_pos = selected_state

            ## The below code checks the agents current state for any award if available.
            ## If the awards is available in the transistioned state, then the reward gets erased in the next iteration.
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
        
        return self.environment.flatten, self.cumulative_reward, self.done_or_not, self.current_time_steps

    # This function has been specifically written for when the agent is supposed to take the greedy action. 
    # Please note the usage of the function *return_state_action_pair_greedy* -> this returns the action-state pair that have max
    # q-value in the q-tables (in our case we are averaging) and we are taking the action-state pair that have max-avg
    # amoung all action pairs.
    def act_on_greedy(self):
        self.epsilon = 0
        self.current_time_steps +=1

        all_possible_actions = self.get_all_possible_actions(self.agent_current_pos)
        states_the_actions_lead_to = self.get_states_for_actions(all_possible_actions,self.agent_current_pos)

        selected_action, selected_state = self.return_state_action_pair_greedy(all_possible_actions,states_the_actions_lead_to)
        
        if self.environment_type == 'stochastic':
            selected_action, selected_state = self.return_final_stochastic(selected_action, selected_state,all_possible_actions,states_the_actions_lead_to)
        
        # The following step is purely for debugging purposes.
#         print("Agent was at {}, he evaluated {} via actions {} and chose state {} via action {}".format(self.agent_current_pos,all_possible_actions,all_possible_actions,selected_state, selected_action))
       
        ## The below code checks the agents current state for any award if available.
        ## If the awards is available in the transistioned state, then the reward gets erased in the next iteration.
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
            self.agent_current_pos = selected_state
        
        return self.environment.flatten, self.cumulative_reward, self.done_or_not, self.current_time_steps

    # This function returns the final action and state that were selected via the epsilon-greedy method in the stochastic 
    # environment. What it does is simple:
    # 1. it receives 4 things - The selected action from policy, the selected state from policy, all_possible_actions that the 
    #    agent can take and states_the_actions_lead_to.
    # 2. It then removes the selected action and state from the available actions and the states those actions lead to.
    # 3. Post that it generates a list of probabilities:
    #   - If there are 2 remaining states (3 in total, one being the greedy state), it divides 20 by 2 and stores it in a list like [.1,.1]
    #   - It then appends .8 to the end of the list to generate a final list like [0.1,0.1,0.8]
    # 4. Post it appends the selected action and state back to the filtered action and state list (as done in step 2).
    # so now we have three lists 1) list_of_probabs 2)remaining_states(which contains all states with the greedy state at the end of the list)
    # 3) remaining actions (which contains all the actions with the greedy action at the end of the list).
    # 5. it then calls random.choice on the remaining_actions with the the list_of_proba as the weight parameter and the random.choice method
    #    abstracts the process of selecting the final state for us.
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
    
    # The following function just returns the state action pair for the which the q-values are max, for generating the final-q value graphs.
    def return_state_action_pair_greedy(self,all_possible_actions,states_the_actions_lead_to):
        action_to_be_returned, state_to_be_returned, _ = self.get_best_state_on_q_value(self.agent_current_pos,all_possible_actions,states_the_actions_lead_to)
        return action_to_be_returned, state_to_be_returned,
        
    # The following function return_state_action_pair returns the selected state action via the policy (epsilon greedy method).
    # If the random.uniform(0, 1) <= self.epsilon, we take a random action or we take the action and state pair returned by "get_best_state_on_q_value".
    # So eventually because of the epsilon decay that we have used, it will be such that we will always take greedy action as epsilon tends to 0.
    def return_state_action_pair(self,all_possible_actions,states_the_actions_lead_to):
        
        random_n_number = random.uniform(0, 1)
        random_action_proba = self.epsilon
        action_to_be_returned, state_to_be_returned, _ = self.get_best_state_on_q_value(self.agent_current_pos,all_possible_actions,states_the_actions_lead_to)
        
        if random_n_number > random_action_proba:
            return action_to_be_returned, state_to_be_returned
            
        else:
            remaining_actions = [x for x in all_possible_actions if x!=action_to_be_returned]
            remaining_states = [x for x in states_the_actions_lead_to if x!= state_to_be_returned]
            
            # remaining actions and remaining states have been aligned in such a way that [1,2,3] will correspond to states that are just below, to the left or the to the right.
            action_to_be_returned = random.choice(remaining_actions)
            state_to_be_returned = remaining_states[remaining_actions.index(action_to_be_returned)]

            return action_to_be_returned, state_to_be_returned

    # This function returns the state and the action which ahve the best q-value from both the q-tables.
    # 1. Initialize the current max, state_to_be_returned and action_to_be_returned
    # 2. Iterate through the all_possible_actions and states_the_actions_lead_to
    # 3. Calculate the current max at each iteration via the average of both the q-value tables for action and state pairs.
    # 4. Select the state action pair that have max average and return those pairs.
    def get_best_state_on_q_value(self,current_state, all_possible_actions,states_the_actions_lead_to):
        current_max = None
        state_to_be_returned = None
        action_to_be_returned = None
        
        for action,state in zip(all_possible_actions,states_the_actions_lead_to):
            
            if current_max == None:    
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

    # A helper function that return the state when given action from the current state or "state" that is part of the parameters for the function.
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

    # Returns the reward from the state_result, if any. Else return 0
    def check_and_get_reward(self, state_result):
        # print('checking rewards for, ', state_result)
        for i in range(len(self.rewards)):
            for key, value in self.rewards[i].items():
                if value == state_result:
                    # self.rewards.pop(i)
                    # print('found reward for state_result', state_result, value)
                    return key

        return 0

    # get_states_for_actions - Returns all the states that can be taken from "agent_state", from the list of given "all_actions"
    def get_states_for_actions(self, all_actions, agent_state):
        
        temp_states = list(agent_state).copy() 
        # vals_to_be_returned = []
        states_considered = []

        for action_to_be_taken in all_actions:

            action_taken = self.get_state_based_on_action(action_to_be_taken,agent_state)
            
            states_considered.append(action_taken)

            agent_state = temp_states.copy()

        return states_considered

    # Returns all the possible actions from a given "agent_current_pos".
    def get_all_possible_actions(self,given_position):
        
        x_pos = given_position[0]
        y_pos = given_position[1]

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

    # Render the environment
    def render(self):
        
        plt.imshow(self.environment)
        plt.show()

    # this function just trains the q-value tables for the given parameters.
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
            
            # these will be useful for tracking later on
            self.epsilons.append(self.epsilon)
            self.time_steps.append(self.current_time_steps)
    
    # This is just the train function, except instead of the step function, we are calling the greedy values only. This is to be called after the train function since we require the
    # trained q-value tables in this.
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
                observation, reward, self.done_or_not, self.current_time_steps = self.act_on_greedy()
                 
                if self.done_or_not:
                    self.reward_per_episode.append(reward)

            self.current_episode+=1
            if self.epsilon > 0:
                self.epsilon -= self.epsilon*self.epsilon_decay
            
            if self.agent_current_pos == [4,4]:
                self.final_goal_reached+=1
            
            self.epsilons.append(self.epsilon)
            self.time_steps.append(self.current_time_steps)