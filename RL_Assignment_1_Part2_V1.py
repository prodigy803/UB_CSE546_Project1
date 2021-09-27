elif self.environment_type == 'stochastic':
            self.current_time_steps +=1
            print('Current Agent POS', str(self.agent_current_pos))

            if action == 0:
                print("Up")
                self.agent_current_pos = self.get_final_action([-1,0],[1,0],[0,1])

            elif action == 1:
                print("Down")
                self.agent_current_pos = self.get_final_action([1,0],[-1,0],[0,-1])

            elif action == 2:
                print("Our Left or the Agents Right")
                self.agent_current_pos = self.get_final_action([0,-1],[-1,0],[0,1])

            elif action == 3:
                print("Our Right or the Agents Left")
                self.agent_current_pos = self.get_final_action([0,1],[1,0],[0,-1])

            else:
                print('action was undefined')

            print('Final Agent POS', str(self.agent_current_pos))
            
            # Here we are calculating the reward (i.e. the cumulative reward) and deleting that reward state from the collection of the reward states.

            breaker = False
            for reward_state_counter in range(len(self.reward_states)):
                for reward, state in self.reward_states[reward_state_counter].items():
                    # if the reward state matches the agents, sum the cum reward and delete that particular reward state space.

                    if state == self.agent_current_pos:
                        self.cumulative_reward += reward
                        del self.reward_states[reward_state_counter]
                        breaker = True
                        break

                if breaker:
                    break
            
            # We are now re-visualizing the environment
            self.environment = np.zeros((5,5)) 
            
            for reward_state_counter in range(len(self.reward_states)):
                for reward, position in self.reward_states[reward_state_counter].items():
                    self.environment[tuple(position)] = reward
        
            self.environment[tuple(self.goal_pos)] = 0.5
            self.environment[tuple(self.agent_current_pos)] = 1
            
            # if the agent has reached the final state then done
            if (self.agent_current_pos == self.goal_pos) or (self.current_time_steps == self.max_timesteps):
                done_or_not = True
            
            else:
                done_or_not = False

            return self.environment.flatten, self.cumulative_reward, done_or_not, self.current_time_steps

    def get_final_action(self, action1, action2, action3):

        """
        This function gets the final action for the "stochastic" modeled environment.
        
        For example take Epsilon as 0.7
        
        Then when we generate the random no, and if it is in the range of 0 - 0.7 (or 0.69999), then we take random exploratory actions.
        When we generate the random no, and if its is in the range of 0.7 - 1, then we take the greedy (or pre-determined) step in our case.
        Please note that at the moment the steps are chosen arbritrarily and not based the next states (like if there is a reward waiting in the 
        next state, then in the current version of the environment, we are not bothering about that, we are still going ahead with predetermined states
        with a chance of taking a random action).
        """
    
        random_n_number = random.uniform(0, 1)
        print('randn is', random_n_number)

        random_action_proba = self.epsilon
        old_pos = self.agent_current_pos

        if random_n_number > random_action_proba:
            self.agent_current_pos = [action1[x]+self.agent_current_pos[x] for x in range(len((self.agent_current_pos)))]
            
        elif (random_n_number >= random_action_proba/2) and (random_n_number < random_action_proba):
            self.agent_current_pos = [action2[x]+self.agent_current_pos[x] for x in range(len((self.agent_current_pos)))]

        elif (random_n_number < random_action_proba/2):
            self.agent_current_pos = [action3[x]+self.agent_current_pos[x] for x in range(len((self.agent_current_pos)))]

        else:
            raise ValueError('A Very Bad Probability thing happened.')

        self.get_action_comparison(old_pos,self.agent_current_pos)
            
        # Here we are clipping the agents position to be in the environment (i.e if the agent goes out of env, we shall clip him to be inside the environment).

        self.agent_current_pos = list(np.clip(self.agent_current_pos, 0, 4))
        return self.agent_current_pos


    def get_action_comparison(self,old_pos, new_pos):
        # This function tells us whether the final step the agent took after completing stochastic decision.

        shift = [old_pos[i]-new_pos[i] for i in range(len(new_pos))]

        if shift == [-1,0]: 
            print('The Agent Ended Up Going Down')

        elif shift == [1,0]:
            print('The Agent Ended Up Going Up')

        elif shift == [0,-1]:
            print('The Agent Ended Up Going Right')

        elif shift == [0,1]:
            print('The Agent Ended Up Going Left')