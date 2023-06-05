import math
import random
import time
import argparse
import numpy as np
from scipy.spatial.distance import cityblock
from gym import Env

from aasma import Agent
from aasma.utils import compare_results
from aasma.wrappers import SingleAgentWrapper
from aasma.simplified_predator_prey import AntColonyEnv

N_ACTIONS = 11
DOWN, LEFT, UP, RIGHT, STAY, DOWN_PHERO, LEFT_PHERO, UP_PHERO, RIGHT_PHERO, COLLECT_FOOD, DROP_FOOD = range(N_ACTIONS)

def run_single_agent(environment: Env, agent: Agent, n_episodes: int, agent_id: int) -> np.ndarray:

    results = np.zeros(n_episodes)

    for episode in range(n_episodes):

        print(f"Episode {episode}")

        steps = 0
        terminal = False
        observation = environment.reset()
        
        while not terminal:
            steps += 1
            print(f"Timestep {steps}")
            agent.see(observation)
            action = agent.action()
            next_observation, reward, terminal, info = environment.step(action)
            environment.render()
            time.sleep(opt.render_sleep_time)
            observation = next_observation

            DeliberativeAgent.express_desire(agent)
            print(f"\tAction: {environment.get_action_meanings()[action]}\n")
            print(f"\tObservation: {observation}")

        
        environment.close()
        results[episode] = steps

    return results


class ReactiveAgent(Agent):
    def __init__(self, agent_id, n_agents):
        super(ReactiveAgent, self).__init__(f"Reactive Agent")
        self.agent_id = agent_id
        self.n_agents = n_agents
        self.n_actions = N_ACTIONS
        self.following_trail = False
        self.promising_pheromone_pos = None

    def action(self) -> int:
        agent_position = self.observation[ : 2]
        colony_position = self.observation[2 : 4] # FOR ONLY 1 COLONY

        foodpiles_in_view = self.observation[4: 29]
        pheromones_in_view = self.observation[29 : 54]

        colony_storage = self.observation[54]
        has_food = self.observation[55]

        if(has_food):
            if(self.check_if_destination_reached(agent_position, colony_position)):
                action = DROP_FOOD
            else:
                action = self.go_to_colony(agent_position, colony_position, has_food)
        elif(self.check_for_foodpiles_in_view(foodpiles_in_view)):
            closest_foodpile_pos = None

            if(self.check_if_destination_reached(agent_position, closest_foodpile_pos)):
                action, closest_foodpile_pos = self.go_to_closest_foodpile(agent_position, foodpiles_in_view)
            else:
                action = COLLECT_FOOD
        elif(self.following_trail):
            action = self.examine_promising_pheromones(agent_position, pheromones_in_view, colony_position)
        elif(self.check_for_intense_pheromones_in_view(agent_position, pheromones_in_view)):
            self.promising_pheromone_pos = self.identify_most_intense_pheromone(agent_position, pheromones_in_view)

            action = self.examine_promising_pheromones(agent_position, pheromones_in_view, colony_position)
            self.following_trail = True
        else:
            action = self.explore_randomly()

        return action

    # ################# #
    # Auxiliary Methods #
    # ################# #

    def find_global_pos(self, agent_pos, object_relative_position_index):
        
        # Test: 23, agent is 4,6

        # Calculate relative row and global row
        if(object_relative_position_index <= 4):
            relative_row = 0 
            global_row = agent_pos[1] - 2
        elif(object_relative_position_index > 4 and object_relative_position_index <= 9):
            relative_row = 1 
            global_row = agent_pos[1] - 1
        elif(object_relative_position_index > 9 and object_relative_position_index <= 14):
             relative_row = 2
             global_row = agent_pos[1]
        elif(object_relative_position_index > 14 and object_relative_position_index <= 19):
            relative_row = 3
            global_row = agent_pos[1] + 1
        elif(object_relative_position_index > 19 and object_relative_position_index <= 24):
            relative_row = 4
            global_row = agent_pos[1] + 2
        # row = 8 -> Correct, relative row is 4

        # Calculate relative column and global column
        relative_column = object_relative_position_index - 5 * relative_row
        global_column = agent_pos[0] + (relative_column - 2)  # column = 5

        global_pos = np.array([global_column, global_row]) # 5, 8
 
        return global_pos
    
    def find_relative_index(self, agent_pos, object_global_position):
        
        # Test: Agent is [4 6] ; Object is [5 8] -> 23

        distances = np.array(object_global_position) - np.array(agent_pos) # [5 8] - [4 6] = [1 2]
        object_relative_position_index = 12 + distances[0] * 1 + distances[1] * 5 # 23
 
        return int(object_relative_position_index)

    def direction_to_go(self, agent_position, point_of_interes_pos, has_food):
        """
        Given the position of the agent and the position of a prey,
        returns the action to take in order to close the distance
        """
        
        distances = np.array(point_of_interes_pos) - np.array(agent_position)
        abs_distances = np.absolute(distances)
        if abs_distances[0] > abs_distances[1]:
            return self._close_horizontally(distances, has_food) 
        elif abs_distances[0] < abs_distances[1]:
            return self._close_vertically(distances, has_food)
        else:
            roll = random.uniform(0, 1)
            return self._close_horizontally(distances, has_food) if roll > 0.5 else self._close_vertically(distances, has_food)
        
    def closest_point_of_interest(self, agent_position, points_of_interest):
        """
        Given the positions of an agent and a sequence of positions of points of interest,
        returns the positions of the point of interest (poi).
        """

        min = math.inf
        closest_poi_position = None
        n_poi = int(len(points_of_interest) / 2)
        for poi_i in range(n_poi):
            poi_position = points_of_interest[poi_i * 2], points_of_interest[(poi_i * 2) + 1]
            distance = cityblock(agent_position,  poi_position)
            if distance < min:
                min = distance
                closest_poi_position =  poi_position
        return closest_poi_position
    
    def farthest_pheromone_of_interest(self, colony_position, agent_position, promising_pheromone_relative_index, pheromones_in_view):
        """
        Given the positions of a colony and a sequence of positions of points of interest,
        returns the positions of the point of interest (poi).
        """ 

        # Find the global positions of the surrounding pheromones
        surrounding_pheromone_down_pos = self.find_global_pos(agent_position, promising_pheromone_relative_index + 5)
        surrounding_pheromone_left_pos = self.find_global_pos(agent_position, promising_pheromone_relative_index - 1)
        surrounding_pheromone_up_pos = self.find_global_pos(agent_position, promising_pheromone_relative_index - 5)
        surrounding_pheromone_right_pos = self.find_global_pos(agent_position, promising_pheromone_relative_index + 1)

        indices = np.array([promising_pheromone_relative_index + 5, promising_pheromone_relative_index - 1, promising_pheromone_relative_index - 5, promising_pheromone_relative_index + 1])

        pos1 = np.concatenate((surrounding_pheromone_down_pos, surrounding_pheromone_left_pos))
        pos2 = np.concatenate((pos1, surrounding_pheromone_up_pos))
        surrounding_pheromones_pos = np.concatenate((pos2, surrounding_pheromone_right_pos))

        # Find the pheromone most distant from the colony while ensuring it has some high level of intensity
        max_dist = 0
        farthest_poi_position = None
        n_poi = int(len(surrounding_pheromones_pos) / 2)
        for poi_i in range(n_poi):
            poi_position = surrounding_pheromones_pos[poi_i * 2], surrounding_pheromones_pos[(poi_i * 2) + 1]
            distance = cityblock(colony_position,  poi_position)
            if distance > max_dist and pheromones_in_view[indices[poi_i]] > 10: # ARBITRARY VALUE
                max_dist = distance
                farthest_poi_position =  poi_position

        if(max_dist == 0): 
            return None
        else:
            return farthest_poi_position

    def check_if_destination_reached(self, agent_position, point_of_interest_pos):
        if(point_of_interest_pos == None):
            return False

        distances = np.array(point_of_interest_pos) - np.array(agent_position)
        abs_distances = np.absolute(distances)
        if abs_distances[0] + abs_distances[1] > 1:
            return False
        elif abs_distances[0] + abs_distances[1] <= 1:
            return True
    
    def go_to_colony(self, agent_position, colony_position, has_food):
        return self.direction_to_go(agent_position, colony_position, has_food)
    
    def explore_randomly(self):
        if(self.steps_exploring == 0): # hasn't been exploring -> choose direction and keep it for 5 steps (arbitrary amount)
            self.current_exploring_action = random.randint(0, 3)

        elif(self.steps_exploring >= 5): # has explored enough in one direction -> choose another which isn't the opposite and isn't the same (better behavior)
                
            new_exploring_action = random.randint(0, 3)
            while(new_exploring_action == self.current_exploring_action + 2 or new_exploring_action == self.current_exploring_action - 2 or new_exploring_action == self.current_exploring_action):
                new_exploring_action = random.randint(0, 3)
            
            self.current_exploring_action = new_exploring_action
            self.steps_exploring = 0 # this action isn't changed in next call because of the += 1 below

        self.steps_exploring += 1

        return self.current_exploring_action

    def check_for_foodpiles_in_view(self, foodpiles_in_view):
        return any(foodpiles_in_view) # if there are any food_piles_in_view
        
    def check_for_intense_pheromones_in_view(self, pheromones_in_view):
        pheromones_of_interest = np.where(pheromones_in_view > 5)[0] # SUBSITUTE FOR initial_pheromone_intensity OF ENV

        return np.any(pheromones_of_interest)

    def go_to_closest_foodpile(self, agent_position, foodpiles_in_view):
        foodpiles_indices = np.where(foodpiles_in_view != 0)[0] # gather for non null indices

        # Get corresponding positions in array format
        foodpiles_positions = np.zeros(len(foodpiles_indices) * 2)


        for foodpile_i in range(len(foodpiles_indices)): 
            foodpile_i_pos = self.find_global_pos(agent_position, foodpiles_indices[foodpile_i])
            foodpiles_positions[foodpile_i * 2] = foodpile_i_pos[0]
            foodpiles_positions[foodpile_i * 2 + 1] = foodpile_i_pos[1]

        # Check closest foodpile position and move there
        closest_foodpile_position = self.closest_point_of_interest(agent_position, foodpiles_positions)

        return self.direction_to_go(agent_position, closest_foodpile_position, False), closest_foodpile_position
    
    def identify_most_intense_pheromone(self, agent_position, pheromones_in_view):

        most_intense_pheromone_index = np.argmax(pheromones_in_view)

        most_intense_pheromone_pos = self.find_global_pos(agent_position, most_intense_pheromone_index)

        return most_intense_pheromone_pos

    def examine_promising_pheromones(self, agent_position, pheromones_in_view, colony_position):

        distances = np.array(self.promising_pheromone_pos) - np.array(agent_position)
        abs_distances = np.absolute(distances)

        if(abs_distances[0] + abs_distances[1] == 1 or (abs_distances[0] == 1 and abs_distances[1] == 1)):
            promising_pheromone_relative_index = self.find_relative_index(agent_position, self.promising_pheromone_pos)

            surrounding_pheromone_down = pheromones_in_view[promising_pheromone_relative_index + 5]
            surrounding_pheromone_left = pheromones_in_view[promising_pheromone_relative_index - 1]
            surrounding_pheromone_up = pheromones_in_view[promising_pheromone_relative_index - 5]
            surrounding_pheromone_right = pheromones_in_view[promising_pheromone_relative_index + 1]

            surrounding_pheromones = np.array([surrounding_pheromone_down, surrounding_pheromone_left, surrounding_pheromone_up, surrounding_pheromone_right])
            next_promising_pheromone = np.argmax(surrounding_pheromones)

            if(surrounding_pheromones[next_promising_pheromone] == 0): # lost trail... 
                self.following_trail = False
                self.promising_pheromone_pos = None
                action = self.explore_randomly()
                return action

    # This option utilizes pheromone levels -> CAN CAUSE INFINTE LOOPS (check file 10)
            # Move into the position of the current promising pheromone, and update the promising pheromone
            #action = self.direction_to_go(agent_position, self.promising_pheromone_pos, False)
            #self.promising_pheromone_pos = self.find_global_pos(agent_position, surrounding_pheromones[next_promising_pheromone])

    # This option utilizes distance from colony (we keep maximizing it)
            self.promising_pheromone_pos = self.farthest_pheromone_of_interest(colony_position, agent_position, promising_pheromone_relative_index, pheromones_in_view)

            if(self.promising_pheromone_pos == None): # lost trail... 
                self.following_trail = False
                action = self.explore_randomly()
                return action

            action = self.direction_to_go(agent_position, self.promising_pheromone_pos, False)
            return action
        
        else:
            action = self.direction_to_go(agent_position, self.promising_pheromone_pos, False)
            return action

    # ############### #
    # Private Methods #
    # ############### #

    def _close_horizontally(self, distances):
        if distances[0] > 0:
            return RIGHT
        elif distances[0] < 0:
            return LEFT
        else:
            return STAY

    def _close_vertically(self, distances):
        if distances[1] > 0:
            return DOWN
        elif distances[1] < 0:
            return UP
        else:
            return STAY
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--render-sleep-time", type=float, default=0.5)
    opt = parser.parse_args()

    # 1 - Setup environment
    environment = AntColonyEnv(
        grid_shape=(10, 10),
        n_agents=1, 
        max_steps=100,
        n_foodpiles=3
    )
    environment = SingleAgentWrapper(environment, agent_id=0)

    # 2 - Setup agents
    agents = [
        ReactiveAgent(agent_id=0, n_agents=1)
    ]

    # 3 - Evaluate agents
    results = {}
    agent_id = 0
    for agent in agents:
        result = run_single_agent(environment, agent, opt.episodes, agent_id)
        results[agent.name] = result
        agent_id += 1

    # 4 - Compare results
    #compare_results(results, title="Agents on 'Predator Prey' Environment", colors=["orange", "green"])
