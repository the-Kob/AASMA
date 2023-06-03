import math
import random
import argparse
import numpy as np
from scipy.spatial.distance import cityblock

from aasma import Agent
from aasma.utils import compare_results
from aasma.wrappers import SingleAgentWrapper
from aasma.simplified_predator_prey import SimplifiedPredatorPrey

from single_random_agent import run_single_agent, RandomAgent

N_ACTIONS = 11
DOWN, LEFT, UP, RIGHT, STAY, DOWN_PHERO, LEFT_PHERO, UP_PHERO, RIGHT_PHERO, COLLECT_FOOD, DROP_FOOD = range(N_ACTIONS)

N_POSSIBLE_DESIRES = 3
GO_TO_COLONY, EXPLORE, FIND_FOODPILE = range(N_POSSIBLE_DESIRES)

class GreedyAgent(Agent):

    """
    A baseline agent for the SimplifiedPredatorPrey environment.
    The greedy agent finds the nearest prey and moves towards it.
    """

    def __init__(self, agent_id, n_agents):
        super(GreedyAgent, self).__init__(f"Greedy Agent")
        self.agent_id = agent_id
        self.n_agents = n_agents
        self.n_actions = N_ACTIONS
        self.desire = None
        self.steps_exploring = 0
        self.current_exploring_action = STAY
        self.following_trail = False
        self.promising_pheromone_pos = None

    def action(self) -> int:

        # [agents position _ colony position _ 25 * foodpiles _ 25 * pheromones _ colonys storage]

        # Make this dependent on agent view mask
        agent_position = self.observation[ : 2]
        colony_position = self.observation[2 : 2 + 2] # FOR ONLY 1 COLONY

        foodpiles_in_view = self.observation[2 + 2 : 2 + 2 + 25]
        pheromones_in_view = self.observation[2 + 2 + 25 : 2 + 2 + 25 + 25]

        colony_storage = self.observation[-2] # FOR ONLY 1 COLONY

        has_food = self.observation[-1]

        # See if there are any noteworthy things in view
        foodpiles_indices = np.where(foodpiles_in_view != 0)[0]
        pheromones_indices = np.where(pheromones_in_view != 0)[0]


        # Determine what the agent should do...
        # WHAT TO DO??? EXPLORE? PHEROMONES? FOOD? COLONY? -> DELIBERATIVE AND REACTIVE

        # WHAT IS THE GLOBAL POSITION OF THIS POINT OF INTEREST?

        # Agent wants to explore first foodpile it sees
        foodpile_global_pos = self.find_global_pos(agent_position, foodpiles_indices[0])

        point_of_interest = foodpile_global_pos

        # WHAT ACTION SHOULD THE AGENT MAKE IN ORDER TO GO TO THE POINT OF INTEREST?
        action = self.direction_to_go(agent_position, foodpile_global_pos) # return this
    
        # If were going after a given target,
        #closest_prey_position = self.closest_prey(agent_position, preys_positions)       

        #return self.direction_to_go(agent_position, closest_prey_position)

        return action

    # ################# #
    # Auxiliary Methods #
    # ################# #

    def find_global_pos(self, agent_pos, object_relative_position_index):
        
        # Test: 23, agent is 4,6

        # Calculate relative row and global row
        if(object_relative_position_index <= 4):
            relative_row = 0 
            global_row = agent_pos[1] + 2
        elif(object_relative_position_index > 4 & object_relative_position_index <= 9):
            relative_row = 1 
            global_row = agent_pos[1] + 1
        elif(object_relative_position_index > 9 & object_relative_position_index <= 14):
             relative_row = 2
             global_row = agent_pos[1] + 0
        elif(object_relative_position_index > 14 & object_relative_position_index <= 19):
            relative_row = 3
            global_row = agent_pos[1] - 1
        elif(object_relative_position_index > 19 & object_relative_position_index <= 24):
            relative_row = 4
            global_row = agent_pos[1] - 2
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
 
        return object_relative_position_index

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
    
    def check_if_destination_reached(self, agent_position, point_of_interest_pos):
        distances = np.array(point_of_interest_pos) - np.array(agent_position)
        abs_distances = np.absolute(distances)
        if abs_distances[0] + abs_distances[1] > 1:
            return False
        elif abs_distances[0] + abs_distances[1] <= 1:
            return True

    def deliberative(self):

        # BELIEFS
        beliefs = self.observation

        agent_position = beliefs[:2]
        colony_position = beliefs[2:4] # FOR ONLY 1 COLONY

        foodpiles_in_view = beliefs[4:29]
        pheromones_in_view = beliefs[29:54]

        colony_storage = beliefs[-2] # FOR ONLY 1 COLONY
        has_food = beliefs[-1]

        # DESIRES
        if(self.desire == None):
            if(has_food or colony_storage == 0): # has food or colony not visible or by default -> go to colony
                self.desire = GO_TO_COLONY 
            else: # near colony
                if(colony_storage < 50): # colony food storage is low -> find foodpile
                    self.desire = FIND_FOODPILE
                else: # colony food storage is high -> explore
                    self.desire = EXPLORE

        # INTENTIONS
        if(self.desire == GO_TO_COLONY):
            if(not self.check_if_destination_reached(agent_position, colony_position)): # if the agent hasn't reached it yet...
                action = self.go_to_colony(agent_position, colony_position, has_food) # move there

            else: # if we have reached it already...
                if(has_food == True): # drop any food, in case the agent is carrying any
                    action = DROP_FOOD
                else: # or just stay - next step the desire will update
                    action = STAY

            self.desire = None # desire accomplished, find a new desire

        elif(self.desire == EXPLORE):
            if(not self.check_for_foodpiles_in_view(foodpiles_in_view)):
                action = self.explore_randomly()
            else:
                desire = FIND_FOODPILE

            # IF SEES HIGH INTENSITY PHEROMONES, FOLLOWS THEM (maybe not) -> Already accounted for in FIND_FOODPILE (when the ant can't find strong pheromones, it randomly explores
            # but this should be different from normal exploring, where the ant is supposed to avoid exploiting other foodpiles)

        if(self.desire == FIND_FOODPILE):
            if(self.check_for_foodpiles_in_view(foodpiles_in_view)): # we have a foodpile in view...
                action, closest_foodpile_pos = self.go_to_closest_foodpile(agent_position, foodpiles_in_view)

                # We don't need to follow a trail anymore
                self.following_trail = False
                self.most_promising_pheromone_pos = None
                
                if(self.check_if_destination_reached(closest_foodpile_pos)):
                    action = COLLECT_FOOD
                    self.desire = None # desire accomplished, find a new desire

            else: # if we don't have a foodpile in view...

                if(self.following_trail): # if we're already following a trail...
                    if(not self.check_for_foodpiles_in_view(foodpiles_in_view)):
                        action = self.examine_promising_pheromones(agent_position, pheromones_in_view)

                elif (self.check_for_intense_pheromones_in_view(pheromones_in_view)): # check for high intensity pheromones
                    action, most_intense_pheromone_pos = self.go_to_most_intense_pheromone(agent_position, pheromones_in_view)
                    self.promising_pheromone_pos = most_intense_pheromone_pos
                    self.following_trail = True

                else: # if we don't have high intensity pheromones in view...
                    action = self.explore_randomly() # we are changing desires but still need to pick an action! -> explore to find pheromones

        return action
    
    
    def go_to_colony(self, agent_position, colony_position, has_food):
        return self.direction_to_go(agent_position, colony_position, has_food)
    
    def explore_randomly(self):
        if(self.steps_exploring == 0): # hasn't been exploring -> choose direction and keep it for 5 steps (arbitrary amount)
            self.current_exploring_action = random.randint(0, 3)

        elif(self.steps_exploring >= 5): # has explored enough in one direction -> choose another which isn't the opposite
                
            new_exploring_action = random.randint(0, 3)
            while(new_exploring_action == self.current_exploring_action + 2 or new_exploring_action == self.current_exploring_action - 2):
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
            foodpile_i_pos = self.find_global_pos(foodpile_i)
            foodpiles_positions[foodpile_i * 2] = foodpile_i_pos[0]
            foodpiles_positions[foodpile_i * 2 + 1] = foodpile_i_pos[1]

        # Check closest foodpile position and move there
        closest_foodpile_position = self.closest_point_of_interest(agent_position, foodpiles_positions)

        return self.direction_to_go(agent_position, closest_foodpile_position, False), closest_foodpile_position
    
    def go_to_most_intense_pheromone(self, agent_position, pheromones_in_view):

        most_intense_pheromone = pheromones_in_view[np.argmax(pheromones_in_view)]

        most_intense_pheromone_pos = self.find_global_pos(agent_position, most_intense_pheromone)

        return self.direction_to_go(agent_position, most_intense_pheromone_pos, False), most_intense_pheromone

    def examine_promising_pheromones(self, agent_position, pheromones_in_view):

        distances = np.array(self.promising_pheromone_pos) - np.array(agent_position)
        abs_distances = np.absolute(distances)

        if(abs_distances[0] == 1 or abs_distances[1] == 1): # WHAT IF THE PHEROMONE IS RIGHT BESIDES THE AGENT?? INCREASE FOOD PHEROM MUCH MORE!!!
            promising_pheromone_relative_index = self.find_relative_index(self.promising_pheromone_pos)

            surrounding_pheromone_down = pheromones_in_view[promising_pheromone_relative_index + 5]
            surrounding_pheromone_left = pheromones_in_view[promising_pheromone_relative_index - 1]
            surrounding_pheromone_up = pheromones_in_view[promising_pheromone_relative_index - 5]
            surrounding_pheromone_right = pheromones_in_view[promising_pheromone_relative_index + 1]

            surrounding_pheromones = np.array([surrounding_pheromone_down, surrounding_pheromone_left, surrounding_pheromone_up, surrounding_pheromone_right])
            action = np.argmax(surrounding_pheromones)

            self.promising_pheromone_pos = self.find_global_pos(agent_position, surrounding_pheromones[action])

            return action
        
        if abs_distances[0] > abs_distances[1]:
            return self._close_horizontally(distances, False) 
        elif abs_distances[0] < abs_distances[1]:
            return self._close_vertically(distances, False)
        else:
            roll = random.uniform(0, 1)
            return self._close_horizontally(distances, False) if roll > 0.5 else self._close_vertically(distances, False)

    # ############### #
    # Private Methods #
    # ############### #

    def _close_horizontally(self, distances, has_food):
        if distances[0] > 0:
            if(has_food):
                return RIGHT_PHERO
            else:
                return RIGHT
        elif distances[0] < 0:
            if(has_food):
                return LEFT_PHERO
            else:
                return LEFT
        else:
            return STAY

    def _close_vertically(self, distances, has_food):
        if distances[1] > 0:
            if(has_food):
                return DOWN_PHERO
            else:
                return DOWN
        elif distances[1] < 0:
            if(has_food):
                return UP_PHERO
            else:
                return UP
        else:
            return STAY

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--episodes", type=int, default=30)
    opt = parser.parse_args()

    # 1 - Setup environment
    environment = SimplifiedPredatorPrey(
        grid_shape=(7, 7),
        n_agents=1,
        max_steps=100, required_captors=1
    )
    environment = SingleAgentWrapper(environment, agent_id=0)

    # 2 - Setup agents
    agents = [
        #RandomAgent(environment.action_space.n),
        GreedyAgent(agent_id=0, n_agents=1)
    ]

    # 3 - Evaluate agents
    results = {}
    agent_id = 0
    for agent in agents:
        result = run_single_agent(environment, agent, opt.episodes)
        results[agent.name] = result
        agent_id += 1

    # 4 - Compare results
    compare_results(results, title="Agents on 'Predator Prey' Environment", colors=["orange", "green"])

