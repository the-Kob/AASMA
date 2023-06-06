import math
import random
import numpy as np
from scipy.spatial.distance import cityblock
from abc import ABC, abstractmethod

N_ACTIONS = 11
DOWN, LEFT, UP, RIGHT, STAY, DOWN_PHERO, LEFT_PHERO, UP_PHERO, RIGHT_PHERO, COLLECT_FOOD, DROP_FOOD = range(N_ACTIONS)

class AntAgent(ABC):
    def __init__(self, name: str, agent_id, n_agents, knowledgeable):
        self.name = name
        self.observation :np.ndarray = np.ndarray([])
        self.agent_id = agent_id
        self.n_agents = n_agents
        self.n_actions = N_ACTIONS
        self.knowledgeable = knowledgeable

        # Exploration variables
        self.steps_exploring = 0
        self.current_exploring_action = STAY
        self.following_trail = False
        self.promising_pheromone_pos = None

    def see(self, observation: np.ndarray):
        self.observation = observation

    @abstractmethod
    def action(self) -> int:
        raise NotImplementedError()
    
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
        distances = np.array(point_of_interest_pos) - np.array(agent_position)
        abs_distances = np.absolute(distances)
        if abs_distances[0] + abs_distances[1] > 1:
            return False
        elif abs_distances[0] + abs_distances[1] <= 1:
            return True
    
    def go_to_colony(self, agent_position, colony_position, has_food):
        return self.direction_to_go(agent_position, colony_position, has_food)
    
    def explore_randomly(self):
        
        if(any([self.observation[55]])): # if agent has food, lay down pheromones
            index_min = 5
            index_max = 8
        else:
            index_min = 0
            index_max = 3

        if(self.steps_exploring == 0): # hasn't been exploring -> choose direction and keep it for 5 steps (arbitrary amount)
            self.current_exploring_action = random.randint(index_min, index_max)

        elif(self.steps_exploring >= 5): # has explored enough in one direction -> choose another which isn't the opposite and isn't the same (better behavior)
                
            new_exploring_action = random.randint(index_min, index_max)
            while(new_exploring_action == self.current_exploring_action + 2 or new_exploring_action == self.current_exploring_action - 2 or new_exploring_action == self.current_exploring_action):
                new_exploring_action = random.randint(index_min, index_max)
            
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

    def avoid_obstacles(self, action, agent_position, colony_position, foodpiles_in_view):

        colony_index = self.find_relative_index(agent_position, colony_position)

        # Go around fixed obstacles, like foodpiles and colony
        if((action == 0 and (foodpiles_in_view[12 + 5] != 0 or colony_index == 12 + 5)) or
            (action == 2 and (foodpiles_in_view[12 - 5] or colony_index == 12 - 5))): # foddpile is obstructing up/down
            action = random.randrange(1, 4, 2) # gives odds (left or right)

        elif((action == 1 and (foodpiles_in_view[12 - 1] != 0 or colony_index == 12 - 1)) or
             (action == 3 and (foodpiles_in_view[12 + 1] or colony_index == 12 + 1))): # object is obstructing left/right
            action = random.randrange(0, 3, 2) # gives evens (up or down)

        elif((action == 5 and (foodpiles_in_view[12 + 5] != 0 or colony_index == 12 + 5)) or
             (action == 7 and (foodpiles_in_view[12 - 5] or colony_index == 12 - 5))): # object is obstructing up_phero/down_phero
            action = random.randrange(6, 9, 2) # gives odds (left phero or right phero)

        elif((action == 6 and (foodpiles_in_view[12 - 1] != 0 or colony_index == 12 - 1)) or
              (action == 8 and (foodpiles_in_view[12 + 1] or colony_index == 12 + 1))): # object is obstructing left_phero/right_phero
            action = random.randrange(5, 8, 2) # gives evens (up phero or down phero)

        return action

    def observation_setup(self):
        agent_position = self.observation[:2]
        colony_position = self.observation[2:4] # FOR ONLY 1 COLONY

        foodpiles_in_view = self.observation[4:29]
        pheromones_in_view = self.observation[29:54]

        colony_storage = self.observation[54] # FOR ONLY 1 COLONY
        has_food = any([self.observation[55]])

        return agent_position, colony_position, foodpiles_in_view, pheromones_in_view, colony_storage, has_food

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