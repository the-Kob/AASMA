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

N_ACTIONS = 5
DOWN, LEFT, UP, RIGHT, STAY = range(N_ACTIONS)

N_POSSIBLE_DESIRES = 3
EXPLORE, FIND_FOODPILE, GO_TO_COLONY = range(N_POSSIBLE_DESIRES)


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


    def action(self) -> int:

        # [agents position _ colony position _ 25 * foodpiles _ 25 * pheromones _ colonys storage]

        # Make this dependent on agent view mask
        agent_position = self.observation[:2]
        colony_position = self.observation[2:4]

        foodpiles_in_view = self.observation[4:29]
        pheromones_in_view = self.observation[29:54]

        colony_storage = self.observation[-1]

        # See if there are any noteworthy things in view
        foodpiles_indices = np.where(foodpiles_in_view != 0)
        pheromones_indices = np.where(pheromones_in_view != 0)

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


    def direction_to_go(self, agent_position, foodpile_global_pos):
        """
        Given the position of the agent and the position of a prey,
        returns the action to take in order to close the distance
        """
        
        distances = np.array(foodpile_global_pos) - np.array(agent_position)
        abs_distances = np.absolute(distances)
        if abs_distances[0] > abs_distances[1]:
            return self._close_horizontally(distances)
        elif abs_distances[0] < abs_distances[1]:
            return self._close_vertically(distances)
        else:
            roll = random.uniform(0, 1)
            return self._close_horizontally(distances) if roll > 0.5 else self._close_vertically(distances)
        
        
    def closest_point_of_interest(self, agent_position, points_of_interest):
        """
        Given the positions of an agent and a sequence of positions of all prey,
        returns the positions of the closest prey.
        If there are no preys, None is returned instead
        """
        min = math.inf
        closest_prey_position = None
        n_preys = int(len(points_of_interest) / 2)
        for p in range(n_preys):
            food_position = points_of_interest[p * 2], points_of_interest[(p * 2) + 1]
            distance = cityblock(agent_position,  food_position)
            if distance < min:
                min = distance
                closest_food_position =  food_position
        return closest_food_position
    
    def deliberative(self):

        # Setup beliefs in accordance with perceptions of the environment
        beliefs = self.observation

        agent_position = beliefs[:2]
        colony_position = beliefs[2:4]

        foodpiles_in_view = beliefs[4:29]
        pheromones_in_view = beliefs[29:54]

        colony_storage = beliefs[-1]

        # Setup desires
        desires = []

        if(self.check_for_intense_pheromones(agent_position, pheromones_in_view)):
            desires.append(FIND_FOODPILE)

        return Exception
    
    def go_to_colony( self, agent_position, colony_position):
        return self.direction_to_go(agent_position, colony_position)
    
    
    def go_to_closest_foodpile(self, agent_position, foodpiles_positions):
        closest_foodpile_position = self.closest_point_of_interest(agent_position, foodpiles_positions)
        return diretction_to_go(agent_position, closest_foodpile_position)
    
 
    
    
    def check_for_intense_pheromones(self, agent_position, pheromones_in_view):
        pheromones_of_interest = np.where(pheromones_in_view > 5) # SUBSITUTE FOR initial_pheromone_intensity OF ENV

        return np.any(pheromones_of_interest)
    

    def got_to_most_intense_pheromones(self, agent_position, pheromones_in_view):

        most_intense_pheromone = pheromones_in_view.index(max(pheromones_in_view))

        most_intense_pheromone_pos = self.find_global_pos(agent_position, most_intense_pheromone)

        return self.direction_to_go(agent_position, most_intense_pheromone_pos)



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
    for agent in agents:
        result = run_single_agent(environment, agent, opt.episodes)
        results[agent.name] = result

    # 4 - Compare results
    compare_results(results, title="Agents on 'Predator Prey' Environment", colors=["orange", "green"])

