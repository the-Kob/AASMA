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

N_POSSIBLE_DESIRES = 3
GO_TO_COLONY, EXPLORE, FIND_FOODPILE = range(N_POSSIBLE_DESIRES)

DESIRE_MEANING = {
    0: "GO_TO_COLONY",
    1: "EXPLORE",
    2: "FIND_FOODPILE"
}

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


class DeliberativeAgent(Agent):

    """
    A baseline agent for the AntColonyEnv environment.
    The deliberative agent has beliefs, desires and intention
    """

    def __init__(self, agent_id, n_agents):
        super(DeliberativeAgent, self).__init__(f"Deliberative Agent")
        self.agent_id = agent_id
        self.n_agents = n_agents
        self.n_actions = N_ACTIONS
        self.beliefs = None
        self.desire = None
        self.steps_exploring = 0
        self.current_exploring_action = STAY
        self.following_trail = False
        self.promising_pheromone_pos = None

    def action(self) -> int:

        # [agents position _ colony position _ 25 * foodpiles _ 25 * pheromones _ colonys storage]

        # MAKE OBSERVATIONS DEPENDENT ON VIEWMASK
        # MAKE THINGS DEPENDET ON INITIAL INTENSITY PHEROMONE LEVEL
        # ONLY WORKS FOR A SINGLE COLONY
        # IN examine_promising_pheromones, WE ARE MERELY USING DISTANCE AND NOT PHEROMONE INTENSITY LEVELS... HOW DO WE CHANGE THIS? Argmin now that we only use food pheromones?
        # REMOVE COLONY POSITION
        # AVOID OBSTACLES IN AGENT INSTEAD OF IN ENV (momentarily turned this off)

        # SOLVED
        # THE AGENT MIGHT MISS RELEVANT HIGH INTENSITY PHEROMONES IF IT DOESN'T GO TO THE COLONY AND MERELY LOOKS AT IT (LINE 189)
        # IN EXPLORE, SHOULD WE ADD A "IF SEES HIGH INTENSITY PHEROMONES, FOLLOWS THEM" (maybe not) -> Already accounted for in FIND_FOODPILE
        #   (when the ant can't find strong pheromones, it randomly explores but this should be different from normal exploring, 
        #   where the ant is supposed to avoid exploiting other foodpiles)
        # INCREASE FOOD PHEROMONE MORE (now there only is food pheromone)

        action_to_perform = self.deliberative_architecture()


        #if(action_to_perform != STAY and action_to_perform != COLLECT_FOOD and action_to_perform != COLLECT_FOOD):
        #    return self.avoid_obstacles(action_to_perform)

        return action_to_perform
    
    def express_desire(self):
        if(self.desire == None):
            print("\tDesire: None")
        else:
            print(f"\tDesire: {DESIRE_MEANING[self.desire]}")

    def deliberative_architecture(self):

        # BELIEFS
        self.beliefs = self.observation

        agent_position = self.beliefs[:2]
        colony_position = self.beliefs[2:4] # FOR ONLY 1 COLONY

        foodpiles_in_view = self.beliefs[4:29]
        pheromones_in_view = self.beliefs[29:54]

        colony_storage = self.beliefs[54] # FOR ONLY 1 COLONY
        has_food = self.beliefs[55]

        # DESIRES
        if(self.desire == None):
            if(has_food or not self.check_if_destination_reached(agent_position, colony_position)): # has food or colony not visible or by default -> go to colony
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
            elif(self.check_for_foodpiles_in_view(foodpiles_in_view) or (colony_storage > 0 and colony_storage < 50)):
                self.desire = FIND_FOODPILE

        if(self.desire == FIND_FOODPILE):

            if(self.check_for_foodpiles_in_view(foodpiles_in_view)):  # we have a foodpile in view...
                action, closest_foodpile_pos = self.go_to_closest_foodpile(agent_position, foodpiles_in_view)

                # We don't need to follow a trail anymore
                self.following_trail = False
                self.promising_pheromone_pos = None
                
                if(self.check_if_destination_reached(agent_position, closest_foodpile_pos)):
                    action = COLLECT_FOOD
                    self.desire = None # desire accomplished, find a new desire

            else: # if we don't have a foodpile in view...

                if(self.following_trail): # if we're already following a trail... # NOT WORKING VERY WELL 
                    action = self.examine_promising_pheromones(agent_position, pheromones_in_view, colony_position)

                elif(self.check_for_intense_pheromones_in_view(pheromones_in_view)): # check for high intensity pheromones

                    self.promising_pheromone_pos = self.identify_most_intense_pheromone(agent_position, pheromones_in_view)

                    action = self.examine_promising_pheromones(agent_position, pheromones_in_view, colony_position)
                    self.following_trail = True

                else: # if we don't have high intensity pheromones in view...
                    action = self.explore_randomly() # we are changing desires but still need to pick an action! -> explore to find pheromones

        return action

    def avoid_obstacles(self, action):

        foodpiles_in_view = self.beliefs[4:29]

        if((action == 0 and foodpiles_in_view[12 + 5] != 0) or (action == 2 and foodpiles_in_view[12 - 5])): # foddpile is obstructing up/down
            action = random.randrange(1, 4, 2) # gives odds (left or right)

        elif((action == 1 and foodpiles_in_view[12 - 1] != 0) or (action == 3 and foodpiles_in_view[12 + 1])): # object is obstructing left/right
            action = random.randrange(0, 3, 2) # gives evens (up or down)

        elif((action == 5 and foodpiles_in_view[12 + 5] != 0) or (action == 7 and foodpiles_in_view[12 - 5])): # object is obstructing up_phero/down_phero
            action = random.randrange(6, 9, 2) # gives odds (left phero or right phero)

        elif(action == 6 or action == 8): # object is obstructing left_phero/right_phero
            action = random.randrange(5, 8, 2) # gives evens (up phero or down phero)

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
                self.desire = EXPLORE
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
                self.desire = EXPLORE
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
        DeliberativeAgent(agent_id=0, n_agents=1)
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


    

