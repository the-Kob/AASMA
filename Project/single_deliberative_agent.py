import math
import random
import time
import argparse
import numpy as np
from scipy.spatial.distance import cityblock
from gym import Env

from aasma.ant_agent import AntAgent
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

def run_single_agent(environment: Env, agent: AntAgent, n_episodes: int, agent_id: int) -> np.ndarray:

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

            DeliberativeAntAgent.express_desire(agent)
            print(f"\tAction: {environment.get_action_meanings()[action]}\n")
            print(f"\tObservation: {observation}")

        
        environment.close()
        results[episode] = steps

    return results


class DeliberativeAntAgent(AntAgent):

    """
    A baseline agent for the AntColonyEnv environment.
    The deliberative agent has beliefs, desires and intention
    """

    def __init__(self, agent_id, n_agents, knowledgeable=True):
        super(DeliberativeAntAgent, self).__init__(f"Deliberative Ant Agent", agent_id, n_agents, knowledgeable)
        
        # Deliberation variables
        self.beliefs = None
        self.desire = None

    def action(self) -> int:

        # [agents position _ colony position _ 25 * foodpiles _ 25 * pheromones _ colonys storage]

        # MAKE OBSERVATIONS DEPENDENT ON VIEWMASK
        # MAKE THINGS DEPENDET ON INITIAL INTENSITY PHEROMONE LEVEL
        # ONLY WORKS FOR A SINGLE COLONY
        # IN examine_promising_pheromones, WE ARE MERELY USING DISTANCE AND NOT PHEROMONE INTENSITY LEVELS... HOW DO WE CHANGE THIS? Argmin now that we only use food pheromones?
        # REMOVE COLONY POSITION
        # AVOID OBSTACLES IN AGENT INSTEAD OF IN ENV (momentarily turned this off) -> DO this for other agents (if action the same for a long time, switch)        # CONTINUES WITH FOOD IN MOUTH?

        # SOLVED
        # THE AGENT MIGHT MISS RELEVANT HIGH INTENSITY PHEROMONES IF IT DOESN'T GO TO THE COLONY AND MERELY LOOKS AT IT (LINE 189)
        # IN EXPLORE, SHOULD WE ADD A "IF SEES HIGH INTENSITY PHEROMONES, FOLLOWS THEM" (maybe not) -> Already accounted for in FIND_FOODPILE
        #   (when the ant can't find strong pheromones, it randomly explores but this should be different from normal exploring, 
        #   where the ant is supposed to avoid exploiting other foodpiles)
        # INCREASE FOOD PHEROMONE MORE (now there only is food pheromone)
        # CONTINUES WITH FOOD IN MOUTH?

        if(self.knowledgeable):
            action_to_perform = self._knowledgeable_deliberative()
        else:
            action_to_perform = self._unknowledgeable_deliberative()

        return action_to_perform

    def _knowledgeable_deliberative(self): # The agent knows its own global position and the colony's position

        agent_position, colony_position, foodpiles_in_view, pheromones_in_view, colony_storage, has_food = self.beliefs_setup()

        # DESIRES
        if(self.desire == None):
            if(has_food or not self.check_if_destination_reached(agent_position, colony_position)): # has food or colony not visible or by default -> go to colony
                self.desire = GO_TO_COLONY 
            else: # near colony
                if(colony_storage < 100): # colony food storage is low -> find foodpile
                    self.desire = FIND_FOODPILE
                else: # colony food storage is high -> explore
                    self.desire = EXPLORE

        # INTENTIONS
        if(self.desire == GO_TO_COLONY):
            if(not self.check_if_destination_reached(agent_position, colony_position)): # if the agent hasn't reached it yet...
                action = self.go_to_colony(agent_position, colony_position, has_food) # move there

            else: # if we have reached it already...
                if(has_food): # drop any food, in case the agent is carrying any
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

                if(self.following_trail): # if we're already following a trail...
                    action = self.knowledgeable_examine_promising_pheromones(agent_position, pheromones_in_view, colony_position)

                elif(self.check_for_intense_pheromones_in_view(pheromones_in_view)): # check for high intensity pheromones

                    self.promising_pheromone_pos = self.identify_most_intense_pheromone(agent_position, pheromones_in_view)

                    action = self.knowledgeable_examine_promising_pheromones(agent_position, pheromones_in_view, colony_position)
                    self.following_trail = True

                else: # if we don't have high intensity pheromones in view...
                    action = self.explore_randomly() # we are still desiring to find food but need to pick an action! -> explore to find pheromones/foodpiles

        # Avoid obstacles
        if(action != STAY and action != COLLECT_FOOD and action != COLLECT_FOOD):
            action = self.avoid_obstacles(action, agent_position, colony_position, foodpiles_in_view)

        return action

    def _unknowledgeable_deliberative(self): # The agent does not know its own global position and the colony's position
        
        agent_position, _, foodpiles_in_view, pheromones_in_view, colony_storage, has_food = self.beliefs_setup()

        
        # DESIRES
        if(self.desire == None):
            if(has_food or colony_storage == 0): # has food or colony not visible or by default -> go to colony
                self.desire = GO_TO_COLONY 
            else: # near colony
                if(colony_storage < 100): # colony food storage is low -> find foodpile
                    self.desire = FIND_FOODPILE
                else: # colony food storage is high -> explore
                    self.desire = EXPLORE

        # INTENTIONS
        if(self.desire == GO_TO_COLONY):
            if(colony_storage == 0): # if the agent hasn't seen it yet...
                action = self.explore_randomly()

            else: # only if it is in our field of view...
                if(not self.check_if_destination_reached(agent_position, _)):
                    action = self.direction_to_go(agent_position, _, has_food)
                else:
                    if(has_food): # drop any food, in case the agent is carrying any
                        action = DROP_FOOD
                    else: # or just stay - next step the desire will update
                        action = STAY

                    self.desire = None # desire accomplished, find a new desire

        elif(self.desire == EXPLORE):
            if(not self.check_for_foodpiles_in_view(foodpiles_in_view)):
                action = self.explore_randomly()
            else:
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

                if(self.following_trail): # if we're already following a trail...
                    action = self.unknowledgeable_examine_promising_pheromones(agent_position, pheromones_in_view)

                elif(self.check_for_intense_pheromones_in_view(pheromones_in_view)): # check for high intensity pheromones

                    self.promising_pheromone_pos = self.identify_most_intense_pheromone(agent_position, pheromones_in_view)

                    action = self.unknowledgeable_examine_promising_pheromones(agent_position, pheromones_in_view)
                    self.following_trail = True

                else: # if we don't have high intensity pheromones in view...
                    action = self.explore_randomly() # we are still desiring to find food but need to pick an action! -> explore to find pheromones/foodpiles

        # Avoid obstacles
        if(action != STAY and action != COLLECT_FOOD and action != COLLECT_FOOD):
            action = self.avoid_obstacles(action, agent_position, _, foodpiles_in_view)

        return action
    
    # ################# #
    # Auxiliary Methods #
    # ################# #

    def express_desire(self):
        if(self.desire == None):
            print("\tDesire: None")
        else:
            print(f"\tDesire: {DESIRE_MEANING[self.desire]}")

    def beliefs_setup(self):

        # BELIEFS
        self.beliefs = self.observation

        agent_position = self.beliefs[:2]
        colony_position = self.beliefs[2:4] # FOR ONLY 1 COLONY

        foodpiles_in_view = self.beliefs[4:29]
        pheromones_in_view = self.beliefs[29:54]

        colony_storage = self.beliefs[54] # FOR ONLY 1 COLONY
        has_food = any([self.beliefs[55]])

        return agent_position, colony_position, foodpiles_in_view, pheromones_in_view, colony_storage, has_food

    def knowledgeable_examine_promising_pheromones(self, agent_position, pheromones_in_view, colony_position):

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
            #  Knowledgeable approach
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

    def unknowledgeable_examine_promising_pheromones(self, agent_position, pheromones_in_view):

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

            #  Unknowledgeable approach
            self.promising_pheromone_pos = self.find_global_pos(surrounding_pheromones[next_promising_pheromone])

            if(self.promising_pheromone_pos == None): # lost trail... 
                self.following_trail = False
                self.desire = EXPLORE
                action = self.explore_randomly()
                return action

        action = self.direction_to_go(agent_position, self.promising_pheromone_pos, False)
        return action

    

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
        DeliberativeAntAgent(agent_id=0, n_agents=1)
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


    

