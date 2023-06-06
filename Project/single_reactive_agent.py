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

            print(f"\tAction: {environment.get_action_meanings()[action]}\n")
            print(f"\tObservation: {observation}")

        
        environment.close()
        results[episode] = steps

    return results

class ReactiveAntAgent(AntAgent):
    def __init__(self, agent_id, n_agents, knowledgeable=True):
        super(ReactiveAntAgent, self).__init__(f"Reactive Ant Agent", agent_id, n_agents, knowledgeable)

    def action(self) -> int:
        '''TODO Unkowledgeable
        if(self.knowledgeable):
            action_to_perform = self._knowledgeable_reactive()
        else:
            action_to_perform = self._unknowledgeable_reactive()
        '''

        action_to_perform = self._knowledgeable_reactive()
            
        return action_to_perform

    def _knowledgeable_reactive(self):
        agent_position, colony_position, foodpiles_in_view, pheromones_in_view, colony_storage, has_food = self.observation_setup()

        if(has_food):
            if(self.check_if_destination_reached(agent_position, colony_position)):
                action = DROP_FOOD
            else:
                action = self.go_to_colony(agent_position, colony_position, has_food)
        elif(self.check_for_foodpiles_in_view(foodpiles_in_view)):
            action, closest_foodpile_pos = self.go_to_closest_foodpile(agent_position, foodpiles_in_view)
            self.following_trail = False
            self.promising_pheromone_pos = None

            if(self.check_if_destination_reached(agent_position, closest_foodpile_pos)):
                action = COLLECT_FOOD
        elif(self.following_trail):
            action = self.knowledgeable_examine_promising_pheromones(agent_position, pheromones_in_view, colony_position)
        elif(self.check_for_intense_pheromones_in_view(pheromones_in_view)):
            self.promising_pheromone_pos = self.identify_most_intense_pheromone(agent_position, pheromones_in_view)
            action = self.knowledgeable_examine_promising_pheromones(agent_position, pheromones_in_view, colony_position)
        else:
            action = self.explore_randomly()

        # Avoid obstacles
        if(action != STAY and action != COLLECT_FOOD and action != COLLECT_FOOD):
            action = self.avoid_obstacles(action, agent_position, colony_position, foodpiles_in_view)

        return action

    '''TODO Unkowledgeable
    def _unknowledgeable_reactive(self):
        agent_position, colony_position, foodpiles_in_view, pheromones_in_view, colony_storage, has_food = self.observation_setup()

        if(has_food):
            if(self.check_if_destination_reached(agent_position, colony_position)):
                action = DROP_FOOD
            else:
                action = self.go_to_colony(agent_position, colony_position, has_food)
        elif(self.check_for_foodpiles_in_view(foodpiles_in_view)):
            action, closest_foodpile_pos = self.go_to_closest_foodpile(agent_position, foodpiles_in_view)
            self.following_trail = False
            self.promising_pheromone_pos = None

            if(self.check_if_destination_reached(agent_position, closest_foodpile_pos)):
                action = COLLECT_FOOD
        elif(self.following_trail):
            action = self.unknowledgeable_examine_promising_pheromones(agent_position, pheromones_in_view, colony_position)
        elif(self.check_for_intense_pheromones_in_view(pheromones_in_view)):
            self.promising_pheromone_pos = self.identify_most_intense_pheromone(agent_position, pheromones_in_view)
            action = self.unknowledgeable_examine_promising_pheromones(agent_position, pheromones_in_view, colony_position)
        else:
            action = self.explore_randomly()

        # Avoid obstacles
        if(action != STAY and action != COLLECT_FOOD and action != COLLECT_FOOD):
            action = self.avoid_obstacles(action, agent_position, colony_position, foodpiles_in_view)

        return action
    '''
        
    # ################# #
    # Auxiliary Methods #
    # ################# #

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

            self.promising_pheromone_pos = self.farthest_pheromone_of_interest(colony_position, agent_position, promising_pheromone_relative_index, pheromones_in_view)

            if(self.promising_pheromone_pos == None): # lost trail... 
                self.following_trail = False
                action = self.explore_randomly()
                return action

        self.following_trail = True

        action = self.direction_to_go(agent_position, self.promising_pheromone_pos, False)
        
        return action

    '''TODO Unkowledgeable
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

            self.promising_pheromone_pos = self.farthest_pheromone_of_interest(colony_position, agent_position, promising_pheromone_relative_index, pheromones_in_view)

            if(self.promising_pheromone_pos == None): # lost trail... 
                self.following_trail = False
                action = self.explore_randomly()
                return action

            self.following_trail = True

        action = self.direction_to_go(agent_position, self.promising_pheromone_pos, False)
        
        return action
    '''
        
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
        ReactiveAntAgent(agent_id=0, n_agents=1, knowledgeable=True)
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
