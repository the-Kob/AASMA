import argparse
import time
import numpy as np
from gym import Env
from typing import Sequence

from aasma.utils import compare_results
from aasma.simplified_predator_prey import AntColonyEnv

from aasma import AntAgent
from single_reactive_agent import ReactiveAntAgent
from single_deliberative_agent import DeliberativeAntAgent
from single_random_agent import RandomAntAgent
from single_deliberative_agent import DeliberativeAntAgent

SEED_MULTIPLIER = 1 

N_ACTIONS = 12
DOWN, LEFT, UP, RIGHT, STAY, DOWN_PHERO, LEFT_PHERO, UP_PHERO, RIGHT_PHERO, COLLECT_FOOD, DROP_FOOD, COLLECT_FOOD_FROM_ANT = range(N_ACTIONS)

N_ROLES = 2
GO_HELP, GO_WORK = range(N_ROLES)

ROLES = {
    0: "GO_HELP",
    1: "GO_WORK",
}


N_POSSIBLE_DESIRES = 4
GO_TO_COLONY, EXPLORE, FIND_FOODPILE, HELP_ANT = range(N_POSSIBLE_DESIRES)

DESIRE_MEANING = {
    0: "GO_TO_COLONY",
    1: "EXPLORE",
    2: "FIND_FOODPILE",
    3: "HELP_ANT",
}

def manhattan_distance(point1, point2):
    distance = 0

    for x1, x2 in zip(point1, point2):
        difference = x2 - x1
        absolute_difference = abs(difference)
        distance += absolute_difference

    return distance

def run_multi_agent_roles(environment: Env, n_episodes: int) -> np.ndarray:
    
    results = {}

    for episode in range(n_episodes):
        
        teams = {

            "Role Team": [
                RoleAgent(agent_id=0, n_agents=4, roles = ROLES),
                RoleAgent(agent_id=1, n_agents=4, roles = ROLES),
                RoleAgent(agent_id=2, n_agents=4, roles =  ROLES),
                RoleAgent(agent_id=3, n_agents=4, roles = ROLES),
            ],
         
        }

        print(f"Episode {episode}")

        results_ep = np.zeros(n_episodes)

        for team, agents in teams.items():
            steps = 0
            terminals = [False for _ in range(len(agents))]
            environment.seed((episode + 1) * SEED_MULTIPLIER) # we use this seed so for each episode the map is equal for every team
            observations = environment.reset()

            while not all(terminals):
                steps += 1
                
                for observations, agent in zip(observations, agents):
                    agent.see(observations)

                actions = [agent.action() for agent in agents]
                
                next_observations, rewards, terminals, info = environment.step(actions)

                environment.render() # ENABLE/DISABLE THIS TO VIEW ENVIRONMENT
                time.sleep(opt.render_sleep_time)

                observations = next_observations

            environment.draw_heat_map(episode, team)

            results_ep[episode] = steps

            environment.close()

        results[team] = results_ep

    return results



class RoleAgent(AntAgent):

    def __init__(self, agent_id, n_agents, roles = ROLES, knowledgeable=True, role_assign_period: int = 1):
        super(RoleAgent, self).__init__(f"Role-based Agent", agent_id, n_agents, knowledgeable=True)
        self.agent_id = agent_id
        self.n_agents = n_agents
        self.roles = roles
        self.role_assign_period = role_assign_period
        self.curr_role = None
        self.steps_counter = 0
        self.steps_carrying_food = 0
        self.desire = None


    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################

    def _knowledgeable_deliberative(self): # The agent knows its own global position and the colony's position

        # BELIEFS
        agent_position, colony_position, foodpiles_in_view, pheromones_in_view, colony_storage, has_food, food_quantity, other_agents_in_view = self.observation_setup()

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
                action = self.go_to_colony(agent_position, colony_position, has_food, food_quantity) # move there

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
                    if(action == STAY):
                        action = self.explore_randomly()

                elif(self.check_for_intense_pheromones_in_view(pheromones_in_view)): # check for high intensity pheromones

                    self.promising_pheromone_pos = self.identify_most_intense_pheromone(agent_position, pheromones_in_view)

                    action = self.knowledgeable_examine_promising_pheromones(agent_position, pheromones_in_view, colony_position)

                else: # if we don't have high intensity pheromones in view...
                    action = self.explore_randomly() # we are still desiring to find food but need to pick an action! -> explore to find pheromones/foodpiles

        elif(self.desire == HELP_ANT):
            # find closest ant in need of help
            # if close enough
                # action = collect food from ant
            # else
                # action = go to ant
        

            if(self.check_for_other_ants_in_view(other_agents_in_view)):  # we have an agent in view...
                action, closest_ant_pos = self.go_to_closest_ant(agent_position, other_agents_in_view)
        
                if(self.check_if_destination_reached(agent_position, closest_ant_pos)):
                    action = COLLECT_FOOD_FROM_ANT
                    self.desire = None # desire accomplished, find a new desire
            else:
                self.desire = FIND_FOODPILE

        # Avoid obstacles
        if(action != STAY and action != COLLECT_FOOD and action != COLLECT_FOOD and action != COLLECT_FOOD_FROM_ANT):
            action = self.avoid_obstacles(action, agent_position, colony_position, foodpiles_in_view, other_agents_in_view)

        return action
    
        
        

    
    

    
    # ################# #
    # Auxiliary Methods #
    # ################# #

    def express_desire(self):
        if(self.desire == None):
            print("\tDesire: None")
        else:
            print(f"\tDesire: {DESIRE_MEANING[self.desire]}")

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
            
        self.following_trail = True

        action = self.direction_to_go(agent_position, self.promising_pheromone_pos, False, 0)

        if(action == STAY): # this avoids ants getting in infinite loop
            action = self.explore_randomly()

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

            if(not any(surrounding_pheromones)): # if there aren't any surrounding pheromones, we lost the trail..
                self.following_trail = False
                self.promising_pheromone_pos = None
                self.desire = EXPLORE
                action = self.explore_randomly()
                return action

            # If there are noteworthy pheromones, we want to find the minimun non null value
            next_promising_pheromone =  np.argmin(surrounding_pheromones[np.where(surrounding_pheromones > 0)])

            self.promising_pheromone_pos = self.find_global_pos(agent_position, surrounding_pheromones[next_promising_pheromone])

            if(self.promising_pheromone_pos == None): # lost trail... 
                self.following_trail = False
                self.desire = EXPLORE
                action = self.explore_randomly()
                return action
            

        action = self.direction_to_go(agent_position, self.promising_pheromone_pos, False)
        return action
    

    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################

    

    def closest_carrying_food_ant(self, agent_position, other_agents_in_view):
            #other_agents_in_view_final = del other_agents_in_view[0]
            other_agents_in_view_copy = np.copy(other_agents_in_view)
            other_agents_in_view_copy[12] = 0
            other_agents_indices = np.where(other_agents_in_view_copy == 2)[0] # gather for non null indices

            # Get corresponding positions in array format
            other_agents_positions = np.zeros(len(other_agents_indices) * 2)


            for other_agent_i in range(len(other_agents_indices)):
                other_agent_i_position = self.find_global_pos(agent_position, other_agents_indices[other_agent_i])
                other_agents_positions[other_agent_i * 2] = other_agent_i_position[0]
                other_agents_positions[other_agent_i * 2 + 1] = other_agent_i_position[1]

            # Check closest foodpile position and move there
            closest_other_agent_position = self.closest_point_of_interest(agent_position, other_agents_positions)

        
            return closest_other_agent_position
            #return DeliberativeAntAgent.direction_to_go(agent_position, closest_other_agent_position, False, 0), closest_other_agent_position
    

    def closest_foodpile_position(self, agent_position, foodpiles_in_view):
        foodpiles_indices = np.where(foodpiles_in_view != 0)[0] # gather for non null indices

        # Get corresponding positions in array format
        foodpiles_positions = np.zeros(len(foodpiles_indices) * 2)


        for foodpile_i in range(len(foodpiles_indices)): 
            foodpile_i_pos = self.find_global_pos(agent_position, foodpiles_indices[foodpile_i])
            foodpiles_positions[foodpile_i * 2] = foodpile_i_pos[0]
            foodpiles_positions[foodpile_i * 2 + 1] = foodpile_i_pos[1]

        # Check closest foodpile position and move there
        closest_foodpile_position = self.closest_point_of_interest(agent_position, foodpiles_positions)

        return closest_foodpile_position


    def potential_function(self, agent_position, role, other_agents_in_view, colony_position, foodpiles_in_view):
        agent_position, colony_position, foodpiles_in_view, pheromones_in_view, colony_storage, has_food,food_quantity, other_agents_in_view = self.observation_setup()
        closest_foodpile = self.closest_foodpile_position(agent_position, foodpiles_in_view)
        closest_ant = self.closest_carrying_food_ant(agent_position, other_agents_in_view)

        if role == GO_HELP:
            # pote
            if(closest_ant == None or has_food == True):
                potential = -100
            else:
                potential =  - manhattan_distance(agent_position,closest_ant)

        elif role == GO_WORK:
            # potential is equal to the distance to the closest foodpile
            if(closest_foodpile == None):
                potential = -50

            else:
                potential =  0
        
        print(f"Agent {self.agent_id} - Role {role} - Potential {potential}")

        return potential


    def role_assignment(self):
        """
        Given the observation vector containing the positions of all predators
        and the prey(s), compute the role-assignment for each of the agents.
        :return: a list with the role assignment for each of the agents
        """
        #agent_positions = self.observation[: self.n_agents * 2]
        #prey_positions = self.observation[self.n_agents * 2 : self.n_agents * 2 + 2]
        #target_adj_locs = self.get_target_adj_locs(prey_positions)

        agent_position = self.observation[:2]
        colony_position = self.observation[2:4]  # FOR ONLY 1 COLONY

        foodpiles_in_view = self.observation[4:29]
        pheromones_in_view = self.observation[29:54]

        colony_storage = self.observation[54]  # FOR ONLY 1 COLONY
        has_food = any([self.observation[55]])
        food_quantity = self.observation[55]

        other_agents_in_view = self.observation[56:]
        print(f"Observations {self.observation}")

        free_agents = [True] * self.n_agents

        role_assignment = []
        #mudar o loop , iterar sobre os agentes primeiro e nao sobre os roles

        agents_potentials = []
        for role_i in range(len(self.roles)):
            # Calculate agent-role pair potential
            agent_i_potential = self.potential_function(agent_position, role_i, other_agents_in_view, colony_position, foodpiles_in_view)
            agents_potentials.append(agent_i_potential)
            
        max_agent_potential_i = agents_potentials.index(max(agents_potentials))

        self.curr_role = max_agent_potential_i
        
        print (f"Role assignment {self.curr_role}")

        return role_assignment


    def action(self) -> int:
        agent_position, colony_position, foodpiles_in_view, pheromones_in_view, colony_storage, has_food, food_quantity, other_agents_in_view = self.observation_setup()

        # Compute potential-based role assignment every `role_assign_period` steps.
        if self.curr_role is None or self.steps_counter % self.role_assign_period == 0:
            role_assignments = self.role_assignment()
            

        self.steps_counter += 1
        if food_quantity == 2:
            self.steps_carrying_food += 1

        action = self.choose_action()
        
        print (f"Action {action}")

        return action

    def choose_action(self):
        print(f"desire: {self.desire}")
        agent_position, colony_position, foodpiles_in_view, pheromones_in_view, colony_storage, has_food, food_quantity, other_agents_in_view = self.observation_setup()
        if (self.curr_role == 0 and has_food == 0): 
            self.desire = HELP_ANT
            return self._knowledgeable_deliberative()
            
        else:
           
            return self._knowledgeable_deliberative()
            

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--episodes", type=int, default=1) # CHANGE THIS (n_episodes)
    parser.add_argument("--render-sleep-time", type=float, default=0.5)
    opt = parser.parse_args()

    # 1 - Setup the environment
    environment = AntColonyEnv(grid_shape=(25, 25), n_agents=4, max_steps=100, n_foodpiles=5)

    result = run_multi_agent_roles(environment, opt.episodes)

    

