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

N_ACTIONS = 11
DOWN, LEFT, UP, RIGHT, STAY, DOWN_PHERO, LEFT_PHERO, UP_PHERO, RIGHT_PHERO, COLLECT_FOOD, DROP_FOOD = range(N_ACTIONS)

N_ROLES = 2
GO_HELP, GO_WORK = range(N_ROLES)

ROLES = {
    0: "GO_HELP",
    1: "GO_WORK",
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

    

    def closest_carrying_food_ant(self, agent_position, other_agents_in_view):
            other_agents_indices = np.where(other_agents_in_view == 2)[0] # gather for non null indices

            # Get corresponding positions in array format
            other_agents_positions = np.zeros(len(other_agents_indices) * 2)


            for other_agent_i in range(len(other_agents_indices)):
                other_agent_i_position = DeliberativeAntAgent.find_global_pos(agent_position, other_agents_indices[other_agent_i])
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

        if role == GO_HELP:
            # potential is equal to the distance to the closest ant carrying food£
            # 
            closest_ant = self.closest_carrying_food_ant(agent_position, other_agents_in_view) 
            if(closest_ant == None):
                potential = 0
            else:
                potential = - manhattan_distance(agent_position,closest_ant)

        elif role == GO_WORK:
            # potential is equal to the distance to the closest foodpile
            closest_foodpile = self.closest_foodpile_position(agent_position, foodpiles_in_view)
            if(closest_foodpile == None):
                potential = 100
            else:
                potential = - manhattan_distance(agent_position, closest_foodpile)
        
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
        # Compute potential-based role assignment every `role_assign_period` steps.
        if self.curr_role is None or self.steps_counter % self.role_assign_period == 0:
            role_assignments = self.role_assignment()
            

        self.steps_counter += 1
        action = self.choose_action()

        return action

    def choose_action(self):
        if (self.curr_role == 0 and self.has_food == 0):
            direction_to_go = DeliberativeAntAgent.direction_to_go(self.agent_position, self.closest_carrying_food_ant(self.agent_position, self.other_agents_in_view), False, 0)
            if DeliberativeAntAgent.check_if_destination_reached(self.agent_position, direction_to_go):
                return COLLECT_FOOD
            

    def get_target_adj_locs(self, loc) -> list():
        target_x = loc[0]
        target_y = loc[1]
        return [
            (target_x, target_y - 1),
            (target_x, target_y + 1),
            (target_x - 1, target_y),
            (target_x + 1, target_y),
        ]


    def advance_to_pos(self, agent_pos: tuple, prey_pos: tuple, agent_dest: int) -> int:
        """
        Choose movement action to advance agent towards the destination around prey
        :param agent_pos: current agent position
        :param prey_pos: prey position
        :param agent_dest: agent destination in relation to prey (0 for NORTH, 1 for SOUTH,
                            2 for WEST, and 3 for EAST)
        :return: movement index
        """

        def _move_vertically(distances) -> int:
            if distances[1] > 0:
                return DOWN
            elif distances[1] < 0:
                return UP
            else:
                return STAY

        def _move_horizontally(distances) -> int:
            if distances[0] > 0:
                return RIGHT
            elif distances[0] < 0:
                return LEFT
            else:
                return STAY

        target_adj_locs = self.get_target_adj_locs(prey_pos)
        distance_dest = np.array(target_adj_locs[agent_dest]) - np.array(agent_pos)
        abs_distances = np.absolute(distance_dest)
        if abs_distances[0] > abs_distances[1]:
            return _move_horizontally(distance_dest)
        elif abs_distances[0] < abs_distances[1]:
            return _move_vertically(distance_dest)
        else:
            roll = np.random.uniform(0, 1)
            return _move_horizontally(distance_dest) if roll > 0.5 else _move_vertically(distance_dest)


    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--episodes", type=int, default=1) # CHANGE THIS (n_episodes)
    parser.add_argument("--render-sleep-time", type=float, default=0.5)
    opt = parser.parse_args()

    # 1 - Setup the environment
    environment = AntColonyEnv(grid_shape=(25, 25), n_agents=4, max_steps=100, n_foodpiles=5)

    result = run_multi_agent_roles(environment, opt.episodes)

    

