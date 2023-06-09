import time
import argparse
import numpy as np
from scipy.spatial.distance import cityblock
from gym import Env

from single_deliberative_agent import DeliberativeAntAgent
from aasma.wrappers import SingleAgentWrapper
from aasma.simplified_predator_prey import AntColonyEnv

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

def run_single_agent(environment: Env, n_episodes: int) -> np.ndarray:

    results = np.zeros(n_episodes)

    for episode in range(n_episodes):

        # Setup agent
        agent = RoleAntAgent(agent_id=0, n_agents=1, knowledgeable=True)

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

            RoleAntAgent.express_desire(agent)
            print(f"\tAction: {environment.get_action_meanings()[action]}\n")
            print(f"\tObservation: {observation}")

        environment.draw_heat_map(episode, "RoleAntAgent")

        environment.close()
        results[episode] = steps

    return results

class RoleAntAgent(DeliberativeAntAgent):
    def __init__(self, agent_id, n_agents, knowledgeable=True, role_assign_period: int = 1):
        super(RoleAntAgent, self).__init__(f"Role-based Agent", agent_id, n_agents)
        self.roles = ROLES
        self.role_assign_period = role_assign_period
        self.curr_role = None
        self.steps_counter = 0

    def action(self) -> int:
        action_to_perform = self._knowledgeable_deliberative()

        return action_to_perform

    def _knowledgeable_deliberative(self): # The agent knows its own global position and the colony's position

        # BELIEFS
        agent_position, colony_position, foodpiles_in_view, pheromones_in_view, colony_storage, has_food, food_quantity, other_agents_in_view = self.observation_setup()

        # Compute potential-based role assignment every `role_assign_period` steps.
        if self.curr_role is None or self.steps_counter % self.role_assign_period == 0:
            self.role_assignment()

        self.steps_counter += 1
        if food_quantity == 2:
            self.steps_carrying_food += 1

        if (self.curr_role == 0 and has_food == 0): 
            self.desire = HELP_ANT

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
                    action = self.knowledgeable_examine_promising_pheromones(agent_position, pheromones_in_view, colony_position, food_quantity)
                    if(action == STAY):
                        action = self.explore_randomly()

                elif(self.check_for_intense_pheromones_in_view(pheromones_in_view)): # check for high intensity pheromones

                    self.promising_pheromone_pos = self.identify_most_intense_pheromone(agent_position, pheromones_in_view)

                    action = self.knowledgeable_examine_promising_pheromones(agent_position, pheromones_in_view, colony_position, food_quantity)

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
                    action = self.knowledgeable_examine_promising_pheromones(agent_position, pheromones_in_view, colony_position, food_quantity)
                    if(action == STAY):
                        action = self.explore_randomly()

                elif(self.check_for_intense_pheromones_in_view(pheromones_in_view)): # check for high intensity pheromones

                    self.promising_pheromone_pos = self.identify_most_intense_pheromone(agent_position, pheromones_in_view)

                    action = self.knowledgeable_examine_promising_pheromones(agent_position, pheromones_in_view, colony_position, food_quantity)

                else: # if we don't have high intensity pheromones in view...
                    action = self.explore_randomly() # we are still desiring to find food but need to pick an action! -> explore to find pheromones/foodpiles


        # Avoid obstacles
        if(action != STAY and action != COLLECT_FOOD and action != COLLECT_FOOD and action != COLLECT_FOOD_FROM_ANT):
            action = self.avoid_obstacles(action, agent_position, colony_position, foodpiles_in_view, other_agents_in_view)

        return action

    # ################# #
    # Auxiliary Methods #
    # ################# #

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

        foodpiles_in_view = self.observation[4:29]

        other_agents_in_view = self.observation[56:]

        role_assignment = []

        agents_potentials = []
        for role_i in range(len(self.roles)):
            # Calculate agent-role pair potential
            agent_i_potential = self.potential_function(agent_position, role_i, other_agents_in_view, foodpiles_in_view)
            agents_potentials.append(agent_i_potential)
            
        max_agent_potential_i = agents_potentials.index(max(agents_potentials))

        self.curr_role = max_agent_potential_i

        return role_assignment

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

    def potential_function(self, agent_position, role, other_agents_in_view, foodpiles_in_view):
        agent_position, _, foodpiles_in_view, _, _, has_food, _, other_agents_in_view = self.observation_setup()
        closest_foodpile = self.closest_foodpile_position(agent_position, foodpiles_in_view)
        closest_ant = self.closest_carrying_food_ant(agent_position, other_agents_in_view)

        if role == GO_HELP:
            # pote
            if(closest_ant == None or has_food == True):
                potential = -100
            else:
                distance = 0

                for x1, x2 in zip(agent_position, closest_ant):
                    difference = x2 - x1
                    absolute_difference = abs(difference)
                    distance += absolute_difference
    
                potential =  - distance

        elif role == GO_WORK:
            # potential is equal to the distance to the closest foodpile
            if(closest_foodpile == None):
                potential = -50

            else:
                potential =  0

        return potential

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--render-sleep-time", type=float, default=0.01)
    opt = parser.parse_args()

    # Setup environment
    environment = AntColonyEnv(
        grid_shape=(10, 10),
        n_agents=1, 
        max_steps=100,
        n_foodpiles=3,
        n_episodes=opt.episodes
    )

    environment = SingleAgentWrapper(environment, agent_id=0)

    # Run single ant
    results = run_single_agent(environment, opt.episodes)
