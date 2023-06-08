import copy
import logging
import random

import numpy as np
logger = logging.getLogger(__name__)

from PIL import ImageColor, Image
import gym
from gym import spaces
from gym.utils import seeding

from ma_gym.envs.utils.action_space import MultiAgentActionSpace
from ma_gym.envs.utils.draw import draw_grid, fill_cell, draw_circle, write_cell_text
from ma_gym.envs.utils.observation_space import MultiAgentObservationSpace

class AntColonyEnv(gym.Env):

    """A simplified version of ma_gym.envs.predator_prey.predator_prey.PredatorPrey
    Observations do not take into account the nearest cells and an extra parameter (required_captors) was added
    """

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, grid_shape=(5, 5), n_agents=2, full_observable=False, penalty=-0.5, step_cost=-0.01, max_steps=100,
                 n_foodpiles=3, foodpile_capture_reward=5, initial_foodpile_capacity=8, foodpile_capacity_decrement=2,
                 n_colonies=1, initial_colonies_storage=100, colonies_storage_decrement=1, colonies_storage_increment=20, colonies_deposit_reward=10,
                 initial_pheromone_intensity=5, food_pheromone_intensity=50, pheromone_evaporation_rate=1, n_episodes=100):
        
        self._grid_shape = grid_shape
        self.n_agents = n_agents
        self._max_steps = max_steps
        self._step_count = None
        self._penalty = penalty
        self._step_cost = step_cost
        self._agent_view_mask = (5, 5)

        # Heat map
        self.heat_map = [[0 for _ in range(self._grid_shape[0])] for row in range(self._grid_shape[1])]

        # Foodpiles
        self.n_foodpiles = n_foodpiles
        self.foodpile_depleted = None
        self.foodpile_pos = {_: None for _ in range(self.n_foodpiles)}
        self.initial_foodpile_capacity = initial_foodpile_capacity
        self.foodpile_capacity = {_: random.randrange(4, self.initial_foodpile_capacity, 2) for _ in range(self.n_foodpiles)}
        self.initial_foodpile_capacities = self.foodpile_capacity
        self.foodpile_capture_reward = foodpile_capture_reward
        self.foodpiles_done = False
        self.foodpile_capacity_decrement = foodpile_capacity_decrement

        # Has food flag
        self.has_food = [0 for _ in range(self.n_agents)]

        # Colonies
        self.n_colonies = n_colonies
        self.colonies_pos = {_: None for _ in range(self.n_colonies)}
        self.initial_colonies_storage = initial_colonies_storage
        self.colonies_storage_decrement = colonies_storage_decrement
        self.colonies_storage = {_: self.initial_colonies_storage for _ in range(self.n_colonies)}
        self.colonies_storage_increment = colonies_storage_increment
        self.colonies_deposit_reward = colonies_deposit_reward

        # Pheromones
        self.pheromones_in_grid = [[0 for _ in range(self._grid_shape[0])] for row in range(self._grid_shape[1])] # keep pheromone level for each grid cell
        self.initial_pheromone_intensity = initial_pheromone_intensity
        self.food_pheromone_intensity = food_pheromone_intensity
        self.pheromone_evaporation_rate = pheromone_evaporation_rate
        self.n_pheromone = 0

        self.action_space = MultiAgentActionSpace([spaces.Discrete(11) for _ in range(self.n_agents)])
        self.agent_pos = {_: None for _ in range(self.n_agents)}

        self._base_grid = self.__create_grid()  # with no agents
        self._full_obs = self.__create_grid()
        self._agent_dones = [False for _ in range(self.n_agents)]
        self.viewer = None
        self.full_observable = full_observable

        # agent pos (2), prey (25), step (1)
        mask_size = np.prod(self._agent_view_mask)
        self._obs_high = np.array([1., 1.] + [1.] * mask_size + [1.0], dtype=np.float32)
        self._obs_low = np.array([0., 0.] + [0.] * mask_size + [0.0], dtype=np.float32)
        if self.full_observable:
            self._obs_high = np.tile(self._obs_high, self.n_agents)
            self._obs_low = np.tile(self._obs_low, self.n_agents)
        self.observation_space = MultiAgentObservationSpace(
            [spaces.Box(self._obs_low, self._obs_high) for _ in range(self.n_agents)])

        self._total_episode_reward = None
        self.seed()

        self.n_episodes = n_episodes # added this

    def simplified_features(self):

        current_grid = np.array(self._full_obs)

        agent_pos = []
        for agent_id in range(self.n_agents):
            tag = f"A{agent_id + 1}"
            row, col = np.where(current_grid == tag)
            row = row[0]
            col = col[0]
            agent_pos.append((col, row))

        # Create tags for grids with foodpiles (possibly can be eliminated?)
        foodpile_pos = []
        for foodpile_id in range(self.n_foodpiles):
            if (not self.foodpile_depleted[foodpile_id]):
                tag = f"F{foodpile_id + 1}"
                row, col = np.where(current_grid == tag)
                row = row[0]
                col = col[0]
                foodpile_pos.append((col, row))

        # Create tags for grids with colonies
        colonies_pos = []
        for colonies_id in range(self.n_colonies):
            tag = f"C{colonies_id + 1}"
            row, col = np.where(current_grid == tag)
            row = row[0] 
            col = col[0]
            colonies_pos.append((col, row))
            
        # At each time step, the agent knows its own position and the colony's position
        features = np.array(agent_pos + colonies_pos).reshape(-1)

        return features

    def reset(self):
        self._total_episode_reward = [0 for _ in range(self.n_agents)]
        self.agent_pos = {}
        self.foodpile_pos = {}
        self.colonies_pos = {}

        self.pheromones_pos = {}

        self.__init_full_obs()
        self._step_count = 0
        self._agent_dones = [False for _ in range(self.n_agents)]
        
        # Reset heat map
        self.heat_map = [[0 for _ in range(self._grid_shape[0])] for row in range(self._grid_shape[1])]

        # Reset foodpiles
        self.foodpile_capacity = {_: random.randrange(4, self.initial_foodpile_capacity, 2) for _ in range(self.n_foodpiles)} 
        self.foodpile_depleted = [False for _ in range(self.n_foodpiles)]
        self.foodpiles_done = False

        # Reset pheromones in grid
        self.pheromones_in_grid = [[0 for _ in range(self._grid_shape[0])] for row in range(self._grid_shape[1])]

        # Reset colonies
        self.colonies_storage = {_: self.initial_colonies_storage for _ in range(self.n_colonies)} 

        # Reset food flag
        self.has_food = [0 for _ in range(self.n_agents)]

        # Concatenate observed environment to features
        observed_environment = self.get_agent_obs() # 77 for each agent
        features = self.simplified_features() # 2 for each agent + 2 for each colony

        separated_full_information = self.format_outgoing_observations(features, observed_environment)

        return separated_full_information

    def step(self, agents_action):
        self._step_count += 1
        rewards = [self._step_cost for _ in range(self.n_agents)]

        # Small penalty for dumb behavior (dropping food without having any)
        for agent_i in range(self.n_agents):
            if(agents_action[agent_i] == 10 and self.has_food[agent_i] == 0):
                rewards[agent_i] += self._penalty

            # Update heat map with current agent pos
            self.heat_map[self.agent_pos[agent_i][0]][self.agent_pos[agent_i][1]] += 1

        # Decrease intensity of pheromones
        for row in range(self._grid_shape[0]):
            for col in range(self._grid_shape[1]):

                if (self.pheromones_in_grid[col][row] > 0):
                   self.pheromones_in_grid[col][row] -= self.pheromone_evaporation_rate

                   if(self.pheromones_in_grid[col][row] < self.pheromone_evaporation_rate):
                        self.pheromones_in_grid[col][row] = 0
                        if('A' not in self._full_obs[col][row]):
                            self._full_obs[col][row] = PRE_IDS['empty'] # this needs to be switched

        for agent_i, action in enumerate(agents_action):
            if not (self._agent_dones[agent_i]):
                self.__update_agent_pos(agent_i, action) # this was also update for the pheromones

        # Update foodpiles
        for foodpile_i in range(self.n_foodpiles):
            if (not self.foodpile_depleted[foodpile_i]):

                # If there are enough agents nearby to capture the foodpile...
                ant_neighbour_count, surrounding_agents_i = self._neighbour_agents(self.foodpile_pos[foodpile_i])
                
                if ant_neighbour_count >= 1: # only takes 1 ant to capture piece of foodpile
                    for i in range(ant_neighbour_count): # if the surrounding agents don't have food and choose to collect food..
                        agent_i = surrounding_agents_i[i]
                        action = agents_action[agent_i]

                        if(self.has_food[agent_i] == 0 and action == 9):

                            # Reduce foodpile capacity
                            self.foodpile_capacity[foodpile_i] -= self.foodpile_capacity_decrement

                            if(self.foodpile_capacity[foodpile_i] < 1):
                                self.foodpile_depleted[foodpile_i] = True
                                row, col = self.foodpile_pos[foodpile_i]
                                self._full_obs[self.foodpile_pos[foodpile_i][0]][self.foodpile_pos[foodpile_i][1]] = PRE_IDS['empty']

                            # Rewards agent which got food
                            rewards[agent_i] += self.foodpile_capture_reward

                            # Signal flag
                            self.has_food[agent_i] = self.foodpile_capacity_decrement


        # Update colonies storage
        for colony_i in range(self.n_colonies):

            # Check what agents are near colony
            ant_neighbour_count, surrounding_agents_i = self._neighbour_agents(self.colonies_pos[colony_i])

            if(ant_neighbour_count >= 1):
                for i in range(ant_neighbour_count):
                    agent_i = surrounding_agents_i[i]
                    action = agents_action[agent_i]

                if(self.has_food[agent_i] != 0 and action == 10): # if one of the surrounding agents decides to drop food...

                    # Increase colony storage
                    self.colonies_storage[colony_i] += self.colonies_storage_increment

                    # Rewards agent which got food
                    rewards[agent_i] += self.colonies_deposit_reward

                    # Signal flag
                    self.has_food[agent_i] = 0

            if(self.colonies_storage[colony_i] > 1):
                self.colonies_storage[colony_i] -= self.colonies_storage_decrement # We consider 1 to be the lowest food capacity possible (so 0 can mean ants can't see the colony)
                
        # If we have reached max steps, if every foodpile has been depleted (and the agents are not holding food), if a colony reaches min capacity, we should also stop
        if (self._step_count >= self._max_steps) or (False not in self.foodpile_depleted and not any(self.has_food)) or (1 in self.colonies_storage):
            for i in range(self.n_agents):
                self._agent_dones[i] = True

        for i in range(self.n_agents):
            self._total_episode_reward[i] += rewards[i]

        if (False not in self.foodpile_depleted): self.foodpiles_done = True

        observed_environment = self.get_agent_obs() # 77 for each agent
        features = self.simplified_features() # 2 for each agent + 2 for each colony

        separated_full_information = self.format_outgoing_observations(features, observed_environment)

        # [agent_pos colony_pos 25*foodpiles 25*pheromones colony_capacity has_food]

        return separated_full_information, rewards, self._agent_dones, {'foodpiles_done': self.foodpiles_done}

    def format_outgoing_observations(self, features, observed_environment):

        # Format the outgoing observations so they are separated by agent
        agents_positions = features[:self.n_agents * 2] # 2 * n_agents
        colonies_positions = features[-2 : ] # 1 COLONY

        #separated_full_information = np.zeros(len(full_information)).reshape(self.n_agents, 2 + 2 + 25 + 25 + 1 + 1 )

        separated_full_information = np.array([])

        for agent_id in range(self.n_agents):

            # Get corresponding information in previous arrays
            true_agent_id = agent_id * 2
            agent_position = agents_positions[true_agent_id:true_agent_id + 2] # 2
            colony_position = colonies_positions # 1 COLONY

            foodpiles_in_view_for_agent_i = observed_environment[agent_id][ : 25] # 25
            pheromones_in_view_for_agent_i = observed_environment[agent_id][25 : 50] # 25
            colony_storage_in_view = observed_environment[agent_id][50:51] # 1
            has_food_agent_i = observed_environment[agent_id][51:52] # 1
            other_agents_in_view = observed_environment[agent_id][52:] # 25

            # Concatenate information and add it to proper place 
            p1 = np.concatenate((agent_position, colony_position))
            p2 = np.concatenate((p1, foodpiles_in_view_for_agent_i))
            p3 = np.concatenate((p2, pheromones_in_view_for_agent_i))
            p4 = np.concatenate((p3, colony_storage_in_view))
            p5 = np.concatenate((p4, has_food_agent_i))
            
            agent_i_full_information = np.concatenate((p5, other_agents_in_view))

            if separated_full_information.size == 0:
                separated_full_information = np.array([agent_i_full_information])
            else:
                separated_full_information = np.append(separated_full_information, [agent_i_full_information], axis=0)

        # separated_full_information[agent_1] = [agent_pos colony_pos 25*foodpiles 25*pheromones colony_capacity has_food 25*other_agents]
        # separated_full_information[agent_id] = [separated_full_information[agent_1] separated_full_information[agent_2] ...]

        return separated_full_information

    def get_action_meanings(self, agent_i=None):
        if agent_i is not None:
            assert agent_i <= self.n_agents
            return [ACTION_MEANING[i] for i in range(self.action_space[agent_i].n)]
        else:
            return [[ACTION_MEANING[i] for i in range(ac.n)] for ac in self.action_space]

    def action_space_sample(self):
        return [agent_action_space.sample() for agent_action_space in self.action_space]

    def __draw_base_img(self):
        self._base_img = draw_grid(self._grid_shape[0], self._grid_shape[1], cell_size=CELL_SIZE, fill=GROUND_COLOR)

    def __create_grid(self):
        _grid = [[PRE_IDS['empty'] for _ in range(self._grid_shape[1])] for row in range(self._grid_shape[0])]
        return _grid

    def __init_full_obs(self):
        self._full_obs = self.__create_grid()

        for agent_i in range(self.n_agents):
            while True:
                pos = [self.np_random.randint(0, self._grid_shape[0] - 1),
                       self.np_random.randint(0, self._grid_shape[1] - 1)]
                if self._is_cell_spawnable(pos):
                    self.agent_pos[agent_i] = pos
                    break
            self.__update_agent_view(agent_i)

        # Randomly choose positions for foodpiles
        for foodpile_i in range(self.n_foodpiles):
            while True:
                pos = [self.np_random.randint(0, self._grid_shape[0] - 1),
                        self.np_random.randint(0, self._grid_shape[1] - 1)]
                if self._is_cell_vacant(pos) and (self._neighbour_agents(pos)[0] == 0):
                    self.foodpile_pos[foodpile_i] = pos
                    break
            self._full_obs[self.foodpile_pos[foodpile_i][0]][self.foodpile_pos[foodpile_i][1]] = PRE_IDS['foodpile'] + str(foodpile_i + 1)

        # Randomly choose positions for colonies
        for colony_i in range(self.n_colonies):
            while True:
                pos = [self.np_random.randint(0, self._grid_shape[0] - 1),
                        self.np_random.randint(0, self._grid_shape[1] - 1)]
                if self._is_cell_vacant(pos) and (self._neighbour_agents(pos)[0] == 0):
                    self.colonies_pos[colony_i] = pos
                    break
            self._full_obs[self.colonies_pos[colony_i][0]][self.colonies_pos[colony_i][1]] = PRE_IDS['colony'] + str(colony_i + 1)

        self.__draw_base_img()

    def get_agent_obs(self):
        _obs = []

        for agent_i in range(self.n_agents):
            pos = self.agent_pos[agent_i]
            #_agent_i_obs = [pos[0] / (self._grid_shape[0] - 1), pos[1] / (self._grid_shape[1] - 1)]  # coordinates

            # check if foodpile is in the view area
            _foodpile_pos = np.zeros(self._agent_view_mask)  # foodpile location in neighbour
            for row in range(max(0, pos[0] - 2), min(pos[0] + 2 + 1, self._grid_shape[0])):
                for col in range(max(0, pos[1] - 2), min(pos[1] + 2 + 1, self._grid_shape[1])):
                    if PRE_IDS['foodpile'] in self._full_obs[row][col]:
                        foodpile_i = int(self._full_obs[row][col].strip('F')) - 1 # from F3 to 2
                        _foodpile_pos[row - (pos[0] - 2), col - (pos[1] - 2)] = self.foodpile_capacity[foodpile_i]  # get relative position for the foodpile loc.
                    
            # check if pheromones is in the view area
            _pheromone_pos = np.zeros(self._agent_view_mask)  # pheromone location in neighbour
            for row in range(max(0, pos[0] - 2), min(pos[0] + 2 + 1, self._grid_shape[0])):
                for col in range(max(0, pos[1] - 2), min(pos[1] + 2 + 1, self._grid_shape[1])):
                    if PRE_IDS['pheromone'] in self._full_obs[row][col]:
                        _pheromone_pos[row - (pos[0] - 2), col - (pos[1] - 2)] = self.pheromones_in_grid[row][col]  # get relative position for the pheromone loc.

            # check if colony is in the view area
            _colonies_storage = np.zeros(self.n_colonies)  # colony
            for row in range(max(0, pos[0] - 2), min(pos[0] + 2 + 1, self._grid_shape[0])):
                for col in range(max(0, pos[1] - 2), min(pos[1] + 2 + 1, self._grid_shape[1])):
                    if PRE_IDS['colony'] in self._full_obs[row][col]:
                        colony_i = int(self._full_obs[row][col].strip('C')) - 1 # from C1 to 0
                        _colonies_storage[colony_i] = self.colonies_storage[colony_i] # get relative position for the colony loc.

            #check if other agents are in the view area
            _other_agents_pos = np.zeros(self._agent_view_mask)
            for row in range(max(0, pos[0] - 2), min(pos[0] + 2 + 1, self._grid_shape[0])):
                for col in range(max(0, pos[1] - 2), min(pos[1] + 2 + 1, self._grid_shape[1])):
                    if PRE_IDS['agent'] in self._full_obs[row][col]:
                        _other_agent_i = int(self._full_obs[row][col].strip('A')) - 1 # from A1 to 0    
                        _other_agents_pos[row - (pos[0] - 2), col - (pos[1] - 2)] = self.has_food[_other_agent_i] # get relative position for the agent loc.

        
            _agent_i_obs = _foodpile_pos.flatten().tolist()  # adding foodpile pos in observable area
            _agent_i_obs += _pheromone_pos.flatten().tolist()  # adding pheromone pos in observable area
            _agent_i_obs += _colonies_storage.flatten().tolist()  # adding colonies pos in observable area
            _agent_i_obs += [self.has_food[agent_i]] # adding has_food flag in observable area
            _agent_i_obs += _other_agents_pos.flatten().tolist() #adding neighbours positions to agent obs

            #_agent_i_obs += [self._step_count / self._max_steps]  # adding time

            _obs.append(_agent_i_obs)

        if self.full_observable:
            _obs = np.array(_obs).flatten().tolist()
            _obs = [_obs for _ in range(self.n_agents)]

        return _obs # [[_agent_1_obs] [_agent_2_obs] ...]

    def __wall_exists(self, pos):
        row, col = pos
        return PRE_IDS['wall'] in self._base_grid[row, col]

    def is_valid(self, pos):
        return (0 <= pos[0] < self._grid_shape[0]) and (0 <= pos[1] < self._grid_shape[1])
    
    def _is_cell_spawnable(self, pos):
        return self.is_valid(pos) and (self._full_obs[pos[0]][pos[1]] == PRE_IDS['empty'])

    def _is_cell_walkable(self, pos):
        return self.is_valid(pos) and ((self._full_obs[pos[0]][pos[1]] == PRE_IDS['empty']) or (self._full_obs[pos[0]][pos[1]] == PRE_IDS['pheromone']))
    
    def _is_cell_vacant(self, pos):
        return self.is_valid(pos) and (self._full_obs[pos[0]][pos[1]] == PRE_IDS['empty'])

    def __update_agent_pos(self, agent_i, move):

        curr_pos = copy.copy(self.agent_pos[agent_i])
        next_pos = None
        if move == 0:  # down
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
        elif move == 1:  # left
            next_pos = [curr_pos[0], curr_pos[1] - 1]
        elif move == 2:  # up
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
        elif move == 3:  # right
            next_pos = [curr_pos[0], curr_pos[1] + 1]
        elif move == 4:  # no-op
            pass
        elif move == 5: # down pheromone (high intensity pheromone)
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
        elif move == 6: # left pheromone (high intensity pheromone)
            next_pos = [curr_pos[0], curr_pos[1] - 1]
        elif move == 7: # up pheromone (high intensity pheromone)
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
        elif move == 8: # right pheromone (high intensity pheromone)
            next_pos = [curr_pos[0], curr_pos[1] + 1]
        elif move == 9: # collect food
            pass
        elif move == 10: # drop food
            pass
        else:
            raise Exception('Action Not found!')
        
        # Make sure we can't perform certain actions if we don't have/have food
        if((move == 5 or move == 6 or move == 7 or move == 8 or move == 10) and self.has_food[agent_i] == 0):
            next_pos = None
        elif((move == 9) and self.has_food[agent_i] != 0):
            next_pos = None
        
        # For COLLECT_FOOD (9)
        # Decrement foodpile is done in step
        # Has food is changed to true in step

        # For DROP_FOOD (10)
        # Increment colony storage is done in step
        # Has food is changed to false in step

        if next_pos is not None and self._is_cell_walkable(next_pos):

            if(move != 9 and move != 10): # movement happens

                self.agent_pos[agent_i] = next_pos
                
                # Add pheromones to last location
                self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty'] # now the last position is going to have the pheromone tag instead of empty

                self.__update_agent_view(agent_i) # this should always happen to prevent pheromone + NOOP => empy cell with agent in there ;(

                if(move == 5 or move == 6 or move == 7 or move == 8):
                    self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['pheromone']
                    self.pheromones_in_grid[curr_pos[0]][curr_pos[1]] += self.food_pheromone_intensity # currently stacks pheromones

    def __update_agent_view(self, agent_i):
        self._full_obs[self.agent_pos[agent_i][0]][self.agent_pos[agent_i][1]] = PRE_IDS['agent'] + str(agent_i + 1)

    def _neighbour_agents(self, pos):
        # check if agent is in neighbour
        _count = 0
        neighbours_xy = []
        if self.is_valid([pos[0] + 1, pos[1]]) and PRE_IDS['agent'] in self._full_obs[pos[0] + 1][pos[1]]:
            _count += 1
            neighbours_xy.append([pos[0] + 1, pos[1]])
        if self.is_valid([pos[0] - 1, pos[1]]) and (PRE_IDS['agent'] in self._full_obs[pos[0] - 1][pos[1]]):
            _count += 1
            neighbours_xy.append([pos[0] - 1, pos[1]])
        if self.is_valid([pos[0], pos[1] + 1]) and PRE_IDS['agent'] in self._full_obs[pos[0]][pos[1] + 1]:
            _count += 1
            neighbours_xy.append([pos[0], pos[1] + 1])
        if self.is_valid([pos[0], pos[1] - 1]) and PRE_IDS['agent'] in self._full_obs[pos[0]][pos[1] - 1]:
            neighbours_xy.append([pos[0], pos[1] - 1])
            _count += 1

        agent_id = []
        for x, y in neighbours_xy:
            agent_id.append(int(self._full_obs[x][y].split(PRE_IDS['agent'])[1]) - 1)
        return _count, agent_id

    def __get_neighbour_coordinates(self, pos):
        neighbours = []
        if self.is_valid([pos[0] + 1, pos[1]]):
            neighbours.append([pos[0] + 1, pos[1]])
        if self.is_valid([pos[0] - 1, pos[1]]):
            neighbours.append([pos[0] - 1, pos[1]])
        if self.is_valid([pos[0], pos[1] + 1]):
            neighbours.append([pos[0], pos[1] + 1])
        if self.is_valid([pos[0], pos[1] - 1]):
            neighbours.append([pos[0], pos[1] - 1])
        return neighbours

    def render_heat_map(self, mode='rgb_array'):
        heat_map_img = copy.copy(self._base_img)
        
        for row in range(self._grid_shape[0]):
            for col in range(self._grid_shape[1]):
                # Draw heat map values
                fill_cell(heat_map_img, [col, row], cell_size=CELL_SIZE, fill=color_lerp(HEAT_MAP_BASE_COLOR, HEAT_MAP_MAX_COLOR, self.heat_map[col][row]/20), margin=0.1)
                if(self.heat_map[col][row]!=0):
                    write_cell_text(heat_map_img, text=str(self.heat_map[col][row]), pos=[col, row], cell_size=CELL_SIZE, fill='white', margin=0.4)
        
        # Draw colonies position
        for colony_i in range(self.n_colonies):
            fill_cell(heat_map_img, self.colonies_pos[colony_i], cell_size=CELL_SIZE, fill=COLONY_COLOR, margin=0.1)
            write_cell_text(heat_map_img, text="C" + str(colony_i), pos=self.colonies_pos[colony_i], cell_size=CELL_SIZE, fill='white', margin=0.4)

        # Draw foodpiles position
        for foodpile_i in range(self.n_foodpiles):
            fill_cell(heat_map_img, self.foodpile_pos[foodpile_i], cell_size=CELL_SIZE, fill=FOOD_COLOR, margin=0.1)
            write_cell_text(heat_map_img, text=str(self.initial_foodpile_capacities[foodpile_i]), pos=self.foodpile_pos[foodpile_i], cell_size=CELL_SIZE,
                               fill='white', margin=0.4)

        
        heat_map_img = np.asarray(heat_map_img)
        if mode == 'rgb_array':
            return heat_map_img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(heat_map_img)
            return self.viewer.isopen
        else:
            raise NotImplementedError 

    def render(self, mode='human'):
        img = copy.copy(self._base_img)

        for row in range(self._grid_shape[0]):
            for col in range(self._grid_shape[1]):

                # Draw pheromones
                if(self.pheromones_in_grid[col][row] >= self.pheromone_evaporation_rate):
                    pheromone_i = self.pheromones_in_grid[col][row]
                    pheromone_pos = [col, row]
                    fill_cell(img, pheromone_pos, cell_size=CELL_SIZE, fill=color_lerp(GROUND_COLOR, PHEROMONE_COLOR, pheromone_i/self.food_pheromone_intensity), margin=0.1)
                    write_cell_text(img, text=str(pheromone_i), pos=pheromone_pos, cell_size=CELL_SIZE,
                            fill='white', margin=0.4)

        # Agent neighborhood render
        for agent_i in range(self.n_agents):
            for neighbour in self.__get_neighbour_coordinates(self.agent_pos[agent_i]):
                fill_cell(img, neighbour, cell_size=CELL_SIZE, fill=AGENT_NEIGHBORHOOD_COLOR, margin=0.1)
            fill_cell(img, self.agent_pos[agent_i], cell_size=CELL_SIZE, fill=AGENT_NEIGHBORHOOD_COLOR, margin=0.1)

        # Agent render
        for agent_i in range(self.n_agents):

            if(self.has_food[agent_i] != 0): ant_color = AGENT_WITH_FOOD_COLOR # ant is purple when carrying food
            else: ant_color = AGENT_COLOR # ant is normally black

            draw_circle(img, self.agent_pos[agent_i], cell_size=CELL_SIZE, fill=ant_color)
            write_cell_text(img, text=str(agent_i + 1), pos=self.agent_pos[agent_i], cell_size=CELL_SIZE,
                            fill='white', margin=0.4)

        # Foodpiles render  
        for foodpile_i in range(self.n_foodpiles):
            if (self.foodpile_depleted[foodpile_i] == False):
                fill_cell(img, self.foodpile_pos[foodpile_i], cell_size=CELL_SIZE, fill=FOOD_COLOR, margin=0.1)

                write_cell_text(img, text=str(self.foodpile_capacity[foodpile_i]), pos=self.foodpile_pos[foodpile_i], cell_size=CELL_SIZE,
                               fill='white', margin=0.4)
        
        # Colonies render 
        for colony_i in range(self.n_colonies):
            fill_cell(img, self.colonies_pos[colony_i], cell_size=CELL_SIZE, fill=COLONY_COLOR, margin=0.1)
            write_cell_text(img, text=str(self.colonies_storage[colony_i]), pos=self.colonies_pos[colony_i], cell_size=CELL_SIZE,
                           fill='white', margin=0.4)

            #write_cell_text(img, text=str(colony_i + 1), pos=self.colonies_pos[colony_i], cell_size=CELL_SIZE,
            #               fill='white', margin=0.4)


        # UNCOMMENT TO VIEW TAGS
        #for row in range(self._grid_shape[0]):
        #    for col in range(self._grid_shape[1]):
        #        write_cell_text(img, text=str(self._full_obs[col][row]), pos=[col, row], cell_size=CELL_SIZE,
        #                    fill='white', margin=0.4)

        img = np.asarray(img)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen
        

    def draw_heat_map(self, curr_episode, team):
        # Check if we are in the first, middle and last episodes to save the heat map
            if(curr_episode == 0 or curr_episode == self.n_episodes/2 or curr_episode == self.n_episodes-1):
                img_heat_map = Image.fromarray(self.render_heat_map(mode='rgb_array'))
                img_heat_map.save('heat_map_' + team + '_' + str(curr_episode) + '.png')

    def seed(self, n=None):
        self.np_random, seed = seeding.np_random(n)
        return [seed]

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


AGENT_COLOR = ImageColor.getcolor('black', mode='RGB')
AGENT_WITH_FOOD_COLOR = 'purple'
AGENT_NEIGHBORHOOD_COLOR = (240, 240, 10)
FOOD_COLOR = 'green'
COLONY_COLOR = 'sienna'
PHEROMONE_COLOR = (10, 240, 240)

GROUND_COLOR = (205, 133, 63)
WALL_COLOR = 'black'

HEAT_MAP_BASE_COLOR = (205, 133, 63)
HEAT_MAP_MAX_COLOR = (255, 0, 0)

CELL_SIZE = 35

ACTION_MEANING = {
    0: "DOWN",
    1: "LEFT",
    2: "UP",
    3: "RIGHT",
    4: "NOOP",
    5: "DOWN_PHERO",
    6: "LEFT_PHERO",
    7: "UP_PHERO",
    8: "RIGHT_PHERO",
    9: "COLLECT_FOOD",
    10: "DROP_FOOD"
}

PRE_IDS = {
    'agent': 'A',
    'wall': 'W',
    'empty': '0',
    'foodpile': 'F',
    'colony': 'C',
    'pheromone': 'I'
}

def color_lerp(color_1, color_2, steps):
    color_1 = np.asarray(color_1)
    color_2 = np.asarray(color_2)
    final_color = color_1 * (1 - steps) + color_2 * steps
    return (int(final_color[0]), int(final_color[1]), int(final_color[2]))
