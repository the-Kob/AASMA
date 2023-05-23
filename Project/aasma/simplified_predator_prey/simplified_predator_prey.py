import copy
import logging
import random

import numpy as np

logger = logging.getLogger(__name__)

from PIL import ImageColor
import gym
from gym import spaces
from gym.utils import seeding

from ma_gym.envs.utils.action_space import MultiAgentActionSpace
from ma_gym.envs.utils.draw import draw_grid, fill_cell, draw_circle, write_cell_text
from ma_gym.envs.utils.observation_space import MultiAgentObservationSpace

class SimplifiedPredatorPrey(gym.Env):

    """A simplified version of ma_gym.envs.predator_prey.predator_prey.PredatorPrey
    Observations do not take into account the nearest cells and an extra parameter (required_captors) was added

    See Also
    --------
    ma_gym.envs.predator_prey

    """

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, grid_shape=(5, 5), n_agents=2, n_preys=1, prey_move_probs=(0.175, 0.175, 0.175, 0.175, 0.3),
                 full_observable=False, penalty=-0.5, step_cost=-0.01, prey_capture_reward=5, max_steps=100, required_captors=2,
                 n_foodpiles=3, foodpile_capture_reward=5, initial_foodpile_capacity=3, n_colonies=1, initial_pheromone_intensity=5, pheromone_evaporation_rate=1):
        
        self._grid_shape = grid_shape
        self.n_agents = n_agents
        self.n_preys = n_preys
        self._max_steps = max_steps
        self._step_count = None
        self._penalty = penalty
        self._step_cost = step_cost
        self._prey_capture_reward = prey_capture_reward
        self._agent_view_mask = (5, 5)
        self._required_captors = required_captors

        # Foodpiles
        self.n_foodpiles = n_foodpiles
        self.foodpile_depleted = None
        self.foodpile_pos = {_: None for _ in range(self.n_foodpiles)}
        self.initial_foodpile_capacity = initial_foodpile_capacity
        self.foodpile_capacity = {_: self.initial_foodpile_capacity for _ in range(self.n_foodpiles)}
        self.foodpile_capture_reward = foodpile_capture_reward

        # Colonies
        self.n_colonies = n_colonies
        self.colonies_pos = {_: None for _ in range(self.n_foodpiles)}

        # Pheromones
        self.pheromones_in_grid = [[0 for _ in range(self._grid_shape[0])] for row in range(self._grid_shape[1])] # keep pheromone level for each grid cell
        self.initial_pheromone_intensity = initial_pheromone_intensity
        self.pheromone_evaporation_rate = pheromone_evaporation_rate
        self.n_pheromone = 0

        self.action_space = MultiAgentActionSpace([spaces.Discrete(5) for _ in range(self.n_agents)])
        self.agent_pos = {_: None for _ in range(self.n_agents)}
        self.prey_pos = {_: None for _ in range(self.n_preys)}
        self._prey_alive = None

        self._base_grid = self.__create_grid()  # with no agents
        self._full_obs = self.__create_grid()
        self._agent_dones = [False for _ in range(self.n_agents)]
        self._prey_move_probs = prey_move_probs
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

    def simplified_features(self):

        current_grid = np.array(self._full_obs)

        agent_pos = []
        for agent_id in range(self.n_agents):
            tag = f"A{agent_id + 1}"
            row, col = np.where(current_grid == tag)
            row = row[0]
            col = col[0]
            agent_pos.append((col, row))

        prey_pos = []
        for prey_id in range(self.n_preys):
            if self._prey_alive[prey_id]:
                tag = f"P{prey_id + 1}"
                row, col = np.where(current_grid == tag)
                row = row[0]
                col = col[0]
                prey_pos.append((col, row))

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
            
        # At each time step, the agent knows its own position, the preys position (to deprecate) and the colony's position
        # Observation: [1 7 5 0 8 5] <-> [col_agent row_agent col_prey row_prey col_colony row_colony]
        features = np.array(agent_pos + colonies_pos).reshape(-1)

        return features

    def reset(self):
        self._total_episode_reward = [0 for _ in range(self.n_agents)]
        self.agent_pos = {}
        self.prey_pos = {}
        self.foodpile_pos = {} # added this
        self.colonies_pos = {} # added this

        self.pheromones_pos = {} # added this

        self.__init_full_obs()
        self._step_count = 0
        self._agent_dones = [False for _ in range(self.n_agents)]
        self._prey_alive = [True for _ in range(self.n_preys)]
        
        # Reset foodpiles
        self.foodpile_capacity = {_: self.initial_foodpile_capacity for _ in range(self.n_foodpiles)} # added this 
        self.foodpile_depleted = [False for _ in range(self.n_foodpiles)] # added this

        # Reset pheromones in grid
        self.pheromones_in_grid = [[0 for _ in range(self._grid_shape[0])] for row in range(self._grid_shape[1])]

        self.get_agent_obs()

        o = np.array(self.get_agent_obs())[0]
        lol = np.concatenate((self.simplified_features(), o))

        return [lol for _ in range(self.n_agents)]

    def step(self, agents_action):
        self._step_count += 1
        rewards = [self._step_cost for _ in range(self.n_agents)]

        # Decrease intensiy of pheromones
        for row in range(self._grid_shape[0]):
            for col in range(self._grid_shape[1]):

                if (self.pheromones_in_grid[col][row] > 0):
                   self.pheromones_in_grid[col][row] -= self.pheromone_evaporation_rate

                   if(self.pheromones_in_grid[col][row] < self.pheromone_evaporation_rate):
                        self.pheromones_in_grid[col][row] = 0
                        self._full_obs[col][row] = PRE_IDS['empty'] # this needs to be switched

        for agent_i, action in enumerate(agents_action):
            if not (self._agent_dones[agent_i]):
                self.__update_agent_pos(agent_i, action) # this was also update for the pheromones

        for prey_i in range(self.n_preys):
            if self._prey_alive[prey_i]:
                predator_neighbour_count, n_i = self._neighbour_agents(self.prey_pos[prey_i])

                if predator_neighbour_count >= self._required_captors:

                    _reward = self._penalty if predator_neighbour_count < self._required_captors else self._prey_capture_reward

                    self._prey_alive[prey_i] = (predator_neighbour_count < self._required_captors)

                    for agent_i in range(self.n_agents):
                        rewards[agent_i] += _reward

                prey_move = None
                if self._prey_alive[prey_i]:
                    prey_move = random.randrange(5)

                self.__update_prey_pos(prey_i, prey_move)

        # Capture foodpiles requirements
        for foodpile_i in range(self.n_foodpiles):
            if (not self.foodpile_depleted[foodpile_i]):
                predator_neighbour_count, n_i = self._neighbour_agents(self.foodpile_pos[foodpile_i])

                if predator_neighbour_count >= self._required_captors:

                    _reward = self._penalty if predator_neighbour_count < self._required_captors else self.foodpile_capture_reward

                    # Reduce foodpile capacity
                    self.foodpile_capacity[foodpile_i] -= 1
                    print("\nFoodpile " + str(foodpile_i) + " now has " + str(self.foodpile_capacity[foodpile_i]) + " food \n")

                    if(self.foodpile_capacity[foodpile_i] < 1):
                        self.foodpile_depleted[foodpile_i] = True
                        row, col = self.foodpile_pos[foodpile_i]
                        self._full_obs[self.foodpile_pos[foodpile_i][0]][self.foodpile_pos[foodpile_i][1]] = PRE_IDS['empty']
                        print("\nFoodpile " + str(foodpile_i) + " was entirely consumed \n")

                    for agent_i in range(self.n_agents):
                        rewards[agent_i] += _reward

        # If every foodpile has been depleted, we should also stop
        if (self._step_count >= self._max_steps) or (False not in self.foodpile_depleted):
            for i in range(self.n_agents):
                self._agent_dones[i] = True

        for i in range(self.n_agents):
            self._total_episode_reward[i] += rewards[i]

        o = np.array(self.get_agent_obs())[0]
        lol = np.concatenate((self.simplified_features(), o))

        return [ lol for _ in range(self.n_agents)], rewards, self._agent_dones, {'prey_alive': self._prey_alive}

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
                if self._is_cell_walkable(pos):
                    self.agent_pos[agent_i] = pos
                    break
            self.__update_agent_view(agent_i)

        for prey_i in range(self.n_preys):
            while True:
                pos = [self.np_random.randint(0, self._grid_shape[0] - 1),
                       self.np_random.randint(0, self._grid_shape[1] - 1)]
                if self._is_cell_walkable(pos) and (self._neighbour_agents(pos)[0] == 0):
                    self.prey_pos[prey_i] = pos
                    break
            self.__update_prey_view(prey_i)

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

            # check if prey is in the view area
            _prey_pos = np.zeros(self._agent_view_mask)  # prey location in neighbour
            for row in range(max(0, pos[0] - 2), min(pos[0] + 2 + 1, self._grid_shape[0])):
                for col in range(max(0, pos[1] - 2), min(pos[1] + 2 + 1, self._grid_shape[1])):
                    if PRE_IDS['prey'] in self._full_obs[row][col]:
                        _prey_pos[row - (pos[0] - 2), col - (pos[1] - 2)] = 1  # get relative position for the prey loc.

            # check if foodpile is in the view area
            _foodpile_pos = np.zeros(self._agent_view_mask)  # foodpile location in neighbour
            for row in range(max(0, pos[0] - 2), min(pos[0] + 2 + 1, self._grid_shape[0])):
                for col in range(max(0, pos[1] - 2), min(pos[1] + 2 + 1, self._grid_shape[1])):
                    if PRE_IDS['foodpile'] in self._full_obs[row][col]:
                        _foodpile_pos[row - (pos[0] - 2), col - (pos[1] - 2)] = 1  # get relative position for the foodpile loc.
                    
            # check if pheromones is in the view area
            _pheromone_pos = np.zeros(self._agent_view_mask)  # pheromone location in neighbour
            for row in range(max(0, pos[0] - 2), min(pos[0] + 2 + 1, self._grid_shape[0])):
                for col in range(max(0, pos[1] - 2), min(pos[1] + 2 + 1, self._grid_shape[1])):
                    if PRE_IDS['pheromone'] in self._full_obs[row][col]:
                        _pheromone_pos[row - (pos[0] - 2), col - (pos[1] - 2)] = 1  # get relative position for the foodpile loc.

        
            #_agent_i_obs += _prey_pos.flatten().tolist()  # adding prey pos in observable area
            _agent_i_obs = _foodpile_pos.flatten().tolist()  # adding foodpile pos in observable area
            _agent_i_obs += _pheromone_pos.flatten().tolist()  # adding pheromone pos in observable area
            #_agent_i_obs += [self._step_count / self._max_steps]  # adding time

            _obs.append(_agent_i_obs)

        if self.full_observable:
            _obs = np.array(_obs).flatten().tolist()
            _obs = [_obs for _ in range(self.n_agents)]
        return _obs

    def __wall_exists(self, pos):
        row, col = pos
        return PRE_IDS['wall'] in self._base_grid[row, col]

    def is_valid(self, pos):
        return (0 <= pos[0] < self._grid_shape[0]) and (0 <= pos[1] < self._grid_shape[1])

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
        else:
            raise Exception('Action Not found!')

        if next_pos is not None and self._is_cell_walkable(next_pos):
            self.agent_pos[agent_i] = next_pos

            # Add pheromones to last location
            self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['pheromone'] # now the last position is going to have the pheromone tag instead of empty
            self.pheromones_in_grid[curr_pos[0]][curr_pos[1]] = self.initial_pheromone_intensity # currently doesn't stack pheromones

        self.__update_agent_view(agent_i) # this should always happen to prevent pheromone + NOOP => empy cell with agent in there ;(

    def __update_prey_pos(self, prey_i, move):
        curr_pos = copy.copy(self.prey_pos[prey_i])
        if self._prey_alive[prey_i]:
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
            else:
                raise Exception('Action Not found!')

            if next_pos is not None and self._is_cell_walkable(next_pos):
                self.prey_pos[prey_i] = next_pos
                self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']
                self.__update_prey_view(prey_i)
            else:
                # print('pos not updated')
                pass
        else:
            self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']

    def __update_agent_view(self, agent_i):
        self._full_obs[self.agent_pos[agent_i][0]][self.agent_pos[agent_i][1]] = PRE_IDS['agent'] + str(agent_i + 1)

    def __update_prey_view(self, prey_i):
        self._full_obs[self.prey_pos[prey_i][0]][self.prey_pos[prey_i][1]] = PRE_IDS['prey'] + str(prey_i + 1)

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

    def render(self, mode='human'):
        img = copy.copy(self._base_img)

        # Draw pheromones 
        for row in range(self._grid_shape[0]):
            for col in range(self._grid_shape[1]):

                if(self.pheromones_in_grid[col][row] >= self.pheromone_evaporation_rate):
                    pheromone_i = self.pheromones_in_grid[col][row]
                    pheromone_pos = [col, row]
                    fill_cell(img, pheromone_pos, cell_size=CELL_SIZE, fill=color_lerp(GROUND_COLOR, PHEROMONE_COLOR, pheromone_i/self.initial_pheromone_intensity), margin=0.1)
                    write_cell_text(img, text=str(pheromone_i), pos=pheromone_pos, cell_size=CELL_SIZE,
                            fill='white', margin=0.4)  


        # Agent neighborhood render
        for agent_i in range(self.n_agents):
            for neighbour in self.__get_neighbour_coordinates(self.agent_pos[agent_i]):
                fill_cell(img, neighbour, cell_size=CELL_SIZE, fill=AGENT_NEIGHBORHOOD_COLOR, margin=0.1)
            fill_cell(img, self.agent_pos[agent_i], cell_size=CELL_SIZE, fill=AGENT_NEIGHBORHOOD_COLOR, margin=0.1)

        # Agent render
        for agent_i in range(self.n_agents):
            draw_circle(img, self.agent_pos[agent_i], cell_size=CELL_SIZE, fill=AGENT_COLOR)
            write_cell_text(img, text=str(agent_i + 1), pos=self.agent_pos[agent_i], cell_size=CELL_SIZE,
                            fill='white', margin=0.4)

        # Prey render
        for prey_i in range(self.n_preys):
            if self._prey_alive[prey_i]:
                draw_circle(img, self.prey_pos[prey_i], cell_size=CELL_SIZE, fill=PREY_COLOR)
                write_cell_text(img, text=str(prey_i + 1), pos=self.prey_pos[prey_i], cell_size=CELL_SIZE,
                                fill='white', margin=0.4)

        # Foodpiles render  
        for foodpile_i in range(self.n_foodpiles):
            if (self.foodpile_depleted[foodpile_i] == False):
                fill_cell(img, self.foodpile_pos[foodpile_i], cell_size=CELL_SIZE, fill=FOOD_COLOR, margin=0.1)
                #write_cell_text(img, text=str(foodpile_i + 1), pos=self.foodpile_pos[foodpile_i], cell_size=CELL_SIZE,
                #                fill='white', margin=0.4)

                write_cell_text(img, text=str(self.foodpile_capacity[foodpile_i]), pos=self.foodpile_pos[foodpile_i], cell_size=CELL_SIZE,
                                fill='white', margin=0.4)
        
        # Colonies render 
        for colony_i in range(self.n_colonies):
            fill_cell(img, self.colonies_pos[colony_i], cell_size=CELL_SIZE, fill=COLONY_COLOR, margin=0.1)
            write_cell_text(img, text=str(colony_i + 1), pos=self.colonies_pos[colony_i], cell_size=CELL_SIZE,
                           fill='white', margin=0.4)


        # UNCOMMENT TO VIEW TAGS
        #for row in range(self._grid_shape[0]):
            #for col in range(self._grid_shape[1]):
                #write_cell_text(img, text=str(self._full_obs[col][row]), pos=[col, row], cell_size=CELL_SIZE,
                #            fill='white', margin=0.4)

        img = np.asarray(img)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def seed(self, n=None):
        self.np_random, seed = seeding.np_random(n)
        return [seed]

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


AGENT_COLOR = ImageColor.getcolor('black', mode='RGB')
AGENT_NEIGHBORHOOD_COLOR = (240, 240, 10)
PREY_COLOR = 'red'
FOOD_COLOR = 'green'
COLONY_COLOR = 'sienna'
PHEROMONE_COLOR = (10, 240, 240)

GROUND_COLOR = (205, 133, 63)
WALL_COLOR = 'black'

CELL_SIZE = 35

ACTION_MEANING = {
    0: "DOWN",
    1: "LEFT",
    2: "UP",
    3: "RIGHT",
    4: "NOOP",
}

PRE_IDS = {
    'agent': 'A',
    'prey': 'P',
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
