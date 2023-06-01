import math
import random
import argparse
import numpy as np
from scipy.spatial.distance import cityblock

from aasma import Agent
from aasma.utils import compare_results
from aasma.wrappers import SingleAgentWrapper

N_ACTIONS = 11
DOWN, LEFT, UP, RIGHT, DOWN_PHERO, LEFT_PHERO, UP_PHERO, RIGHT_PHERO, COLLECT_FOOD, DROP_FOOD, NOOP = range(N_ACTIONS)

class ReactiveAgent(Agent):
    def __init__(self, agent_id, n_agents):
        super(ReactiveAgent, self).__init__(f"Reactive Agent")
        self.agent_id = agent_id
        self.n_agents = n_agents
        self.n_actions = N_ACTIONS

    def action(self) -> int:
        # [agents position _ colony position _ 25 * foodpiles _ 25 * pheromones _ colonys storage]

        # Make this dependent on agent view mask
        agent_position = self.observation[ : 2]
        colony_position = self.observation[2 : 2 + 2] # FOR ONLY 1 COLONY

        foodpiles_in_view = self.observation[2 + 2 : 2 + 27]
        pheromones_in_view = self.observation[2 + 27 : 2 + 52]

        colony_storage = self.observation[-2]

        has_food = self.observation[-1]

        # See if there are any noteworthy things in view
        foodpiles_indices = np.where(foodpiles_in_view != 0)
        pheromones_indices = np.where(pheromones_in_view != 0)

        # TODO - DECISION LOGIC GOES HERE

        return Exception
