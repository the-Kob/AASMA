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

def run_single_agent(environment: Env, n_episodes: int) -> np.ndarray:

    results = np.zeros(n_episodes)

    for episode in range(n_episodes):

        # Setup agent
        agent = RandomAntAgent(agent_id=0, n_agents=1, knowledgeable=True)

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


        environment.draw_heat_map(episode, "RandomAntAgent")
 
        environment.close()
        results[episode] = steps

    return results

class RandomAntAgent(AntAgent):

    def __init__(self, agent_id, n_agents, knowledgeable=True):
        super(RandomAntAgent, self).__init__(f"Random Ant Agent", agent_id, n_agents, knowledgeable)

    def action(self) -> int:
        return np.random.randint(self.n_actions)

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
        n_foodpiles=3
    )
    environment = SingleAgentWrapper(environment, agent_id=0)


    results = run_single_agent(environment, opt.episodes)

