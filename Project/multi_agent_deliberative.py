import argparse
import time
import numpy as np
from gym import Env
from typing import Sequence

from aasma import Agent
from aasma.utils import compare_results
from aasma.simplified_predator_prey import AntColonyEnv

from single_deliberative_agent import DeliberativeAntAgent

# from SOMEWHERE import ReactiveAgent REPLACE!!!


def run_multi_agent(environment: Env, agents: Sequence[Agent], n_episodes: int) -> np.ndarray:

    results = np.zeros(n_episodes)

    for episode in range(n_episodes):

        steps = 0
        terminals = [False for _ in range(len(agents))]
        observations = environment.reset()

        while not all(terminals):
            steps += 1
            actions = np.zeros(len(agents))
            
            for i in range(len(agents)):
                agents[i].see(observations[i])
                actions[i] = agents[i].action()
            
            next_observations, rewards, terminals, info = environment.step(actions)
            environment.render() # ENABLE/DISABLE THIS
            time.sleep(opt.render_sleep_time)
            observations = next_observations

        results[episode] = steps

        environment.close()

    return results


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--episodes", type=int, default=1) # CHANGE THIS (n_episodes)
    parser.add_argument("--render-sleep-time", type=float, default=0.1)
    opt = parser.parse_args()

    # 1 - Setup the environment
    environment = AntColonyEnv(grid_shape=(25, 25), n_agents=4, max_steps=200, n_foodpiles=5)

    # 2 - Setup the teams
    teams = {

        "Deliberative Team": [
            DeliberativeAntAgent(agent_id=0, n_agents=4),
            DeliberativeAntAgent(agent_id=1, n_agents=4),
            DeliberativeAntAgent(agent_id=2, n_agents=4),
            DeliberativeAntAgent(agent_id=3, n_agents=4),
        ],

        #"Reactive Team": [
        #    ReactiveAgent(agent_id=0, n_agents=4),
        #    ReactiveAgent(agent_id=1, n_agents=4),
        #    ReactiveAgent(agent_id=2, n_agents=4),
        #    ReactiveAgent(agent_id=3, n_agents=4)
        #],

        #"1 Deliberative + 3 Reactive": [
        #    DeliberativeAgent(agent_id=0, n_agents=4),
        #    ReactiveAgent(environment.action_space[1].n),
        #    ReactiveAgent(environment.action_space[2].n),
        #    ReactiveAgent(environment.action_space[3].n)
        #]
    }

    # 3 - Evaluate teams
    results = {}
    for team, agents in teams.items():
        result = run_multi_agent(environment, agents, opt.episodes)
        results[team] = result

    # 4 - Compare results
    compare_results(
        results,
        title="Teams Comparison on 'Ant Colony' Environment",
        colors=["orange", "green", "blue"]
    )

