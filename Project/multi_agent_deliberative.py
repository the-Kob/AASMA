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


def run_multi_agent(environment: Env, n_episodes: int) -> np.ndarray:
    
    results = {}

    for episode in range(n_episodes):
        
        teams = {

            "Deliberative Team": [
                DeliberativeAntAgent(agent_id=0, n_agents=4),
                DeliberativeAntAgent(agent_id=1, n_agents=4),
                DeliberativeAntAgent(agent_id=2, n_agents=4),
                DeliberativeAntAgent(agent_id=3, n_agents=4),
            ] 
         
        }

        results_ep = np.zeros(n_episodes)

        for team, agents in teams.items():
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

            results_ep[episode] = steps

            environment.close()

        results[team] = results_ep

    return results

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--episodes", type=int, default=100) # CHANGE THIS (n_episodes)
    parser.add_argument("--render-sleep-time", type=float, default=0.1)
    opt = parser.parse_args()

    # 1 - Setup the environment
    environment = AntColonyEnv(grid_shape=(25, 25), n_agents=4, max_steps=100, n_foodpiles=5)

    # 2 - Setup the teams # ISOLATE IN FUNCTION (return agents)
    

    # FOR EPISODE, FOR TEAM (RUN 1 EPISODE FOR EVERY TEAM, THEN THE NEXT EPISODE)

    # 3 - Evaluate teams
    results = run_multi_agent(environment, opt.episodes)

    # 4 - Compare results
    compare_results(
        results,
        title="Teams Comparison on 'Ant Colony' Environment",
        colors=["orange", "green", "blue"]
    )

