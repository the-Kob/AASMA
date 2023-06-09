import argparse
import time
import numpy as np
from gym import Env
from typing import Sequence

from aasma.utils import compare_results_teams, compare_results_storage
from aasma.simplified_predator_prey import AntColonyEnv

from aasma import AntAgent
from single_reactive_agent import ReactiveAntAgent
from single_deliberative_agent import DeliberativeAntAgent
from single_random_agent import RandomAntAgent

SEED_MULTIPLIER = 1 # CHANGE THIS IF YOU WANT TO TEST A DIFFERENT SET OF MAPS!

def run_multi_agent(environment: Env, n_episodes: int, max_steps: int) -> np.ndarray:
    results_colonies_storage = {"Random Team" : np.zeros(max_steps), 
                                "Deliberative Team": np.zeros(max_steps), 
                                "Reactive Team": np.zeros(max_steps), 
                                "Hybrid Team": np.zeros(max_steps)}

    results_teams = {"Random Team" : np.zeros(n_episodes), 
                    "Deliberative Team": np.zeros(n_episodes), 
                    "Reactive Team": np.zeros(n_episodes), 
                    "Hybrid Team": np.zeros(n_episodes)}
    
    for episode in range(n_episodes):
        
        teams = {

            "Random Team": [
                RandomAntAgent(agent_id=0, n_agents=4),
                RandomAntAgent(agent_id=1, n_agents=4),
                RandomAntAgent(agent_id=2, n_agents=4),
                RandomAntAgent(agent_id=3, n_agents=4),
            ],

            "Deliberative Team": [
                DeliberativeAntAgent(agent_id=0, n_agents=4),
                DeliberativeAntAgent(agent_id=1, n_agents=4),
                DeliberativeAntAgent(agent_id=2, n_agents=4),
                DeliberativeAntAgent(agent_id=3, n_agents=4),
            ],

            "Reactive Team": [
                ReactiveAntAgent(agent_id=0, n_agents=4),
                ReactiveAntAgent(agent_id=1, n_agents=4),
                ReactiveAntAgent(agent_id=2, n_agents=4),
                ReactiveAntAgent(agent_id=3, n_agents=4),
            ],

            "Hybrid Team": [
                ReactiveAntAgent(agent_id=0, n_agents=4),
                ReactiveAntAgent(agent_id=1, n_agents=4),
                DeliberativeAntAgent(agent_id=2, n_agents=4),
                DeliberativeAntAgent(agent_id=3, n_agents=4),
            ] 
         
        }

        print(f"Episode {episode}")

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

                results_colonies_storage[team][steps - 1] += info['colony_storage']

                #environment.render() # ENABLE/DISABLE THIS TO VIEW ENVIRONMENT
                #time.sleep(opt.render_sleep_time)

                observations = next_observations
            
            environment.draw_heat_map(episode, team)
            environment.close()

            results_teams[team][episode] = steps

    for team in results_colonies_storage.keys():
         for i in range(max_steps):
            results_colonies_storage[team][i] /= n_episodes

    results_final = [results_teams, results_colonies_storage]

    return results_final

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--episodes", type=int, default=100) # CHANGE THIS (n_episodes)
    parser.add_argument("--steps", type=int, default=100) # CHANGE THIS (max_steps)
    parser.add_argument("--render-sleep-time", type=float, default=0.1)
    opt = parser.parse_args()

    # 1 - Setup the environment
    environment = AntColonyEnv(grid_shape=(16, 16), n_agents=4, max_steps=opt.steps, n_foodpiles=4, n_episodes=opt.episodes, pheromone_evaporation_rate=2)

    # 3 - Evaluate teams
    results = run_multi_agent(environment, opt.episodes, opt.steps)

    # 4 - Compare results
    compare_results_teams(
        results,
        title="Teams Comparison on 'Ant Colony' Environment",
        colors=["orange", "green", "blue", "red"]
    )

    compare_results_storage(
        results,
        title="Teams Comparison on 'Ant Colony' Environment",
        colors=["orange", "green", "blue", "red"]
    )
