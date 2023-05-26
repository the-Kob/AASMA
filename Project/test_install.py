import time
import argparse
import numpy as np

from aasma.wrappers import SingleAgentWrapper
from aasma.simplified_predator_prey import SimplifiedPredatorPrey


if __name__ == '__main__':

    """
    Demo for usage of OpenAI Gym's interface for environments.
    --episodes: Number of episodes to run
    --render-sleep-time: Seconds between frames
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--render-sleep-time", type=float, default=0.5)
    opt = parser.parse_args()

    # 1 - Setup the environment
    environment = SimplifiedPredatorPrey(
        grid_shape=(10, 10), # try 40x40
        n_agents=1, 
        max_steps=100, required_captors=1,
        n_foodpiles=5
    )
    environment = SingleAgentWrapper(environment, agent_id=0)

    n_actions = environment.action_space.n

    # Run
    for episode in range(opt.episodes):
        
        print(f"Episode {episode}")
        
        n_steps = 0

        observation = environment.reset()

        environment.render()
        time.sleep(opt.render_sleep_time)

        terminal = False
        while not terminal:

            n_steps += 1
            action = np.random.randint(n_actions)
            next_observation, reward, terminal, info = environment.step(action)

            print(f"Timestep {n_steps}")
            print(f"\tObservation: {observation}")
            print(f"\tAction: {environment.get_action_meanings()[action]}\n")
            environment.render()
            time.sleep(opt.render_sleep_time)

            observation = next_observation

        environment.close()
