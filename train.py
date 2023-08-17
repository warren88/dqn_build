import numpy as np
import logging
from agent import Agent
from environment import Environment
from data import get_random_samples
from config import Config

print("training Initialized...")
def run_training():
    # Initialize the environment and the agent
    try:
        env = Environment(Config.ACCESS_TOKEN, Config.ACCOUNT_ID, Config.INSTRUMENT)
    except Exception as e:
        logging.error(f"Error initializing environment: {e}")
        raise

    try:
        agent = Agent(env.state_size)
    except Exception as e:
        logging.error(f"Error initializing agent: {e}")
        raise
# THIS NEEDS IMPLEMENTING IN THE GET_DATA METHOD IN ENVIRONMENT.PY
    # Get historic data
    try:
        historic_data = env._get_data()
    except Exception as e:
        logging.error(f"Error getting historic data: {e}")
        raise

    # Get random samples from historic data
    try:
        train_data, test_data = get_random_samples(historic_data)
    except Exception as e:
        logging.error(f"Error getting random samples: {e}")
        raise

    # Number of episodes for training
    episodes = len(train_data)

    for e in range(episodes):
        try:
            print("Episode " + str(e) + "/" + str(episodes))
            state = env.reset(train_data[e])
            state = np.reshape(state, [1, env.state_size])

            # Time steps in each episode
            for t in range(env.time_steps):
                try:
                    action = agent.act(state)
                except Exception as e:
                    logging.error(f"Error during agent action: {e}")
                    raise

                try:
                    next_state, reward, done, _ = env.step(action)
                    next_state = np.reshape(next_state, [1, env.state_size])
                except Exception as e:
                    logging.error(f"Error during environment step: {e}")
                    raise

                agent.remember(state, action, reward, next_state, done)
                state = next_state

                if done:
                    print("Episode: {}/{}, Score: {}".format(e, episodes, env.score))
                    break

                if len(agent.memory) > 32:
                    agent.expReplay(32)
        except Exception as e:
            logging.error(f"Error in episode {e}/{episodes}: {e}")
            raise

        # Save the model weights after each episode
        try:
            if e % 10 == 0:
                agent.save("weights.h5")
        except Exception as e:
            logging.error(f"Error saving weights: {e}")
            raise
    return agent

if __name__ == "__main__":
    run_training()