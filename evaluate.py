# evaluate.py
import numpy as np
from environment import Environment
from agent import Agent
from keras.models import load_model
from config import Config

def run_evaluation(agent):
    env = Environment(Config.TEST_DATA_PATH, initial_balance=Config.INITIAL_BALANCE)
    state_size = env.state_size
    action_size = env.action_size

    # Load agent model from file if no agent is passed
    if agent is None:
        agent = Agent(state_size, action_size)
        agent.model = load_model(Config.MODEL_SAVE_PATH)

    state = env.reset()
    done = False
    total_profit = 0
    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        total_profit += info['profit']
        state = next_state

        if done:
            print("Total profit: {}".format(total_profit))
            return total_profit

if __name__ == "__main__":
    agent = None  # Default agent to None when running evaluate.py directly
    run_evaluation(agent)
