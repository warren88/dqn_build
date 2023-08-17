import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class Agent:
    def __init__(self, state_size, is_eval=False, model_name=""):
        self.state_size = state_size  # dimension of state
        self.action_size = 3  # hold, buy, sell
        self.memory = deque(maxlen=1000)  # internal memory of agent, used for Experience Replay
        self.inventory = []  # list to store the stock bought
        self.model_name = model_name  # model name if evaluation is being done on a pre-trained model
        self.is_eval = is_eval  # indicates whether the agent is in evaluation mode or not

        # hyperparameters
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01  # minimum exploration rate
        self.epsilon_decay = 0.995  # exploration decay rate

        # build model
        self.model = self._model()

    def _model(self):
        # Neural Net for Deep-Q Learning
        try:
            model = Sequential()
            model.add(Dense(units=64, input_dim=self.state_size, activation="relu"))
            model.add(Dense(units=32, activation="relu"))
            model.add(Dense(units=8, activation="relu"))
            model.add(Dense(self.action_size, activation="linear"))
            model.compile(loss="mse", optimizer=Adam())

            return model
        except Exception as e:
            print(f"Exception occurred in model creation: {str(e)}")

    def act(self, state):
        try:
            if not self.is_eval and random.random() <= self.epsilon:  # if agent is not in evaluation mode, do exploration
                return random.randrange(self.action_size)

            # else predict the reward value based on current state and choose the action having maximum predicted reward
            options = self.model.predict(state)
            return np.argmax(options[0])
        except Exception as e:
            print(f"Exception occurred while choosing action: {str(e)}")

    def expReplay(self, batch_size):
        try:
            mini_batch = []
            l = len(self.memory)
            for i in range(l - batch_size, l):
                mini_batch.append(self.memory[i])

            # replay the experiences of the agent
            for state, action, reward, next_state, done in mini_batch:
                target = reward
                if not done:  # if the game is not over, predict the future discounted reward
                    target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

                # make the agent to approximately map
                # the current state to future discounted reward
                target_f = self.model.predict(state)
                target_f[0][action] = target

                # train the Neural Net with state and target_f
                self.model.fit(state, target_f, epochs=1, verbose=0)

            # reduce the exploration rate after each replay
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        except Exception as e:
            print(f"Exception occurred in experience replay: {str(e)}")
