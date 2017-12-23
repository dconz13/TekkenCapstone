# Main file for the neural network
# references:
# https://arxiv.org/pdf/1312.5602.pdf
# https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/
# https://github.com/DavidSanwald/DDQN

import os
import random
import math
import numpy as np
import time
from SumTree import SumTree
from keras.models import Sequential
from keras.models import load_model
from keras.utils import plot_model
from keras.layers import Conv2D, Dense, Flatten
from keras.optimizers import *
from windowsInputHandler import *

IMAGE_STACK = 2
IMAGE_WIDTH = 84
IMAGE_HEIGHT = 84
HUBER_LOSS_DELTA = 2.0
LEARNING_RATE = 0.00025

def huber_loss(y_true, y_pred):
    err = y_true - y_pred

    cond = K.abs(err) < HUBER_LOSS_DELTA
    L2 = 0.5*K.square(err)
    L1 = HUBER_LOSS_DELTA * (K.abs(err) - 0.5 * HUBER_LOSS_DELTA)

    loss = tf.where(cond, L2, L1)

    return K.mean(loss)

class Model:

    def __init__(self, input_shape, actionCnt, model=None, target_model=None):
        self.input_shape = input_shape
        self.actionCnt = actionCnt

        if model is not None: # a model was loaded from memory.
            self.model = model
            self.target_model = target_model
        else: # First time run
            self.model = self._createModel()
            self.target_model = self._createModel()

    def _createModel(self):
        model = Sequential()

        # channels_first mode (batch, channels, height, width)
        model.add(Conv2D(32, (8,8), strides=(4,4), activation='relu', input_shape=(self.input_shape), data_format='channels_first'))
        model.add(Conv2D(64, (4,4), strides=(2,2), activation='relu'))
        model.add(Conv2D(64, (3,3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(units=512, activation='relu'))
        model.add(Dense(units=self.actionCnt, activation='linear'))

        opt = RMSprop(lr=LEARNING_RATE)
        model.compile(loss=huber_loss, optimizer=opt)

        return model

    def train(self, x, y, epochs=1, verbose=0):
        self.model.fit(x, y, batch_size=32, epochs=epochs, verbose=verbose)

    def predict(self, s, target=False):
        if target:
            return self.target_model.predict(s)
        else:
            return self.model.predict(s)

    def predict_one(self, s, target=False):
        return self.predict(s.reshape(1, IMAGE_STACK, IMAGE_WIDTH, IMAGE_HEIGHT), target).flatten()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

MEMORY_CAPACITY = 400000
BATCH_SIZE = 32
GAMMA = 0.99
MAX_EPSILON = 1
MIN_EPSILON = 0.1

EXPLORATION_STOP = 500000
LAMBDA = - math.log(0.01) / EXPLORATION_STOP

UPDATE_TARGET_FREQUENCY = 10000

class LearningAgent:
    steps = 0
    latest_Q = 0
    epsilon = MAX_EPSILON

    def __init__(self, learning=False, epsilon=1.0, alpha=0.5):
        #super(LearningAgent, self).__init__()
        self.input_handler = InputHandler()
        self.learning = learning
        self.inputShape = (IMAGE_STACK, IMAGE_WIDTH, IMAGE_HEIGHT)
        self.numActions = len(valid_actions)
        self.model = Model(self.inputShape, self.numActions)
        self.memory = Memory(MEMORY_CAPACITY)
        #self.valid_actions = []

    def observe(self, sample): #(s, a, r, s_)
        x, y, errors = self.get_targets([(0, sample)])
        self.memory.add(errors[0], sample)

        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
            self.model.update_target_model()

        self.steps = self.steps + 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def get_targets(self, batch):
        no_state = np.zeros(self.inputShape)

        states = np.array([o[1][0] for o in batch])
        states_ = np.array([(no_state if o[1][3] is None else o[1][3]) for o in batch])

        # p is the prediction for a given state
        p = agent.model.predict(states)

        p_ = agent.model.predict(states_, target=False) # target set to false means predict from model
        pTarget_ = agent.model.predict(states_, target=True) # target true means predict from target model

        x = np.zeros((len(batch), IMAGE_STACK, IMAGE_WIDTH, IMAGE_HEIGHT))
        y = np.zeros((len(batch), self.numActions))
        errors = np.zeros(len(batch))

        for i in range(len(batch)):
            o = batch[i][1]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
            t = p[i]
            oldVal = t[a]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * pTarget_[i][np.argmax(p_[i])] # DDQN portion

            x[i] = s
            y[i] = t
            errors[i] = abs(oldVal - t[a])
            self.latest_Q = t[a]

        return (x, y, errors)

    def choose_action(self, state):
        # call when the agent must make a decision based on the state
        self.state = state
        self.steps += 1
        action = None

        if self.learning == False:
            #action = self.input_handler.get_action(random.randint(0,len(valid_actions)-1))
            action = random.randint(0, len(valid_actions)-1)
        else:
            if random.uniform(0,1) < self.epsilon:
                #action = self.input_handler.get_action(random.randint(0,len(valid_actions)-1))
                action = random.randint(0, len(valid_actions)-1)
            else:
                action = np.argmax(self.model.predict_one(state))
        return action

    def execute_action(self, action):
        self.input_handler.execute_action(action)

    def replay(self):
        batch = self.memory.sample(BATCH_SIZE)
        x, y, errors = self.get_targets(batch)

        for i in range(len(batch)):
            idx = batch[i][0]
            self.memory.update(idx, errors[i])

        self.model.train(x, y)

    def play(self, state):
        self.state = state
        action = np.argmax(self.model.predict_one(state))
        return action

# Gets screen data for HP/screen monitoring and reward feedback
import mss
from PIL import Image
class Vision:
    screen = {'top': 40, 'left': 0, 'width': 640, 'height': 350}
    leftHPCapture = {'top': 62, 'left': 84, 'width': 201, 'height': 12}
    rightHPCapture = {'top': 62, 'left': 360, 'width': 201, 'height': 12}

    positive = 1    # AI hit the opponent
    negative = -1   # AI took a hit

    def __init__(self, side):
        self.side = side
        with mss.mss() as sct:
            self.prevLeftHP = self.numpy_img_to_gray(np.array(sct.grab(self.leftHPCapture)))
            self.prevRightHP = self.numpy_img_to_gray(np.array(sct.grab(self.rightHPCapture)))

    # Dot product to reduce the pixel values to their grayscale equivlalent
    def numpy_img_to_gray(self, img):
        return np.dot(img[...,:3], [0.299,0.587,0.114])

    def get_current_screen(self):
        with mss.mss() as sct:
            sct_img = sct.grab(self.screen)
            img = Image.frombytes('RGB', sct_img.size, sct_img.rgb)
            img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.LANCZOS)
            currScreen = np.array(img)
        return self.numpy_img_to_gray(currScreen)

    def get_reward(self):
        with mss.mss() as sct:
            reward = 0

            currLeftHP = np.array(sct.grab(self.leftHPCapture))
            currRightHP = np.array(sct.grab(self.rightHPCapture))
            # Convert to gray
            currLeftHP = self.numpy_img_to_gray(currLeftHP)
            currRightHP = self.numpy_img_to_gray(currRightHP)
            # get the difference in previous vs current
            diffLeftHP = self.prevLeftHP - currLeftHP
            diffRightHP = self.prevRightHP - currRightHP
            # round negative values up to 0
            diffLeftHP = diffLeftHP.clip(min=0)
            diffRightHP = diffRightHP.clip(min=0)
            # If hit, there are typically more than 10 pixels with values > 120
            if((diffLeftHP > 125).sum() > 10):
                if(self.side == 'left'):
                    reward = reward - 1
                else:
                    reward = reward + 1
            # If hit, there are typically more than 10 pixels with values > 120
            if((diffRightHP > 125).sum() > 10):
                if(self.side == 'right'):
                    reward = reward - 1
                else:
                    reward = reward + 1
            # Set previous frame data to current frame data
            self.prevLeftHP = currLeftHP
            self.prevRightHP = currRightHP

        return reward

# SumTree of previous decisions
class Memory:
    def __init__(self, capacity, epsilon=1.0, alpha=0.5):
        self.tree = SumTree(capacity)
        self.epsilon = epsilon
        self.alpha = alpha

    def _getPriority(self, error):
        return (error + self.epsilon) ** self.alpha

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append((idx, data))

        return batch

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)

def import_model(agent):
    try:
        agent.model = Model(agent.inputShape, agent.numActions,
            load_model('TekkenBotDDQN_3.h5', custom_objects={'huber_loss':huber_loss}), load_model('TekkenBotDDQN_Target_3.h5', custom_objects={'huber_loss':huber_loss}))
        print('Model loaded for agent.')
        #plot_model(agent.model, to_file='TekkenBotDDQNRound1.png')
        #print('Loaded model plotted.')
    except Exception as e:
        print('Model failed to load.')
        print(e)

# Used when not learning.
def play(agent):
    agent.input_handler.activate_remap()
    vision = Vision('left')
    w = vision.get_current_screen()
    state = np.array([w,w])
    try:
        while True:
            actionIndex = agent.play(state)
            agent.execute_action(actionIndex)
            screenCap = vision.get_current_screen()
            statePrime = np.array((state[1], screenCap))
            state = statePrime

    except KeyboardInterrupt:
        print('break')


TRIAL = 6 # necessary for correct model backup
TOTAL_TESTS = 3 # 3 12 hour segments of testing
TOTAL_EPISODES = 540 # 720 = 12 hours
def save_models(agent, curr_test):

    directory = 'model backup/trial {}'.format(TRIAL)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + "/TekkenBotDDQN_{}.h5".format(curr_test)
    agent.model.model.save(filename, overwrite=True)
    print("Model saved as: TekkenBotDDQN_{}.h5".format(curr_test))

    filename = directory + "/TekkenBotDDQN_Target_{}.h5".format(curr_test)
    agent.model.target_model.save(filename, overwrite=True)
    print("Target Model saved as: TekkenBotDDQN_Target_{}.h5".format(curr_test))

# lets start the show. Used when learning.
def run(agent):
    agent.input_handler.activate_remap()
    # character is on the left side of the screen
    vision = Vision('left')
    w = vision.get_current_screen()
    state = np.array([w,w])
    start_time = time.time()

    # total reward for the episode
    rewardTotal = 0
    # initial episode
    episode = 0
    true_episode = 0
    curr_test = 1
    record_size = TOTAL_EPISODES*TOTAL_TESTS # How big to initialize the array depending on time running
    reward_per_episode = np.zeros((record_size, 4))
    i = 0 # current np array index
    agent.input_handler.reset_players()

    # Press Ctrl + Alt followed by Ctrl + C to break out of the program.
    try:
        while True:
            actionIndex = agent.choose_action(state) #index in valid_actions array
            agent.execute_action(actionIndex)
            screenCap = vision.get_current_screen()
            reward = vision.get_reward()

            statePrime = np.array((state[1], screenCap))

            agent.observe((state, actionIndex, reward, statePrime))
            agent.replay()

            state = statePrime
            rewardTotal = rewardTotal + reward

            if time.time() - start_time > 59: # reset players every minute. That's how arcade mode works.
                reward_per_episode[i] = [true_episode, rewardTotal, agent.steps, agent.latest_Q]
                print("Episode {}".format(episode) + " Ended. Reward earned: {}".format(rewardTotal))
                episode += 1
                true_episode += 1
                rewardTotal = 0
                i += 1
                agent.input_handler.reset_players()
                start_time = time.time()

            if episode >= TOTAL_EPISODES:
                if curr_test >= TOTAL_TESTS:
                    save_models(agent, curr_test)
                    agent.input_handler.quit_game()
                    agent.input_handler.activate_remap()
                    raise KeyboardInterrupt('Episodes limit reached')
                else:
                    save_models(agent, curr_test)
                    agent.input_handler.reset_map(curr_test)
                    curr_test += 1
                    episode = 0
                    start_time = time.time()


    except KeyboardInterrupt:
        # finish up reward array and save it.
        #reward_per_episode[i] = [episode, rewardTotal, agent.steps, agent.latest_Q]
        #print("Episode {}".format(episode) + " Ended. Reward earned: {}".format(rewardTotal))
        np.savetxt('model backup/trial {}/episodesAndRewards.txt'.format(TRIAL), reward_per_episode, fmt='%d')
        print("Episodes and rewards saved to episodesAndRewards.txt")

if __name__ == '__main__':
    try:
        agent = LearningAgent(learning=True, epsilon=MAX_EPSILON, alpha=LEARNING_RATE)
        import_model(agent)
        #run(agent)
        play(agent)
    finally:
        print('Thanks for playing!')
        #i = InputHandler()
        #i.activate_remap()
