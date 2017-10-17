# Main file for the neural network
# atari network
# https://arxiv.org/pdf/1312.5602.pdf
# https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/
import os
import random
import numpy as np
from SumTree import SumTree
from keras.models import Sequential
from keyboardCombos import KeyboardCombos
from windowsInputHandler import InputHandler

class LearningAgent(Agent):



    __init__(self, learning=false, epsilon=1.0, alpha=0.5):
        super(LearningAgent, self).__init__()
        self.inputHandler = KeyboardCombos(self)
        self.valid_actions = []



# Gets screen data for HP/screen monitoring and reward feedback
import mss
class Vision:
    screen = {'top': 40, 'left': 0, 'width': 640, 'height': 350}
    leftHPCapture = {'top': 62, 'left': 84, 'width': 201, 'height': 12}
    rightHPCapture = {'top': 62, 'left': 360, 'width': 201, 'height': 12}

    positive = 1    # AI hit the opponent
    negative = -1   # AI took a hit

    def __init__(self, side):
        self.side = side
        with mss as sct():
            self.prevLeftHP = np.array(sct.grab(leftHPCapture))
            self.prevRightHP = np.array(sct.grab(rightHPCapture))

    # Dot product to reduce the pixel values to their grayscale equivlalent
    def numpyImgToGray(self, img):
        return np.dot(img[...,:3], [0.299,0.587,0.114])

    def getCurrentScreen(self):
        with mss as sct():
            currScreen = np.array(sct.grab(screen))
            return numpyImgToGray(currScreen)

    def getReward(self):
        with mss as sct():
            reward = 0

            currLeft = np.array(sct.grab(leftHPCapture))
            currRight = np.array(sct.grab(rightHPCapture))
            # Convert to gray
            currLeft = numpyImgToGray(currLeft)
            currRight = numpyImgToGray(currRight)
            # get the difference in previous vs current
            diffLeft = self.prevLeft - currLeft
            diffRight = self.prevRight - currRight
            # round negative values up to 0
            diffLeft = diffLeft.clip(min=0)
            diffRight = diffRight.clip(min=0)
            # If hit, there are typically more than 10 pixels with values > 120
            if((diffLeft > 125).sum() > 10):
                if(side == 'left'):
                    reward = reward - 1
                else:
                    reward = reward + 1
            # If hit, there are typically more than 10 pixels with values > 120
            if((diffRight > 125).sum() > 10):
                if(side == 'right'):
                    reward = reward - 1
                else:
                    reward = reward + 1
            # Set previous frame data to current frame data
            self.prevLeft = currLeft
            self.prevRight = currRight

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

# lets start the show
def run():

if __name__ == '__main__':
    run()
