from sklearn.cluster import DBSCAN
from PIL import Image, ImageOps
import os
import numpy as np
import matplotlib.pyplot as plt

class letter_pixels():
    def __init__(self, pixels):
        self._X = pixels

        # Create bounding box
        self._top_left = np.array([0, 0])
        self._bot_right = np.array([0, 0])
        self._bot_right[0] = np.max(self._X[:, 0])
        self._top_left[0] = np.min(self._X[:, 0])
        self._bot_right[1] = np.max(self._X[:, 1])
        self._top_left[1] = np.min(self._X[:, 1])
        self._center = (self._top_left + self._bot_right) / 2

        # Separate into quad clusters
        # -----------
        # | q2 | q1 |
        # -----------
        # | q3 | q4 |
        # -----------
        q1 = self._X[(self._X[:, 0] < self._center[0]) & ( self._X[:, 1] >= self._center[1]), :]
        q2 = self._X[(self._X[:, 0] < self._center[0]) & ( self._X[:, 1] < self._center[1]), :]
        q3 = self._X[(self._X[:, 0] >= self._center[0]) & ( self._X[:, 1] < self._center[1]), :]
        q4 = self._X[(self._X[:, 0] >= self._center[0]) & ( self._X[:, 1] >= self._center[1]), :]

        ## Create a plot with the quads in different colors
        #plot_array = np.zeros(np.shape(img_array)) + 255
        #plot_array[q1[:, 0], q1[:, 1]] = 0
        #plot_array[q2[:, 0], q2[:, 1]] = 50
        #plot_array[q3[:, 0], q3[:, 1]] = 100
        #plot_array[q4[:, 0], q4[:, 1]] = 150
        #show_image(plot_array)

        # Calculate quad cluster means
        if len(q1) > 0:
            self._q1_mean = np.array([np.mean(q1[:, 0]), np.mean(q1[:, 1])]) - np.array([self._top_left[0], self._center[1]])
        else:
            self._q1_mean = np.array([0, 0])
        if len(q2) > 0:
            self._q2_mean = np.array([np.mean(q2[:, 0]), np.mean(q2[:, 1])]) - self._top_left
        else:
            self._q2_mean = np.array([0, 0])
        if len(q3) > 0:
            self._q3_mean = np.array([np.mean(q3[:, 0]), np.mean(q3[:, 1])]) - np.array([self._center[0], self._top_left[1]])
        else:
            self._q3_mean = np.array([0, 0])
        if len(q4) > 0:
            self._q4_mean = np.array([np.mean(q4[:, 0]), np.mean(q4[:, 1])]) - self._center
        else:
            self._q4_mean = np.array([0, 0])

    def get_encoded_centers(self):
        '''Return centers as a list'''
        return np.reshape(np.array([self._q1_mean, self._q2_mean, self._q3_mean, self._q4_mean]), -1)


