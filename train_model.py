
# Imports
from sklearn.cluster import DBSCAN
from PIL import Image, ImageOps
import os
import numpy as np
import matplotlib.pyplot as plt

# CONSTANTS
LEARNING_IMG_PATH = os.path.join('C:\\Users', 'kylev', 'Desktop', 'Education', 'INFO_523', 'Project', 'INFO_523_Project', 'train', 'train.PNG')

# helper functions
def show_image(img):
    plt.imshow(img)
    plt.show()

# Load img into grayscale pixel map
im = ImageOps.grayscale(Image.open(LEARNING_IMG_PATH))
img_array = np.asarray(im)

# Create a boolean map
bool_array = np.zeros(np.shape(img_array))
bool_array[img_array < np.max(img_array)] = 1

# Assemble True pixels into single variable
X = np.transpose(np.asarray(img_array < np.max(img_array)).nonzero())

# Cluster using DBSCAN
db = DBSCAN(eps=4, min_samples=3).fit(X)
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print('Number of clusters found = ', n_clusters_)

## Shows letters using the label as color
#plot_array = np.zeros(np.shape(img_array))
#for l in set(labels):
#    X_l = X[labels == l, :]
#    plot_array[X_l[:, 0], X_l[:, 1]] = (l + 2) * 3
#show_image(plot_array)

## SHOWS EACH LETTER ONE AT A TIME
#for l in set(labels):
#    X_l = X[labels == l, :]
#    plot_array = np.zeros(np.shape(img_array))
#    plot_array[X_l[:, 0], X_l[:, 1]] = 1
#    show_image(plot_array)

import pdb; pdb.set_trace()
