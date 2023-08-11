
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

def show_cluster(X, row_bool):
    X_l = X[row_bool, :]
    plot_array = np.zeros(np.shape(img_array))
    plot_array[X_l[:, 0], X_l[:, 1]] = 1
    show_image(plot_array)


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

# Order labels
weights = np.zeros(len(set(labels)))
ii = 0
for l in set(labels):
    X_l = X[labels == l, :]
    min_horiz = np.min(X_l[:, 1])
    min_vert = np.min(X_l[:, 0])
    weights[ii] = (round(min_vert / 10) * 10) * 100000 + min_horiz
    ii += 1

label_set_ordered = np.asarray(list(set(labels)))[np.argsort(np.array(weights))]

## Shows letters using the label as color
#plot_array = np.zeros(np.shape(img_array))
#for l in set(labels):
#    X_l = X[labels == l, :]
#    plot_array[X_l[:, 0], X_l[:, 1]] = (l + 2) * 3
#show_image(plot_array)

## SHOWS EACH LETTER ONE AT A TIME
#for l in label_set_ordered:
#    X_l = X[labels == l, :]
#    plot_array = np.zeros(np.shape(img_array))
#    plot_array[X_l[:, 0], X_l[:, 1]] = 1
#    show_image(plot_array)

import pdb; pdb.set_trace()
