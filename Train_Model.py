
# Imports
import time
print('->Begin Imports')
tic = time.time()
from sklearn.cluster import DBSCAN
from PIL import Image, ImageOps
import os
import numpy as np
import matplotlib.pyplot as plt
from common.types import *
print(f'->End Imports ({time.time() - tic} seconds)')

# CONSTANTS
#LEARNING_IMG_PATH = os.path.join('C:\\Users', 'kylev', 'Desktop', 'Education', 'INFO_523', 'Project', 'INFO_523_Project', 'train', 'train.PNG')
LEARNING_IMG_PATH = os.path.join('C:\\Users', 'kylev', 'Desktop', 'Education', 'INFO_523', 'Project', 'INFO_523_Project', 'train', 'trainLARGE.PNG')
LABELS_PATH = os.path.join('C:\\Users', 'kylev', 'Desktop', 'Education', 'INFO_523', 'Project', 'INFO_523_Project', 'train', 'labels.csv')
TRAINED_MODEL_PATH = os.path.join('C:\\Users', 'kylev', 'Desktop', 'Education', 'INFO_523', 'Project', 'INFO_523_Project', 'train', 'model.csv')

# helper functions
def show_cluster(X, row_bool):
    X_l = X[row_bool, :]
    plot_array = np.zeros(np.shape(img_array))
    plot_array[X_l[:, 0], X_l[:, 1]] = 1
    show_image(plot_array)



print('->Begin Loading Image')
tic = time.time()
# Load img into grayscale pixel map
im = ImageOps.grayscale(Image.open(LEARNING_IMG_PATH))
img_array = np.asarray(im)
print(f'->End Loading Image ({time.time() - tic} seconds)')

print('->Begin Clustering Letter Pixels')
tic = time.time()
# Create a boolean map
bool_array = np.zeros(np.shape(img_array))
bool_array[img_array < (np.min(img_array) + 150)] = 1

# Assemble True pixels into single variable
X = np.transpose(np.asarray(bool_array).nonzero())

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
print(f'->End Clustering Letter Pixels ({time.time() - tic} seconds)')

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

# Check that the number of clusters matches the number of letters we are training to
with open(LABELS_PATH, 'r') as f:
    lbls = f.readline().strip().split(',')
assert len(lbls) == len(label_set_ordered), 'ERROR: Number of clusters in the training set does not match the number of providied labels'

# Calculate quad centers for each letter
trained_model = []
for ii, l in enumerate(label_set_ordered):
    X_l = X[labels == l, :]
    q = letter_pixels(X_l)
    centers = q.get_encoded_centers()
    # Round to 3 decimal places
    trained_model.append([lbls[ii]] + [round(thing * 1000) / 1000 for thing in centers])

with open(TRAINED_MODEL_PATH, 'w') as f:
    for dataline in trained_model:
        f.write(','.join(str(thing) for thing in dataline) + '\n')

