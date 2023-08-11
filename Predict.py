# Imports
import time
print('->Begin Imports')
tic = time.time()
from common.types import *
from sklearn.cluster import DBSCAN
from PIL import Image, ImageOps
import os
import numpy as np
import matplotlib.pyplot as plt
print(f'->End Imports ({time.time() - tic} seconds)')

# CONSTANTS
PREDICTION_IMAGE = os.path.join('C:\\Users', 'kylev', 'Desktop', 'Education', 'INFO_523', 'Project', 'INFO_523_Project', 'predict', 'temp.PNG')
#PREDICTION_IMAGE = os.path.join('C:\\Users', 'kylev', 'Desktop', 'Education', 'INFO_523', 'Project', 'INFO_523_Project', 'train', 'train.PNG')
MODEL_PATH = os.path.join('C:\\Users', 'kylev', 'Desktop', 'Education', 'INFO_523', 'Project', 'INFO_523_Project', 'train', 'model.csv')

# Load model
model = dict()
with open(MODEL_PATH, 'r') as f:
    for line in f:
        line_list = line.strip().split(',')
        model[line_list[0]] = np.array(list(map(float, line_list[1:])))

print('->Begin Loading Image')
tic = time.time()
# Load img into grayscale pixel map
im = ImageOps.grayscale(Image.open(PREDICTION_IMAGE))
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
print(f'->End Clustering Letter Pixels ({time.time() - tic} seconds)')

# Perform prediction
for ii, l in enumerate(set(labels)):
    err = np.zeros(len(model))
    X_l = X[labels == l, :]
    q = letter_pixels(X_l)
    centers = q.get_encoded_centers()
    for jj, category in enumerate(model):
        err[jj] = np.linalg.norm(model[category] - centers)
    letters_ordered_by_probability = np.array([key for key in model.keys()])[np.argsort(err)]
    print(f'Prediction: "{letters_ordered_by_probability[0]}"')
    
    plot_array = np.zeros(np.shape(img_array))
    plot_array[X_l[:, 0], X_l[:, 1]] = 1
    show_image(plot_array)
    
    #import pdb; pdb.set_trace()

