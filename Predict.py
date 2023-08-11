# Imports
import time
print('->Begin Imports')
tic = time.time()
from sklearn.cluster import DBSCAN
from PIL import Image, ImageOps
import os
import numpy as np
import matplotlib.pyplot as plt
print(f'->End Imports ({time.time() - tic} seconds)')

# CONSTANTS
PREDICTION_IMAGE = os.path.join('C:\\Users', 'kylev', 'Desktop', 'Education', 'INFO_523', 'Project', 'INFO_523_Project', 'predict', 'N.PNG')
MODEL_PATH = os.path.join('C:\\Users', 'kylev', 'Desktop', 'Education', 'INFO_523', 'Project', 'INFO_523_Project', 'train', 'model.csv')

# Load model
model = dict()
with open(MODEL_PATH, 'r') as f:
    for line in f:
        line_list = line.strip().split(',')
        model[line_list[0]] = np.array(list(map(float, line_list[1:])))
import pdb; pdb.set_trace()

print('->Begin Loading Image')
tic = time.time()
# Load img into grayscale pixel map
im = ImageOps.grayscale(Image.open(PREDICTION_IMAGE))
img_array = np.asarray(im)
print(f'->End Loading Image ({time.time() - tic} seconds)')


