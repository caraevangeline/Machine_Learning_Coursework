import numpy as np
import os
import matplotlib.pyplot as plt
from print_values import *
from plot_data_all_phonemes import *
from plot_data import *
import random
from sklearn.preprocessing import normalize
from get_predictions import *
from plot_gaussians import *
from model import *
# File that contains the data
data_npy_file = 'data/PB_data.npy'

# Loading data from .npy file
data = np.load(data_npy_file, allow_pickle=True)
data = np.ndarray.tolist(data)

# Make a folder to save the figures
figures_folder = os.path.join(os.getcwd(), 'figures')
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder, exist_ok=True)

# Array that contains the phoneme ID (1-10) of each sample
phoneme_id = data['phoneme_id']
# frequencies f1 and f2
f1 = data['f1']
f2 = data['f2']

# Initialize array containing f1 & f2, of all phonemes.
X_full = np.zeros((len(f1), 2))
#########################################
# Write your code here
# Store f1 in the first column of X_full, and f2 in the second column of X_full
for i in range(len(X_full)):
    X_full[i, 0] = f1[i]
    X_full[i, 1] = f2[i]
########################################/
X_full = X_full.astype(np.float32)

# number of GMM components
k = 6
mu1, p1, s1 = model(k, 1)
mu2, p2, s2 = model(k, 2)
#########################################
# Write your code here

# Create an array named "X_phonemes_1_2", containing only samples that belong to phoneme 1 and samples that belong to phoneme 2.
# The shape of X_phonemes_1_2 will be two-dimensional. Each row will represent a sample of the dataset, and each column will represent a feature (e.g. f1 or f2)
# Fill X_phonemes_1_2 with the samples of X_full that belong to the chosen phonemes
# To fill X_phonemes_1_2, you can leverage the phoneme_id array, that contains the ID of each sample of X_full
n1 = np.sum(phoneme_id==1)
n2 = np.sum(phoneme_id==2)
# X_phonemes_1_2 = ...
X_phonemes_1_2 = np.zeros((n1+n2, 2))
j = 0
for i in range(len(phoneme_id)):
      if phoneme_id[i] == 1:
           X_phonemes_1_2[j, 0] = f1[i]
           X_phonemes_1_2[j, 1] = f2[i]
           j = j+1
           
for i in range(len(phoneme_id)):
      if phoneme_id[i] == 2:
           X_phonemes_1_2[j, 0] = f1[i]
           X_phonemes_1_2[j, 1] = f2[i]
           j = j+1
########################################/

# Plot array containing the chosen phonemes

# Create a figure and a subplot
fig, ax1 = plt.subplots()

title_string = 'Phoneme 1 & 2'
# plot the samples of the dataset, belonging to the chosen phoneme (f1 & f2, phoneme 1 & 2)
plot_data(X=X_phonemes_1_2, title_string=title_string, ax=ax1)
# save the plotted points of phoneme 1 as a figure
plot_filename = os.path.join(os.getcwd(), 'figures', 'dataset_phonemes_1_2.png')
plt.savefig(plot_filename)


#########################################
# Write your code here
X = X_phonemes_1_2.copy()
# get number of samples
N = X.shape[0]
# get dimensionality of our dataset
D = X.shape[1]
Z1 = np.zeros((N,k))
Z2 = np.zeros((N,k))
# Get predictions on samples from both phonemes 1 and 2, from a GMM with k components, pretrained on phoneme 1
Z1 = get_predictions(mu1, s1, p1, X)
#Z1 = normalize(Z1, axis=1, norm='l1')
# Get predictions on samples from both phonemes 1 and 2, from a GMM with k components, pretrained on phoneme 2
Z2 = get_predictions(mu2, s2, p2, X)
#Z2 = normalize(Z2, axis=1, norm='l1')

Z1_new = np.zeros((Z1.shape[0],1))
Z2_new = np.zeros((Z1.shape[0],1))

for i in range(Z1.shape[0]):
    Z1_new[i] = Z1[i, 0] + Z1[i, 1] + Z1[i, 2]
    Z2_new[i] = Z2[i, 0] + Z2[i, 1] + Z2[i, 2]
    
# Compare these predictions for each sample of the dataset, and calculate the accuracy, and store it in a scalar variable named "accuracy"
#print(len(X_phonemes_1_2))
correct_classify = 0
for i in range(len(X_phonemes_1_2)):
    if i < len(X_phonemes_1_2)/2:
       if Z1_new[i] > Z2_new[i]:
          correct_classify = correct_classify+1
    else:
       if Z2_new[i] > Z1_new[i]: 
          correct_classify = correct_classify+1
########################################/
#print(cc)
accuracy = (correct_classify/len(X_phonemes_1_2))*100
error = (1-(correct_classify/len(X_phonemes_1_2)))*100
print('Accuracy using GMMs with {} components: {:.2f}%'.format(k, accuracy))
print('Misclassification error using GMMs with {} components: {:.2f}%'.format(k, error))
################################################
# enter non-interactive mode of matplotlib, to keep figures open
plt.ioff()
plt.show()
