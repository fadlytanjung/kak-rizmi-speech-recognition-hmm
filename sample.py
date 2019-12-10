
import os
import urllib

import numpy as np
from numpy.lib.stride_tricks import as_strided
import matplotlib.pyplot as plt
import scipy
from scipy.io import wavfile
from sklearn.model_selection import StratifiedShuffleSplit
from hmmlearn import hmm

# %matplotlib inline

#######################
## import audio file ##
#######################

link = 'https://dl.dropboxusercontent.com/u/15378192/audio.tar.gz'
dlname = 'audio.tar.gz'
testfile = urllib.URLopener()
testfile.retrieve(link, dlname)
os.system('tar xzf %s' % dlname)

fpaths = []
labels = []
spoken = []
for f in os.listdir('audio'):
    for w in os.listdir('audio/' + f):
        fpaths.append('audio/' + f + '/' + w)
        labels.append(f)
        if f not in spoken:
            spoken.append(f)
print('Words spoken:', spoken)
# ('Words spoken:', ['apple', 'banana', 'kiwi', 'lime', 'orange', 'peach', 'pineapple'])

data = np.zeros((len(fpaths), 32000))
maxsize = -1
for n, file in enumerate(fpaths):
    _, d = wavfile.read(file)
    data[n, :d.shape[0]] = d
    if d.shape[0] > maxsize:
        maxsize = d.shape[0]
data = data[:, :maxsize]

all_labels = np.zeros(data.shape[0])
for n, l in enumerate(set(labels)):
    all_labels[np.array([i for i, _ in enumerate(labels) if _ == l])] = n

#######################
##  plot audio file  ##
#######################

plt.plot(data[0, :], color='steelblue')
plt.title('Timeseries example for %s' % labels[0])
plt.xlim(0, 3500)
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude (signed 16 bit)')
plt.figure()
