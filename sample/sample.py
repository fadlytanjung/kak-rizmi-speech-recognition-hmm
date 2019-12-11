
import os
import urllib3

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

# link = 'https://dl.dropboxusercontent.com/u/15378192/audio.tar.gz'
# dlname = 'audio.tar.gz'
# testfile = urllib3.PoolManager()
# r = testfile.request('GET', link, preload_content=False)
# print(r)
# with open(dlname, 'wb') as out:
#     while True:
#         data = r.read(chunk_size)
#         if not data:
#             break
#         out.write(data)

# r.release_conn()
# os.system('tar xzf %s' % dlname)
# exit()
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
# plt.figure()
plt.show()

#######################
# feature engineering #
#######################


def stft(x, fftsize=64, overlap_pct=.5):
    #Modified from http://stackoverflow.com/questions/2459295/stft-and-istft-in-python
    hop = int(fftsize * (1 - overlap_pct))
    w = scipy.hanning(fftsize + 1)[:-1]
    raw = np.array([np.fft.rfft(w * x[i:i + fftsize])
                    for i in range(0, len(x) - fftsize, hop)])
    return raw[:, :(fftsize // 2)]

#Peak detection using the technique described here: http://kkjkok.blogspot.com/2013/12/dsp-snippets_9.html


def peakfind(x, n_peaks, l_size=3, r_size=3, c_size=3, f=np.mean):
    win_size = l_size + r_size + c_size
    shape = x.shape[:-1] + (x.shape[-1] - win_size + 1, win_size)
    strides = x.strides + (x.strides[-1],)
    xs = as_strided(x, shape=shape, strides=strides)

    def is_peak(x):
        centered = (np.argmax(x) == l_size + int(c_size/2))
        l = x[:l_size]
        c = x[l_size:l_size + c_size]
        r = x[-r_size:]
        passes = np.max(c) > np.max([f(l), f(r)])
        if centered and passes:
            return np.max(c)
        else:
            return -1
    r = np.apply_along_axis(is_peak, 1, xs)
    top = np.argsort(r, None)[::-1]
    heights = r[top[:n_peaks]]
    #Add l_size and half - 1 of center size to get to actual peak location
    top[top > -1] = top[top > -1] + l_size + int(c_size / 2.)
    return heights, top[:n_peaks]


all_obs = []
for i in range(data.shape[0]):
    d = np.abs(stft(data[i, :]))
    n_dim = 6
    obs = np.zeros((n_dim, d.shape[0]))
    for r in range(d.shape[0]):
        _, t = peakfind(d[r, :], n_peaks=n_dim)
        obs[:, r] = t.copy()
    if i % 10 == 0:
        print("Processed obs %s" % i)
    all_obs.append(obs)

all_obs = np.atleast_3d(all_obs)

#######################
## plot new features ##
#######################

plot_data = np.abs(stft(data[0, :]))[15, :]
values, locs = peakfind(plot_data, n_peaks=6)
fp = locs[values > -1]
fv = values[values > -1]
plt.plot(plot_data, color='steelblue')
plt.plot(fp, fv, 'x', color='darkred')
plt.title('Peak location example')
plt.xlabel('Frequency (bins)')
plt.ylabel('Amplitude')
plt.show()

#######################
# split training data #
#######################

sss = StratifiedShuffleSplit(n_splits=2, test_size=0.1, random_state=0)
sss.get_n_splits(all_obs, all_labels)

for n, i in enumerate(all_obs):
    all_obs[n] /= all_obs[n].sum(axis=0)

for train_index, test_index in sss.split(all_obs, all_labels):
    X_train, X_test = all_obs[train_index, ...], all_obs[test_index, ...]
    y_train, y_test = all_labels[train_index], all_labels[test_index]

print('Size of training matrix:', X_train.shape)
# ('Size of training matrix:', (94, 6, 216))
print('Size of testing matrix:', X_test.shape)
# ('Size of testing matrix:', (11, 6, 216))

#######################
##  estimate model   ##
#######################

ys = set(all_labels)
models = [hmm.GaussianHMM(n_components=6) for y in ys]
_ = [m.fit(X_train[y_train == y, :, :].mean(axis=2))
     for m, y in zip(models, ys)]

#######################
## predict test data ##
#######################

logprob = np.array([[m.score(i) for i in X_test.mean(axis=2)] for m in models])
y_hat = np.argmax(logprob, axis=0)
missed = (y_hat != y_test)
print('Test accuracy: %.2f percent' % (100 * (1 - np.mean(missed))))
# Test Accuracy: 63.64%
