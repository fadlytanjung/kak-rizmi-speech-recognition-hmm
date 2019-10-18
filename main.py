from python_speech_features import mfcc
from scipy.io import wavfile
from hmmlearn import hmm
from sklearn.externals import joblib
import numpy as np
import os

def gen_wavlist(wavpath):
    lab = set()
    wavdict = {}
    labeldict = {}
    for (dirpath, dirnames, filenames) in os.walk(wavpath):
        for filename in filenames:
            if filename.endswith('.wav'):
                filepath = os.sep.join([dirpath, filename])
                fileid = filename.strip('.wav')
                wavdict[fileid] = filepath
                label = fileid.split('_')[1]
                labeldict[fileid] = label
                lab.add(label)

    return wavdict, labeldict

def compute_mfcc(file):
	fs, audio = wavfile.read(file)
	mfcc_feat = mfcc(audio, samplerate=(fs/2), numcep=100)
	return mfcc_feat

class Model():
	"""docstring for Model"""
	def __init__(self, CATEGORY=None, n_comp=3, n_mix = 3, cov_type='diag', n_iter=1000):
		super(Model, self).__init__()
		self.CATEGORY = CATEGORY
		self.category = len(CATEGORY)
		self.n_comp = n_comp
		self.n_mix = n_mix
		self.cov_type = cov_type
		self.n_iter = n_iter
		
		self.models = []
		for k in range(self.category):
			model = hmm.GMMHMM(n_components=self.n_comp, n_mix = self.n_mix, 
								covariance_type=self.cov_type, n_iter=self.n_iter)
			self.models.append(model)


	def train(self, wavdict=None, labeldict=None):
		for k in range(self.category):
			subdata = []
			model = self.models[k]
			for x in wavdict:
				if labeldict[x] == self.CATEGORY[k]:
					mfcc_feat = compute_mfcc(wavdict[x])
					model.fit(mfcc_feat)

	
	def test(self, wavdict=None, labeldict=None):
		result = []
		for k in range(self.category):
			subre = []
			label = []
			model = self.models[k]
			for x in wavdict:
				mfcc_feat = compute_mfcc(wavdict[x])
				
				re = model.score(mfcc_feat)
				subre.append(re)
				label.append(labeldict[x])
			
			result.append(subre)
		
		result = np.vstack(result).argmax(axis=0)
	
		result = [self.CATEGORY[label] for label in result]
		print('hasil：\n',result)
		print('label：\n',label)
		
		totalnum = len(label)
		correctnum = 0
		for i in range(totalnum):
		 	if result[i] == label[i]:
		 	 	correctnum += 1 
		print('akurasi :', correctnum/totalnum)


	def save(self, path="models.pkl"):
		
		joblib.dump(self.models, path)


	def load(self, path="models.pkl"):
		
		self.models = joblib.load(path)

CATEGORY = ['Q3', 'D3', 'Q4', 'D4', 'Q1', 'D1', 'Q2', 'D2']
wavdict, labeldict = gen_wavlist('train_data')
testdict, testlabel = gen_wavlist('test_data')	
models = Model(CATEGORY=CATEGORY)
models.train(wavdict=wavdict, labeldict=labeldict)
models.save()
models.load()
models.test(wavdict=wavdict, labeldict=labeldict)
models.test(wavdict=testdict, labeldict=testlabel)