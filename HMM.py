from python_speech_features import mfcc
from scipy.io import wavfile
from hmmlearn import hmm
from sklearn.externals import joblib
import numpy as np
import os

class HMM:
    """docstring for Model"""

    def __init__(self, CATEGORY=None, n_comp=15, n_mix=3, cov_type='diag', n_iter=1000):
        super(HMM, self).__init__()
        self.CATEGORY = CATEGORY
        self.category = len(CATEGORY)
        self.n_comp = n_comp
        self.n_mix = n_mix
        self.cov_type = cov_type
        self.n_iter = n_iter
        self.models = []
        for k in range(self.category):
            model = hmm.GMMHMM(n_components=self.n_comp, n_mix=self.n_mix,
                            covariance_type=self.cov_type, n_iter=self.n_iter)
            self.models.append(model)

    def gen_wavlist(self, wavpath):

        lab = set()
        wavdict = {}
        labeldict = {}
      
        for (dirpath, dirnames, filenames) in os.walk(wavpath):
            for filename in filenames:
                if filename.endswith('.wav'):
                    filepath = os.sep.join([dirpath, filename])
                    fileid = filename.strip('.wav')
                    wavdict[fileid] = filepath
                    label = dirpath.split('/')[1]
                    labeldict[fileid] = label
                    lab.add(label)
        # self.CATEGORY = list(set(labeldict.values()))
        return wavdict,labeldict

    def compute_mfcc(self, file):
        fs, audio = wavfile.read(file)
        mfcc_feat = mfcc(audio, samplerate=(fs/2), numcep=100)
        # nfft=700)
        return mfcc_feat

    def train(self, wavdict=None, labeldict=None):

        filemodel = {}
        path_model = 'models/'

        for k in range(self.category):
            subdata = []
            model = self.models[k]
            
            for x in wavdict:
               
                if labeldict[x] == self.CATEGORY[k]:
                    mfcc_feat = self.compute_mfcc(wavdict[x])
                    model.fit(mfcc_feat)
            
            filemodel[self.CATEGORY[k]] = model
            self.save(model, path_model+self.CATEGORY[k]+'.pkl')
        return 'model saved'

    def save(self, model, path):
        return joblib.dump(model, path)
    
    def load(self, path):

        return joblib.load(path)

    def single_test(self, path_model, path_data):
        model = self.load(path_model)
        # print(model)
        # for i in model:
        #     print(i)

        # exit()
        mfcc_feat = self.compute_mfcc(path_data)
        # print(mfcc_feat)
        # log = np.array(mfcc_feat)
        # print(log)
        # y_hat = np.argmax(log, axis=0)
        # print(y_hat)
        # exit()
        re = model.score(mfcc_feat)
        import math
        logprob, seq = model.decode(mfcc_feat)
        # print(math.exp(logprob),logprob)
        # print(seq)
        # print(re*-1/100000)
        result = model.predict(mfcc_feat)
        temp = {}
        for i in result:
            if i in temp:
                temp[i]+=1
            else:
                temp[i] = 1
        
        # print(max(temp),max(temp.values()),temp)
        # print(model.sample(n_samples=9))
        # sampel = model.sample(n_samples=9)
        # score_sampel = model.score_samples(sampel[0])[1]
        
        # for i in sampel:
        #     print(i)    
        
        # for i in range(9):
        #     print(model.score_samples(np.reshape(sampel,(9,1)),lengths=9))
        

    def test(self, wavdict=None, labeldict=None):
        result = []
        for k in range(self.category):
            subre = []
            label = []
            model = self.models[k]
            for x in wavdict:
                mfcc_feat = self.compute_mfcc(wavdict[x])
                re = model.score(mfcc_feat)
                subre.append(re)
                label.append(labeldict[x])
            result.append(subre)
            
        
        result = np.vstack(result).argmax(axis=0)
        print(result)
        result = [self.CATEGORY[label] for label in result]
        print('hasil：\n', result)
        print('label：\n', label)

        totalnum = len(label)
        correctnum = 0
        for i in range(totalnum):
            if result[i] == label[i]:
                correctnum += 1
        print('akurasi :', correctnum/totalnum)

if __name__ == "__main__":

    CATEGORY = ['D1_4', 'D2_3', 'D3_2', 'D3_1', 'D4_4', 'D3_3',
                'D2_2', 'D1_3', 'D2_4', 'D1_2', 'D4_1', 'D3_4',
                'D2_1', 'D4_2', 'D1_1', 'D4_3']
    obj = HMM(CATEGORY=CATEGORY)
    wavdict, labeldict = obj.gen_wavlist('train')
    testdict, testlabel = obj.gen_wavlist('test')
   
    obj.train(wavdict=wavdict, labeldict=labeldict)
    obj.test(wavdict=testdict, labeldict=testlabel)
    # exit()
    # obj.single_test('models/D2_1.pkl','input/tempData/L1_D4_1.wav')


