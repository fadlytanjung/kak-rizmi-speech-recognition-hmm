from python_speech_features import mfcc
from scipy.io import wavfile
from hmmlearn import hmm
from sklearn.externals import joblib
import numpy as np
import os

class HMM:
    """docstring for Model"""

    def __init__(self, CATEGORY=None, n_comp=3, n_mix=3, cov_type='diag', n_iter=1000):
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
                    label = dirpath.split('/')[len(dirpath.split('/'))-1]
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
        subre = []
        label = []
        for i in path_model:

            model = self.load('models/'+i+'.pkl')
        
            mfcc_feat = self.compute_mfcc(path_data)
            score = model.score(mfcc_feat)
            label.append(i)
            subre.append(score)
        result = np.vstack(subre).argmax(axis=0)
        # print(result,subre,label)

        # re = model.score(mfcc_feat)
        # import math
        # logprob, seq = model.decode(mfcc_feat)
       
        # result = model.predict(mfcc_feat)
        return label[result[0]]
       

    def test(self, wavdict=None, labeldict=None):
        result = []
        for k in range(self.category):
            subre = []
            label = []
            model = self.load('models/'+self.CATEGORY[k]+'.pkl')
            alamat = {}
            o = 0
            for x in wavdict:
                # print(self.CATEGORY[k], labeldict[x])
                mfcc_feat = self.compute_mfcc(wavdict[x])
                alamat[o] = wavdict[x]
                o += 1
                re = model.score(mfcc_feat)
                subre.append(re)
                label.append(labeldict[x])
            result.append(subre)
       
        
        result = np.vstack(result).argmax(axis=0)
        result = [self.CATEGORY[label] for label in result]
        # print('hasil：\n', result)
        # print('label：\n', label)

        # totalnum = len(label)
        # correctnum = 0
        # for i in range(totalnum):
        #     if result[i] == label[i]:
        #         correctnum += 1
        # print('akurasi :', correctnum/totalnum)

        totalnum = len(label)
        correctnum = 0
        kelompok = {}
        false = {}
        for k in label:
            kelompok[k] = 0
        for i in range(totalnum):
            if result[i] == label[i]:
                correctnum += 1
                try:
                    kelompok[label[i]] += 1
                except NameError:
                    kelompok[label[i]] = 0
            else:
                print(alamat[i] + ' =======> ' + result[i] +
                      '(' + str(result[i] == label[i]) + ')')
                false[alamat[i]] = result[i]
                '(' + str(result[i] == label[i]) + ')'
        detail = {}
        for j in kelompok:
            print(j + ': ' + str(kelompok[j]))
            detail[j] = str(kelompok[j])
        print('akurasi :' + str(correctnum) + '/' +
              str(totalnum) + ':', correctnum/totalnum)

        return [false, detail, correctnum/totalnum]

if __name__ == "__main__":

    CATEGORY = ['idgham','iqlab']
    obj = HMM(CATEGORY=CATEGORY)
    # wavdict, labeldict = obj.gen_wavlist('data/training')
    # testdict, testlabel = obj.gen_wavlist('data/testing')

    # obj.train(wavdict=wavdict, labeldict=labeldict)
    # obj.test(wavdict=testdict, labeldict=testlabel)
    predict = obj.single_test(CATEGORY,'input/tempData/L4_D2_3.wav')
    print(predict)


