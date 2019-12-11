import numpy as np
import pandas as pd
import os, random
from python_speech_features import mfcc
from scipy.io import wavfile
from hmmlearn import hmm
from sklearn.externals import joblib
from KNN import KNN

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

    def improvment_single_test(self, label, data=None):
        Q_label = ['Q_1', 'Q_2', 'Q_3', 'Q_4']
        D_label = ['D2_4', 'D1_1', 'D1_2', 'D1_4', 'D2_2', 'D2_3', 'D3_3', 'D4_3',
                   'D3_2', 'D4_1', 'D1_3', 'D2_1', 'D4_4', 'D4_2', 'D3_1', 'D3_4']

        if label.split('_')[0][0] == 'D':
            compare_label = random.choice(Q_label)
        else:
            compare_label = random.choice(D_label)

        score_data_label = []
        score_compare_data_label = []
        model = self.load('models/'+label+'.pkl')

        if data != None:
            mfcc_feat = self.compute_mfcc(data)
        else:
            mfcc_feat = self.compute_mfcc('input/tempData/data.wav')

        score_data = model.score(mfcc_feat)

        label_df = []
        values_df = []
        for filename in os.listdir('train/'+label):
            mfcc_file = self.compute_mfcc('train/'+label+'/'+filename)
            mfcc_feat_file = model.score(mfcc_file)
            score_data_label.append(mfcc_feat_file)
            values_df.append(mfcc_feat_file)
            label_df.append(label)

        for filename in os.listdir('train/'+compare_label):
            mfcc_file = self.compute_mfcc('train/'+compare_label+'/'+filename)
            mfcc_feat_file = model.score(mfcc_file)
            score_compare_data_label.append(mfcc_feat_file)
            values_df.append(mfcc_feat_file)
            label_df.append(compare_label)

        
        print(label_df)
        print(values_df)
        data_df = {'values':values_df,'label':label_df}
        df = pd.DataFrame(data=data_df)
        

        
        # min_val = min(score_data_label)
        # max_val = max(score_data_label)

        # if round(score_data) in range(round(min_val),round(max_val)):
        #     print(True)
        # else:
        #     print(False)
        return df.to_csv('dataKNN/'+label+'.csv', index=False)
    
    def predict(self, label, data=None):

        model = self.load('models/'+label+'.pkl')
        if data != None:
            mfcc_feat = self.compute_mfcc(data)
        else:
            mfcc_feat = self.compute_mfcc('input/tempData/data.wav')

        score_data = model.score(mfcc_feat)
       
        data_df = {0: [score_data]}
        df = pd.DataFrame(data=data_df)
        
        obj = KNN('dataKNN/'+label+'.csv')
        path = 'modelsKNN/'+label+'.pkl'
        predict = obj.predict(path,df)

        return predict[0]



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

    # CATEGORY = ['idgham','iqlab']
    CATEGORY = ['Q_3', 'D2_4', 'D1_1', 'D1_2', 'D1_4', 'D2_2', 'D2_3', 'D3_3', 'D4_3',
                'Q_4', 'D3_2', 'Q_1', 'D4_1', 'D1_3', 'D2_1', 'D4_4', 'D4_2', 'D3_1', 'Q_2', 'D3_4']
    obj = HMM(CATEGORY=CATEGORY)
    wavdict, labeldict = obj.gen_wavlist('train')
    # print(list(set(labeldict.values())))
    # testdict, testlabel = obj.gen_wavlist('data/testing')

    # obj.train(wavdict=wavdict, labeldict=labeldict)
    # obj.test(wavdict=testdict, labeldict=testlabel)
    # predict = obj.single_test(CATEGORY,'input/tempData/L4_D2_3.wav')
    # for i in CATEGORY:
    #     predict = obj.improvment_single_test(i)
    # print(predict)
    predict_final = obj.predict('D1_1', 'input/tempData/data.wav')


