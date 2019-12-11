import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib


class KNN:

    def __init__(self, data=None):
        self.data = self.read_csv(data)
        self.name_file = data.split('/')[1].split('.')[0]

    def read_csv(self, data):
        return pd.read_csv(data, delimiter=",", encoding="utf8", header=None)

    def xylabel(self):
        X = self.data.iloc[1:, :-1]
        Y = self.data.iloc[1:, -1]
        return X, Y

    def random_split(self, data_vector):
        X, Y = data_vector
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.20)

        return X_train, X_test, y_train, y_test

    def train(self, data_split):
        X_train, X_test, y_train, y_test = data_split
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train, y_train)
        print(X_test)
        exit()
        y_pred = model.predict(X_test)
        report = classification_report(y_pred, y_test)
        score = accuracy_score(y_pred, y_test)
        # score = int(score)*100

        print(y_test)
        print(report)
        print(score)

        return self.save(model, 'modelsKNN/'+self.name_file+'.pkl')

    def save(self, model, path):
        return joblib.dump(model, path)

    def load(self, path):

        return joblib.load(path)
        # return pickle.load(open(path, 'rb'))

    def predict(self, model, data):
        model = self.load(model)
        result = model.predict(data)
        
        return result


if __name__ == "__main__":

    obj = KNN('dataKNN/Q_4.csv')
    # xylabel = obj.xylabel()
    # split = obj.random_split(xylabel)
    # model = obj.train(split)

    # exit()
    data_df = {0: [-47941.713536]}
    df = pd.DataFrame(data=data_df)
    
    # df = df.iloc[:, -1]
    # df.to_csv('input/tempDataCsv/Q_4.csv', index=False)

    path = 'modelsKNN/Q_4.pkl'
    # data_test = KNN('input/tempDataCsv/Q_4.csv')
    # df_test = data_test.data.iloc[1:]
    # data_test.data.drop(data_test..data.index[[2, 3]])
    predict = obj.predict(path,df)
    print(predict)
