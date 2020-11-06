import numpy as np
import pandas as pd 
from sklearn import feature_extraction, linear_model, model_selection, preprocessing, metrics
import sklearn
import os
import sys

import sklearn
from sklearn.metrics.classification import precision_score



#os.chdir(os.path.dirname(os.path.realpath(__file__)))
#os.chdir("/Users/hango/Desktop/UCDavis(2019-)/Fall2020/ECS289/NLPwithDisasterTweets/code")


class ClassificationReport(): 

    def __init__(self, train_data, test_data):
        self.X_train, self.y_train = train_data
        self.X_test,  self.y_test  = test_data

        self.train_precision_score  = []
        self.train_f1_score         = []
        self.train_recall_score     = [] 

        self.test_preision_score    = []
        self.test_f1_score          = []
        self.test_recal_score       = []


    def report(self, model):
        prediction_train = model.predict(self.X_train)
        self.train_precision_score = metrics.precision_score(self.y_train, prediction_train, average = 'binary')
        self.train_f1_score        = metrics.f1_score(self.y_train, prediction_train, average = 'binary')
        self.train_recall_score    = metrics.recall_score(self.y_train, prediction_train, average = 'binary')

        prediction_test = model.predict(self.X_test)
        self.test_precision_score  = metrics.precision_score(self.y_test, prediction_test, average = 'binary')
        self.test_f1_score         = metrics.f1_score(self.y_test, prediction_test, average = 'binary')
        self.test_recall_score     = metrics.recall_score(self.y_test, prediction_test, average = 'binary')

        print("Training data f1_score: {:.4}, precision_score: {:.4}, recall_score: {:.4}".format(self.train_precision_score, self.train_f1_score, self.train_recall_score))
        print("Testing data  f1_score: {:.4}, precision_score: {:.4}, recall_score: {:.4}".format(self.test_precision_score, self.test_f1_score, self.test_recall_score))


class LogisticModel():

    def __init__(self, X_data, y_data):
        #divide up the training data into training set and validation set 
        self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(X_data, y_data, test_size = 0.2, random_state = 1)

    def build_model(self):
        clf = linear_model.LogisticRegression(random_state=0).fit(self.X_train, self.y_train)
        Metrics_logis = ClassificationReport((self.X_train, self.y_train), (self.X_test, self.y_test))
        Metrics_logis.report(clf)



if __name__ == "__main__":
    train_df = pd.read_csv("../data/train.csv")
    test_df  = pd.read_csv("../data/test.csv")

    #train_df.head()
    #train_df[train_df["target"] == 0]["text"].values[1]
    #train_df[train_df["target"] == 1]["text"].values[1]


    vectorizer = feature_extraction.text.CountVectorizer()#
    train_x = vectorizer.fit_transform(train_df["text"]).toarray()
    test_x = vectorizer.transform(test_df["text"]).toarray()


    Model_logic = LogisticModel(train_x, train_df["target"])
    Model_logic.build_model()










