# -*- coding: utf-8 -*-

import os, sys
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, model_selection
from sklearn import svm
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import precision_recall_fscore_support
import time
from copy import deepcopy
import matplotlib.pyplot as plt



def init():
    neg = open("/home/paulwu/COMP 550/rt-polaritydata/rt-polaritydata/rt-polarity.neg", "r")
    pos = open("/home/paulwu/COMP 550/rt-polaritydata/rt-polaritydata/rt-polarity.pos", "r")
    neg_lines = neg.readlines()
    pos_lines = pos.readlines()
    neg.close()
    pos.close()
    return neg_lines, pos_lines

def remove_stopwords_and_symbols(stop_words, line):
    return [w for w in line if w.isalpha() and not w in stop_words]

def remove_symbols(line):
    return [w for w in line if w.isalpha()]

def lemmatize(lemmatizer,line):
    return [lemmatizer.lemmatize(word) for word in line]

# modify
def create_data(pos_lines, neg_lines, tfidf=True, ngram_range = (1,1), max_features=5000):
    labels = np.concatenate((np.ones(len(pos_lines)),np.zeros(len(neg_lines))), axis = 0)
    data = pos_lines + neg_lines

    if tfidf:
        tfidf_vect = TfidfVectorizer(ngram_range = ngram_range, max_features = max_features)
        X = tfidf_vect.fit_transform(data)
        # print X.shape

    else:
        count_vect = CountVectorizer(ngram_range = ngram_range, max_features = max_features)
        X = count_vect.fit_transform(data)
        # print(X.shape)

    return X, labels

def print_report(method, report):
    print(method +":")
    print("Precision: " + str(report[0]) + " Recall: " + str(report[1]) + " f1-score: " + str(report[2]) + "\n")

def plot_f1_score():
    plt.xlabel('Maximum features')
    plt.ylabel('F1-score')
    plt.title('Histogram of IQ')
    plt.axis([2000, 10000])
    plt.show()

if __name__ == "__main__":

    # Read the dataset:
    neg_lines, pos_lines = init()

    # Declear a set of parameters to experiment with:
    remove_stopwords = [False, True]
    # ngrams = [(1,1),(1,2),(2,2)]
    ngrams = [(1,2)]
    max_features = range(2000, 10500, 500)

    log_reg_f1s = {"False": {"1,1": [], "1,2": [], "2,2": []}, "True": {"1,1": [], "1,2": [], "2,2": []}}
    svm_f1s = {"False": {"1,1": [], "1,2": [], "2,2": []}, "True": {"1,1": [], "1,2": [], "2,2": []}}
    multinomialNB_f1s = {"False": {"1,1": [], "1,2": [], "2,2": []}, "True": {"1,1": [], "1,2": [], "2,2": []}}

    for remove_stopword in remove_stopwords:

        log_reg_f1 = list()
        svm_f1 = list()
        multinomialNB_f1 = list()

        # Preprocessing:
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()

        if remove_stopword:
            print("With stopwords removed:")
            pos_filtered = [remove_stopwords_and_symbols(stop_words, k.strip("\n").split()) for k in pos_lines]
            neg_filtered = [remove_stopwords_and_symbols(stop_words, k.strip("\n").split()) for k in neg_lines]

        else:
            print("Without stopwords removed:")
            pos_filtered = [remove_symbols(k.strip("\n").split()) for k in pos_lines]
            neg_filtered = [remove_symbols(k.strip("\n").split()) for k in neg_lines]

        pos_lemmatized = [lemmatize(lemmatizer, line) for line in pos_filtered]
        neg_lemmatized = [lemmatize(lemmatizer, line) for line in neg_filtered]

        pos_data = [" ".join(s) for s in pos_lemmatized]
        neg_data = [" ".join(s) for s in neg_lemmatized]

        # Training with set of parameters:
        for ngram in ngrams:
            for max_feature in max_features:
                # Construct dataset X and labels
                X, labels = create_data(pos_data, neg_data, True, ngram, max_feature)

                print("Result with parameters max features: {}, gram number: from {} to {}.".format(max_feature, ngram[0], ngram[1]))

                predicted_log_reg = model_selection.cross_val_predict(LogisticRegression(), X, labels, cv=10)
                report = precision_recall_fscore_support(labels, predicted_log_reg, average='weighted')
                print_report("Logistic Regression", report)
                log_reg_f1.append(deepcopy(report[2]))

                predicted_svm = model_selection.cross_val_predict(svm.LinearSVC(), X, labels, cv=10)
                report = precision_recall_fscore_support(labels, predicted_svm, average='weighted') 
                print_report("SVM", report)
                svm_f1.append(deepcopy(report[2]))

                # start = time.time()
                # predicted_nb = model_selection.cross_val_predict(GaussianNB(), X.toarray(), labels, cv=10)
                # end = time.time()
                # print metrics.classification_report(labels, predicted_nb) 
                # print("Running time of NB:" + str(end - start))


                start = time.time()
                predicted_nb = model_selection.cross_val_predict(MultinomialNB(), X.toarray(), labels, cv=10)
                end = time.time()
                # print metrics.classification_report(labels, predicted_nb) 
                report = precision_recall_fscore_support(labels, predicted_nb, average='weighted') 
                print_report("Multinomial naive bayes", report)
                multinomialNB_f1.append(deepcopy(report[2]))

                # print("Running time of MultinomialNB:" + str(end - start) + '\n')
        log_reg_f1s.append(deepcopy(log_reg_f1))
        svm_f1s.append(deepcopy(svm_f1))
        multinomialNB_f1s.append(deepcopy(multinomialNB_f1))




    # With stopwords removed



    # Plot Logistic Regression f1-score across max_features
    # Plot SVM f1-score across max_features

    # Plot MultinomialNB f1-score across max_features

