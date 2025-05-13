# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
import numpy as np
import math
from tqdm import tqdm
from collections import Counter
import reader

"""
This is the main entry point for MP4. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""


"""
  load_data calls the provided utility to load in the dataset.
  You can modify the default values for stemming and lowercase, to improve performance when
       we haven't passed in specific values for these parameters.
"""
 
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming is {stemming}")
    print(f"Lowercase is {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels



# Keep this in the provided template
def print_paramter_vals(laplace,pos_prior):
    print(f"Unigram Laplace {laplace}")
    print(f"Positive prior {pos_prior}")


"""
You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
Notice that we may pass in specific values for these parameters during our testing.
"""

def naiveBayes(train_set, train_labels, dev_set, laplace=0.008, pos_prior=0.8,silently=False):
    # laplace smoothing parameter is tuned manually to 0.008 to get the best performance
    # pos_prior is calculated with the # of positive and negative comments in dev_set
    # total # of positive comments in dev_set = 4000
    # total # of negative comments in dev_set = 1000
    # pos_prior = 4000/(4000+1000) = 0.8 to get the best performance

    # Keep this in the provided template
    print_paramter_vals(laplace,pos_prior)

    # get the total count of a word in positive dataset
    word_count_pos = getWordCount(train_set, train_labels, 1)
    # get the total count of a word in negative dataset
    word_count_neg = getWordCount(train_set, train_labels, 0)

    # get the probability of a word given the dataset is positive
    # get UNK word probability given the dataset is positive
    word_prob_pos, UNK_pos = calcProb(word_count_pos, laplace)
    # get the probability of a word give the dataset is negative
    # get UNK word probability given the dataset is negative
    word_prob_neg, UNK_neg = calcProb(word_count_neg, laplace)

    # storing the final result
    yhats = []
    for doc in tqdm(dev_set,disable=silently):

        pos_prob,neg_prob=unigramModel(word_prob_pos,UNK_pos,word_prob_neg,UNK_neg,pos_prior,doc)

        # the dataset seems to be positive
        if (pos_prob > neg_prob):
            yhats.append(1)
        else:
            yhats.append(0)

    return yhats


# Keep this in the provided template
def print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior):
    print(f"Unigram Laplace {unigram_laplace}")
    print(f"Bigram Laplace {bigram_laplace}")
    print(f"Bigram Lambda {bigram_lambda}")
    print(f"Positive prior {pos_prior}")


# main function for the bigrammixture model
def bigramBayes(train_set, train_labels, dev_set, unigram_laplace=0.005, bigram_laplace=0.005, bigram_lambda=0.5,pos_prior=0.5, silently=False):
    # Keep this in the provided template
    print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)

    # get the total count of a word in positive dataset
    word_count_pos = getWordCount(train_set, train_labels, 1)
    # get the total count of a word in negative dataset
    word_count_neg = getWordCount(train_set, train_labels, 0)

    # get the probability of a word given the dataset is positive
    # get UNK word probability given the dataset is positive
    word_prob_pos, UNK_pos = calcProb(word_count_pos, unigram_laplace)
    # get the probability of a word give the dataset is negative
    # get UNK word probability given the dataset is negative
    word_prob_neg, UNK_neg = calcProb(word_count_neg, unigram_laplace)

    # get the total count of bigram in positive dataset
    bigram_count_pos = getBigram(train_set, train_labels, 1)
    # get the total count of bigram in negative dataset
    bigram_count_neg = getBigram(train_set, train_labels, 0)

    # get the probability of a bigram given the dataset is positive
    # get the probability of a UNK bigram give the dataset is positive
    bigram_prob_pos, bigram_UNK_pos = calcProb(bigram_count_pos, bigram_laplace)

    # get the probability of a bigram given the dataset is negative
    # get the probability of a UNK bigram give the dataset is negative
    bigram_prob_neg, bigram_UNK_neg = calcProb(bigram_count_neg, bigram_laplace)

    # storing the final result
    yhats = []
    dev_pos = []
    dev_neg = []
    bigram_dev_pos = []
    bigram_dev_neg = []

    for doc in tqdm(dev_set,disable=silently):

        # get the pos and neg probability of each doc in dev_set
        pos_prob,neg_prob=unigramModel(word_prob_pos,UNK_pos,word_prob_neg,UNK_neg,pos_prior,doc)

        dev_pos.append(pos_prob)
        dev_neg.append(neg_prob)

        # get the pos and neg probability of each doc in dev_set
        bigram_pos_prob,bigram_neg_prob=bigramModel(bigram_prob_pos,bigram_UNK_pos,bigram_prob_neg,bigram_UNK_neg,pos_prior,doc)

        bigram_dev_pos.append(bigram_pos_prob)
        bigram_dev_neg.append(bigram_neg_prob)

    for i in range(len(dev_set)):
        # calculate the pos and neg probability with bigram lambda
        positive_prob = (1-bigram_lambda)*dev_pos[i]+bigram_lambda*bigram_dev_pos[i]
        negative_prob = (1-bigram_lambda)*dev_neg[i]+bigram_lambda*bigram_dev_neg[i]

        # the dataset seems to be positive
        if positive_prob > negative_prob:
            yhats.append(1)
        else:
            yhats.append(0)

    return yhats

def getWordCount(train_set, train_labels, isPos):
    """
    Calculate the total # of a word appeared in a given pos/neg dataset

    :param train_set:    the dataset that is going to be trained
    :param train_labels: the train label of each word (eg. pos = 1, neg = 0)
    :param isPos: const  used to check if the given word is pos/neg
    :return:             the total # of a word in a pos/neg dataset stored in dictionary
    """

    # create word count dictionary
    word_count = {}

    for i in range(len(train_labels)):

        # check the train_label to see if it's from pos/neg dataset
        if(train_labels[i] != isPos):
            continue

        # get the current dataset
        cur_set = train_set[i]

        for word in cur_set:
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1

    return word_count

def calcProb(word_count, laplace):
    """
    Calculate the probability of a word given it is in a positive/negative dataset
    Add the key:value -> word:probability into probability dictionary

    :param word_count: dictionary with the total # of each words in the pos/neg dataset
    :param laplace: laplace smoothing parameter (required tuning manually)
    :return: the probability of a word given it is in a pos/neg dataset stored in dictionary
    """

    # create a probability dictionary
    prob = {}
    total_words = 0
    total_types = len(word_count)

    # calculate the total # of words in word_count dictionary
    for word in word_count:
        total_words += word_count[word]

    #                                     laplace_smoothing_parameter
    # P(W|dataset) =  ----------------------------------------------------------------------
    #                  total # number of words in dataset + LSP*(total # of word types +1 )
    UNK = laplace/(total_words+laplace*(total_types+1))

    #                          # of times W appears + laplace_smoothing_parameter
    # P(W|dataset) =  ----------------------------------------------------------------------
    #                  total # number of words in dataset + LSP*(total # of word types +1 )
    for word in word_count:
        probability = (laplace+word_count[word])/(total_words+laplace*(total_types+1))

        # add the probability of 'word' into dictionary
        prob[word] = probability

    return prob, UNK

def unigramModel(word_prob_pos, UNK_pos, word_prob_neg, UNK_neg, pos_prior, doc):
    """
    Calculate the probability of a given word in doc is positive or negative

    :param word_prob_pos: the probability of a word given the dataset is positive
    :param UNK_pos:       the probability of a UNK word given the dataset is positive
    :param word_prob_neg: the probability of a word given the dataset is negative
    :param UNK_neg:       the probability of a UNK word given the dataset is negative
    :param pos_prior:     probability of positive from prior experiment
    :param doc:           the file contains all the words being trained
    :return:              the positive probability and negative probability of a word
    """

    # create variable to store pos/neg probability
    pos_prob = neg_prob = 0

    for word in doc:
        if word in word_prob_pos:
            pos_prob += np.log(word_prob_pos[word])
        else:
            pos_prob += np.log(UNK_pos)

        if word in word_prob_neg:
            neg_prob += np.log(word_prob_neg[word])
        else:
            neg_prob += np.log(UNK_neg)

    # adjust the probability with the prior pos probability to improve accuracy
    pos_prob += np.log(pos_prior)
    neg_prob += np.log(1-pos_prior)

    return pos_prob, neg_prob

def getBigram(train_set, train_labels, isPos):
    """
    Calculate the total # of a bigram appeared in a given pos/neg dataset

    :param train_set:    the dataset that is going to be trained
    :param train_labels: the train label of each word (eg. pos = 1, neg = 0)
    :param isPos: const  used to check if the given word is pos/neg
    :return:             the total # of a word in a pos/neg dataset stored in dictionary
    """
    bigram = {}

    for i in range(len(train_labels)):
        if train_labels[i] != isPos:
            continue

        cur_set = train_set[i]
        for j in range(len(cur_set)-1):

            # get the bigram from the current set
            bg = tuple((cur_set[j],cur_set[j+1]))
            if bg in bigram:
                bigram[bg] += 1
            else:
                bigram[bg] = 1
    return bigram

def bigramModel(bigram_prob_pos,bigram_UNK_pos,bigram_prob_neg,bigram_UNK_neg,pos_prior,doc):
    """
    Calculate the probability of a given bigram in doc is positive or negative

    :param bigram_prob_pos: the probability of a bigram given the dataset is positive
    :param bigram_UNK_pos:  the probability of a UNK bigram given the dataset is positive
    :param bigram_prob_neg: the probability of a bigram given the dataset is negative
    :param bigram_UNK_neg:  the probability of a UNK bigram given the dataset is negative
    :param pos_prior:       probability of positive from prior experiment
    :param doc:             the file contains all the words being trained
    :return:                the positive probability and negative probability of a bigram
    """

    bigram_pos_prob = bigram_neg_prob = 0

    for word in range(len(doc)-1):

        bg = tuple((doc[word],doc[word+1]))

        if bg in bigram_prob_pos:
            bigram_pos_prob += np.log(bigram_prob_pos[bg])
        else:
            bigram_pos_prob += np.log(bigram_UNK_pos)

        if bg in bigram_prob_neg:
            bigram_neg_prob += np.log(bigram_prob_neg[bg])
        else:
            bigram_neg_prob += np.log(bigram_UNK_neg)

    # adjust the probability with the prior pos probability to improve accuracy
    bigram_pos_prob += np.log(pos_prior)
    bigram_neg_prob += np.log(1-pos_prior)

    return bigram_pos_prob, bigram_neg_prob