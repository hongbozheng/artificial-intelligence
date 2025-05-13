"""
Part 4: Here should be your best version of viterbi, 
with enhancements such as dealing with suffixes/prefixes separately
"""

import numpy as np
import math

LAPLACE = 0.001

def viterbi_3(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''

    TAG, tag_count, start_tag_count = getTagCount(train)
    tag_trans_count = getTransitionCount(train)
    WORD, word_count, word_tag, tag_emis_count = getEmissionCount(train)
    tag_init_prob = getInitProb(train, tag_count, start_tag_count)
    HAPAX, hapax_tag_count = getHapax(word_count, word_tag)
    hapax_prob = getHapaxProb(TAG, HAPAX, hapax_tag_count)
    trans_prob = getTransitionProb(tag_count, tag_trans_count)
    emis_prob = getEmissionProb(tag_count, WORD, tag_emis_count, hapax_prob)

    test_pred = []

    for sentence in test:
        V = np.zeros((len(TAG), len(sentence)))
        backtrace = np.zeros(V.shape, dtype=int)

        for i in range(len(TAG)):
            if sentence[0] in emis_prob[TAG[i]]:
                V[i][0] = tag_init_prob[TAG[i]]+emis_prob[TAG[i]][sentence[0]]
            else:
                V[i][0] = tag_init_prob[TAG[i]]+emis_prob[TAG[i]]['UNK']

        for col in range(1, V.shape[1]):
            curr_word = sentence[col]

            for row in range(V.shape[0]):
                prob = np.zeros(V.shape[0])

                for prev_row in range(V.shape[0]):
                    if curr_word in emis_prob[TAG[row]]:
                        prob[prev_row] = V[prev_row][col-1]+trans_prob[(TAG[prev_row],TAG[row])]+\
                                         emis_prob[TAG[row]][curr_word]
                    else:
                        prob[prev_row] = V[prev_row][col-1]+trans_prob[(TAG[prev_row],TAG[row])]+\
                                         emis_prob[TAG[row]]['UNK']

                best_prob = np.argmax(prob)
                V[row][col] = prob[best_prob]
                backtrace[row, col] = best_prob

        row = np.argmax(V[:,-1])  # pick the best row

        sentence_pred = []
        for col in reversed(range(1, V.shape[1])):
            sentence_pred.append((sentence[col], TAG[row]))
            row = backtrace[row, col]
        sentence_pred.append((sentence[0], TAG[0]))
        sentence_pred.reverse()

        test_pred.append(sentence_pred)

    return test_pred


def getTagCount(train):
    TAG = []
    tag_count = {}
    start_tag_count = {}

    for sentence in train:
        for word, tag in sentence:
            if tag in tag_count:
                tag_count[tag] += 1
                if tag == 'START':
                    start_tag_count[tag] += 1
            else:
                tag_count[tag] = 1
                if tag == 'START':
                    start_tag_count[tag] = 1
                else:
                    start_tag_count[tag] = 0
                TAG.append(tag)

    return TAG, tag_count, start_tag_count


def getInitProb(train, tag_count, start_tag_count):
    tag_init_prob = {}
    for tag in tag_count:
        tag_init_prob[tag] = math.log((LAPLACE+start_tag_count[tag])/(len(train)+LAPLACE*(len(tag_count)+1)))
    return tag_init_prob


def getTransitionCount(train):
    tag_trans_count = {}

    for sentence in train:
        for i in range(len(sentence)-1):
            word0, tag0 = sentence[i]
            word1, tag1 = sentence[i+1]

            if (tag0, tag1) in tag_trans_count:
                tag_trans_count[(tag0, tag1)] += 1
            else:
                tag_trans_count[(tag0, tag1)] = 1

    return tag_trans_count


def getTransitionProb(tag_count, tag_trans_count):
    trans_prob = {}

    for tag0 in tag_count:
        for tag1 in tag_count:
            if (tag0, tag1) in tag_trans_count:
                trans_prob[(tag0, tag1)] = math.log((LAPLACE+tag_trans_count[(tag0,tag1)]) /
                                                    (tag_count[tag0]+LAPLACE*(1+len(tag_trans_count))))
            else:
                trans_prob[(tag0, tag1)] = math.log(LAPLACE/(tag_count[tag0]+LAPLACE*(1+len(tag_trans_count))))

    return trans_prob


def getEmissionCount(train):
    WORD = set()
    word_count = {}
    word_tag = {}
    tag_emis_count = {}

    for sentence in train:
        for word, tag in sentence:
            if tag not in tag_emis_count:
                tag_emis_count[tag] = {}
            if word in tag_emis_count[tag]:
                tag_emis_count[tag][word] += 1
            else:
                tag_emis_count[tag][word] = 1
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1
            if word not in WORD:
                WORD.add(word)
            word_tag[word] = tag
    WORD.add('UNK')

    return WORD, word_count, word_tag, tag_emis_count


def getEmissionProb(tag_count, WORD, tag_emis_count, hapax_prob):
    emis_prob = {}

    for tag in tag_count:
        emis_prob[tag] = {}
        for word in WORD:
            if word in tag_emis_count[tag]:
                emis_prob[tag][word] = math.log((LAPLACE*hapax_prob[tag]+tag_emis_count[tag][word])/
                                                (tag_count[tag]+LAPLACE*len(WORD)*hapax_prob[tag]))
            else:
                emis_prob[tag][word] = math.log(LAPLACE*hapax_prob[tag]/
                                                (tag_count[tag]+LAPLACE*len(WORD)*hapax_prob[tag]))

    return emis_prob


def getHapax(word_count, word_tag):
    hapax_tag_count = {}
    HAPAX = 0

    for word in word_count:
        if word_count[word] == 1:
            HAPAX += 1
            if word_tag[word] in hapax_tag_count:
                hapax_tag_count[word_tag[word]] += 1
            else:
                hapax_tag_count[word_tag[word]] = 1

    return HAPAX, hapax_tag_count


def getHapaxProb(TAG, HAPAX, hapax_tag_count):
    hapax_prob = {}
    for tag in TAG:
        if tag in hapax_tag_count:
            hapax_prob[tag] = (LAPLACE+hapax_tag_count[tag])/(HAPAX+LAPLACE*(1+len(TAG)))
        else:
            hapax_prob[tag] = (LAPLACE)/(HAPAX+LAPLACE*(1+len(TAG)))

    return hapax_prob
