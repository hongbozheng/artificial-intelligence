"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

from collections import Counter

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''

    word_count = {}
    tag_count = {}
    pred = []

    for sentence in train:
        for pair in sentence:
            word,tag = pair
            if tag in tag_count:
                tag_count[tag] += 1
            else:
                tag_count[tag] = 1

            if word not in word_count:
                word_count[word] = {}
            if tag in word_count[word]:
                word_count[word][tag] += 1
            else:
                word_count[word][tag] = 1

    max_tag = getMaxTag(tag_count)

    for sentence in test:
        sentence_pred = []
        for word in sentence:
            if word in word_count:
                word_tag_count = word_count[word]
                word_max_tag = getMaxTag(word_tag_count)
                sentence_pred.append((word,word_max_tag))
            else:
                sentence_pred.append((word,max_tag))
        pred.append(sentence_pred)

    return pred

def getMaxTag(tag_count):
    return max(tag_count.keys(),key=(lambda key:tag_count[key]))