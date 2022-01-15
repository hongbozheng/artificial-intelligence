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

    # -------------------------construct model-----------------------------
    all_tag = []  # record all tags
    all_word = {}  # record all words
    transfer_count = {}  # [tag1][tag2]  tag1 -----> tag2
    emission_count = {}  # [tag1][word]  tag1 generate word
    initial_count = {}  # [tag]
    # -------------------------construct emission--------------------------
    for sentence in train:
        for word in sentence:
            if word[1] not in all_tag:  # add to all_tag
                all_tag.append(word[1])
            if word[0] not in all_word:
                all_word[word[0]] = {}
                all_word[word[0]][word[1]] = 1
            else:
                if word[1] not in all_word[word[0]]:
                    all_word[word[0]][word[1]] = 1
                else:
                    all_word[word[0]][word[1]] += 1
            if word[1] not in emission_count:
                emission_count[word[1]] = {}
                emission_count[word[1]][word[0]] = 1
            else:
                if word[0] not in emission_count[word[1]]:
                    emission_count[word[1]][word[0]] = 1
                else:
                    emission_count[word[1]][word[0]] += 1
    # ---------------------construct transfer & initial--------------------
    for sentence in train:
        if (sentence[0][1] not in initial_count):
            initial_count[sentence[0][1]] = 1
        else:
            initial_count[sentence[0][1]] += 1
        for num in range(len(sentence) - 1):
            curr_tag = sentence[num][1]
            next_tag = sentence[num + 1][1]
            if curr_tag not in transfer_count:
                transfer_count[curr_tag] = {}
                transfer_count[curr_tag][next_tag] = 1
            else:
                if next_tag not in transfer_count[curr_tag]:
                    transfer_count[curr_tag][next_tag] = 1
                else:
                    transfer_count[curr_tag][next_tag] += 1
    # -------------------------formalize everything------------------------
    hapax = get_hapax(emission_count, all_tag)
    initial_count["unknown"] = 0
    factor1 = 0.000009
    factor2 = 0.00001
    factor3 = 0.000005
    initial_count = smooth(initial_count, factor1)
    for tag in all_tag:
        emission_count[tag]["unknown"] = 0
        emission_count[tag] = smooth_part2(emission_count[tag], factor2, hapax, tag)
        if tag != 'END':
            transfer_count[tag]["unknown"] = 0
            transfer_count[tag] = smooth(transfer_count[tag], factor3)
    # -------------------------make predications---------------------------
    predicts = []
    for sentence in test:
        temp = []
        dp_array = [[0 for i in range(len(all_tag))] for j in range(len(sentence))]
        record_array = [[0 for i in range(len(all_tag))] for j in range(len(sentence))]
        i = 0
        for j in range(len(all_tag)):
            dp_array[0][j] = get_initial(initial_count, all_tag[j]) + get_emission(emission_count, all_tag[j],
                                                                                   sentence[i])
        i += 1
        while (i < len(sentence)):
            for j in range(len(all_tag)):  # next col
                dp_array[i][j] = dp_array[i - 1][0] + get_transfer(transfer_count, all_tag[0],
                                                                   all_tag[j]) + get_emission(emission_count,
                                                                                              all_tag[j],
                                                                                              sentence[i])
                record_array[i][j] = 0
                for k in range(len(all_tag)):  # pre col
                    cmp = dp_array[i - 1][k] + get_transfer(transfer_count, all_tag[k], all_tag[j]) + get_emission(
                        emission_count, all_tag[j], sentence[i])
                    if (cmp > dp_array[i][j]):
                        dp_array[i][j] = cmp
                        record_array[i][j] = k
            i += 1
        # back track
        i = len(sentence) - 1
        dp_row = dp_array[i]
        spot = dp_row.index(max(dp_row))
        temp.append((sentence[i], all_tag[spot]))
        while (i >= 1):
            temp.append((sentence[i - 1], all_tag[record_array[i][spot]]))
            spot = record_array[i][spot]
            i -= 1
        temp = temp[::-1]
        predicts.append(temp)
    # return predicts
    # -------------------------extra credit implemetion---------------------------------
    final = []
    for sentence in predicts:
        temp = []
        for tuple in sentence:
            if tuple[0] not in all_word:
                if tuple[0][len(tuple[0]) - 2:len(tuple[0])] == "ly":
                    temp.append((tuple[0], "ADV"))
                elif tuple[0][len(tuple[0]) - 4:len(tuple[0])] == "wise" and len(tuple[0]) > 4:
                    temp.append((tuple[0], "ADV"))
                elif tuple[0][len(tuple[0]) - 2:len(tuple[0])] == "ed":
                    temp.append((tuple[0], "ADJ"))
                elif tuple[0][len(tuple[0]) - 3:len(tuple[0])] == "ous":
                    temp.append((tuple[0], "ADJ"))
                elif tuple[0][len(tuple[0]) - 3:len(tuple[0])] == "ble":
                    temp.append((tuple[0], "ADJ"))
                elif tuple[0][len(tuple[0]) - 2:len(tuple[0])] == "fy":
                    temp.append((tuple[0], "VERB"))
                elif tuple[0][len(tuple[0]) - 3:len(tuple[0])] == "ize":
                    temp.append((tuple[0], "VERB"))
                elif tuple[0][len(tuple[0]) - 3:len(tuple[0])] == "ise":
                    temp.append((tuple[0], "VERB"))
                elif tuple[0][len(tuple[0]) - 4:len(tuple[0])] == "iate":
                    temp.append((tuple[0], "VERB"))
                else:
                    temp.append((tuple[0], "NOUN"))
            else:
                if tuple[0] == "what":
                    temp.append(("what", max(all_word["what"], key=all_word["what"].get)))
                elif tuple[0] == "as":
                    temp.append(("as", max(all_word["as"], key=all_word["as"].get)))
                else:
                    temp.append(tuple)
        final.append(temp)
    return final

def get_hapax(emission_count, all_tag):
    result = {}  # [word][tag]
    for tag in emission_count:
        for word in emission_count[tag]:
            if emission_count[tag][word] == 1:
                if tag in result:
                    result[tag] += 1
                else:
                    result[tag] = 1
    all = sum(result.values())
    for tag in all_tag:
        if tag in result:
            result[tag] = result[tag] / all
        else:
            result[tag] = 0.000001
    return result

def smooth(map, factor):
    all = sum(map.values())
    for key in map:
        all += factor
        map[key] += factor
    for key in map:
        map[key] = map[key] / all
        map[key] = math.log(map[key])
    return map

def smooth_part2(map, factor, hapax, tag):  # emission[tag][word] key is word
    all = sum(map.values())
    for key in map:
        all += factor * hapax[tag]
        map[key] += factor * hapax[tag]
    for key in map:
        map[key] = map[key] / all
        map[key] = math.log(map[key])
    return map

def get_initial(initial_count, tag):
    if tag in initial_count:
        return initial_count[tag]
    else:
        return initial_count["unknown"]

def get_emission(emission_count, tag, word):
    if word in emission_count[tag]:
        return emission_count[tag][word]
    else:
        return emission_count[tag]["unknown"]

def get_transfer(transfer_count, tag1, tag2):
    if tag1 != 'END' and tag2 != 'END':
        if tag2 in transfer_count[tag1]:
            return transfer_count[tag1][tag2]
        else:
            return transfer_count[tag1]["unknown"]
    return 0.001
