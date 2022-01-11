import numpy as np
from hmm import HMM

def model_training(train_data, tags):
    """
    Train an HMM based on training data

    Inputs:
    - train_data: (1*num_sentence) a list of sentences, each sentence is an object of the "line" class 
            defined in data_process.py (read the file to see what attributes this class has)
    - tags: (1*num_tags) a list containing all possible POS tags

    Returns:
    - model: an object of HMM class initialized with parameters (pi, A, B, obs_dict, state_dict) calculated 
            based on the training dataset
    """

    # unique_words.keys() contains all unique words
    unique_words = get_unique_words(train_data)
    word2idx = {}
    tag2idx = dict()
    S = len(tags)
    ###################################################
    # TODO: build two dictionaries
    #   - from a word to its index 
    #   - from a tag to its index 
    # The order you index the word/tag does not matter, 
    # as long as the indices are 0, 1, 2, ...
    ###################################################
    idx = 0
    for word in unique_words.keys():
        word2idx[word] = idx
        idx += 1
    for i in range(len(tags)):
        tag2idx[tags[i]] = i

    pi = np.zeros(S)
    A = np.zeros((S, S))
    B = np.zeros((S, len(unique_words)))
    ###################################################
    # TODO: estimate pi, A, B from the training data.
    #   When estimating the entries of A and B, if  
    #   "divided by zero" is encountered, set the entry 
    #   to be zero.
    ###################################################
    for sentence in train_data:
        for i in range(len(sentence.tags)):
            pos = sentence.tags[i]
            pi[tag2idx[pos]] += 1
            for pos2 in sentence.tags[i+1:]:
                A[tag2idx[pos]][tag2idx[pos2]] += 1
        for i in range(len(sentence.words)):
            B[tag2idx[sentence.tags[i]]][word2idx[sentence.words[i]]] += 1

    pi = pi/pi.sum()
    A = A/A.sum(axis=1)[:, None]
    B = B/B.sum(axis=1)[:, None]
    # DO NOT MODIFY BELOW
    model = HMM(pi, A, B, word2idx, tag2idx)
    return model


def sentence_tagging(test_data, model, tags):
    """
    Inputs:
    - test_data: (1*num_sentence) a list of sentences, each sentence is an object of the "line" class
    - model: an object of the HMM class
    - tags: (1*num_tags) a list containing all possible POS tags

    Returns:
    - tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
    """
    tagging = []
    next_idx = max(model.obs_dict.values()) + 1
    p = np.full((len(tags), 1), 10**(-6))
    ######################################################################
    # TODO: for each sentence, find its tagging using Viterbi algorithm.
    #    Note that when encountering an unseen word not in the HMM model,
    #    you need to add this word to model.obs_dict, and expand model.B
    #    accordingly with value 1e-6.
    ######################################################################
    for sentence in test_data:
        for word in sentence.words:
            if word not in model.obs_dict.keys():
                model.obs_dict[word] = next_idx
                next_idx += 1
                model.B = np.hstack((model.B, p))
        tagging.append(model.viterbi(sentence.words))
    return tagging

# DO NOT MODIFY BELOW
def get_unique_words(data):

    unique_words = {}

    for line in data:
        for word in line.words:
            freq = unique_words.get(word, 0)
            freq += 1
            unique_words[word] = freq

    return unique_words
