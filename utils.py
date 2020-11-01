import math
import json
import os
from collections import OrderedDict

import numpy as np
from PIL import Image


def create_prob_dict(alphabet_list, path='./lab1_data/frequencies.json'):
    """
    Open json-file and create corresponding probability dictionary
    :param alphabet: a list of string symbols
    :param path: path to json
    :return: a dictionary, where keys: all possible bigrams,
        values: probabilites of corresponding bigrams
    """
    with open(path) as f:
        prob_dict = json.load(f)
    total_count = 0
    for _, v in prob_dict.items():
        total_count += v
    for k, _ in prob_dict.items():
        prob_dict[k] /= total_count

    for i in alphabet_list:
        for j in alphabet_list:
            if i + j not in prob_dict.keys():
                prob_dict[i + j] = 1e-128
    return prob_dict


def create_char_to_arr_dict(alphabet_list, path='./lab1_data/alphabet/'):
    """
    Create array, which contains a matrix respresantation of chars
    :param alphabet_list: a list of string symbols
    :param path: path to images
    :return: a dictionary, where keys: a char from 'alphabet_list',
        values: a corresponding matrix
    """
    char_to_arr_dict = dict()
    for character_name in alphabet_list:
        if character_name == " ":
            path_to_image = os.path.join(path, "space.png")
        else:
            path_to_image = os.path.join(path, character_name + ".png")
        character_arr = np.array(Image.open(path_to_image), dtype=np.int8)
        char_to_arr_dict[character_name] = character_arr
    return char_to_arr_dict


def read_input_sentence(index, path='./lab1_data/input/'):
    """
    Read an input image as numpy array
    :param index: int in range [0, 1, 2, ..., 15]
    :param path: path to input images
    :return: numpy array
    """
    list_input_txt = os.listdir(path)
    input_txt_file = list_input_txt[index]
    path_to_image = "./lab1_data/input/{}".format(input_txt_file)
    input_arr = np.array(Image.open(path_to_image), dtype=np.int8)
    print("You have read = ", input_txt_file)
    return input_arr


def calculate_ln_prob(noised_arr, label_arr, p):
    """
    Calculate probability
    :param noised_arr: n*m numpy matrix with 0 and 1
    :param label_arr: n*m numpy matrix with 0 and 1
    :param p: float probability in (0, 1)
    :return: float
    >>> calculate_ln_prob(np.array([[1]]), np.array([[0]]), 0.5)
    -0.6931471805599453
    >>> calculate_ln_prob(np.array([[1]]), np.array([[0]]), 0.2)
    -1.6094379124341003
    >>> calculate_ln_prob(np.array([[1, 0]]), np.array([[0, 1]]), 0.75)
    0.810930216216329
    >>> calculate_ln_prob(np.array([[1, 1]]), np.array([[1, 1]]), 0.75)
    -1.3862943611198906
    >>> calculate_ln_prob(np.array([[0, 0]]), np.array([[0, 0]]), 0.75)
    -1.3862943611198906
    >>> calculate_ln_prob(np.array([[1, 0],[ 1, 0]]), np.array([[0, 1],[1, 0]]), 0.75)
    0.810930216216329
    >>> calculate_ln_prob(np.array([[1, 0],[ 1, 0]]), np.array([[1, 1],[1, 1]]), 0.75)
    0.810930216216329
    >>> calculate_ln_prob(np.array([[1, 1],[ 1, 1]]), np.array([[1, 1],[1, 1]]), 0.75)
    -1.3862943611198906
    """
    assert noised_arr.shape == label_arr.shape
    assert 0 < p < 1
    xor_sum = (noised_arr ^ label_arr).sum()
    return math.log(p/(1-p))*xor_sum + math.log(1-p)


def create_indexes_dict(char_to_arr_dict, input_arr, char_distr, p):
    """
    Create the data structure (dictionary of dictionaries), which being used
    for computation. Keys: integers from 0, to m+1. Values: a dictionary
    where keys: a char (exist 'word', where the last symbol is this char),
    where values:  [(float) a probability of the 'word',
    (string)'word', string where the last symbol is a key of a dict
    (int)step on which the word has been created]
    :param char_to_arr_dict: a dictionary, where keys: a char from 'alphabet_list',
        values: a corresponding matrix
    :param input_arr: n*m numpy array
    :param char_distr: a dictionary, where keys: all possible bigrams,
        values: probabilites of corresponding bigrams
    :param p: float probability in (0, 1)
    :return: a dictionary, where keys: ints from 0, to m+1,
        values: a dictionary, where keys: char (exist 'word', where the last
                                                symbol this char),
                valuse:[(float)prob, (string)'word',
                        (int)step on which has been created]

    """
    indexes_dict = dict()
    for i in range(input_arr.shape[1]+1):
        indexes_dict[i] = None

    for k, v in char_to_arr_dict.items():
        current_width = v.shape[1]
        input_character = input_arr[:, :current_width]
        prob = calculate_ln_prob(input_character, v, p)
        if char_distr is not None:
            prob += math.log(char_distr[" " + k])

        if indexes_dict[current_width] is None:
            indexes_dict[current_width] = OrderedDict()
            indexes_dict[current_width][k] = [prob, k, 0]
        else:
            indexes_dict[current_width][k] = [prob, k, 0]
    return indexes_dict


def update_indexes_dict(indexes_dict, char_to_arr_dict, input_arr,
                        char_distr, p, n_step):
    """
    Update for 'indexes_dict'.
    :param indexes_dict: a dictionary, where keys: ints from 0, to m+1,
        values: a dictionary, where keys: char (exist 'word', where the last
                                                symbol this char),
                valuse:[(float)prob, (string)'word',
                        (int)step on which has been created]
    :param char_to_arr_dict: a dictionary, where keys: a char from
        'alphabet_list', values: a corresponding matrix
    :param input_arr: n*m numpy array
    :param char_distr: a dictionary, where keys: all possible bigrams,
        values: probabilites of corresponding bigrams
    :param p: float probability in (0, 1)
    :param n_step: a step for updating 'indexes_dict'
    :return: None
    """
    for ind, probs_dicts in indexes_dict.items():
        if probs_dicts is not None:
            update_one_index(indexes_dict, int(ind), probs_dicts,
                            char_to_arr_dict, input_arr, char_distr, p, n_step)


def update_one_index(indexes_dict, index, probs_dicts, char_to_arr_dict,
                     input_arr, char_distr, p, n_step):
    """
    Update 'indexes_dict', means update probabilities for all indexes
    ('index' + length a word from alpabet)
    :param indexes_dict: a dictionary, where keys: ints from 0, to m+1,
        values: a dictionary, where keys: char (exist 'word', where the last
                                                symbol this char),
                valuse:[(float)prob, (string)'word',
                        (int)step on which has been created]
    :param index: int from 0 to m+1
    :param probs_dicts: a dictionary, where keys: char (exist 'word', where
                    the last symbol this char),
                valuse:[(float)prob, (string)'word',
                        (int)step on which has been created]
    :param char_to_arr_dict: a dictionary, where keys: a char from
        'alphabet_list', values: a corresponding matrix
    :param input_arr: n*m numpy array
    :param char_distr: a dictionary, where keys: all possible bigrams,
        values: probabilites of corresponding bigrams
    :param p: float probability in (0, 1)
    :param n_step: a step for updating 'indexes_dict'
    :return: None
    """
    # loop thorugh alphabet dict
    for new_char, new_char_arr in char_to_arr_dict.items():
        # loop through previous chars
        for prev_char, prev_data in probs_dicts.items():
            prev_prob, prev_str, prev_step = prev_data
            curent_width = new_char_arr.shape[1]
            new_length = index + curent_width
            # for step n_step we using previous symols from n_step - 1
            if prev_step != n_step - 1 or new_length > input_arr.shape[1]:
                continue
            input_character = input_arr[:, index:new_length]
            current_prob = calculate_ln_prob(input_character, new_char_arr, p)
            new_prob = current_prob + prev_prob
            if char_distr is not None:
                new_prob += char_distr[prev_char + new_char]
            # position 'new_length' appears first time in 'indexes_dict'
            if indexes_dict[new_length] is None:
                indexes_dict[new_length] = OrderedDict()
                indexes_dict[new_length][new_char] = [new_prob,
                                                      prev_str + new_char,
                                                      n_step]
            # 'new_char' appears first time in 'indexes_dict[new_length]'
            if new_char not in indexes_dict[new_length].keys():
                indexes_dict[new_length][new_char] = [new_prob,
                                                      prev_str + new_char,
                                                      n_step]

            # update probs and with better word
            if new_prob > indexes_dict[new_length][new_char][0]:
                indexes_dict[new_length][new_char] = [new_prob,
                                                      prev_str + new_char,
                                                      n_step]


def predict(indexes_dict):
    """
    Get a sequence from a value of the 'indexes_dict's last key with the
    bigggest probability
    :param indexes_dict: a dictionary, where keys: ints from 0, to m+1,
        values: a dictionary, where keys: char (exist 'word', where the last
                                                symbol this char),
                valuse:[(float)prob, (string)'word',
                        (int)step on which has been created]
    :return: string
    """
    last_ind = max(indexes_dict.keys())
    max_pred = -1e+16
    best_sqc = None
    for prob, sqc, _ in indexes_dict[last_ind].values():
        if prob > max_pred:
            max_pred = prob
            best_sqc = sqc
    return best_sqc