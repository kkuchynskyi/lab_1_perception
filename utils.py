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
                prob_dict[i + j] = 1e-16
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
            indexes_dict[current_width][k] = [prob, k]
        else:
            indexes_dict[current_width][k] = [prob, k]
    return indexes_dict


def update_one_index(indexes_dict, char_to_arr_dict,
                     input_arr, char_distr, p, index):
    """
    Update 'indexes_dict' for 'index'
    Update 'indexes_dict', means update probabilities for all indexes
    ('index' + length a word from alpabet)
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
    :param index: int from 0 to m+1
    :return: None
    """
    # loop thorugh alphabet dict
    for new_char, new_char_arr in char_to_arr_dict.items():
        curent_width = new_char_arr.shape[1]
        if index < curent_width or indexes_dict[index - curent_width] is None:
            continue
        best_prob = -1e+16
        best_seq = None
        # loop through previous chars
        for prev_char, prev_data in indexes_dict[index - curent_width].items():
            prev_prob, prev_str = prev_data
            input_character = input_arr[:, (index - curent_width):index]
            current_prob = calculate_ln_prob(input_character, new_char_arr, p)
            new_prob = current_prob + prev_prob
            if char_distr is not None:
                new_prob += char_distr[prev_char + new_char]
            if new_prob > best_prob:
                best_prob = new_prob
                best_seq = prev_str + new_char

        if indexes_dict[index] is None:
            indexes_dict[index] = OrderedDict()
        indexes_dict[index][new_char] = [best_prob, best_seq]


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
    for prob, sqc in indexes_dict[last_ind].values():
        if prob > max_pred:
            max_pred = prob
            best_sqc = sqc
    return best_sqc