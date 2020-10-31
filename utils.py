import math
import json
import os

import numpy as np
from PIL import Image


def create_prob_dict(alphabet, path='./lab1_data/frequencies.json'):
    with open(path) as f:
        prob_dict = json.load(f)
    total_count = 0
    for _, v in prob_dict.items():
        total_count += v
    for k, _ in prob_dict.items():
        prob_dict[k] /= total_count

    for i in alphabet:
        for j in alphabet:
            if i + j not in prob_dict.keys():
                prob_dict[i + j] = 1e-16
    return prob_dict


def create_char_to_arr_dict(alphabet_list, path='./lab1_data/alphabet/'):
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
    list_input_txt = os.listdir(path)
    input_txt_file = list_input_txt[index]
    path_to_image = "./lab1_data/input/{}".format(input_txt_file)
    input_arr = np.array(Image.open(path_to_image), dtype=np.int8)
    print("You have read = ", input_txt_file)
    return input_arr


def calculate_ln_prob(noised_arr, label_arr, p):
    assert noised_arr.shape == label_arr.shape
    assert 0 < p < 1
    xor_sum = (noised_arr ^ label_arr).sum()
    return math.log(p/(1-p))*xor_sum + math.log(1-p)


def create_dict_char_prob_width(char_to_arr_dict, input_arr, char_distr, p):
    char_to_prob_width = dict()
    for k, v in char_to_arr_dict.items():
        current_width = v.shape[1]
        input_character = input_arr[:, :current_width]
        prob = calculate_ln_prob(input_character, v, p) + math.log(char_distr[k + " "])
        char_to_prob_width[k] = [prob, current_width]
    return char_to_prob_width


def recalculate_probs(char_to_prob_width,
                      char_to_arr_dict,
                      input_arr,
                      char_distr,
                      p):
    char_to_prob_width_new = dict()
    # loop through an entire alphabet and an corresponding array
    for new_k, char_arr in char_to_arr_dict.items():
        max_prob = -1e+10
        predicted_digit = None
        new_shift = None
        # loop throught alphabet and previous prob
        for old_k, old_v in char_to_prob_width.items():
            curent_width = char_arr.shape[1]
            old_prob, old_shift = old_v
            input_character = input_arr[:, old_shift:(old_shift + curent_width)]
            current_prob = calculate_ln_prob(input_character, char_arr, p)
            prob_sum = current_prob + old_prob
            if prob_sum > max_prob:
                max_prob = prob_sum
                predicted_digit = old_k
                new_shift = old_shift + curent_width
        char_to_prob_width_new[predicted_digit + new_k] = [max_prob, new_shift]
    return char_to_prob_width_new