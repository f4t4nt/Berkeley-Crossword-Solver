import json
import os
import random
import string
import csv

random.seed(0)

def clean_clue(clue):
    if "(" not in clue:
        return clue, None
    suffix = clue[clue.rindex("("):]
    sz = suffix[1:-1]
    clue = clue[:clue.rindex("(")]
    return clue, sz

def parse_row(row):
    clue = row[1]
    ans = row[2]
    defn = row[3]
    clue, sz = clean_clue(clue)
    return clue, sz, ans, defn

def parse_file(file_path, data):
    with open(file_path, "r") as f:
        file_data = json.load(f)
        rows = file_data["rows"]
        for row in rows:
            clue, sz, ans, defn = parse_row(row)
            data.append((clue, sz, ans, defn))
    return data

def unwrap_data(data, print_err=False):
    clue, sz, ans, defn = data
    if not clue or not defn or not ans:
        if print_err:
            print(f"Error: {clue} -> {defn}")
        return None, None, None, None, None
    clue = clue.strip()
    defn = defn.strip()
    ans = ans.strip()
    clue = clue.translate(str.maketrans('', '', string.punctuation + '\xa0' + '–'))
    defn = defn.translate(str.maketrans('', '', string.punctuation + '\xa0' + '–'))
    ans = ans.translate(str.maketrans('', '', string.punctuation + '\xa0' + '–'))
    clue = clue.lower()
    if "(" in clue:
        clue = clue[:clue.rindex("(")]
    clue = clue.strip()
    defn = defn.lower()
    ans = ans.lower()
    if clue[:len(defn)] == defn:
        nondef = clue[len(defn):]
    elif clue[-len(defn):] == defn:
        nondef = clue[:-len(defn)]
    else:
        if print_err:
            print(f"Error: {clue} -> {defn}")
        return None, None, None, None, None
    nondef = nondef.strip()
    if len(ans) == 1:
        return None, None, None, None, None
    return clue, nondef, defn, ans, sz

def parse_dir(dir_path):
    data = []
    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)
        data = parse_file(file_path, data)
    data = [unwrap_data(d) for d in data]
    data = [d for d in data if [e for e in d if e is None] == []]
    return data
        
def load_data(load_type="data", randomize=False):
    data = parse_dir("data/georgeho/")
    
    if randomize:
        random.shuffle(data)        

    if load_type == "data":
        return data
    elif load_type == "all":
        train_data = data[:int(len(data) * 0.8)]
        val_data = data[int(len(data) * 0.8):int(len(data) * 0.9)]
        test_data = data[int(len(data) * 0.9):]
        return train_data, val_data, test_data
    elif load_type == "train":
        train_data = data[:int(len(data) * 0.9)]
        val_data = data[int(len(data) * 0.9):]
        return train_data, val_data
    else:
        raise ValueError(f"Invalid load type: {load_type}")

def load_words(only_ans=False):
    if only_ans:
        data = parse_dir("data/georgeho/")
        words = set([d[3] for d in data])
    else:
        with open('words.txt', 'r') as f:
            word_list = f.read().splitlines()
        words = set(word_list)
    return words

# read lines from csv file such as
# 17,alternation,at intervals,"[304791](/data/clues/304791), [370799](/data/clues/370799), [660785](/data/clues/660785), [665444](/data/clues/665444)"
# and add {"at intervals": 4} to indic_dict["alternation"]
def load_indicators(): # TODO: also count how many times indicator phrases appear without being an indicator
    # open indicators.csv
    indic_dict = {}
    with open('indicators.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            # # row[0] is the idx
            # # row[1] is the type
            # # row[2] is the indicator
            # # row[3] is the list of clues where the indicator appears
            # if row[1] not in indic_dict:
            #     indic_dict[row[1]] = {}
            # indic_dict[row[1]][row[2]] = len(row[3].split(','))
            if row[2] not in indic_dict:
                indic_dict[row[2]] = []
            indic_dict[row[2]] += [(row[1], len(row[3].split(',')))]
    return indic_dict

if __name__ == "__main__":
    data = load_data()
    print(len(data))