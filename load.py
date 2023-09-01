import json
import os
import string

def load_data(load_type="data"):
    def clean_clue(clue):
        if "(" not in clue:
            return clue, None
        suffix = clue[clue.rindex("("):]
        length = suffix[1:-1]
        clue = clue[:clue.rindex("(")]
        return clue, length

    def parse_row(row):
        clue = row[1]
        answer = row[2]
        definition = row[3]
        clue, length = clean_clue(clue)
        return clue, length, answer, definition

    data = []

    def parse_file(file_path):
        with open(file_path, "r") as f:
            file_data = json.load(f)
            rows = file_data["rows"]
            for row in rows:
                clue, length, answer, definition = parse_row(row)
                data.append((clue, length, answer, definition))

    def parse_dir(dir_path):
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            parse_file(file_path)

    parse_dir("data/georgeho/")
    
    data = [d for d in data if d[0] and d[1] and d[2] and d[3]]

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

def unwrap_data(data, print_err=False):
    clue, length, ans, defn = data
    if not clue or not defn:
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
    return clue, nondef, defn, ans, length

def load_words():
    with open('words.txt', 'r') as f:
        word_list = f.read().splitlines()
    words = set(word_list)
    return words

if __name__ == "__main__":
    data = load_data()
    print(len(data))