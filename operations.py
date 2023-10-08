import pandas as pd
import re
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer, T5ForConditionalGeneration
import tokenizers
import json
import puz
import os
import numpy as np
import streamlit as st
import scipy

import sys
import subprocess
import copy
import json

from itertools import zip_longest
from copy import deepcopy
import regex

from solver.Crossword import Crossword
from solver.BPSolver import BPSolver
from models import setup_closedbook, setup_t5_reranker, DPRForCrossword
from solver.Utils import print_grid

from utils import puz_to_json

import load
import models
import random

import re
import itertools

# input -> answers
# NOT answer, input -> probability

dpr = models.setup_closedbook(0)
cw_dict = load.load_words(only_ans=True)

# . bank0: original clue
# v bank1: bank0.split()
# v bank2: cleaned, no punctuation, lowercase, can have multiple options for each word
# v bank3: single option for each word

def clean(bank0):
    bank1 = []
    for i in range(len(bank0)):
        if isinstance(bank0[i], list):
            bank1.append([re.sub(r'[^\w\s]', '', w).lower() for w in bank0[i]])
        else:
            bank1.append(re.sub(r'[^\w\s]', '', bank0[i]).lower())
    return bank1

# yields next bank with only words
def next_bank(bank1, i=0, root=True):
    if i >= len(bank1):
        yield []
    else:
        if isinstance(bank1[i], list):
            for word in bank1[i]:
                for bank2 in next_bank(bank1, i+1, False):
                    yield [word] + bank2
        else:
            for bank2 in next_bank(bank1, i+1, False):
                yield [bank1[i]] + bank2
        
# yields random bank with only words
def rand_bank(bank0):
    while True:
        bank1 = []
        for i in range(len(bank0)):
            if isinstance(bank0[i], list):
                bank1.append(random.choice(bank0[i]))
            else:
                bank1.append(bank0[i])
        yield bank1

################################################
# if ans is None, returns all possible answers #
# else return probability                      #
################################################

def Alternation(bank0, indic=None, in_dict=True, ans=None):
    bank1 = clean(bank0)
    output = set()
    for bank2 in next_bank(bank1):
        merged = ''.join(bank2)
        for i in range(len(merged)):
            rv = ""
            for j in range(i, len(merged), 2):
                rv += merged[j]
                if not in_dict or rv in cw_dict:
                    output.add(rv)
    if ans is None:
        return list(output)
    else:
        if ans in output:
            return 1
        else:
            return 0

def Anagram(bank0, indic=None, in_dict=True, ans=None):
    bank1 = clean(bank0)
    output = set()
    for bank2 in next_bank(bank1):
        merged = ''.join(bank2)
        merged = sorted(merged)
        if in_dict:
            for word in cw_dict:
                word = re.sub(r'[^\w\s]', '', word).lower().replace(" ", "")
                if len(word) == len(merged) and sorted(word) == merged:
                    output.add(word)
        else:
            for perm in itertools.permutations(merged): # O(n!), fix later
                output.add(''.join(perm))
    if ans is None:
        return list(output)
    else:
        if ans in output:
            return 1
        else:
            return 0

def Concatenation(bank0, indic=None, in_dict=True, ans=None):
    bank1 = clean(bank0)
    output = set()
    for bank2 in next_bank(bank1):
        merged = ''.join(bank2)
        if not in_dict or merged in cw_dict:
            output.add(merged)
    if ans is None:
        return list(output)
    else:
        if ans in output:
            return 1
        else:
            return 0
    
def Container(bank0, indic=None, in_dict=True, ans=None):
    if ans is None:
        return []
    else:
        return 0

def Deletion(bank0, indic=None, in_dict=True, ans=None):
    if ans is None:
        return []
    else:
        return 0
    
def Hidden(bank0, indic=None, in_dict=True, ans=None):
    bank1 = clean(bank0)
    output = set()
    for bank2 in next_bank(bank1):
        merged = ''.join(bank2)
        lo = len(bank2[0])
        hi = len(merged) - len(bank2[-1])
        # for i in range(len(merged)): # could maybe make sure all words are at least partially included
        for i in range(lo):
            # for j in range(i+1, len(merged)+1):
            for j in range(min(i, hi)+1, len(merged)+1):
                rv = merged[i:j]
                if not in_dict or rv in cw_dict:
                    output.add(rv)
    if ans is None:
        return list(output)
    else:
        if ans in output:
            return 1
        else:
            return 0
    
# similar to hidden
def Initialism(bank0, indic=None, in_dict=True, ans=None):
    bank1 = clean(bank0)
    output = set()
    for bank2 in next_bank(bank1):
        word = ""
        for w in bank2:
            word += w[0]
        if not in_dict or word in cw_dict:
            output.add(word)
    if ans is None:
        return list(output)
    else:
        if ans in output:
            return 1
        else:
            return 0

def Terminals(bank0, indic=None, in_dict=True, ans=None):
    bank1 = clean(bank0)
    output = set()
    for bank2 in next_bank(bank1):
        word = ""
        for w in bank2:
            word += w[-1]
        if not in_dict or word in cw_dict:
            output.add(word)
    if ans is None:
        return list(output)
    else:
        if ans in output:
            return 1
        else:
            return 0

def Homophone(bank0, indic=None, in_dict=True, ans=None):
    if ans is None:
        return []
    else:
        return 0

def Insertion(bank0, indic=None, in_dict=True, ans=None):
    if ans is None:
        return []
    else:
        return 0

def Reversal(bank0, indic=None, in_dict=True, ans=None):
    bank1 = clean(bank0)
    output = set()
    for bank2 in next_bank(bank1):
        merged = ''.join(bank2)
        merged = merged[::-1]
        if not in_dict or merged in cw_dict:
            output.add(merged)
    if ans is None:
        return list(output)
    else:
        if ans in output:
            return 1
        else:
            return 0

def soft(target, all):
    return np.exp(target) / np.sum(np.exp(np.array(all)))

def Substitution(bank0, indic=None, in_dict=True, thresh=1000, ans=None):
    bank1 = clean(bank0)
    output = set()
    for bank2 in next_bank(bank1):
        merged = ' '.join(bank2)
        preds, preds_scores = models.answer_clues(dpr, [merged], 999999999, output_strings=True)
        preds = preds[0]
        preds_scores = preds_scores[0]
        cnt = 0
        for pred, pred_score in zip(preds, preds_scores):
            pred = re.sub(r'[^\w\s]', '', pred).lower()
            if not in_dict or pred in cw_dict:
                output.add((pred_score, pred))
                cnt += 1
            if cnt >= thresh:
                break
    output = list(output)
    output.sort(reverse=True)
    if ans is None:
        output = [o[1] for o in output]
        return output
    else:
        target = -1e9
        for pred_score, pred in output:
            if pred == ans:
                target = pred_score
                break
        all = [o[0] for o in output]
        return soft(target, all)

# yields random tree of operators
def rand_op_tree(operators):
    d = {}
    for i in range(len(operators)):
        if operators[i] in d:
            d[operators[i]] += 1
        else:
            d[operators[i]] = 0
        operators[i] = (operators[i], str(d[operators[i]]))
    while True:
        random.shuffle(operators)
        tree = dict()
        used = [operators[0]]
        for i in range(1, len(operators)):
            parent = random.choice(used)
            if parent not in tree:
                tree[parent] = []
            tree[parent].append(operators[i])
            used.append(operators[i])
        yield operators[0], tree

def print_tree(root, tree, depth=0):
    print("  " * depth, list(root))
    if root in tree:
        for child in tree[root]:
            print_tree(child, tree, depth+1)

def unwrap_tree(root, tree):
    if root not in tree:
        return [root]
    else:
        output = [root]
        for child in tree[root]:
            output += unwrap_tree(child, tree)
            output += [root]
        return output

def rand_assign(bank, root, tree):
    unwrapped = unwrap_tree(root, tree)
    assigned = {}
    for word in bank: # TODO: consecutive words should (probably) be in the same part
        op = random.choice(unwrapped)
        if op not in assigned:
            assigned[op] = []
        assigned[op].append(word)
    return assigned

# TODO
# 18 	Easily annoyed a couple of times aboard railway (5) https://times-xwd-times.livejournal.com/2242176.html
# 	RATTY - A and TT (couple of times) contained by (aboard) RY (railway).

# gas one two bottle
# one -(two)-> oneone -(bottle)-> neon

if __name__ == "__main__":
    # bank = ["a", ["bc", "de", "ef"], "g", ["hi", "jk"], "l", "m", ["no", "pq", "rs"]]
    # for cnt, bank2 in enumerate(next_bank(bank)):
    #     print(cnt, bank2)
    # print()
    # print(Alternation(["Odd", "stuff", "of", "Mr.", "Waugh", "is", "set" "for"]))
    # print()
    # print(Anagram(["defeat", "faced", "in"]))
    # print()
    # print(Anagram([Alternation(["adaeafaeaaata"]), "faced", "in"]))
    # print()
    # print(Concatenation([Substitution(["Outlaw"]), Substitution(["leader"])]))
    # print()
    # print(Hidden(["Found", "ermine,", "deer"]))
    # print()
    # print(Concatenation([Substitution(["left"]), Alternation(["judgeable"], in_dict=False)]))
    # print()
    # print(Hidden(["circumspect,"]))
    # print()
    # print(Hidden(["circumspect,", "really"]))
    # print()
    for cnt, (root, tree) in enumerate(rand_op_tree(["anagram", "concat", "alternation", "substitution", "hidden"])):
        print(cnt)
        print_tree(root, tree)
        print(unwrap_tree(root, tree))
        print(rand_assign(["a", "b", "c", "d", "e", "f", "g"], root, tree))
        print()
        if cnt >= 5:
            break