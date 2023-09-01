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
import pickle
import spacy

random.seed()

nlp = spacy.load("en_core_web_lg")
data = load.load_data()
words = load.load_words()
dpr = models.setup_closedbook(0)

def split_to_substrings(s):
    if len(s) == 0:
        return []
    rv = []
    for i in range(1, len(s)):
        for rest in split_to_substrings(s[i:]):
            rv.append([s[:i]] + rest)
    rv.append([s])
    return rv

def filter_substrings(arr):
    rv = []
    for partition in arr:
        ok = True
        for word in partition:
            if word not in words:
                ok = False
                break
        if ok:
            rv.append(partition)
    return rv

def get_valid_substrings_old(s):
    return filter_substrings(split_to_substrings(s))

def get_valid_substrings(s):
    # assume all words are length 2 or greater
    if len(s) < 2:
        return []
    rv = []
    for i in range(2, len(s)):
        if s[:i] in words:
            for rest in get_valid_substrings(s[i:]):
                rv.append([s[:i]] + rest)
    if s in words:
        rv.append([s])
    return rv

def split_to_size(words, sz):
    if sz == 0:
        return [[]]
    rv = []
    for i in range(len(words)):
        for rest in split_to_size(words[i+1:], sz-1):
            rv.append([words[i]] + rest)
    return rv

def fit_partition(nondefn, partition):
    if len(nondefn.split()) < len(partition):
        return (-1e18, None, None)
    subseqs = split_to_size(nondefn.split(), len(partition))
    best_subseq = (-1e18, None)
    for subseq in subseqs:
        if len(subseq) == 1:
            continue
        score = 0
        for i in range(len(subseq)):
            score += nlp(subseq[i]).similarity(nlp(partition[i]))
        best_subseq = max(best_subseq, (score, subseq))
    if best_subseq[0] > 0:
        return (best_subseq[0], best_subseq[1], partition)
    else:
        return (-1e18, None, None)
    
def find_mapping(nondefn, ans):
    partitions = get_valid_substrings(ans)
    rv = (-1e18, None, None)
    for partition in partitions:
        rv = max(rv, fit_partition(nondefn, partition))
    return rv

nondefn, defn, ans = "outlaw leader", "managing money", "banking"
score, mapping, converted = find_mapping(nondefn, ans)
print("nondef:", nondefn)
print("selected:", mapping)
print("synonyms:", converted)
print("score:", score)
print("avg score:", score / len(mapping))
print()

defn, nondefn, ans = "agriculture", "in remote chinese dynasty", "farming"
score, mapping, converted = find_mapping(nondefn, ans)
print("nondef:", nondefn)
print("selected:", mapping)
print("synonyms:", converted)
print("score:", score)
print("avg score:", score / len(mapping))
print()

nondefn, defn, ans = "a combo on", "leave", "abandon"
score, mapping, converted = find_mapping(nondefn, ans)
print("nondef:", nondefn)
print("selected:", mapping)
print("synonyms:", converted)
print("score:", score)
print("avg score:", score / len(mapping))
print()

random.shuffle(data)

for datapoint in data:
    clue, nondef, defn, ans, sz = load.unwrap_data(datapoint)
    if not clue or not defn or not ans:
        continue
    score, mapping, converted = find_mapping(nondef, ans)
    if mapping and score / len(mapping) > 0.3:
        print("clue:", clue)
        print("nondef:", nondef)
        print("ans:", ans)
        print("selected:", mapping)
        print("synonyms:", converted)
        print("score:", score)
        print("avg score:", score / len(mapping))
        print()