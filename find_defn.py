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

max_answers = 1000000

def get_possible_defns(clue):
    words = clue.split()
    rv = [clue]
    for i in range(1, len(words)):
        rv.append(' '.join(words[:i]))
        rv.append(' '.join(words[i:]))
    return rv

random.shuffle(data)

for datapoint in data:
    clue, nondef, defn, ans, sz = load.unwrap_data(datapoint)
    if not clue or not defn or not ans:
        continue
    ans = ans.lower().replace(' ', '')
    possible_defns = get_possible_defns(clue)
    possible_ans, scores = models.answer_clues(dpr, possible_defns, max_answers, output_strings=True)
    # how to directly compare ans to defn? versus computing all possible answers and finding ranking
    for i in range(len(possible_ans)):
        possible_ans[i] = [ansn.lower().replace(' ', '') for ansn in possible_ans[i] if len(ansn) - ansn.count(' ') == len(ans)]
    for i in range(len(possible_defns)):
        # possible_defns[i] = (possible_defns[i], possible_ans[i].index(ans) if ans in possible_ans[i] else 1e9)
        possible_defns[i] = (possible_defns[i], scores[i][possible_ans[i].index(ans)] if ans in possible_ans[i] else -1e9)
    possible_defns.sort(key=lambda x: x[1], reverse=True)
    print("clue:    ", clue)
    print("ans: ", ans)
    if possible_defns[0][1] != -1e9:
        print("most probable defn:  ", possible_defns[0][0])
        print("score:   ", possible_defns[0][1])
    else:
        print("most probable defn:  N/A")
        print("score:   N/A")
    print("actual defn:", defn)
    # find actual defn in possible defns
    actual_idx = 0
    while actual_idx < len(possible_defns) and possible_defns[actual_idx][0] != defn:
        actual_idx += 1
    if actual_idx < len(possible_defns):
        print("actual defn rank:", actual_idx)
        print("actual defn score:", possible_defns[actual_idx][1])
    else:
        print("actual defn rank: N/A")
        print("actual defn score: N/A")
    print()