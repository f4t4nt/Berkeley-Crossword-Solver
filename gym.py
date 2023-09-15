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

train_data, val_data, test_data = load.load_data(load_type="all", randomize=True)
cw_dict = load.load_words(only_ans=True)
indic_dict = load.load_indicators()

test_data = test_data[:100]

def evaluate(method, get_ranks=False):
    ranks = []
    for clue, nondef, defn, ans, sz in test_data:
        ans_cut = ans.lower().replace(' ', '')
        ranklist = method(clue, len(ans_cut), cw_dict, indic_dict)
        if ans_cut in ranklist:
            print(f"{clue} -> {ans_cut}: {ranklist[ans_cut]}")
            ranks.append(ranklist[ans_cut][0] + 1)
        else:
            print(f"Error: {clue} -> {ans_cut}")
            ranks.append(len(ranklist) + 1)
    if get_ranks:
        return ranks
    return 1 / np.mean([1 / r for r in ranks])

import methods

print("Basic:", evaluate(methods.basic))
# print()
# print("Baseline:", evaluate(methods.baseline))