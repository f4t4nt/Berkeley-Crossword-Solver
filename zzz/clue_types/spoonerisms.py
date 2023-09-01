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
import spacy

nlp = spacy.load("en_core_web_lg")
spooners = [
    nlp("spooner"),
    nlp("spoon")
] 

data = load.load_data()

def find_spoonerism(clue, ans):
    words = clue.split()
    for idx0 in range(len(words)):
        for idx1 in range(idx0, len(words)):
            if len(words[idx0]) == 1 or len(words[idx1]) == 1:
                continue
            if words[idx1][0] + words[idx0][1:] + words[idx0][0] + words[idx1][1:] == ans:
                indicator = (-1e18, '')
                other = words[:idx0] + words[idx0+1:idx1] + words[idx1+1:]
                for j in range(len(other)):
                    word_nlp = nlp(other[j])
                    indicator = max(indicator, (sum([word_nlp.similarity(r) for r in spooners]), other[j]))
                return ans, indicator
    return None, None

cnt, idx = 0, 0
for datapoint in data:
    clue, nondef, defn, ans, sz = load.unwrap_data(datapoint)
    if not clue or not defn or not ans:
        continue
    anas, indicator = find_spoonerism(clue, ans)
    # isis appears a lot, very few examples of spoonerisms
    if anas:
        print("clue:", clue)
        print("ans:", ans)
        print("spoonerism:", anas)
        print("indicator:", indicator)
        print()