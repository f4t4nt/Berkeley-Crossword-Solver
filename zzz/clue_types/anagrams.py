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
anagrams = [
    nlp("anagram"),
    nlp("scramble"),
    nlp("mixed"),
    nlp("rearrange"),
    nlp("shuffle"),
    nlp("wacky"),
    nlp("jumble"),
    nlp("disorder"),
    nlp("disarrange")
] 

data = load.load_data()

def find_reverse(clue, ans):
    words = clue.split()
    ans_decomp = list(ans)
    ans_decomp.sort()
    for i in range(len(words)):
        word_decomp = list(words[i])
        word_decomp.sort()
        if word_decomp == ans_decomp:
            indicator = (-1e18, '')
            other = words[:i] + words[i+1:]
            for j in range(len(other)):
                for k in range(j, len(other)):
                    joint = ' '.join(other[j:k+1])
                    joint_nlp = nlp(joint)
                    indicator = max(indicator, (sum([joint_nlp.similarity(r) for r in anagrams]), joint))
            return f"{words[i]} -> {ans}", indicator
    return None, None

for datapoint in data:
    clue, nondef, defn, ans, sz = load.unwrap_data(datapoint)
    if not clue or not defn or not ans:
        continue
    anas, indicator = find_reverse(clue, ans)
    if anas:
        print("clue:", clue)
        print("ans:", ans)
        print("anagram:", anas)
        print("indicator:", indicator)
        print()