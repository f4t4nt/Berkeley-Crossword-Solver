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
terminals = [
    nlp("last"),
    nlp("final"),
    nlp("end"),
    nlp("finish"),
    nlp("conclude")
]

data = load.load_data()

def find_terminals(clue, ans):
    words = clue.split()
    for i in range(len(words)):
        jw, ja = i, 0
        while jw < len(words) and ja < len(ans):
            if words[jw][-1] == ans[ja]:
                jw += 1
                ja += 1
            else:
                break
        if ja == len(ans):
            indicator = (-1e18, '')
            other = []
            for k in range(len(words)):
                if k >= i and k < jw:
                    words[k] = words[k][:-1] + words[k][-1].upper()
                else:
                    other.append(words[k])
            for k in range(len(other)):
                for l in range(k, len(other)):
                    joint = ' '.join(other[k:l+1])
                    joint_nlp = nlp(joint)
                    indicator = max(indicator, (sum([terminal.similarity(joint_nlp) for terminal in terminals]), joint))
            return ' '.join(words), indicator[1]
    return None, None

for datapoint in data:
    clue, nondef, defn, ans, sz = load.unwrap_data(datapoint)
    if not clue or not defn or not ans:
        continue
    terminals, indicator = find_terminals(clue, ans)
    if terminals:
        print("clue:", clue)
        print("ans:", ans)
        print("initials:", terminals)
        print("indicator:", indicator)
        print()