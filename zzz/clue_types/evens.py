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
alts = [
    nlp("even"),
    nlp("odd"),
    nlp("alternate"),
    nlp("switch")
] 

data = load.load_data()

def find_alt(clue, ans):
    cur = ''
    pos = []
    for d in range(2):
        i = d
        skip = False
        while i < len(clue):
            if skip:
                while i < len(clue) and not clue[i].isalpha():
                    i += 1
                i += 1
                if i >= len(clue):
                    break
                skip = False
                continue
            else:
                while i < len(clue) and not clue[i].isalpha():
                    i += 1
                if i >= len(clue):
                    break
                pos.append(i)
            cur += clue[i]
            if clue[i].islower():
                clue = clue[:i] + clue[i].upper() + clue[i+1:]
            if len(cur) > len(ans):
                clue = clue[:pos[0]] + clue[pos[0]].lower() + clue[pos[0]+1:]
                cur = cur[1:]
                pos = pos[1:]
            if cur == ans:
                indicator = (-1e18, '')
                words = clue.split()
                for k in range(len(words)):
                    for l in range(k, len(words)):
                        joint = ' '.join(words[k:l+1])
                        if joint != joint.lower():
                            continue
                        joint_nlp = nlp(joint)
                        if joint_nlp.vector_norm == 0:
                            continue
                        indicator = max(indicator, (sum([alt.similarity(joint_nlp) for alt in alts]), joint))
                return ' '.join(words), indicator[1]
            i += 1
            skip = True
    return None, None

for datapoint in data:
    clue, nondef, defn, ans, sz = load.unwrap_data(datapoint)
    if not clue or not defn or not ans:
        continue
    evens, indicator = find_alt(clue, ans)
    if evens:
        print("clue:", clue)
        print("ans:", ans)
        print("evens:", evens)
        print("indicator:", indicator)
        print()