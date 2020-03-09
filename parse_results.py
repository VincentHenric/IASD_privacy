#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 11:09:41 2020

@author: henric
"""

import pandas as pd
import numpy as np

import re
from scipy import stats
import os

def extract_result_dict(lines2):

    results = {}
    results['index'] = []
    results['aux'] = []
    results['top_index'] = []
    results['score'] = []
    results['ecc'] = []
    for i,line in enumerate(lines2):
        if i==0:
            continue
        else:
            try:
                l = line.split(',')
                results['index'].append(l[0])
                results['aux'].append(parse_aux_dataframe(l[1]))
                if len(l)>=3:
                    results['top_index'].append(l[2])
                    results['score'].append(l[3])
                    results['ecc'].append(l[4])
                else:
                    results['top_index'].append(None)
                    results['score'].append(None)
                    results['ecc'].append(None)
            except:
                print(line)
                continue
    return results

def parse_results(filename, with_header=True):
    with open("results/{}".format(filename), "r") as f:
        lines = f.readlines()
        
    l = np.array([i for i,line in enumerate(lines) if line.startswith('custId')])
    freq = stats.mode(np.diff(l)).mode[0]
        
    lines2 = ['']
    l = ''
    for i,line in enumerate(lines):
        lines2[-1] += line
        if i%freq==0:
            lines2.append('')

    if lines2[-1]=='':
        lines2.pop(-1)
        
    results = extract_result_dict(lines2)
    df = pd.DataFrame(results)
    df = df.astype({'index':np.int32, 'aux':'object', 'top_index':np.int32, 'score':np.float32, 'ecc':np.float32})
    return df

def parse_aux_dataframe(s):
    l2 = s.split('\n')
    col_names = re.sub("\s\s+", " ", l2[0]).strip().split(' ')
    idx_name = l2[1].strip()
    records = [re.sub("\s\s+", " ", line).split(' ') for line in l2[2:]]

    df = pd.DataFrame(records, columns=[idx_name]+col_names).set_index(idx_name)
    return df


if __name__ == '__main__':
    filenames = sorted(os.listdir('results'))
    
    experiment_map = {'exp_0.txt':[2,2,3],
                      'exp_1.txt':[3,4,3],
                      'exp_2.txt':[6,8,3],
                      'exp_3.txt':[2,2,14],
                      'exp_4.txt':[3,4,14],
                      'exp_5.txt':[6,8,14]}
    
    experiments = {}
    
    for filename in filenames:
        try:
            df = parse_results(filename, with_header=True)
            print('nb experiments for {}: {}'.format(filename, len(df)))
            experiments[filename] = df
        except:
            print('parsing failed for {}'.format(filename))
            continue