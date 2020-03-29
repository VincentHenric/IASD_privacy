#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 19:15:53 2020

@author: henric
"""

import sys

line_nb = 0
movie_id = -1
print('movieId,custId,rating,date')

for line in sys.stdin:
    line = line.strip()
    
    if line.endswith(':'):
        movie_id = line[:-1]
    
    elif ',' in line:
        row = '{},{}'.format(movie_id, line)
        print(row)
        
    line_nb += 1
    
   # if line_nb >100:
   #     break