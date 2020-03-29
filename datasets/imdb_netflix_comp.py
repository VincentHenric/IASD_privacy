#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 21:54:36 2020

@author: henric
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyspark.sql import *
import pyspark.sql.functions as F

spark = SparkSession \
    .builder \
    .appName("Privacy Project") \
    .getOrCreate()
    
netflix = spark.read.csv("./netflix.ratings.csv", inferSchema=True, header=True)
imdb    = spark.read.csv("./imdb.ratings.csv", inferSchema=True, header=True, sep=",")

netflix = netflix.drop('custId', 'date').groupBy('movieId').agg(F.mean('rating').alias('averageRating_netflix'),
                                                      F.count('rating').alias('numVotes_netflix'))

netflix.join(imdb.drop('_c0'), 'movieId').toPandas().to_csv('common_movies.csv', index=False)
