from pyspark.sql import *
import pyspark.sql.functions as F
import pandas as pd
import numpy as np

spark = SparkSession \
    .builder \
    .appName("Privacy Project") \
    .getOrCreate()

netflix_names = spark.read.csv("./netflix-prize-data/movie_titles.csv", inferSchema=True) # id, date, name
imdb_names    = spark.read.csv("title.basics.tsv", inferSchema=True, header=True, sep="\t")
imdb_ratings  = spark.read.csv("title.ratings.tsv", inferSchema=True, header=True, sep="\t")

print(netflix_names.columns)
print(imdb_names.columns)
print(imdb_ratings.columns)

res = netflix_names\
    .join(imdb_names, (imdb_names.primaryTitle == netflix_names._c2) & (imdb_names.startYear == netflix_names._c1))\
    .join(imdb_ratings, "tconst")

res = res.select('_c0', 'averageRating', 'numVotes').withColumnRenamed('_c0', 'movieId')
print("{}/{} movies have been found in IMDB.".format(res.count(), netflix_names.count()))
res.toPandas().to_csv("imdb.ratings.csv")
