from pyspark.sql import *
import pyspark.sql.functions as F
from pyspark.sql.types import LongType, TimestampType, StringType, StructField, StructType, BooleanType, ArrayType, IntegerType
import pandas as pd
import numpy as np

import privacy_spark as privacy
from typing import List

SCHEMA = StructType([
    StructField("movieId", IntegerType(), True),        
    StructField("custId", IntegerType(), True),
    StructField("rating", IntegerType(), True),
    StructField("date", TimestampType(), True)
])

#StructType(List(StructField(movieId,IntegerType,true),
#                StructField(custId,IntegerType,true),
#                StructField(rating,IntegerType,true),
#                StructField(date,TimestampType,true)))

class Experiment():
    def __init__(self, spark: SparkSession):
        self.spark = spark

    def load_dataset(self, filename: str, nrows=None):
        if nrows:
            df = self.spark.read.csv(filename, schema=SCHEMA, header=True).limit(nrows).cache()
        else:
            df = self.spark.read.csv(filename, schema=SCHEMA, header=True).cache() # load whole dataset in memory
            #df = self.spark.read.csv(filename, inferSchema=True, header=True).cache() # load whole dataset in memory
        # figure out date range
        MIN_DATE = df.agg({'date': 'min'}).collect()[0]["min(date)"]
        MAX_DATE = df.agg({'date': 'max'}).collect()[0]["max(date)"]
        df = df.withColumn('days', (((F.col('date').cast('long') - MIN_DATE.timestamp())/(3600*24)).cast("long")))
        self.df = df.drop('date')
        self.df.cache()

    def generate_auxiliary_data(self, req: List[privacy.Auxiliary], N):
        return privacy.Generate.generate(self.df, req, N)

    def get_scoring(self, similarity, with_movie=True):
        if similarity == "general":
            sim_fn = privacy.general_similarity()
        elif similarity == "equal":
            sim_fn = privacy.equal_similarity()
        elif similarity == "netflix":
            sim_fn = privacy.netflix_similarity()
        else:
            raise "Unknown similarity function."
        
        if with_movie:
            return privacy.Scoreboard_RH(sim_fn, self.df)
        else:
            return privacy.Scoreboard_RH_without_movie(sim_fn, self.df)
    
    def compute_score(self, auxiliary, similarity: str = "general", with_movie=True):
        scoring = self.get_scoring(similarity, with_movie)
        scores = scoring.compute_score(self.spark.createDataFrame(auxiliary), self.df)
        scores.cache()
        return scores

    def evaluate(self, req: List[privacy.Auxiliary], N=2, similarity="general", mode="best-guess", with_movie=True):
        scoring = self.get_scoring(similarity, with_movie)
        aux = self.generate_auxiliary_data(req, N)
        scores = self.compute_score(aux, similarity)
        

        if mode == "best-guess":
            match = scoring.matching_set(scores, 0.5)
            return 100*match.filter("custId_1 == custId_2").count() / N
        else:
            raise "Error"
        return 0


