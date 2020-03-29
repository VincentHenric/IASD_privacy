from pyspark.sql import *
import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.sql.types import LongType, TimestampType, StringType, StructField, StructType, BooleanType, ArrayType, IntegerType
import pandas as pd
import numpy as np
from typing import List
import time

from config import SPARK_CONF
import privacy_spark as privacy

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
    """Experiment manager

    Wrapper around de-anonymisation algorithms to easily run experiments.
    """
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
        
    def get_subset(self, nb_reviews=3000):
        """
        get subset of the dataset
        with nbReviews >= 3000, we get 1,015,364 records, for 89 movies
        """
        window = Window.partitionBy('movieId')
        self.df = self.df.withColumn('nbReviews', F.count('rating').over(window))
        self.df = self.df.filter('nbReviews>={}'.format(nb_reviews))

    def generate_auxiliary_data(self, req: List[privacy.Auxiliary], N):
        return privacy.Generate.generate(self.df, req, N)

    def get_scoring(self, similarity, with_movie=True):
        if similarity == "general":
            sim_fn = privacy.general_similarity()
        elif similarity == "equal":
            sim_fn = privacy.equal_similarity()
        elif similarity == "netflix":
            sim_fn = privacy.netflix_similarity()
        elif similarity == "netflix_weighted":
            sim_fn = privacy.netflix_similarity_weighted()
        else:
            raise "Unknown similarity function."
        
        if with_movie:
            return privacy.Scoreboard_RH(sim_fn, self.df)
        else:
            return privacy.Scoreboard_RH_without_movie(sim_fn, self.df)
    
    def compute_score(self, auxiliary, similarity: str = "general", with_movie=True, tol=15):
        scoring = self.get_scoring(similarity, with_movie)
        scores = scoring.compute_score(self.spark.createDataFrame(auxiliary), self.df, tol)
        scores
        return scores

    def evaluate(self, req: List[privacy.Auxiliary], N=2, similarity="general", mode="best-guess", with_movie=True, tol=15):
        """De-anonymisation evaluator

        Given a list of Auxiliary requests and a number of sampled customers, evaluate
        de-anonymisation performance. There are two modes:
        - 'best-guess': returns true positive rate for a fixed threshold. 
        - 'entropic': returns the entropy of the probability distribution.
        """
        scoring = self.get_scoring(similarity, with_movie)
        aux = self.generate_auxiliary_data(req, N)
        scores = self.compute_score(aux, similarity, with_movie, tol)
        
        if mode == "best-guess":
            match = scoring.matching_set(scores, 0.5)
            return 100*match.filter("custId_1 == custId_2").count() / N
        elif mode == "entropic":
            probas      = scoring.output(scores, mode="entropic")
            withEntropy = probas.groupBy("custId_1").agg((-F.sum(F.col("probas") * F.log2(F.col("probas")))).alias("entropy"))
            return withEntropy.groupBy().avg('entropy').collect()
        else:
            raise "Invalid argument."

    def evaluate_all(self, req: List[privacy.Auxiliary], N=100, similarity="general", mode="best-guess", with_movie=True, tol=15):
        scoring = self.get_scoring(similarity, with_movie)
        aux = self.generate_auxiliary_data(req, N)
        scores = self.compute_score(aux, similarity, with_movie, tol)
        custIds = aux.custId.unique()
        
        
        if mode == "best-guess": # {aux, custId, score, excentricity }
            match = scoring.matching_set(scores, 0.0).toPandas().set_index("custId_1")
            return [{
                "id": custId,
                "aux": aux.set_index("custId").loc[custId],
                "matchedId": int(match.loc[custId]["custId_2"]),
                "score": match.loc[custId]["value_1"],
                "eccentricity": match.loc[custId]["eccentricity"],
            } for custId in custIds]
        elif mode == "entropic":
            scores.cache()
            probas      = scoring.output(scores, mode="entropic")
            match       = scoring.matching_set(scores, 0.0).toPandas().set_index("custId_1")
            
            withEntropy = probas.groupBy("custId_1").agg((-F.sum(F.col("probas") * F.log2(F.col("probas")))).alias("entropy")).toPandas().set_index("custId_1")
            
            return [
                {
                    "id": custId,
                    "aux": aux.set_index("custId").loc[custId],
                    "matchedId": int(match.loc[custId]["custId_2"]),
                    "score": match.loc[custId]["value_1"],
                    "eccentricity": match.loc[custId]["eccentricity"],
                    "entropy": withEntropy.loc[custId]
                }
                for custId in custIds
            ]
        else:
            raise "Invalid argument."

import pickle

import os

if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("Privacy Project") \
        .config(conf=SPARK_CONF) \
        .getOrCreate()

    t0 = time.time()

    exp = Experiment(spark)
    exp.load_dataset("./datasets/netflix.ratings.csv")#, nrows=100000)
    window = Window.partitionBy('movieId')
    exp.df = exp.df.withColumn('avgMovieRating',F.avg('rating').over(window))
    exp.df = exp.df.withColumn('nbMovieReviews', F.count('rating').over(window))
    window2 = Window.partitionBy('custId')
    exp.df = exp.df.withColumn('nbCustReviews', F.count('rating').over(window2))
    exp.df = exp.df.repartition('custId').cache()
    print("Loaded dataset!")

    print("Time to load: {}s".format(time.time() - t0))

    no_info = privacy.Auxiliary(False, False, 0, 0)

    mode = "best-guess" # entropic | best-guess
    withMovie = True
    N=1000

    experiments = {}
    for (n_info, n_no_info) in [(2,0), (3,1), (6,2)]:
        for days in [3, 14]:
            t0 = time.time()
            if withMovie:
                name = "{}-{}-{}-{}".format(mode, n_info, n_info+n_no_info, days)
            else:
                name = "withoutMovie_{}-{}-{}-{}".format(mode, n_info, n_info+n_no_info, days)
            fname = "experiments/{}.pkl".format(name)
            if not os.path.exists(fname):
                info = privacy.Auxiliary(True, True, 0, days)
                aux_req = n_info*[info] + n_no_info*[no_info]
                
                results = exp.evaluate_all(aux_req, N=N, mode=mode, similarity="netflix", with_movie=withMovie, tol=2*days+1)
                if mode == "best-guess":
                    print("{}: {}".format(name, 100*sum([r["id"] == r["matchedId"] for r in results])/len(results)))
                else:
                    print("{}: {}".format(name, sum([r["entropy"] for r in results])/len(results)))
                r = open(fname, "wb")
                pickle.dump(results, r)
                r.close()
            print("Done in: {}s".format(time.time() - t0))
            
# Uncomment for with_movie experiment (can be very long)        
#    experiments = {}
#    for (n_info, n_no_info) in [(10,0)]:
#        for days in [3]:
#            t0 = time.time()
#            name = "{}-{}-{}".format(n_info, n_info+n_no_info, days)
#            fname = "experiments/{}-{}.pkl".format(mode, name)
#            if not os.path.exists(fname):
#                info = privacy.Auxiliary(True, True, 0, days)
#                aux_req = n_info*[info] + n_no_info*[no_info]
#                results = exp.evaluate_all(aux_req, N=10, mode=mode, similarity="netflix", with_movie=False, tol=2*days+1)
#                if mode == "best-guess":
#                    print("{}: {}".format(name, 100*sum([r["id"] == r["matchedId"] for r in results])/len(results)))
#                else:
#                    print("{}: {}".format(name, sum([r["entropy"] for r in results])/len(results)))
#                r = open(fname, "wb")
#                pickle.dump(results, r)
#                r.close()
#            print("Done in: {}s".format(time.time() - t0))
