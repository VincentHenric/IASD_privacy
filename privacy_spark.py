#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from pyspark.sql import *
import pyspark.sql.functions as F
from pyspark.sql.functions import broadcast
from pyspark.sql.window import Window
from scipy.stats import binom
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
import numpy as np
from pyspark.sql.functions import col, create_map, lit
from itertools import chain
from scipy.special import comb

RANGE_RATING = list(range(1, 6))
RANGE_DATE = list(range(0, 2242))

def binom_cdf(p=15/11210, s=8):
    def func(n):
        return 1/(1-binom.cdf(s+1, n, p).item()+1)
    return func

def proba_2(p=15/11210, s=8):
    def func(n):
        return sum([comb(s,k)*(1-k*p)**n*(-1)**k for k in range(s+1)])
    return func
#binom_cdf_udf = udf(binom_cdf(14/11210, 8), DoubleType())

def equal_similarity(r):
    """Equal similarity function"""
    return r['rating_1'] == r['rating_2']


def general_similarity(margin_rating=0, margin_date=14):
    """General similarity function"""
    def similarity(r):
        D_1 = (F.abs(r['rating_1'] - r['rating_2'])
               <= margin_rating).cast('int')
        D_2 = (F.abs(r['days_1'] - r['days_2']) <= margin_date).cast('int')
        return D_1 + D_2
    return similarity


def netflix_similarity(r0=1.5, d0=30):
    """Netflix similarity function
    
    Custom similarity as presented in the de-anonymisation paper.

    """
    def similarity(r):
        D_1 = F.exp(-(F.abs(r['rating_1'] - r['rating_2'])/r0))
        D_2 = F.exp(-(F.abs(r['days_1']-r['days_2'])/d0))
        return D_1 + D_2
    return similarity

def netflix_similarity_weighted(r0=1.5, d0=30, avgr0=1):
    """Weighted Netflix similarity
    
    Netflix similarity with scaling according to the rating closeness between the two movies ratings. 
    This is used in the unknown movie mapping case.
    
    """
    def similarity(r):
        D_1 = F.exp(-(F.abs(r['rating_1'] - r['rating_2'])/r0))
        D_2 = F.exp(-(F.abs(r['days_1']-r['days_2'])/d0))
        D_3 = F.exp(-(F.abs(r['avgMovieRating_1']-r['avgMovieRating_2'])/avgr0))
        return D_1 + D_2 + D_3
    return similarity


def prepare_join(df, suffix, with_movieId=False):
    """Prepare a Spark join by adding suffixes to the dataframe columns"""

    df = df.withColumnRenamed('custId', 'custId'+suffix)
    df = df.withColumnRenamed('rating', 'rating'+suffix)
    df = df.withColumnRenamed('days', 'days'+suffix)
    if with_movieId:
        df = df.withColumnRenamed('movieId', 'movieId'+suffix)
        df = df.withColumnRenamed('avgMovieRating', 'avgMovieRating'+suffix)
        df = df.withColumnRenamed('nbMovieReviews', 'nbMovieReviews'+suffix)
        df = df.withColumnRenamed('nbCustReviews', 'nbCustReviews'+suffix)
    return df


class Scoreboard_RH:
    """Scoreboard-RH de-anonymisation algorithm.

    It works by ranking matches between auxiliary information and the full dataset. 
    The scores are weighted by the statistical probability of such a match.
    """
    def __init__(self, similarity_func, df):
        self.similarity_func = similarity_func
        # Count number of ratings for each movie, along with wt score
        # (movieId, count, wt)
        self.wt = df.groupBy('movieId').count()\
            .withColumn('wt', 1/F.log(F.col('count')))

    def compute_score(self, aux, df_records, tol=None):
        """Compute scoreboard of auxiliary information aux inside record df_records.
        
        Both must be spark dataframes.
        Parameters:
            - aux: DF (custId, movieId, rating, days)
            - df_records: DF (custId, movieId, rating, days)
        Returns a spark dataframe.
        """
        # (custId, movieId, rating, days, count, wt)
        aux_with_wt = aux.join(self.wt, 'movieId', 'left')
        # (custId_1, rating_1, days_1, custId_2, rating_2, days_2, movieId, count, wt)
        merged = broadcast(prepare_join(aux_with_wt, '_1')).join(
            prepare_join(df_records, '_2'), 'movieId', 'left')

        # (..., similarity)
        merged = merged.withColumn('similarity', self.similarity_func(merged))
        # (..., value)
        merged = merged.withColumn('value', merged.wt * merged.similarity)
        # (custId_1, custId_2, sum(value))
        merged = merged.groupBy('custId_1', 'custId_2').sum('value')
        # (custId_1, custId_2, value)
        merged = merged.withColumnRenamed('sum(value)', 'value')
        return merged

    def matching_set(self, scores, thresh=1.5):
        """Best-guess method with eccentricity threshold.
        
        Scores is a spark dataframe that may contain multiple custId.
        Parameters:
            - scores: DF(custId_1, custId_2, value)
        """
        # Sigma (custId_1, std): Standard deviation for each attacked customer.
        sigma = scores.groupBy('custId_1').agg(
            F.stddev(scores.value).alias('std'))
        window = Window.partitionBy(
            scores.custId_1).orderBy(scores.value.desc())
        # Top scores (custId_1, custId_2, value, rank): 
        # For each customer, the two closest customers in the dataset.
        top_scores = scores.select(
            '*', F.row_number().over(window).alias('rank')).filter(F.col('rank') <= 2)
        # First match: (custId_1, custId_2, value_1)
        top_1 = top_scores.filter('rank == 1').drop(
            'rank').withColumnRenamed('value', 'value_1')
        # Second match: (custId_1, custId_3, value_2)
        top_2 = top_scores.filter('rank == 2').drop('rank').withColumnRenamed(
            'value', 'value_2').withColumnRenamed('custId_2', 'custId_3')

        # (custId_1, custId_2, custId_3, value_1, value_2, std)
        scores = top_1.join(top_2, ['custId_1']).join(sigma, 'custId_1')
        # (..., eccentricity)
        # For each customer, the two closests customers in the dataset, along with the
        # eccentricity measure.
        scores_w_eccentricity = scores.withColumn(
            'eccentricity', (F.col('value_1') - F.col('value_2'))/F.col('std'))
        return scores_w_eccentricity.filter('eccentricity >= {}'.format(thresh))

    def output(self, scores, thresh=1.5, mode="best-guess"):
        """Standard output of the algorithm

        De-anonymisation has two modes: entropic (keeps the full distribution) or 
        best-guess (matching with threshold).
        """
        if mode == "best-guess":
            return self.matching_set(scores, thresh)
        elif mode == "entropic":
            # (custId_1, std)
            sigma = scores.groupBy('custId_1').agg(
                F.stddev(scores.value).alias('std'))
            # (custId_1, custId_2, probas_raw)
            probas_raw = scores\
                .join(sigma, ['custId_1'])\
                .withColumn("probas_raw", F.exp(F.col('value')/F.col('std')))\
                .select(['custId_1', 'custId_2', 'probas_raw', 'std'])
            # (custId_1, probas_z)
            probas_z   = probas_raw.groupBy('custId_1').agg(F.sum(probas_raw.probas_raw).alias('probas_z'))
            # (custId_1, custId_2, probas)
            return scores\
                .join(probas_raw, ['custId_1', 'custId_2'])\
                .join(probas_z, ['custId_1'])\
                .withColumn("probas", F.col('probas_raw')/F.col('probas_z'))\
                .select(['custId_1', 'custId_2', 'probas', 'value', 'std'])
        else:
            raise "Mode '{}' is invalid.".format(mode)


class Scoreboard_RH_without_movie:
    def __init__(self, similarity_func, df):
        self.similarity_func = similarity_func
        self.max_nb_review_per_cust = 17653
        self.nb_combination = len(RANGE_DATE)*len(RANGE_RATING)

    def compute_score(self, aux, df_records, tol=15):
        """
        Compute scoreboard of auxiliary information aux inside record df_records.
        Both must be spark dataframes.
        Returns a spark dataframe.
        """
        s = aux.groupby('custId').count().take(1)[0][1]
        #mapping = {n:binom_cdf(p=tol/self.nb_combination, s=s)(n) for n in range(0,self.max_nb_review_per_cust+1)}
        mapping_2 = {n:proba_2(p=tol/self.nb_combination, s=s)(n) for n in range(0, self.max_nb_review_per_cust+1)}
        #mapping_expr = create_map([lit(x) for x in chain(*mapping.items())])
        mapping_expr_2 = create_map([lit(x) for x in chain(*mapping_2.items())])

        merged = broadcast(prepare_join(aux, '_1', True)).crossJoin(
            prepare_join(df_records, '_2', True))

        merged = merged.withColumn('similarity', self.similarity_func(merged))
        #merged = merged.withColumn('value', 1/F.log(F.log(merged.nbCustReviews_2+100)) * merged.similarity)
        #merged = merged.withColumn('value', 1/F.log(merged.nbMovieReviews_2) * merged.value)
        #merged = merged.withColumn('value', binom_cdf_udf(merged.nbCustReviews_2) * merged.similarity)
        merged = merged.withColumn('value', mapping_expr_2.getItem(col('nbCustReviews_2')) * merged.similarity)
        #merged = merged.withColumn('value', merged.similarity)
        merged = merged.groupBy('custId_1', 'custId_2', 'movieId_1').max('value')
        merged = merged.withColumnRenamed('max(value)', 'value')
        merged = merged.groupBy('custId_1', 'custId_2').sum('value')
        merged = merged.withColumnRenamed('sum(value)', 'value')
        return merged

    def compute_individual_score(self, aux, df_records):
        """
        Compute scoreboard of auxiliary information aux inside record df_records.
        Both must be spark dataframes.
        Returns a spark dataframe.
        """

        merged = broadcast(prepare_join(aux, '_1', True)).crossJoin(
            prepare_join(df_records, '_2', True))

        merged = merged.withColumn('similarity', self.similarity_func(merged))
        #merged = merged.withColumn('value', merged.wt * merged.similarity)
        merged = merged.withColumn('value', merged.similarity)
        return merged

    def matching_set(self, scores, thresh=1.5):
        """
        best-guess method with eccentricity threshold.
        scores is a spark dataframe that may contain multiple custId.
        """
        sigma = scores.groupBy('custId_1').agg(
            F.stddev(scores.value).alias('std'))
        window = Window.partitionBy(
            scores.custId_1).orderBy(scores.value.desc())
        top_scores = scores.select(
            '*', F.row_number().over(window).alias('rank')).filter(F.col('rank') <= 2)
        top_1 = top_scores.filter('rank == 1').drop(
            'rank').withColumnRenamed('value', 'value_1')
        top_2 = top_scores.filter('rank == 2').drop('rank').withColumnRenamed(
            'value', 'value_2').withColumnRenamed('custId_2', 'custId_3')

        scores = top_1.join(top_2, ['custId_1']).join(sigma, 'custId_1')
        scores_w_eccentricity = scores.withColumn(
            'eccentricity', (F.col('value_1') - F.col('value_2'))/F.col('std'))
        return scores_w_eccentricity.filter('eccentricity >= {}'.format(thresh))

    def output(self, scores, thresh=1.5, mode="best-guess"):
        if mode == "best-guess":
            return self.matching_set(scores, thresh)
        elif mode == "entropic":
            # (custId_1, std)
            sigma = scores.groupBy('custId_1').agg(
                F.stddev(scores.value).alias('std'))
            # (custId_1, custId_2, probas_raw)
            probas_raw = scores\
                .join(sigma, ['custId_1'])\
                .withColumn("probas_raw", F.exp(F.col('value')/F.col('std')))\
                .select(['custId_1', 'custId_2', 'probas_raw', 'std'])
            # (custId_1, probas_z)
            probas_z   = probas_raw.groupBy('custId_1').agg(F.sum(probas_raw.probas_raw).alias('probas_z'))
            # (custId_1, custId_2, probas)
            return scores\
                .join(probas_raw, ['custId_1', 'custId_2'])\
                .join(probas_z, ['custId_1'])\
                .withColumn("probas", F.col('probas_raw')/F.col('probas_z'))\
                .select(['custId_1', 'custId_2', 'probas', 'value', 'std'])
        else:
            raise "Mode '{}' is invalid.".format(mode)


class Auxiliary:
    def __init__(self, rating, date, margin_rating, margin_date):
        self.rating = rating  # boolean: True if we want right value in a margin
        self.date = date  # boolean: True if we want right value in a margin
        self.margin_rating = margin_rating
        self.margin_date = margin_date

    def generate_aux_record(self, record):
        rating = record['rating']
        date = record['days']

        if self.rating:
            rating = np.random.choice(
                [i for i in RANGE_RATING if i >= rating-self.margin_rating and i <= rating+self.margin_rating])
        else:
            rating = np.random.choice(
                [i for i in RANGE_RATING if i != self.rating])

        if self.date:
            date = np.random.choice(
                [i for i in RANGE_DATE if i >= date-self.margin_date and i <= date+self.margin_date])
        else:
            date = np.random.choice(
                [i for i in RANGE_DATE if i != self.rating])

        record['rating'] = rating
        record['days'] = date

        return record


class Generate:
    @staticmethod
    def generate(df, aux_list, N=1, loc=0, scale=0.4):
        """
        generate a complete auxiliary information record
        :params df: the dataframe of ratings (SPARK Dataframe)
        :params aux_list: the informations about the records (Auxiliary)
        :params custId: the id of the customer (Int | None)
        :params movieId_list: the list of Id of movies to apply the aux info to (Int[])

        returns a pandas dataframe
        """

        N_requested_ratings = len(aux_list)
        customers = df.groupBy("custId").count().where(
            "count >= {}".format(N_requested_ratings)).cache()
        custId = customers.sample(False, min(
            1., 10*N/customers.count())).limit(N)
        records = custId.join(df, 'custId').cache()

        window = Window.partitionBy(F.col('custId')).orderBy(F.col('rng'))

        # for each user, sample len(aux_list) ratings.
        records_movies_sampled = records\
            .withColumn('rng', F.rand())\
            .withColumn('rnw', F.row_number().over(window))\
            .where('rnw <= {}'.format(N_requested_ratings))\
            .drop('rng')

        aux_record = records_movies_sampled.toPandas()
        aux_record['avgMovieRating'] = aux_record['avgMovieRating'] + np.random.normal(loc=loc, scale=scale, size=len(aux_record))
        return pd.DataFrame.from_records([aux_list[aux['rnw']-1].generate_aux_record(aux) for aux in aux_record.to_dict('records')]).astype(aux_record.dtypes)
