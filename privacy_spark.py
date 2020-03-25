#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from pyspark.sql import *
import pyspark.sql.functions as F
from pyspark.sql.functions import broadcast
from pyspark.sql.window import Window
import numpy as np
import itertools

RANGE_RATING = list(range(1, 6))
RANGE_DATE = list(range(0, 2242))


def equal_similarity(r):
    return r['rating_1'] == r['rating_2']


def general_similarity(margin_rating=0, margin_date=14):
    def similarity(r):
        D_1 = (F.abs(r['rating_1'] - r['rating_2'])
               <= margin_rating).cast('int')
        D_2 = (F.abs(r['days_1'] - r['days_2']) <= margin_date).cast('int')
        return D_1 + D_2
    return similarity


def netflix_similarity(r0=1.5, d0=30):
    def similarity(r):
        D_1 = F.exp(-(F.abs(r['rating_1'] - r['rating_2'])/r0))
        D_2 = F.exp(-(F.abs(r['days_1']-r['days_2'])/d0))
        return D_1 + D_2
    return similarity

def netflix_similarity_weighted(r0=1.5, d0=30, avgr0=1):
    def similarity(r):
        D_1 = F.exp(-(F.abs(r['rating_1'] - r['rating_2'])/r0))
        D_2 = F.exp(-(F.abs(r['days_1']-r['days_2'])/d0))
        D_3 = F.exp(-(F.abs(r['avgMovieRating_1']-r['avgMovieRating_2'])/avgr0))
        return D_1 + D_2 + D_3
    return similarity


def prepare_join(df, suffix, with_movieId=False):
    df = df.withColumnRenamed('custId', 'custId'+suffix)
    df = df.withColumnRenamed('rating', 'rating'+suffix)
    df = df.withColumnRenamed('days', 'days'+suffix)
    if with_movieId:
        df = df.withColumnRenamed('movieId', 'movieId'+suffix)
    return df


class Scoreboard_RH:
    def __init__(self, similarity_func, df):
        self.similarity_func = similarity_func
        self.wt = df.groupBy('movieId').count().withColumn(
            'wt', 1/F.log(F.col('count')))

    def compute_score(self, aux, df_records):
        """
        Compute scoreboard of auxiliary information aux inside record df_records.
        Both must be spark dataframes.
        Returns a spark dataframe.
        """
        aux_with_wt = aux.join(self.wt, 'movieId', 'left')
        merged = broadcast(prepare_join(aux_with_wt, '_1')).join(
            prepare_join(df_records, '_2'), 'movieId', 'left')

        merged = merged.withColumn('similarity', self.similarity_func(merged))
        merged = merged.withColumn('value', merged.wt * merged.similarity)
        merged = merged.groupBy('custId_1', 'custId_2').sum('value')
        merged = merged.withColumnRenamed('sum(value)', 'value')
        return merged.cache()

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
        return scores_w_eccentricity.filter('eccentricity >= {}'.format(thresh)).cache()

    def output(self, scores, thresh=1.5, mode="best-guess"):
        if mode == "best-guess":
            return self.matching_set(scores, thresh)
        elif mode == "entropic":
            sigma = scores.groupBy('custId_1').agg(
                F.stddev(scores.value).alias('std'))
            probas = scores.withColumn(
                "probas_raw", F.exp(scores.value/sigma.std))
            return scores.withColumn("probas", scores.probas_raw/F.sum(scores.groupBy('custId_1').probas_raw))
        else:
            raise "Mode '{}' is invalid.".format(mode)


class Scoreboard_RH_without_movie:
    def __init__(self, similarity_func, df):
        self.similarity_func = similarity_func
        self.wt = df.groupBy('movieId').count().withColumn(
            'wt', 1/F.log(F.col('count')))

    def compute_score(self, aux, df_records):
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
        merged = merged.groupBy('custId_1', 'custId_2', 'movieId_1').max('value')
        merged = merged.withColumnRenamed('max(value)', 'value')
        merged = merged.groupBy('custId_1', 'custId_2').sum('value')
        merged = merged.withColumnRenamed('sum(value)', 'value')
        return merged.cache()

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
        return merged.cache()

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
        return scores_w_eccentricity.filter('eccentricity >= {}'.format(thresh)).cache()

    def output(self, scores, thresh=1.5, mode="best-guess"):
        if mode == "best-guess":
            return self.matching_set(scores, thresh)
        elif mode == "entropic":
            sigma = scores.groupBy('custId_1').agg(
                F.stddev(scores.value).alias('std'))
            probas = scores.withColumn(
                "probas_raw", F.exp(scores.value/sigma.std))
            return scores.withColumn("probas", scores.probas_raw/F.sum(scores.groupBy('custId_1').probas_raw))
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
        customers =  df.groupBy("custId").count().where("count >= {}".format(N_requested_ratings)).cache()
        custId = customers.sample(False, min(1., 10*N/customers.count())).limit(N)
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
