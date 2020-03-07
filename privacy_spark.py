#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from pyspark.sql import *
import pyspark.sql.functions as F
import numpy as np
import itertools

RANGE_RATING = list(range(1, 6))
RANGE_DATE = list(range(0, 2242))
MIN_DATE = '1999-11-11'
MAX_DATE = '2005-12-31'


class Score():
    def __init__(self, similarity_func):
        self.similarity_func = similarity_func

    def similarity(self, record1, record2):
        """
        computes the similarity between two records
        """
        pass

    def compute_similarity(self, record, df_records):
        """
        compute the similarity between one record and all other records
        """
        pass

    def get_matching_set(self):
        """

        """
        pass

    def calculate_extentricity(self):
        pass

    def predict(self):
        pass


def equal_similarity(r):
    return r['rating_1'] == r['rating_2']


def general_similarity(margin_rating=0, margin_date=14):
    def similarity(r):
        D_1 = (F.abs(r['rating_1'] - r['rating_2']) <= margin_rating).cast('int')
        D_2 = (F.abs(r['days_1'] - r['days_2']) <= margin_date).cast('int')
        return D_1 + D_2
    return similarity


def netflix_similarity(r0=1.5, d0=30):
    def similarity(r):
        return (np.exp(-np.abs(r['rating_1']-r['rating_2'])/r0) + np.exp(-np.abs(r['days_1']-r['days_2'])/d0)).fillna(0)
    return similarity


def netflix_similarity_old(r0=1.5, d0=30):
    def similarity(r):
        return (np.exp((r['rating_1']-r['rating_2'])/r0) + np.exp((r['days_1']-r['days_2'])/d0)).fillna(0)
    return similarity


class Score_simple():
    def __init__(self, similarity_func):
        self.similarity_func = similarity_func

    def similarity(self, record1, record2):
        """
        computes the similarity between two records
        """
        merged = pd.merge(record1, record2, how='outer',
                          left_on='movieId', right_on='movieId', suffixes=('_1', '_2'))
        # after outer merge, both records have same cardinality
        card = len(merged)
        return self.similarity_func(merged.dropna()).sum()/card

    def compute_score(self, record, df_records):
        """
        compute the similarity between one record and all other records
        returns a series with the custId of other records as index, and similarity as value
        """

        merged = pd.merge(record.reset_index().set_index('movieId'),
                          df_records.reset_index().set_index('movieId'),
                          how='outer',
                          left_index=True,
                          right_index=True,
                          suffixes=('_1', '_2'),
                          indicator='present')
        merged['similarity'] = self.similarity_func(merged)  # .astype('int')
        merged['right_only'] = 1*(merged['present'] == 'right_only')
        grouped = merged.groupby('custId_2').agg(
            {'similarity': ['sum'], 'right_only': 'sum'})
        grouped.columns = [x[0] if x[0] !=
                           'right_only' else 'sum' for x in grouped.columns]
        grouped['sum'] += len(record)

        return grouped['similarity']/grouped['sum']


class Scoreboard:
    def __init__(self, similarity_func):
        self.similarity_func = similarity_func

    def similarity(self, record1, record2):
        """
        computes the similarity between two records
        """
        merged = pd.merge(record1, record2, how='left', left_on='movieId',
                          right_on='movieId', suffixes=('_1', '_2'))
        return self.similarity_func(merged).min()

    def compute_score(self, record, df_records):

        df_records = (
            df_records
            .set_index([df_records.index, 'movieId'])
            .reindex(itertools.product(np.unique(df_records.index),
                                       record['movieId'].values)
                     )
        ).reset_index().set_index('custId', drop=True)

        merged = pd.merge(record.reset_index().set_index('movieId'),
                          df_records.reset_index().set_index('movieId'),
                          how='left',
                          left_index=True,
                          right_index=True,
                          suffixes=('_1', '_2')
                          )
        merged['similarity'] = 1*self.similarity_func(merged)  # .astype('int')
        merged = merged.groupby('custId_2')['similarity'].min()

        return merged.sort_values(ascending=False)

    def matching_set(self, scores, thresh):
        return scores[scores > thresh]

    def output(self, scores, thresh, best_guess=True):
        matched_scores = self.matching_set(scores, thresh)
        if best_guess:
            return matched_scores.sort_values(ascending=False).iloc[[0]]
        else:
            matched_scores = matched_scores/matched_scores.sum()
            return matched_scores


class Scoreboard_RH:
    def __init__(self, similarity_func, df):
        self.similarity_func = similarity_func
        self.wt = df.groupBy('movieId').count().withColumn('wt', 1/F.log(F.col('count')))

    def similarity(self, record1, record2):
        """
        computes the similarity between two records
        """
        merged = pd.merge(record1.reset_index().set_index('movieId'),
                          record2.reset_index().set_index('movieId'),
                          how='left',
                          left_index=True,
                          right_index=True,
                          suffixes=('_1', '_2'))
        merged['similarity'] = self.similarity_func(merged)
        merged = pd.merge(self.similarity_func(merged), self.wt,
                          how='left', left_index=True, right_index=True)
        return (merged['wt'] * merged['similarity']).sum()

    def compute_score(self, aux, df_records):
        def prepare_join(df, suffix):
            df = df.withColumnRenamed('custId', 'custId'+suffix)
            df = df.withColumnRenamed('rating', 'rating'+suffix)
            df = df.withColumnRenamed('days', 'days'+suffix)
            return df

        aux_with_wt = aux.join(self.wt, 'movieId', 'left')
        merged = prepare_join(aux_with_wt, '_1').join(prepare_join(df_records, '_2'), 'movieId', 'left')

        merged = merged.withColumn('similarity', self.similarity_func(merged))  # .astype('int')
        merged = merged.withColumn('value', merged.wt * merged.similarity)
        merged = merged.groupBy('custId_2').sum('value').withColumnRenamed('sum(value)', 'value')

        return merged.sort('value', ascending=False)

    def matching_set(self, scores, thresh=1.5):
        sigma = scores.select(F.stddev(F.col('value')).alias('std')).collect()[0]['std']
        print(sigma)
        top_scores = scores.collect()[:2]
        eccentricity = (top_scores[0]['value'] - top_scores[1]['value'])/sigma
        if eccentricity < thresh:
            return None
        return top_scores[0], eccentricity

    def output(self, scores, thresh=1.5, best_guess=True):
        if best_guess:
            result = self.matching_set(scores, thresh)
            if result:
                return result[0]
            else:
                return None
        else:
            sigma = np.std(scores.values)
            probas = np.exp(scores/sigma)
            return probas/probas.sum()


class Auxiliary:
    def __init__(self, rating, date, margin_rating, margin_date):
        self.rating = rating  # boolean: True if we want right value in a margin
        self.date = date  # boolean: True if we want right value in a margin
        self.margin_rating = margin_rating
        self.margin_date = margin_date

    def generate_aux_record(self, record):
        record = record.copy()
        #rating = record['rating']
        #date = record['days']
        rating = record.iloc[0, record.columns.get_loc('rating')]
        date = record.iloc[0, record.columns.get_loc('days')]

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

        #record['rating'] = rating
        #record['days'] = date
        record.iloc[0, record.columns.get_loc('rating')] = rating
        record.iloc[0, record.columns.get_loc('days')] = date

        return record


class Generate:
    @staticmethod
    def generate(df, aux_list, custId=None, movieId_list=None):
        """
        generate a complete auxiliary information record
        :params df: the dataframe of ratings (SPARK Dataframe)
        :params aux_list: the informations about the records (Auxiliary)
        :params custId: the id of the customer (Int | None)
        :params movieId_list: the list of Id of movies to apply the aux info to (Int[])

        returns a pandas dataframe
        """
        if movieId_list == None:
            movieId_list = []

        if not custId:
            N_requested_ratings = len(aux_list)
            custId = df.groupBy("custId").count().where("count >= {}".format(
                N_requested_ratings)).rdd.takeSample(False, 1)[0]["custId"]
            print("Picked customer", custId)

        record = df.where("custId == {}".format(custId))
        print(record.collect(), record.count(), len(aux_list), len(movieId_list))
        if record.count() - len(movieId_list) < len(aux_list):
            raise ValueError('The customer has not reviewed enough movies')

        nb_remaining_movie_id = 0
        if len(movieId_list) < len(aux_list):
            nb_remaining_movie_id = len(aux_list)-len(movieId_list)

        movieIds = np.array([x["movieId"]
                    for x in record.select("movieId").distinct().collect()])
                    
        remaining_movie_ids = list(np.random.choice(movieIds[~np.isin(movieIds, movieId_list)],
                                                    size=nb_remaining_movie_id,
                                                    replace=False))
        movieId_list += remaining_movie_ids

        #aux_record = pd.concat([aux.generate_aux_record(record.loc[record['movieId']==movieId].iloc[0]) for aux, movieId in zip(aux_list, movieId_list)])
        aux_records_list = []
        for aux, movieId in zip(aux_list, movieId_list):
            entry = record.where("movieId == {}".format(movieId)).toPandas()
            aux_result = aux.generate_aux_record(entry)
            aux_records_list.append(aux_result)
        
        aux_record = pd.concat(aux_records_list)
        return aux_record
