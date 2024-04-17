# -*- coding: utf-8 -*-
import sys
sys.path.append("../src/")

from adding_metadata import replies

import polars as pl

import numpy as np

from langchain_community.embeddings import HuggingFaceEmbeddings

def add_agree_disagree_distances(df:pl.DataFrame):
    """
    Add columns to the dataframe that show the average distance of each chunk
    and each reddit_name text to agree and disagree statements

    Parameters:
    -----------
        df: pl.DataFrame
            A dataframe with a column "vector" that contains the embeddings
            of the text chunks
    
    Returns:
    --------
        pl.DataFrame
            A clone of the input dataframe with the following columns added:
                - "chunk_agree_distance": A list of distances of each chunk to
                    the agree statements
                - "chunk_disagree_distance": A list of distances of each chunk 
                    to the disagree statements
                - "chunk_agree_distance_avg": The average distance of each chunk
                    to the agree statements
                - "chunk_disagree_distance_avg": The average distance of each 
                    chunk to the disagree statements
                - "agree_distance_avg": The average distance of all chunks from 
                    the same reddit_name text to the agree statements
                - "disagree_distance_avg": The average distance of all chunks 
                    from the same reddit_name text to the disagree statements
                - "reply_agree_distances": A list of the average distances of 
                    the replies to each reddit_name text to the agree statements
                - "reply_disagree_distances": A list of the average distances of
                    the replies to each reddit_name text to the disagree 
                    statements
                - "top_reply_agree_distance": The average distance of the top 
                    reply to the agree statements
                - "top_reply_disagree_distance": The average distance of the top
                    reply to the disagree statements
                - "avg_reply_agree_distance": The average distance of all 
                    replies to the agree statements
                - "avg_reply_disagree_distance": The average distance of all
                    replies to the disagree statements
    """
    ## Embedding model
    MODEL = HuggingFaceEmbeddings(model_name="thenlper/gte-base")

    ## Define agree and disagree statements
    agree_statements = [
    "This is correct",
    "This is true", 
    "I agree",
    "This is helpful"]

    disagree_statements = [
        "This is incorrect",
        "This is false",
        "I disagree",
        "This is not helpful"]

    ## Embed agree and disagree statements
    agree_embeddings = MODEL.embed_documents(agree_statements)
    disagree_embeddings = MODEL.embed_documents(disagree_statements)

    ## Define a function to calculate distances from agree statements
    def get_agree_scores(vector):
        agree_scores = []
        for statement in agree_embeddings:
            agree_array = np.array(statement)
            agree_scores.append(np.linalg.norm(vector - agree_array))    

        return agree_scores

    ## Define a function to calculate distances from disagree statements
    def get_disagree_scores(vector):
        disagree_scores = []

        for statement in disagree_embeddings:
            disagree_array = np.array(statement)
            disagree_scores.append(np.linalg.norm(vector - disagree_array))

        return disagree_scores
    
    ## Calculate distances to chunks
    df = df.with_columns(
        chunk_agree_distance = pl.col("vector")\
            .map_elements(get_agree_scores),
        chunk_disagree_distance = pl.col("vector")\
            .map_elements(get_disagree_scores))
    
    ## Calculate average distances for each chunk
    df = df.with_columns(
        chunk_agree_distance_avg = pl.col(
            "chunk_agree_distance").list.mean(),
        chunk_disagree_distance_avg = pl.col(
            "chunk_disagree_distance").list.mean())

    ## Calculate average distances for all chunks from the same reddit_name text
    grouped = df.group_by("reddit_name").agg(
        pl.col("chunk_agree_distance_avg").mean()\
            .alias("agree_distance_avg"), 
        pl.col("chunk_disagree_distance_avg").mean()\
            .alias("disagree_distance_avg"))

    ## Join the grouped data back to the original dataframe
    df = df.join(grouped, on="reddit_name", how="left")

    ## Add agree and disagree distances for replies to each reddit_name
    def get_reply_agree_distances(reply_list):
        reply_agree_distances = []
        for reddit_name in reply_list:
            reply_agree_distances\
                .append(df.filter(
                    pl.col("reddit_name")==reddit_name)["agree_distance_avg"]\
                        .mean())    
        return reply_agree_distances

    ## Add agree and disagree distances for replies to each reddit_name
    def get_reply_disagree_distances(reply_list):
        reply_disagree_distances = []
        for reddit_name in reply_list:
            reply_disagree_distances\
                .append(df.filter(
                    pl.col("reddit_name")==reddit_name)["disagree_distance_avg"]\
                        .mean())    
        return reply_disagree_distances

    ## Add reply distances to the dataframe
    df = df.with_columns(
        reply_agree_distances = pl.col("reply_ids")\
            .map_elements(get_reply_agree_distances),
        reply_disagree_distances = pl.col("reply_ids")\
            .map_elements(get_reply_disagree_distances))
    
    ## Define functions to get the distance of the top reply
    def get_top_reply_distance(reply_agreement_distance_list):
        if reply_agreement_distance_list[0]:
            return reply_agreement_distance_list[0]
        else:
            return None
    
    ## Add top reply distances and average distances
    df = df.with_columns(
        top_reply_agree_distance = pl.col("reply_agree_distances")\
            .map_elements(get_top_reply_distance),
        top_reply_disagree_distance = pl.col("reply_disagree_distances")\
            .map_elements(get_top_reply_distance),
        avg_reply_agree_distance = pl.col("reply_agree_distances")\
            .list.mean(),
        avg_reply_disagree_distance = pl.col("reply_disagree_distances")\
            .list.mean())
    
    return df.clone()