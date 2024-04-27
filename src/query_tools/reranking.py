import polars as pl
import numpy as np

def rerank_by_agree_distance(df: pl.DataFrame, 
                             agree_threshold: float = 0.25,
                             agree_bump: float = 1.5):
    '''
    This function reranks the dataframe based on the agree distance. Results
    that are within the agree threshold will be bumped up by the agree bump
    before reranking.

    df: pl.DataFrame, the dataframe to rerank
    agree_threshold: float, the threshold for the agree distance
    agree_bump: float, the bump for the agree distance

    It is expected that the dataframe has the following columns:
    - avg_reply_agree_distance
    '''
    
    df = df.with_row_index("rank")
    df = df.with_columns(
        pl.when((pl.col("avg_reply_agree_distance")!=0) 
                & (pl.col("avg_reply_agree_distance")<=agree_threshold))\
            .then(pl.col("rank")-agree_bump)\
            .otherwise(pl.col("rank"))\
            .alias("rank")
    )

    return df.sort("rank").drop("rank").clone()

def rerank_by_disagree_distance(df: pl.DataFrame,
                                disagree_threshold: float = 0.25,
                                disagree_bump: float = 1.5):
    '''
    This function reranks the dataframe based on the disagree distance. Results
    that are within the disagree threshold will be bumped down by the disagree 
    bump before reranking.

    df: pl.DataFrame, the dataframe to rerank
    disagree_threshold: float, the threshold for the disagree distance
    disagree_bump: float, the bump for the disagree distance

    It is expected that the dataframe has the following columns:
    - avg_reply_disagree_distance
    '''
    
    df = df.with_row_index("rank")
    df = df.with_columns(
        pl.when((pl.col("avg_reply_disagree_distance")!=0) 
                & (pl.col("avg_reply_disagree_distance")<=disagree_threshold))\
            .then(pl.col("rank")+disagree_bump)\
            .otherwise(pl.col("rank"))\
            .alias("rank")
    )

    return df.sort("rank").drop("rank").clone()

def rerank_by_sentiment(df: pl.DataFrame,
                        sentiment_bump: float = 0.75):
    '''
    This function reranks the dataframe based on the summed sentiments. Results
    that are negative will be bumped down by the sentiment bump, while results
    that are positive will be bumped up by the sentiment bump.

    df: pl.DataFrame, the dataframe to rerank
    sentiment_bump: float, the bump for the sentiment
    
    It is expected that the dataframe has the following columns:
    - summed_sentiments
    '''
    
    df = df.with_row_index("rank")
    df = df.with_columns(
        pl.when((pl.col("summed_sentiments")<0))\
            .then(pl.col("rank") - 
                  np.log2(-pl.col("summed_sentiments")) - 
                  sentiment_bump)\
            .when((pl.col("summed_sentiments")>0))\
            .then(pl.col("rank") + 
                  np.log2(pl.col("summed_sentiments")) + 
                  sentiment_bump)\
            .otherwise(pl.col("rank"))\
            .alias("rank")
    )

    return df.sort(["rank","_distance"]).drop("rank").clone()