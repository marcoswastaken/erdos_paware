import polars as pl
import re
from typing import Optional, List, Tuple

## For adding our modules to the system path
import sys
sys.path.append("../src/")

from adding_metadata import replies

def preprocess_data(DATA_RAW: pl.DataFrame)->pl.DataFrame:
    '''
    Complete basic preprocessing tasks:
        - Drop rows with comments|submissions that have been dropped|deleted
        - Drop rows with comments|submissions automatically generated by a bot
        - Drop rows with comments that are empty
        - Replace 'reddit_text' with 'reddit_title' in rows with sumbissions 
            that are empty
        - Add reply_id column that lists the reddit_name of replies to that row

    Parameters:
    -----------
        DATA_RAW: pl.DataFrame
            A dataframe loaded from the raw Aware data provided
    
    Returns:
    --------
        pl.DataFrame
            A clone of the raw dataframe that has been preprocessed as 
            described above

    Example:
    --------
    ## Load raw data
    DATA_PARQUET = "../temp_data/raw_data_subset_gm.parquet"
    DATA_RAW = pl.read_parquet(DATA_PARQUET)
    ## Implement preprocessing
    data_preprocessed = preprocess_data(DATA_RAW=DATA_RAW)
    '''
    
    ## Clone raw data
    data_preprocessed = DATA_RAW.clone()
    
    ## Drop rows with comments|submissions that have been dropped|deleted

    values_to_drop = ["[deleted]", "[removed]"]
    
    for value in values_to_drop:
        
        mask = (data_preprocessed["reddit_text"] != value)
        
        print(f"Dropping {data_preprocessed.filter(~mask).shape[0]} rows with "+
            f"reddit_text=='{value}'")
        
        data_preprocessed = data_preprocessed.filter(mask)

    ## Drop rows with comments|submissions automatically generated by a bot

    data_preprocessed = filter_out_bots(data_preprocessed)

    ## Drop rows with comments|submissions that are empty

    data_preprocessed = replace_null_text_with_title(data_preprocessed)

    data_preprocessed = replies.add_reply_list(data_preprocessed)
    
    return data_preprocessed

def filter_out_bots(DATA_RAW: pl.DataFrame, 
                    min_len:int=35, 
                    min_freq:int=7)->pl.DataFrame:
    '''
        A function that filters out likely bots and memes.

        Parameters:
        -----------
            DATA_RAW: pl.DataFrame 
                A DataFrame with a scheme similar to the raw dataset from Aware   
            
            min_len:int=35
                All filtered posts have more than min_length characters.
                Default: 35
                    
            min_freq:int
                All filtered posts appear more than min_freq times.
                Default: 7
        
        Returns:
        --------
            pl.DataFrame
            A DataFrame after applying the described filters.
    '''
    ## Count how many times each value appears in the 'reddit_text' column
    value_counts = DATA_RAW["reddit_text"].value_counts()

    ## Focus on the ones that show up more than once
    non_unique_texts = value_counts.filter(pl.col("count")>1)

    ## Add a column that hold the length of the 'reddit_text' string
    non_unique_texts = non_unique_texts.with_columns(
        text_length = non_unique_texts["reddit_text"].str.len_bytes()
        )
    
    ## Apply the filter to catch likely bots and memes
    likely_bots = non_unique_texts.filter(
        pl.col("text_length")>min_len).filter(
            pl.col("count")>min_freq)
    
    ## Filter likely bots and memes out of the dataset
    filtered_data = DATA_RAW.filter(
        ~pl.col("reddit_text").is_in(likely_bots["reddit_text"]))
    
    ## Count how many rows were filtered out
    filter_count = DATA_RAW.shape[0]-filtered_data.shape[0]
    print(f"Dropping {filter_count} rows that are likely bots or memes")

    ## Return the filtered data
    return filtered_data

def replace_null_text_with_title(DATA_RAW: pl.DataFrame)->pl.DataFrame:
    '''
        A function that handles rows with 'reddit_text'="" or 'reddit_text'=" ".
            - For 'aware_post_type'="comment": 
                drop this row
            - For aware_post_type="submission": 
                replace 'reddit_text' value with 'reddit_title' value

        Parameters:
        -----------
            DATA_RAW: pl.DataFrame 
                A DataFrame with a scheme similar to the raw dataset from Aware   
        
        Returns:
        --------
            pl.DataFrame
            A DataFrame after applying the described filters.
    '''

    ## Drop rows with empty comments
    text_is_empty = DATA_RAW["reddit_text"]==""
    text_is_a_comment = DATA_RAW["aware_post_type"]=="comment"
    mask = (text_is_a_comment & text_is_empty)
    text_inferred = DATA_RAW.filter(~mask)
    num_dropped = DATA_RAW.shape[0] - text_inferred.shape[0]
    print(f"Dropping {num_dropped} rows with 'reddit_text'=='' "+
          "and 'aware_post_type'=='comment'")
    
    text_is_empty = DATA_RAW["reddit_text"]==" "
    text_is_a_comment = DATA_RAW["aware_post_type"]=="comment"
    mask = (text_is_a_comment & text_is_empty)
    text_inferred = DATA_RAW.filter(~mask)
    num_dropped = DATA_RAW.shape[0] - text_inferred.shape[0]
    print(f"Dropping {num_dropped} rows with 'reddit_text'==' ' "+
          "and 'aware_post_type'=='comment'")
    
    ## Replace empty submissions with their titles
    blanks = text_inferred.filter(pl.col("reddit_text")=="").shape[0]
    blanks += text_inferred.filter(pl.col("reddit_text")==" ").shape[0]

    text_inferred = text_inferred.with_columns(
        pl.when(pl.col("reddit_text")=="") \
            .then(pl.col("reddit_title")) \
            .otherwise(pl.col("reddit_text")) \
            .alias("reddit_text")
    )
    text_inferred = text_inferred.with_columns(
        pl.when(pl.col("reddit_text")==" ") \
            .then(pl.col("reddit_title")) \
            .otherwise(pl.col("reddit_text")) \
            .alias("reddit_text")
    )

    print(f"Replacing 'reddit_text' with 'reddit_title' in {blanks} rows "+
          "with 'reddit_text'=='' or 'reddit_text'==' '")
    
    return text_inferred