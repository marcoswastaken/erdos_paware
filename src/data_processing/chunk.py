from langchain_text_splitters import RecursiveCharacterTextSplitter
import polars as pl
from typing import Optional, Tuple

def chunk_preprocessed_data(data_preprocessed: pl.DataFrame,
                            chunk_size: int,
                            chunk_overlap_pct: float)->pl.DataFrame:
    """
    Given a pl.DataFrame with reddit data, break target data into chunks
    suitable for embedding.

    Parameters:
    -----------
    data_raw: pl.DataFrame
        A dataframe containing the preprocessed data
    chunk_size: int
        The maximum size of any text chunk
    chunk_overlap_pct: float
        The percent adjacent text chunks should overlap

    Returns:
    --------
    pl.DataFrame
        A clone of the input dataframe with a new column added to it:
            "text_chunk": A chunk of text determined by the splitter

    Example:
    --------
    data_chunked = chunk_preprocessed_data(data_preprocessed=data_preprocessed,
                                           chunk_size=512,
                                           chunk_overlap_pct=0.2)                          
    """

    ## Clone the data
    data_chunked = data_preprocessed.clone()

    ## Initialize the splitter    
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size*chunk_overlap_pct),
            length_function=len,
            is_separator_regex=False,
            strip_whitespace=True)
    
    ## Break longer texts into chunks
    data_chunked = data_chunked.with_columns(
        text_chunk=pl.col("reddit_text").map_elements(
            lambda x:text_splitter.split_text(x)))
    
    ## Explode the text chunks
    data_chunked = data_chunked.explode("text_chunk")

    ## Check for null chunks and announce their presence
    mask = (data_chunked["text_chunk"].is_null())
    null_chunk_count = data_chunked.filter(mask).shape[0]
    if null_chunk_count>0:
        print(f"There are {null_chunk_count} rows with "+
                    f"text_chunk is null")
    
    ## Return the chunked dataframe
    return data_chunked