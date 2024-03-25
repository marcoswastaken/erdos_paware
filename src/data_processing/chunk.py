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

def chunk_preprocessed_data_with_subreddit(
        data_preprocessed: pl.DataFrame,
        chunk_size: int,
        chunk_overlap_pct: float)->pl.DataFrame:
    """
    Given a pl.DataFrame with reddit data, break target data into chunks
    suitable for embedding. Add the subreddit title to the start of each chunk.

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

    ## Define labels
    '''
    Here we have to decide how to use the subreddit title as a useful bit of
    metadata that we can add to each text chunk. The idea is that, by adding
    this information before embedding, a query like "How much time do Walmart 
    employees get off every year?" will be closer to "WalMart \n\n I get 15 days 
    off" than just "I get 15 days off" or "CVS \n\n I get 15 days off".
    '''

    meta_label_dict= {
        'RiteAid':"RiteAid",
        'cabincrewcareers':"Flight Attendants\n\n",
        'Lowes': "Lowes\n\n",
        'KrakenSupport': "Kraken\n\n",
        'Fedexers': "FedEx\n\n",
        'GameStop': "GameStop\n\n",
        'disney': "Disney\n\n",
        'walmart': "Walmart\n\n",
        'TalesFromYourBank':"Bank\n\n",
        'DisneyWorld':"Disney\n\n",
        'WalmartEmployees':"Walmart\n\n",
        'Panera':"Panera\n\n",
        'FedEmployees':"Federal\n\n",
        'sysadmin':"System Administrator\n\n",
        'Target':"Target\n\n",
        'starbucksbaristas':"Starbucks\n\n",
        'UPSers':"UPS\n\n",
        'starbucks':"Starbucks\n\n",
        'McDonaldsEmployees':"McDonalds\n\n",
        'BestBuyWorkers':"Best Buy\n\n",
        'Disneyland':"Disney\n\n",
        'Chase':"Chase\n\n",
        'WaltDisneyWorld':"Disney\n\n",
        'wholefoods':"Whole Foods\n\n",
        'cybersecurity':"Cybersecurity\n\n",
        'TjMaxx':"TJ Maxx\n\n",
        'nursing':"Nursing\n\n",
        'CVS':"CVS\n\n",
        'Bestbuy':"Best Buy\n\n",
        'fidelityinvestments':"Fidelity\n\n",
        'GeneralMotors':"General Motors, GM\n\n",
        'McLounge':"McDonalds\n\n",
        'DollarTree':"Dollar Tree\n\n",
        'PaneraEmployees':"Panera\n\n"
        }
    
    ## Adjust chunk length to account for maximum label length
    lengths = []
    for value in meta_label_dict.values():
        lengths.append(len(value))
    
    max_label_length = max(lengths)
    chunk_size -= max_label_length

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
    
    ## Add metadata label to the start of each chunk
    data_chunked = data_chunked.with_columns(
        text_chunk = pl.col("reddit_subreddit").replace(meta_label_dict)\
              + pl.col("text_chunk"))
    
    ## Return the chunked dataframe with metadata labels
    return data_chunked