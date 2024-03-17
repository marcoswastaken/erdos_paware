import matplotlib.pyplot as plt

from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from transformers import AutoTokenizer

import polars as pl

from typing import Optional, List, Tuple

def vectorize_chunked_data(data_chunked: pl.DataFrame, 
                           embeddings: HuggingFaceEmbeddings=HuggingFaceEmbeddings(
                               model_name="thenlper/gte-base")
                           )->pl.DataFrame:
    '''
        Vectorize the chunked texts

        Parameters:
        -----------
            data_chunked: pl.DataFrame
                A dataframe with a chunk of text to be embedded
            embeddings: HuggingFaceEmbeddings
                A sentence-transformer model from the Hugging Face library 
                https://huggingface.co/models?library=sentence-transformers
                Default: HuggingFaceEmbeddings(model_name="thenlper/gte-base")

        Returns:
        --------
            pl.DataFrame
                A clone of the input datafram with a new column:
                    - "vector": The vector representing the text_chunk
                        determined by the embdeddings model
        
        Example:
        --------
            from langchain_community.embeddings import HuggingFaceEmbeddings
            model = HuggingFaceEmbeddings(model_name="thenlper/gte-base")
            data_vectorized = vectorize.vectorize_chunked_data(data_chunked=data_chunked,
                                                               embeddings=model)
    '''
    ## Clone the data
    data_vectorized = data_chunked.clone()
    
    ## Drop any rows with null chunks and announce if this is done
    mask = (data_vectorized["text_chunk"].is_null())
    if data_vectorized.filter(mask).shape[0]>0:
        print(f"Dropping {data_vectorized.filter(mask).shape[0]} rows with "+
                    f"text_chunk is null")
    data_vectorized = data_vectorized.filter(~mask)
    
    ## Compute embeddings
    text_chunks = data_vectorized["text_chunk"].to_list()
    vectors = embeddings.embed_documents(text_chunks)
    
    ## Add embeddings to the dataframe
    data_vectorized = data_vectorized.with_columns(
        pl.Series(
            name="vector",
            values=vectors
        )
    )

    return data_vectorized

def check_token_lengths(data_chunked:pl.DataFrame, 
                        model:str="thenlper/gte-base"):
    """ 
    Make sure that the text chunks can be handled (vectorized) by the 
    chosen model.

    Prints the maximum length of a tokenized input allowed by the model, and the
    maximum length of the tokenized content observed in the input pl.DataFrame.
    Prints the maximum token length of the chunks of text, and graphs the 
    distribution of chunked text lengths (in tokens).

    Parameters:
    -----------
        data_chunked: pl.DataFrame
            A dataframe with a chunk of text to be embedded
        model: str
            The name of a sentence-transformer model from the Hugging Face 
            library https://huggingface.co/models?library=sentence-transformers
            Default: "thenlper/gte-base"

    Returns:
    --------
    None

    Example:
    --------
    check_token_lengths(data_chunked=data_chunked, model="thenlper/gte-base")
    """

    print(f"Model's maximum sequence length",
          f": {SentenceTransformer(model).max_seq_length}")
    
    ## Define a tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model)
    
    ## Get a list of text chunks
    text_chunks = data_chunked["text_chunk"].to_list()

    ## Count the length of the sequence of tokens after tokenizing each chunk
    lengths = []
    for chunk in text_chunks:
        if chunk:
            lengths.append(len(tokenizer.encode(chunk)))
    
    print(f"Max sequence length observed after tokenizing"+
          f" chunked text: {max(lengths)}")
    
    # Plot the distribution of tokenized chunk sequence lengths
    fig = plt.hist(lengths)
    plt.title("Distribution of chunk lengths in"+
              " the knowledge base (in count of tokens)")
    plt.xlabel("Chunk Length (in tokens)")
    plt.ylabel("Frequency")
    plt.show()