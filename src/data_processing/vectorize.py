from langchain_community.embeddings import HuggingFaceEmbeddings
import polars as pl
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from typing import Optional, List, Tuple

def vectorize_chunked_data(data_chunked: pl.DataFrame, 
                           embeddings: HuggingFaceEmbeddings=HuggingFaceEmbeddings(model_name="thenlper/gte-base")
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
                                                               model=model)
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