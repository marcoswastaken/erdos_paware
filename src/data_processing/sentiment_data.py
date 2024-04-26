import polars as pl
import os

def copy_sentiment_data(vec_dir, sen_dir, force_try_all=False):
    '''
    This function copies the sentiment data to the vectorized data
    
    vec_dir: str, path to the vectorized data
    sen_dir: str, path to the sentiment data

    It is expected that the sentiment data has the following columns:
    - reddit_id
    - summed_sentiments
    - absolute_summed_sentiment

    It is expected that the vectorized data has the following columns:
    - reddit_id
    - vector

    It is also expected that the filename structure will be:
    For vectorized data: vectorized_[subreddit]_data_complete.parquet
    For sentiment data: [subreddit]_data.parquet
    '''
    ## files to update
    files = [f for f in os.listdir(sen_dir) if f[0]!="."]

    for f in files:
        raw_filename = vec_dir+"vectorized_"+f.split(".")[0]+"_complete.parquet"
        
        raw_df = pl.read_parquet(raw_filename)
        if "summed_sentiments" in raw_df.columns:
            if force_try_all:
               print("Sentiment data already copied ... skipping")
               continue
            else:
                print("Some sentiment data already copied ... stopping")
                print("If you want to force copy all sentiment data, set force_try_all=True")
                break
        sen_df = pl.read_parquet(sen_dir+f)[
            ["reddit_id", 'summed_sentiments', 'absolute_summed_sentiment']]
        
        full_df = raw_df.join(sen_df, on="reddit_id", how="left")
        col_order = [col for col in full_df.columns if col != "vector"]+["vector"]
        full_df = full_df[col_order].clone()
        os.remove(raw_filename)
        full_df.write_parquet(raw_filename)