import polars as pl
import typing

def add_reply_list(df:pl.DataFrame)->pl.DataFrame:
    '''
    A function that adds a column called 'reply_ids' that contains a list of the 
    'reddit_name' ids that are replies to the post in that row. If a there are 
    no replies, the value in this column is 'null'.

    Parameters:
    -----------
        df:pl.DataFrame
            A pl.DataFrame with a schema similar to the raw Aware data
    
    Returns:
    --------
        pl.DataFrame
            The input dataframe along with a new 'reply_ids' column, as 
            described above.
    '''

    ## Group the data by 'reddit_parent_id'
    group = df.group_by("reddit_parent_id")

    ## Aggregate the list of 'reddit_names' for each 'reddit_parent_id'
    replies = group.agg(pl.col("reddit_name"))
    ## Rename the columns so that they are aligned for joining. 
    ## We want to add the list of replies to a 'reddit_parent_id' to the
    ## row whose 'reddit_name' matches that value.
    new_names = {"reddit_name":"reply_ids","reddit_parent_id":"reddit_name"}
    replies = replies.rename(new_names)

    ## Join the list of 'reddit_name' values that are replies
    return df.join(replies, on="reddit_name", how="left")