import os
import polars as pl

humans = ["mo", "kk", "kp", "dr", "sr"]
vote_cols = ["mo_label", "kk_label", "kp_label", "dr_label", "sr_label"]

LABELED_DATA_DIR = "../data_labeling/labeled_results/"

def get_merged_labels_and_votes(config:str)->pl.DataFrame:
    """
    Create a dataframes of merged labels for a given config

    Args:
        config (str): 
            The configuration used to generate the query/result pairs 
            for labeling.
    
    Returns:
        pl.DataFrame: 
            A dataframe with the merged labels for the given config.
    """
    ## Get filenames for labeled data in specified config
    files = [
        f for f in os.listdir("../data_labeling/labeled_results/") 
        if f.startswith("labeled_"+config)]
    
    ## Read the first file to get the schema
    df = pl.read_parquet(
        LABELED_DATA_DIR+files[0]).sort(by=["query_text", "reddit_name"])

    ## Merge all the files
    for f in files[1:]:

        ## Get the current columns
        df_cols = df.columns

        ## Get the next human name from the filename
        human = f.split("_")[2][:2]

        ## Read the next file
        next_df = pl.read_parquet(LABELED_DATA_DIR+f)
        ## Join the next file to the current dataframe
        df = df.join(next_df, on=["query_text", "reddit_name"], how="outer")

        ## Update the labels from the next human with the current data
        df = df.with_columns(
            pl.when(pl.col(human+"_label_right")\
                .is_not_null())\
                .then(pl.col(human+"_label_right"))\
                .otherwise(None)\
                .alias(human+"_label")
        )[df_cols]
    
    ## Add a column that has a list of all votes cast
    df = df.with_columns(
        pl.struct([pl.col(column_name) for column_name in vote_cols])\
            .map_elements(lambda s: [value for value in s.values() 
                                 if value is not None])
                                 .alias("votes"))
    
    return df

def get_majority_vote(df:pl.DataFrame)->pl.DataFrame:
    """
    Get the majority vote for each query_text and reddit_name pair

    Args:
        df (pl.DataFrame): 
            A dataframe with the merged labels for the given config.
    
    Returns:
        pl.DataFrame: 
            A dataframe with the majority vote for each query_text and 
            reddit_name pair.
    """
    ## Set up the voting machine
    def voting_machine(row: pl.Series) -> int:
        votes = row.to_list()
        votes_1 = votes.count(1)
        votes_2 = votes.count(2)
        votes_3 = votes.count(3)
        
        ## Clear winners
        if (votes_1 > votes_2) & (votes_1 > votes_3):
            return 1
        elif (votes_2 > votes_1) & (votes_2 > votes_3):
            return 2
        elif (votes_3 > votes_2) & (votes_3 > votes_1):
            return 3

        ## Dealing with ties
        if votes_1 == votes_2 > votes_3:
            return 2

        if votes_1 == votes_3 > votes_2:
            return 3

        if votes_2 == votes_3 > votes_1:
            return 2

        if votes_1 == votes_2 == votes_3:
            return 2

    # Apply the voting machine function
    df = df.with_columns(
        df['votes'].map_elements(
            voting_machine, return_dtype=int).alias("relevance_rating"))
    
    return df