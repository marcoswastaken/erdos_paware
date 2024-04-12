import os
import polars as pl

HUMAN = "mo" ## "mo", "kk", "kp", "dr", "sr"
CONFIG = "02"

RAW_RESULTS_PATH = "../data_labeling/raw_results/query_results_config_"+CONFIG+".parquet"
SAVE_PATH = "../data_labeling/labeled_results/labeled_"+CONFIG+"_"+HUMAN+".parquet"

## Check if a label file already exists.
if os.path.exists(SAVE_PATH):
  ## If it does, load that
  results_to_rate = pl.read_parquet(SAVE_PATH)
  print("The file", SAVE_PATH, " has been loaded.\n\n")
else:
  ## If not, start fresh
  raw_results = pl.read_parquet(RAW_RESULTS_PATH)
  columns_to_consider=["reddit_name", "query_text"]
  results_to_rate = raw_results.unique(columns_to_consider)
  print(f"No previously labeled results found at {SAVE_PATH}. Starting from scratch...\n\n")

## Find unrated results:
keep_going = True

while keep_going == True:
    ## Get unrated query/post pairs:
    mask = results_to_rate[HUMAN+"_label"].is_null()
    unrated = results_to_rate.filter(mask).sample(fraction=1, shuffle=True)
    if unrated.shape[0]==1:
        keep_going = False
    
    ## Get the info for the current pair
    post_id = unrated["reddit_name"][0]
    query = unrated["query_text"][0]
    subreddit = unrated["reddit_subreddit"][0]
    post_type = unrated["aware_post_type"][0]
    post_text = unrated["reddit_text"][0]

    ## Clear the screen on Windows
    if os.name == "nt":
        os.system("cls")

    ## Clear the screen on Linux or macOS
    else:
        os.system("clear")
    
    ## Serve the query and the pair
    print("-"*80+"\n")
    print(f"Query: {query} \n")
    print("-"*80+"\n")
    print(f"Result is a {post_type} from the [[[{subreddit}]]] subreddit:\n\n")
    print(f"{post_text}\n")
    print("-"*80+"\n")

    ## Ask for a rating
    print("Please label this result as:")
    print("(1) Relevant to the query")
    print("(2) Related to the query, but not relevant to the query")
    print("(3) Not related to the query\n")
    human_rating = input("Your label: ")
    print("\n\n")

    ## Update the results_to_rate dataframe
    is_same_post = (results_to_rate["reddit_name"]==post_id)
    is_same_query = (results_to_rate["query_text"]==query)
    should_update = is_same_post & is_same_query
    results_to_rate = results_to_rate.with_columns(
        pl.when(should_update)\
        .then(pl.lit(int(human_rating)))\
        .otherwise(pl.col(HUMAN+"_label"))
        .alias(HUMAN+"_label")
    )

    ## Save the results_to_rate dataframe
    results_to_rate.write_parquet(
        "../data_labeling/labeled_results/labeled_"+CONFIG+"_"+HUMAN+".parquet")