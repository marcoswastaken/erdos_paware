import os
import time
import warnings

## For adding our modules to the system path
import sys
sys.path.append("../src/")

## Loading our modules
from data_processing import preprocess, chunk, vectorize

## For data handling
import polars as pl

## For embedding models
from langchain_community.embeddings import HuggingFaceEmbeddings

## For vector database
import lancedb

CONFIG = "02"
VERBOSE = True
## Read settings: 
RAW_DATA_PATH = "../temp_data/raw_data_subset_of_13.parquet"

## Write settings for vectorized data:
BATCH_SIZE = 200000 ## Number of rows to vectorize per batch 
                    ##(max: ~200000 on M2 Max with 64GB RAM)

VECTORIZED_SAVE_DIR = "../vectorized_data/config_"+CONFIG+"/"
FILE_PREFIX = "vectorized_config_"+CONFIG+"_data_"

## Write settings for database and tables
DATABASE_DIR = "../db_data/db_"+CONFIG

## Define chunking hyperparameters
CHUNK_SIZE = 512
CHUNK_OVERLAP_PCT = 0.2
CHUNK_WITH_METADATA = True

## Define model hyperparameters
MODEL = HuggingFaceEmbeddings(model_name="thenlper/gte-base")

## Define ANN Hyperparameters
## https://lancedb.github.io/lancedb/ann_indexes/#creating-an-ivf_pq-index
NUM_PARTITIONS=1024
NUM_SUB_VECTORS=96
ACCELERATOR="mps" ## "mps" to use Apple Metal, default: None

master_start_time = time.time()
start_time = time.time()
if VERBOSE: print("Loading and chunking...")
## Load raw data
DATA_RAW = pl.read_parquet(RAW_DATA_PATH)

## Precprocess raw data
data_preprocessed = preprocess.preprocess_data(DATA_RAW=DATA_RAW)

## Chunk preproccessed data
if CHUNK_WITH_METADATA:
    ## Chunk with metadata
    data_chunked = chunk.chunk_preprocessed_data_with_subreddit(
        data_preprocessed=data_preprocessed, 
        chunk_size=CHUNK_SIZE, 
        chunk_overlap_pct=CHUNK_OVERLAP_PCT)
else:
    ## Chunk without metadata
    data_chunked = chunk.chunk_preprocessed_data(
        data_preprocessed=data_preprocessed, 
        chunk_size=CHUNK_SIZE, 
        chunk_overlap_pct=CHUNK_OVERLAP_PCT)

if VERBOSE: print("...Done.\n")
end_time = time.time()
if VERBOSE: print(f"Runtime: {end_time-start_time:.4f} seconds\n\n")

start_time = time.time()
if VERBOSE: print("Vectorizing batches...")

## Vectorize data in batches and save the resulting dataframes in parquets
vectorize.batch_vectorize_and_save(
    data_chunked=data_chunked,
    batch_size=BATCH_SIZE,
    save_dir=VECTORIZED_SAVE_DIR,
    file_prefix=FILE_PREFIX
)
if VERBOSE: print("...Done.\n")
end_time = time.time()
if VERBOSE: print(f"Runtime: {end_time-start_time:.4f} seconds")
file_path_list = os.listdir(VECTORIZED_SAVE_DIR)
if VERBOSE: print(f"Files created:\n:{file_path_list}\n")

start_time = time.time()
if VERBOSE: print("Building Vector Database...")

## Initialize a database
db = lancedb.connect(DATABASE_DIR)

## Load the first data file
data = pl.read_parquet(VECTORIZED_SAVE_DIR+file_path_list[0])

## Initialize a table in the database
table = db.create_table("table_"+CONFIG, data=data)

## Load the remaining data into the table, one file at a time
for i in range(1, len(file_path_list)):
    data = pl.read_parquet(VECTORIZED_SAVE_DIR+file_path_list[i])
    table.add(data)

## Build the ANN Index
## Ignoring a UserWarning that is out of my control
with warnings.catch_warnings(category=UserWarning):
    table.create_index(
        num_partitions=NUM_PARTITIONS,
        num_sub_vectors=NUM_SUB_VECTORS,
        accelerator=ACCELERATOR
    )

if VERBOSE: print("...Done.\n")
end_time = time.time()
if VERBOSE: print(f"Runtime: {end_time-start_time:.4f} seconds\n\n")

master_end_time = time.time()
if VERBOSE: print(
    f"Full processing time: {master_end_time-master_start_time:.4f} seconds")