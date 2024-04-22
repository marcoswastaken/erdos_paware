## For adding our modules to the system path
import sys
sys.path.append("../src/")

## For file handling
import os

## For warnings
import warnings

## Loading our modules
from data_processing import preprocess, chunk, vectorize

## For data handling
import polars as pl

## For embedding models
from langchain_community.embeddings import HuggingFaceEmbeddings

## For vector database
import lancedb

class PawEmbedding:
    def __init__(
            self,
            CONFIG_NAME: str,
            RAW_DATA_PATH: str,
            EMBEDDED_SAVE_DIR: str,
            BATCH_SIZE: int,
            CHUNK_WITH_METADATA: bool,
            CHUNK_SIZE: int,
            CHUNK_OVERLAP_PCT: float,
            ) -> None:
        
        self.config_name = CONFIG_NAME
        self.raw_data_path = RAW_DATA_PATH
        self.embedded_save_dir = EMBEDDED_SAVE_DIR+"/config_"\
            +self.config_name+"/"
        self.batch_size = BATCH_SIZE
        self.chunk_with_metadata = CHUNK_WITH_METADATA
        self.chunk_size = CHUNK_SIZE
        self.chunk_overlap_pct = CHUNK_OVERLAP_PCT

        self.file_prefix = "vectorized_config_"+self.config_name+"_data_"

    def embed_data(self, verbose: bool = False):
        if verbose: print("Loading and chunking...")
        ## Load raw data
        data_raw = pl.read_parquet(self.raw_data_path)

        ## Precprocess raw data
        data_preprocessed = preprocess.preprocess_data(DATA_RAW=data_raw)

        ## Chunk preproccessed data
        if self.chunk_with_metadata:
            ## Chunk with metadata
            data_chunked = chunk.chunk_preprocessed_data_with_subreddit(
                data_preprocessed=data_preprocessed, 
                chunk_size=self.chunk_size, 
                chunk_overlap_pct=self.chunk_overlap_pct)
        else:
            ## Chunk without metadata
            data_chunked = chunk.chunk_preprocessed_data(
                data_preprocessed=data_preprocessed, 
                chunk_size=self.chunk_size, 

                chunk_overlap_pct=self.chunk_overlap_pct)
        if verbose: print("... done loading and chunking.\n")

        if verbose: print("Vectorizing and saving...")
        ## Vectorize data in batches and save the results in parquets
        vectorize.batch_vectorize_and_save(
            data_chunked=data_chunked,
            batch_size=self.batch_size,
            save_dir=self.embedded_save_dir,
            file_prefix=self.file_prefix
        )
        if verbose: print("... done vectorizing and saving.\n")

        ## List the files created
        file_path_list = os.listdir(self.embedded_save_dir)
        if verbose: print(f"Files created:\n:{file_path_list}\n")
    
class PawIndex:
    def __init__(
            self,
            EMBEDDING_CONFIG_NAME: str,
            EMBEDDING_DIR: str,
            INDEX_CONFIG_NAME: str,
            DB_SAVE_DIR: str,
            METRIC: str,
            NUM_PARTITIONS: int = 1024,
            NUM_SUB_VECTORS: int = 96,
            ACCELERATOR: str = None,
            ) -> None:
        
        self.embedding_config_name = EMBEDDING_CONFIG_NAME
        self.embedding_dir = EMBEDDING_DIR
        self.index_config_name = INDEX_CONFIG_NAME
        self.db_save_dir = DB_SAVE_DIR
        
        self.metric = METRIC
        self.num_partitions = NUM_PARTITIONS
        self.num_sub_vectors = NUM_SUB_VECTORS
        self.accelerator = ACCELERATOR

        self.vector_dir = self.embedding_dir+"config_"\
            +self.embedding_config_name+"/"
        self.vector_files = os.listdir(self.vector_dir)
        self.database_dir = self.db_save_dir+"/db_"\
            +self.embedding_config_name\
                +self.index_config_name

        self.config_name = self.embedding_config_name+self.index_config_name

    def index_data(self, verbose: bool = False):
        ## Initialize a database
        db = lancedb.connect(self.database_dir)

        ## Load the first data file
        data = pl.read_parquet(self.vector_dir+self.vector_files[0])

        ## Initialize a table in the database
        table = db.create_table("table_"+self.config_name, data=data)

        ## Load the remaining data into the table, one file at a time
        for i in range(1, len(self.vector_files)):
            data = pl.read_parquet(self.vector_dir+self.vector_files[i])
            table.add(data)

        ## Build the ANN Index
        ## Ignoring a UserWarning that is out of my control
        with warnings.catch_warnings(category=UserWarning):
            table.create_index(
                metric=self.metric,
                num_partitions=self.num_partitions,
                num_sub_vectors=self.num_sub_vectors,
                accelerator=self.accelerator
            )

class PawQuery:
    standard_queries = [
    "How do General Motors employees feel about RTO?",
    "What kind of benefits does GM offer?",
    "When should you apply for a promotion at GM?",
    "How much does a driver make with UPS?",
    "How long is a typical UPS shift? OR Should I work a double shift at UPS?",
    "How do UPS employees feel about route cuts?",
    "Is it better to work at fedex express or fedex ground?",
    "How do FedEx employees feel about route cuts?",
    "How often do you get a raise at Lowes?",
    "Does your schedule get changed often at Lowes?",
    "What is the worst drink to make for Starbucks baristas?",
    "Does Starbucks pay overtime?",
    "What is your favorite thing about working for Starbucks?",
    "How do Whole Foods workers feel about store managers?",
    "What job perks for Whole Foods employees value most?",
    #"Do Kraken employees see themselves staying at the company for the long term?",
    #"What do Kraken employees find frustrating in their day to day work?",
    "What benefits do Chase employees value most?",
    "Do Chase employees see opportunities for promotion and professional growth at the company?",
    "What causes bank employees the most stress at work?",
    "What are some reasons that bank employees quit their jobs?",
    "Do Fidelity employees want to work remotely?",
    "Do GameStop employees feel valued by the company?",
    "What does a typical day look like when working for GameStop?",
    "Do CVS employees feel safe at work?",
    "What do CVS workers do if they notice theft?"]

    embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-base")

    def __init__(
            self,
            CONFIG_NAME: str,
            DB_DIR: str,
            QUERY_SAVE_DIR: str,
            QUERY_NAME: str,
            METRIC: str,
            LIMIT: int,
            NPROBES: int,
            REFINE_FACTOR: int,
    ) -> None:
        
        self.config_name = CONFIG_NAME
        self.db_dir = DB_DIR+"db_"+self.config_name
        self.query_save_dir = QUERY_SAVE_DIR
        self.query_name = QUERY_NAME
        self.metric = METRIC
        self.limit = LIMIT
        self.nprobes = NPROBES
        self.refine_factor = REFINE_FACTOR

        self.db_table = "table_"+self.config_name
        self.query_file = self.query_save_dir\
            +"queries_"+self.config_name\
            +self.query_name+".parquet"

        self.db = lancedb.connect(self.db_dir)
        self.table = self.db.open_table(self.db_table)

    def ask_a_query(self, query: str):
        ## Embed the query
        query_embedding = PawQuery.embedding_model.embed_query(query)

        result = self.table.search(query_embedding)\
            .metric(self.metric)\
            .limit(self.limit)\
            .nprobes(self.nprobes)\
            .refine_factor(self.refine_factor)\
            .to_polars()
        
        return result

    def ask_standard_queries(self):
        results = self.ask_a_query(PawQuery.standard_queries[0])

        results = results.with_columns(
            query_text = pl.lit(PawQuery.standard_queries[0])
        )

        for i in range(1, len(PawQuery.standard_queries)):
            next = self.ask_a_query(PawQuery.standard_queries[i])
            next = next.with_columns(
                query_text = pl.lit(PawQuery.standard_queries[i])
            )
            results = pl.concat([results, next])
        
        return results