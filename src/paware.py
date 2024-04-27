## For adding our modules to the system path
import sys
sys.path.append("../src/")

## For file handling
import os

## For warnings
import warnings

## Loading our modules
from data_processing import preprocess, chunk, vectorize
from evaluating_results import process_labels
from adding_metadata import agree_disagree_distances
from query_tools import reranking

## For data handling
import polars as pl

## For math
import numpy as np
import math

## For embedding models
from langchain_community.embeddings import HuggingFaceEmbeddings

## For vector database
import lancedb

class PawEmbedding:
    '''
    This class is used to embed data using a given configuration.

    Parameters:
        CONFIG_NAME: str
            The name of the configuration to be used.
        RAW_DATA_PATH: str
            The path to the raw data to be embedded.
        EMBEDDED_SAVE_DIR: str
            The directory where the embedded data will be saved.
        BATCH_SIZE: int
            The size of the batches to be used for vectorization.
        CHUNK_WITH_METADATA: bool
            Whether to chunk the data with metadata.
        CHUNK_SIZE: int
            The size of the chunks to be used for vectorization.
        CHUNK_OVERLAP_PCT: float
            The percentage of overlap between chunks.
        
        Returns:
            None

        Methods:
            embed_data(prefix:str = None, verbose: bool = False)
                This method is used to embed the data.
            embed_from_subs(subs_dir:str)
                This method is used to embed data from subreddits stored 
                individually.
            add_agree_disagree_distances()
                This method is used to add agree and disagree distances to the 
                data.
            copy_agree_disagree_distances(finished_dir:str)
                This method is used to copy agree and disagree distances from a 
                file.
    '''

    def __init__(
            self,
            CONFIG_NAME: str,
            RAW_DATA_DIR: str,
            EMBEDDED_SAVE_DIR: str,
            BATCH_SIZE: int,
            CHUNK_WITH_METADATA: bool,
            CHUNK_SIZE: int,
            CHUNK_OVERLAP_PCT: float,
            ) -> None:
        
        self.config_name = CONFIG_NAME
        self.raw_data_dir = RAW_DATA_DIR
        self.embedded_save_dir = EMBEDDED_SAVE_DIR+"config_"\
            +self.config_name+"/"
        self.batch_size = BATCH_SIZE
        self.chunk_with_metadata = CHUNK_WITH_METADATA
        self.chunk_size = CHUNK_SIZE
        self.chunk_overlap_pct = CHUNK_OVERLAP_PCT

    def embed_data(self, prefix:str = None, verbose: bool = False):
        '''
        This method is used to embed the data using the given configuration.

        Parameters:
            verbose: bool
                Whether to print the progress of the process.
        
        Returns:
            None
        '''
        if prefix:
            file_prefix = "vectorized_"+prefix+"_data_"
        else:
            file_prefix = "vectorized_config_"+self.config_name+"_data_"
        if verbose: print("Loading and chunking...")
        ## Load raw data
        data_files = [f for f in os.listdir(self.raw_data_dir) if f[0]!="."]
        if prefix:
            data_raw = pl.read_parquet(self.raw_data_dir+prefix+"_data.parquet")
        else:
            data_raw = pl.read_parquet(self.raw_data_path+data_files[0])

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
            file_prefix=file_prefix
        )

        ## Combine the parquet files
        if verbose: print("Combining parquet files...")
        files = [f for f in os.listdir(self.embedded_save_dir) 
                 if (not f.endswith("complete.parquet")) and (f[0]!=".")]  
        
        df = pl.read_parquet(self.embedded_save_dir+files[0])
        for f in files[1:]:
            df = pl.concat([df, pl.read_parquet(self.embedded_save_dir+f)])
        
        df.write_parquet(
            self.embedded_save_dir+file_prefix+"complete.parquet")

        for f in files:
            os.remove(self.embedded_save_dir+f)

        if verbose: print("... done vectorizing and saving.\n")
    
    def embed_from_subs(self, subs_dir:str):
        '''
        This method is used to embed data from subreddits stored individually.

        Parameters:
            None
        
        Returns:
            None
        '''
        files = [f for f in os.listdir(subs_dir) if f[0]!="."]
        for f in files:
            prefix = f.split("_")[0]
            self.embed_data(prefix=prefix, verbose=True)
        
        return None

    def add_agree_disagree_distances(self):
        '''
        This method is used to add agree and disagree distances to the data.

            Parameters:
                None
            
            Returns:
                None
        '''
        ## Load the data
        files = [f for f in os.listdir(self.embedded_save_dir) if f[0]!="."]

        for f in files:
            df = pl.read_parquet(self.embedded_save_dir+f)
            df = agree_disagree_distances.add_agree_disagree_distances(df)
            os.remove(self.embedded_save_dir+f)
            df.write_parquet(self.embedded_save_dir+f)
        
        return None

    def copy_agree_disagree_distances(self, finished_dir:str):
        '''
        This method is used to copy agree and disagree distances from a file.

        Parameters:
            file: str
                The path to the file from which to copy the distances.
        
        Returns:
            pl.DataFrame
                The data with the copied distances.
        '''

        ## Load the data
        before_files = [f for f in os.listdir(self.embedded_save_dir) 
                        if f[0]!="."]
        for f in before_files:
            df_before = pl.read_parquet(self.embedded_save_dir+f)
            df_before = agree_disagree_distances.copy_agree_disagree_distances(
                file=finished_dir+f, df=df_before)
            os.remove(self.embedded_save_dir+f)
            df_before.write_parquet(self.embedded_save_dir+f)
        
        return None        
    
class PawIndex:
    '''
    This class is used to index data using a given configuration.

    Generally this will work with any set of files in the embedding directory, 
    but it is designed to work with the output of the PawEmbedding class.

    Parameters:
        EMBEDDING_CONFIG_NAME: str
            The name of the embedding configuration to be indexed.
        EMBEDDING_DIR: str
            The directory where the embedded data is saved.
        INDEX_CONFIG_NAME: str
            The name of the index configuration.
        DB_SAVE_DIR: str
            The directory where the database will be saved.
        METRIC: str
            The metric to be used for the index. Default is "cosine".
        NUM_PARTITIONS: int
            The number of Voronoi partitions to be used for the inverted file 
            index.
        NUM_SUB_VECTORS: int
            The number of sub-vectors to be used for the product quantization 
            index.
        ACCELERATOR: str
            The accelerator to be used for the index. 'mps' for metal, 
            otherwise None.
        
        Returns:
            None

        Methods:
            index_data(verbose: bool = False)
                This method is used to index the data.
    '''
    def __init__(
            self,
            EMBEDDING_CONFIG_NAME: str,
            EMBEDDING_DIR: str,
            INDEX_CONFIG_NAME: str,
            DB_SAVE_DIR: str,
            METRIC: str = "cosine",
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
        self.vector_files = [f for f in os.listdir(self.vector_dir) 
                             if f[0]!="."]
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

        ## Load the rest of the data files
        for f in self.vector_files[1:]:
            data = pl.read_parquet(self.vector_dir+f)
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
    '''
    This class is used to query data using a given configuration.

    Parameters:
        CONFIG_NAME: str
            The name of the configuration to be used.
        DB_DIR: str
            The directory where the database is saved.
        QUERY_SAVE_DIR: str
            The directory where the query results will be saved.
        QUERY_NAME: str
            The name of the query.
        METRIC: str
            The metric to be used for the query.
        LIMIT: int
            The number of results to be returned.
        NPROBES: int
            The number of probes to be used for the query.
        REFINE_FACTOR: int
            The refine factor to be used for the query.
        FILTER_SUBMISSIONS: bool
            Whether to filter submissions.
        FILTER_SHORT_QUESTIONS: bool
            Whether to filter short questions.
        RERANK_SENTIMENT: bool
            Whether to rerank by sentiment.
        RERANK_AGREE_DISTANCE: bool
            Whether to rerank by agree distance.
        RERANK_DISAGREE_DISTANCE: bool
            Whether to rerank by disagree distance.
        
        Returns:
            None

        Methods:
            ask_a_query(query: str)
                This method is used to ask a query.
            ask_standard_queries()
                This method is used to ask standard queries.
    '''
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
            LIMIT: int,
            NPROBES: int,
            REFINE_FACTOR: int,
            FILTER_SUBMISSIONS: bool = False,
            FILTER_SHORT_QUESTIONS: bool = False,
            RERANK_SENTIMENT: bool = False,
            RERANK_AGREE_DISTANCE: bool = False,
            RERANK_DISAGREE_DISTANCE: bool = False,
            METRIC: str = "cosine",
    ) -> None:
        
        self.config_name = CONFIG_NAME
        self.db_dir = DB_DIR+"db_"+self.config_name
        self.query_save_dir = QUERY_SAVE_DIR
        self.query_name = QUERY_NAME
        self.metric = METRIC
        self.limit = LIMIT
        self.nprobes = NPROBES
        self.refine_factor = REFINE_FACTOR
        self.filter_submissions = FILTER_SUBMISSIONS
        self.filter_short_questions = FILTER_SHORT_QUESTIONS
        self.rerank_sentiment = RERANK_SENTIMENT
        self.rerank_agree_distance = RERANK_AGREE_DISTANCE
        self.rerank_disagree_distance = RERANK_DISAGREE_DISTANCE
        ## TODO: Add reranking by sentiment then agree distance
        ## TODO: Add reranking by sentiment then disagree distance
        ## TODO: Add reranking by agree distance then sentiment
        ## TODO: Add reranking by disagree distance then sentiment

        self.db_table = "table_"+self.config_name
        self.query_file = self.query_save_dir\
            +"queries_"+self.config_name\
            +self.query_name+".parquet"

        self.db = lancedb.connect(self.db_dir)
        self.table = self.db.open_table(self.db_table)

        self.prefilter = None

        if self.filter_short_questions:
            if self.filter_submissions:
                self.prefilter = "(is_short_question = False) AND"+\
                    " (aware_post_type = 'comment')"
            self.prefilter = "(is_short_question = False)"
        elif self.filter_submissions:
            self.prefilter = "(aware_post_type = 'comment')"

    def ask_a_query(self, query: str):
        ## Embed the query
        query_embedding = PawQuery.embedding_model.embed_query(query)

        result = self.table.search(query_embedding)\
            .metric(self.metric)\
            .limit(self.limit)\
            .nprobes(self.nprobes)\
            .refine_factor(self.refine_factor)\
            .to_polars()
        
        ## Rerank the results
        if self.rerank_sentiment:
            result = reranking.rerank_by_sentiment(result)
        if self.rerank_agree_distance:
            result = reranking.rerank_by_agree_distance(result)
        if self.rerank_disagree_distance:
            result = reranking.rerank_by_disagree_distance(result)
        if not self.rerank_sentiment and not self.rerank_agree_distance\
            and not self.rerank_disagree_distance:
            result = result.sort(by="_distance")

        return result.clone()

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
        
        results.write_parquet(self.query_file)

        return results
    
class PawScores:
    '''
    This class is used to compute scores for the results of a query.

    Parameters:
        RESULTS_FILE_PATH: str
            The path to the file containing the results of a query.
        
        Returns:
            None

        Methods:
            compute_mext_rr_scores()
                This method is used to compute the MEXT-RR scores.
            compute_rr_scores()
                This method is used to compute the RR scores.
            compute_dcg_scores()
                This method is used to compute the DCG scores.
            get_mext_rr_scores()
                This method is used to get the MEXT-RR scores.
            get_rr_scores()
                This method is used to get the RR scores.
            get_dcg_scores()
                This method is used to get the DCG scores.
    '''
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

    def __init__(
            self,
            RESULTS_FILE_PATH: str,
            ) -> None:
        
        self.results_file_path = RESULTS_FILE_PATH

        ## Load the query results
        self.results = pl.read_parquet(self.results_file_path)

        ## Load the labels and votes for the 00 config
        self.labeled_00_df = process_labels\
            .get_merged_labels_and_votes(config="00")
        self.labeled_00_df = process_labels\
            .get_majority_vote(self.labeled_00_df)

        ## Load the labels and votes for the 02 config
        self.labeled_02_df = process_labels\
            .get_merged_labels_and_votes(config="02")
        self.labeled_02_df = process_labels\
            .get_majority_vote(self.labeled_02_df)

        ## Concatenate the two dataframes
        self.labeled_df = pl.concat([self.labeled_00_df, 
                                     self.labeled_02_df])

        ## Group the labeled dataframe by query_text
        self.grouped = self.labeled_df.group_by("query_text")

        ## Add a counts of relevant results, lists of relevant results
        self.relevant_results = self.grouped.agg(
            num_relevant = pl.col("relevance_rating")\
                    .filter(pl.col("relevance_rating")==1).len(),
            relevant_names = pl.col("reddit_name")\
                    .filter(pl.col("relevance_rating")==1)).clone()
        
        self.mext_rr_scores = {}
        self.rr_scores = {}
        self.dcg_scores = {}

    def compute_mext_rr_scores(self):
        for i in range(self.relevant_results.shape[0]):
            query_text = self.relevant_results[i]["query_text"][0]
            num_relevant = self.relevant_results[i]["num_relevant"][0]
            
            if query_text:
                query_results = self.results\
                    .filter(pl.col("query_text")==query_text)
                
                query_score = 0
                for j in range(query_results.shape[0]):
                    if query_results[j]["reddit_name"] in \
                        self.relevant_results[i]["relevant_names"][0]:

                        if j < num_relevant:
                            query_score += 1
                        else:
                            query_score += 1/(j-num_relevant+1)
                
                if num_relevant > 0:
                    self.mext_rr_scores[query_text] = query_score/num_relevant
                else:
                    self.mext_rr_scores[query_text] = 0
    
    def compute_rr_scores(self):
        for i in range(self.relevant_results.shape[0]):
            query_text = self.relevant_results[i]["query_text"][0]
            num_relevant = self.relevant_results[i]["num_relevant"][0]
            
            if query_text:
                query_results = self.results\
                    .filter(pl.col("query_text")==query_text)
                
                query_score = 0
                for j in range(query_results.shape[0]):
                    if query_results[j]["reddit_name"] in \
                        self.relevant_results[i]["relevant_names"][0]:

                        query_score = 1/(j+1)
                        break

                
                if num_relevant > 0:
                    self.rr_scores[query_text] = query_score
                else:
                    self.rr_scores[query_text] = 0

    def ndcg_calc(self, array):

        ndcg = np.zeros(len(array))
        ndcg_ideal = np.zeros(len(array))
    
        sort_ind = np.argsort(array)
        sorted_arr = np.take(array,sort_ind[::-1])
        rel = [2**array[i]-1 for i in range(len(array))]
        rel_ideal = [2**sorted_arr[i]-1 for i in range(len(sorted_arr))]
    
        for i in range(len(ndcg)):
            for j in range(len(array)):
                ndcg[i] = ndcg[i]+rel[j]/math.log2(j+2)

        for i in range(len(ndcg_ideal)):
            for j in range(len(sorted_arr)):
                ndcg_ideal[i] = ndcg_ideal[i]+rel_ideal[j]/math.log2(j+2)+1e-8

        # Add the 1e-8 to control division by 0 error. 
        # This results when all the entries are irrelevant.
    
        return np.mean(np.divide(ndcg,ndcg_ideal))
    
    def compute_dcg_scores(self):
        # Apply thresolding, chuck out everything from the 00 and 02 config 
        # results that are not relevant
        self.old_rating = [1,2,3]
        self.new_rating = [1,0,0]
        self.labeled_thrs_df = self.labeled_df.with_columns(
            pl.when(pl.col("relevance_rating") == self.old_rating[1])\
            .then(self.new_rating[1])\
            .when(pl.col("relevance_rating") == self.old_rating[2])\
            .then(self.new_rating[2])\
            .otherwise(pl.col("relevance_rating")).alias("relevance_rating")
)
        ## labeled_thrs_df is the thresolded df now, now we remove all the entries
        ## that have 0 relevance rating.
        self.labeled_thrs_df = self.labeled_thrs_df.filter(
            pl.col("relevance_rating") == 1)
    
        ## We would like to compare this "relevant" dataframe with our new query
        ## reply pair, all matching relevant replies get a score of 1, 
        ## everything else is set to 0

        for i in range(self.results.shape[0]):
            query_text = self.results[i]["query_text"][0]

            if query_text:
        
               # First, we filter out the dataframe based on the specific query, 
               ## both for new (self.results) and labelled, thresolded one 
               ## (self.labeled_thrs_df)
               self.results_query = self.results.filter(
                   pl.col("query_text") == query_text)
               
               self.df_lab = self.labeled_thrs_df.filter(
                   pl.col("query_text") == query_text)
               
               self.relevant_names = self.df_lab["reddit_name"].to_list()

               #compare if the replies from redditors with name "reddit_name", 
               ## in the new config is present in the list "self.relevant 
               ## names". If they do, we score it 1, otherwise we score it 0
    
               self.df_r = self.results_query.with_columns(
                    pl.col("reddit_name").is_in(self.relevant_names)\
                    .alias("is_relevant"))
               
               # is_relevant is now a boolean columnn, consisting of True and 
               ## False. For numerically socring it, we substitute True = 1, 
               ## False = 0
               self.df_r = self.df_r.with_columns(
                    pl.when(pl.col("is_relevant") == True)\
                    .then(1)\
                    .when(pl.col("is_relevant") == False)\
                    .then(0)\
                    .otherwise(pl.col("is_relevant")).alias("relevance_score"))
               
               self.rating = self.df_r["relevance_score"].to_numpy()
               
               self.dcg_scores[query_text] = self.ndcg_calc(self.rating)

    def get_mext_rr_scores(self):
        return self.mext_rr_scores
    
    def get_rr_scores(self):
        return self.rr_scores
    
    def get_dcg_scores(self):
        return self.dcg_scores
        
