# load the necessary packages
from gensim.models import ldaseqmodel
from gensim.corpora import Dictionary, bleicorpus
from gensim.test.utils import datapath
import numpy
from gensim.matutils import hellinger
import pandas as pd
import nltk
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer 
import logging
from gensim.models import LdaSeqModel
import time
import numpy as np

# show logging when running the dynamic LDA
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# read in the limited dataset 
df_limited = pd.read_parquet('limited_dataset_sequential_lda.gzip')


# determine time slices; these follow the following structure: [n papers year 2000, n papers year 2001, ...,]
df_limited = df_limited.sort_values('year')
time_slice = df_limited.groupby('year').count()['title'].to_list()

# create list of words for each document
texts_list = [[word for word in document.split()] for document in df_limited['body_lemmatized'].str.lower().tolist()]
    
## construct word <-> id mappings used by the LDA model
# start with dictionary of all words
dictionary = Dictionary(texts_list) 
    
# then, filter words that occur in more than 50% of the provided documents
# keep_n = keep n most frequent tokens
dictionary.filter_extremes(no_above=0.5, keep_n=100000)  
dictionary.compactify()

# convert them into bag of words format, to be used in dynamic lda model
corpus = [dictionary.doc2bow(text) for text in texts_list]
_ = dictionary[0] # need this line to force-load the data in the kernel
id2word = dictionary.id2token # to convert ids to words, adds interpretation


# try for 10 topics, start time to count
n_topics = 10
start = time.time()

# dynamic LDA model, print topics after done
lda_seq_model = LdaSeqModel(corpus=corpus,id2word=id2word, time_slice=time_slice, num_topics=n_topics, chunksize=1) #iterations=1000, passes=100, chunksize=2400, minimum_probability=0.0)
# default values:
# passes = 10, we would want a much larger number of passes on the final model (25+)
# em_min_iter = 6
lda_seq_model.print_topics()

# save how much time has passed
end = time.time()
time_consumed=end-start;
time_consumed