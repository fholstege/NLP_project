# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 14:35:19 2021

@author: flori
"""

from gensim.models import ldaseqmodel
from gensim.corpora import Dictionary, bleicorpus
from gensim.test.utils import datapath
import numpy
from gensim.matutils import hellinger
import pandas as pd
import nltk
import matplotlib.pyplot as plt
# nltk.download('wordnet') # uncomment if wordnet not installed
from nltk.stem import WordNetLemmatizer 
import logging
from gensim.models import LdaSeqModel
import time

import numpy as np
from helpfunctions_models import check_n_topic_scores_CV

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# stack data from all journals
df = pd.read_parquet('../Data/clean/all_journals_adject_nouns_merged.gzip')
df = df.reset_index()

# remove 2 missings in year
df = df.dropna(subset = ['year'])

# get limited df for illustrative purposes ; 5 papers
df_limited = df.sample(25, random_state = 1)

# determine time slices
df_limited = df_limited.sort_values('year')
time_slice = df_limited.groupby('year').count()['title'].to_list()



# create list of words for each document
texts_list = [[word for word in document.split()] for document in df_limited['body_lemmatized'].str.lower().tolist()]
    
# construct word <-> id mappings used by the LDA model
dictionary = Dictionary(texts_list) 
    
# filter words that occur in more than 50% of the provided documents
# keep_n = keep n most frequent tokens
dictionary.filter_extremes(no_above=0.5, keep_n=100000)  
dictionary.compactify()
# convert into bag of words format
corpus = [dictionary.doc2bow(text) for text in texts_list]
_ = dictionary[0] # need this line to force-load the data in the kernel
id2word = dictionary.id2token # to convert ids to words, adds interpretation


# try for 10 topics, start time to count
n_topics = 10
start = time.time()

# sequential LDA model, print topics after done
lda_seq_model = LdaSeqModel(corpus=corpus,id2word=id2word, time_slice=time_slice, num_topics=n_topics, chunksize=1) #iterations=1000, passes=100, chunksize=2400, minimum_probability=0.0)
lda_seq_model.print_topics()

# save how much time has passed
end = time.time()
time_consumed=end-start;


# get the outputs per document
outputs_seq_model = []
for i in range(0,len(corpus)):
    
    output_seq_model = lda_seq_model[corpus[i]]
    outputs_seq_model.append(output_seq_model)
    
df_topic_allocation = pd.DataFrame((outputs_seq_model))
df_topic_allocation.columns = ['topic_' + str(i) for i in range(1,n_topics+1)]
