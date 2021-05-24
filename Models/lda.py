from gensim.models import ldaseqmodel
from gensim.corpora import Dictionary, bleicorpus
from gensim.test.utils import datapath
import numpy
from gensim.matutils import hellinger
import pandas as pd
import nltk
# nltk.download('wordnet') # uncomment if wordnet not installed
from nltk.stem import WordNetLemmatizer 
import logging
from gensim.models import ldamodel

from helpfunctions_models import get_average_perplexity_from_LDA_CV

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

file_list = ['journalofmarketing',
             'journalofmarketingresearch',
             'journalofconsumerresearch',
             'journalofconsumerpsych',
             'journalacademyofmarketingscience']

# stack data from all journals
df = pd.read_parquet('../data/clean/' + file_list[0] + '_merged.gzip')
for f in file_list[1:len(file_list)]:
    df = pd.concat([df, pd.read_parquet('../data/clean/' + f + '_merged.gzip')]) 
df = df.reset_index()

# check articles per journal
df['Source title'].value_counts()

# determine time slices
df = df.sort_values('Year')

# remove 2 missings in year
df = df.dropna(subset = ['Year'])
df['Year'].value_counts()
time_slice = df.groupby('Year').count()['title'].to_list()

# init the Wordnet Lemmatizer
lemmatizer = WordNetLemmatizer()

# create list of words for each document
texts_list = [[lemmatizer.lemmatize(word) for word in document.split()] for document in df['body'].tolist()]
    
# construct word <-> id mappings used by the LDA model
dictionary = Dictionary(texts_list) 
    
# filter words that occur in less than 10% the documents (follow dynamic model paper)
# filter words that occur in more than 50% of the provided documents
# keep_n = keep n most frequent tokens
dictionary.filter_extremes(no_below=250, no_above=0.5, keep_n=100000)  
dictionary.compactify()
# convert into bag of words format
corpus = [dictionary.doc2bow(text) for text in texts_list]
_ = dictionary[0] # need this line to force-load the data in the kernel
id2word = dictionary.id2token
len(id2word)

# # save as blei corpus (lda-c format)
# bleicorpus.BleiCorpus.serialize('../data/clean/lda_corpus.lda-c', corpus)

# # save time_slices
# textfile = open("time_slices.txt", "w")
# for element in time_slice:
#     textfile.write(str(element) + "\n")
# textfile.close()


# estimate model
#lda_model = ldaseqmodel.LdaSeqModel(corpus=corpus, time_slice=time_slice, id2word = id2word, num_topics=2, chunksize=100, passes = 1)

##################################################################

# save model
#temp_file = datapath("seq_model")
#lda_model.save(temp_file)

# load model
#lda = LdaModel.load(temp_file)


# test to see how dynamic LDA works
#from gensim.test.utils import common_corpus
#lda_model = ldaseqmodel.LdaSeqModel(corpus=common_corpus, time_slice=[2, 4, 3], num_topics=2, chunksize=1)


# implement cross validation
result_per_n_topics = []
K = 4

for n_topics in range(2, 20+1):
    
    result = get_average_perplexity_from_LDA_CV(corpus, n_topics, id2word, K)
    result_per_n_topics.append(result)


# original LDA model
lda_model = ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=20)

lda_model.print_topics()

lda_model.get_topic_terms(1)

lda_model.show_topics()

lda_model.log_perplexity(corpus)

# import pyLDAvis
# import pyLDAvis.gensim_models as gensimvis
# import matplotlib.pyplot as plt

# pyLDAvis.disable_notebook()
# vis = gensimvis.prepare(test, corpus, id2word)
# vis.show()

# TODO
# do CV on normal LDA
# use gensim coherence model to evaluate fit
# prepare corpus for each fold separately?
# visualisation
# show ranking of words in topics over time
# find most typical paper for each year and topic?