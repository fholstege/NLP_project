from gensim.models import ldaseqmodel
from gensim.corpora import Dictionary
import numpy
from gensim.matutils import hellinger
import pandas as pd
import nltk
# nltk.download('wordnet') # uncomment if wordnet not installed
from nltk.stem import WordNetLemmatizer 



file_list = ['journalofmarketing',
             'journalofmarketingresearch',
             'journalofconsumerresearch',
             'journalofconsumerpsych',
             'journalacademyofmarketingscience']

# stack data from all journals
df = pd.read_parquet('../data/clean/' + file_list[0] + '_merged.gzip')
for f in file_list[1:len(file_list)]:
    df = pd.concat([df, pd.read_parquet('../data/clean/' + f + '_merged.gzip')]) 
    
# check articles per journal
df['Source title'].value_counts()

# determine time slices
df = df.sort_values('Year')
df['Year'].value_counts()
time_slice = df.groupby('Year').count()['title'].to_list()

# init the Wordnet Lemmatizer
lemmatizer = WordNetLemmatizer()

# create list of words for each document
texts_list = [[lemmatizer.lemmatize(word) for word in document.split()] for document in df['body'].tolist()]
    
# construct word <-> id mappings used by the LDA model
dictionary = Dictionary(texts_list) 
    
# filter words that occur in less than 10% the documents
# filter words that occur in more than 50% of the provided documents
# keep_n = keep n most frequent tokens
dictionary.filter_extremes(no_below=round(df.shape[0]*0.1), no_above=0.5, keep_n=None)  

# convert into bag of words format
corpus = [dictionary.doc2bow(text) for text in texts_list]
_ = dictionary[0] # need this line to force-load the data in the kernel
id2word = dictionary.id2token

lda_model = ldaseqmodel.LdaSeqModel(corpus=corpus, id2word=id2word, time_slice=time_slice, num_topics=10)


from gensim.models import ldamodel
test = ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=8)

test.print_topics()

test.get_topic_terms(1)

test.show_topics()

# import pyLDAvis
# import pyLDAvis.gensim_models as gensimvis
# import matplotlib.pyplot as plt

# pyLDAvis.disable_notebook()
# vis = gensimvis.prepare(test, corpus, id2word)
# vis.show()

# take evaluation set first (30%)

# do CV on remaining 70%

# use gensim coherence model to evaluate fit
    
# prepare corpus for each fold separately?

# visualisation
# show ranking of words in topics over time