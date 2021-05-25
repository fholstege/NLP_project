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
import numpy as np
from helpfunctions_models import check_n_topic_scores_CV

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
K = 4
range_n_topics = range(2, 31)
result_cv = check_n_topic_scores_CV(corpus, range_n_topics, id2word, K, coherence_measure = 'u_mass')



# original LDA model
lda_model = ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=3, minimum_probability=0)
lda_model.print_topics()
lda_corpus = lda_model[corpus]
lda_model.get_topics()

def edit_tuple(x):
    return x[1]
df_topics = pd.DataFrame(lda_corpus).applymap(edit_tuple)
df_topics.columns = ['Topic 1', 'Topic 2', 'Topic 3']


def get_max(doc):
        idx,l = zip(*doc)
        return idx[np.argmax(l)]

data['doc_topic'] = [get_max(doc) for doc in model.get_document_topics(model_corpus)]


