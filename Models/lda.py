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
from gensim.models import ldamodel
import numpy as np
from helpfunctions_models import check_n_topic_scores_CV

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# stack data from all journals
df = pd.read_parquet('../Data/clean/data_journals_adject_nouns_merged.gzip')
df = df.reset_index()


# check articles per journal
df['Source title'].value_counts()

# determine time slices
df = df.sort_values('Year')

# remove 2 missings in year
df = df.dropna(subset = ['Year'])
df['Year'].value_counts()
time_slice = df.groupby('Year').count()['title'].to_list()

# create list of words for each document
texts_list = [[word for word in document.split()] for document in df['body_lemmatized'].tolist()]
    
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


# implement cross validation
K = 4
range_n_topics = range(2, 5)
result_cv = check_n_topic_scores_CV(corpus, range_n_topics, id2word, K, coherence_measure = 'u_mass')



n_topics = 3
# original LDA model
lda_model = ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=n_topics, minimum_probability=0)
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

doc_topics = [get_max(doc) for doc in lda_model.get_document_topics(corpus)]
df_topics['Topic'] = doc_topics
df_topics['Topic'] = df_topics['Topic']  + 1

df_combined_info = pd.concat([df,df_topics], axis=1)
df_combined_info.to_parquet("../Data/clean/data_journals_merged_topics.gzip", compression='gzip')

df_topics_journals = df_combined_info[['Source title', 'Topic 1', 'Topic 2', 'Topic 3']].groupby('Source title').mean()

X = np.arange(5)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, df_topics_journals['Topic 1'].to_list(), color = 'b', width = 0.25, label = '1')
ax.bar(X + 0.25, df_topics_journals['Topic 2'].to_list(), color = 'g', width = 0.25, label = '2')
ax.bar(X + 0.50, df_topics_journals['Topic 3'].to_list(), color = 'r', width = 0.25, label = '3')
ax.set_xticklabels(df_topics_journals.index)
ax.tick_params(axis='x', rotation=45)
plt.legend(loc = 'upper right')


### plots over time
df_topics_over_time = df_combined_info[['Year', 'Topic 1', 'Topic 2', 'Topic 3']].groupby('Year').mean()

plt.plot(df_topics_over_time.index, df_topics_over_time['Topic 1'], label = 'Topic 1', color = 'red')
plt.plot(df_topics_over_time.index, df_topics_over_time['Topic 2'], label = 'Topic 2', color = 'blue')
plt.plot(df_topics_over_time.index, df_topics_over_time['Topic 3'], label = 'Topic 3', color = 'green')
plt.xlabel("Year Of Publication")
plt.ylabel("Average Probability of Belonging to a Topic")
plt.legend(loc="lower right")



######## old code



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

