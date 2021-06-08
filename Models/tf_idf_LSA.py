from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# get data
df = pd.read_parquet('../data/clean/all_journals_adject_nouns_merged_withLemmatized.gzip')
corpus_body_text = df['body_lemmatized'].to_list()

# create train and test set, save which ones are train and test
random_indeces = list(np.random.permutation(len(corpus_body_text)))
train_size = int(round(len(corpus_body_text) * 0.7,0))
train_indeces = random_indeces[:train_size]
test_indeces = random_indeces[train_size:]
train_bool =  [1 if i in train_indeces else 0 for i in range(0,max(random_indeces)+1)]
train_texts = [corpus_body_text[i] for i in train_indeces]
test_texts = [corpus_body_text[j] for j in test_indeces]

# get tf-idf
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(corpus_body_text)
feature_names = vectorizer.get_feature_names()
len(feature_names)



# reduce dimensions
n_dim = 10
svd = TruncatedSVD(n_components=n_dim,
                   algorithm = 'randomized',
                   random_state=123)
result_LSA = svd.fit(vectors)

result_LSA.explained_variance_

explained_var = np.sort(result_LSA.explained_variance_ratio_)[::-1]

# show the 'elbow plot'
plt.plot(range(1,n_dim+1),explained_var)

# per document show LSA representation
LSA_representation = svd.transform(vectors)
df_LSA_representation = pd.DataFrame(LSA_representation)
df_LSA_selected = df_LSA_representation[[1,2,3,4]]

df_LSA_selected.columns = ['PC1', 'PC2', 'PC3', 'PC4']
df_LSA_selected['citations'] = df['citations']
df_LSA_selected['altmetric_score'] = df['altmetric_score']


df_LSA_selected.to_parquet('../data/clean_all_journals_LSA_representation.gzip', compression='gzip')