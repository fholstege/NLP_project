import multiprocessing
from gensim.models import Word2Vec
from nltk import tokenize
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib as plt
import numpy as np


df = pd.read_parquet('../Data/clean/data_journals_merged.gzip')
body_texts = df['body'].tolist()


texts = [row.split() for row in df['body']]

random_indeces = list(np.random.permutation(len(texts)))
train_size = int(round(len(texts) * 0.7,0))

train_indeces = random_indeces[:train_size]
test_indeces = random_indeces[train_size:]

train_texts = [texts[i] for i in train_indeces]
test_texts = [texts[j] for j in test_indeces]

cores = multiprocessing.cpu_count()

window = 2
w2v_model = Word2Vec(train_texts,
                     window=window,
                     size=100,
                     alpha = 0.001,
                     workers=cores-1)




# summarize model
# summarize vocabulary
vocab = list(w2v_model.wv.vocab)

with open("test_data_window.txt", 'w', encoding="utf-8") as f:
    
    f.write(': articles \n')
    
    for text_list in test_texts:
        four_word_combination = []

        for word in text_list:

            four_word_combination.append(word)
            
            if len(four_word_combination) == 2* window:
                line = " ".join(four_word_combination)
                four_word_combination = []
                f.write(line + '\n')

                

accuracy_test_set = w2v_model.wv.accuracy('test_data_window.txt')
n_correct = len(accuracy_test_set[1]['correct'])

n_incorrect = len(accuracy_test_set[1]['incorrect'])

n_correct / (n_correct + n_incorrect)






# save model
w2v_model.save('model.bin')
# load model
new_model = Word2Vec.load('model.bin')
print(new_model)