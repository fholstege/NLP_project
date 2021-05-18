
from collections import Counter
from nltk import word_tokenize
from nltk.util import ngrams
import pandas as pd
from helpfunctions_summary_stats import take

# read the zipped files
df_jom = pd.read_parquet('../Data/clean/journalofmarketing_words.gzip')
df_jomr = pd.read_parquet('../Data/clean/journalofmarketingresearch_words.gzip')
df_jcr = pd.read_parquet('../Data/clean/journalofconsumerresearch_words.gzip')
df_jcp = pd.read_parquet('../Data/clean/journalofconsumerpsych_words.gzip')
df_jam = pd.read_parquet('../Data/clean/journalacademyofmarketingscience_words.gzip')



body_text = ' '.join(df_jam['body'])
body_text_tokenized = word_tokenize(body_text)
n_gram = ngrams(body_text_tokenized,1)


word_counter = Counter(n_gram)
top_words = word_counter.most_common(30)
top_words


