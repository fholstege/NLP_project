import pandas as pd
from transformers import *
import nltk
# nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import torch
from tqdm import tqdm

# load data
df = pd.read_parquet('../data/clean/all_journals_BERT_merged.gzip')
df = df[['DOI', 'citations', 'altmetric_score', 'body']]

# load pre-trained model
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

# split body into sentences
df['sentences'] = df['body'].apply(sent_tokenize)

# for each document batch sentences together
N = 8
df['grouped_sentences'] = df['sentences'].apply(lambda x: [' '.join(x[n:n+N]) for n in range(0, len(x), N)])

# make each 8 sentences as separate row
df_expl = df.explode('grouped_sentences')

# init corpus result dataframe to save embeddings + outcomes + DOI
corpus_results = pd.DataFrame()

# loop over each document
for doi in tqdm(list(df['DOI'])):

    # get text (sentences in groups of 8)
    input_text = df_expl.loc[df_expl['DOI'] == doi, 'grouped_sentences'].to_list()
    
    # init embeddings vector to collect embeddings
    embeds = []

    for sentence_group in input_text:
        tokenized_text = tokenizer(sentence_group, padding='max_length', 
                                   truncation = True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            output = model(input_ids=tokenized_text['input_ids'], 
                           attention_mask=tokenized_text['attention_mask'])
        embeds.append(output['pooler_output'])
    
    # concatenate embeddings along dim 0
    embed_torch = torch.cat(embeds)
    
    # mean pooling for sentence groups of one document (along dim 0)
    mean_values =  list(torch.mean(embed_torch, 0).numpy())
    
    # max pooled
    max_values, max_indices =  torch.max(embed_torch, 0)
    max_values =  list(max_values.numpy())
    
    # min pooled
    min_values, min_indices =  torch.min(embed_torch, 0)
    min_values =  list(min_values.numpy())
    
    # create row corresponding to document infos
    doc_results = pd.DataFrame({'DOI': doi, 'mean_BERT': [mean_values], 'max_BERT': [max_values],
                                'min_BERT': [min_values]})
    doc_results['citations'] = int(df.loc[df['DOI']==doi, 'citations'])
    doc_results['altmetric_score'] = df.loc[df['DOI']==doi, 'altmetric_score']
    
    # append to corpus results
    corpus_results = corpus_results.append(doc_results, ignore_index=True)
    
corpus_results.shape

# save results
corpus_results.to_parquet('../data/representations/BERT_all_representations.gzip', compression='gzip')




# get embeddings from different layers and compare predictive performance



# bigbird

# scibert tuned

# compare what is more important to attend to: length of docs or domain of docs