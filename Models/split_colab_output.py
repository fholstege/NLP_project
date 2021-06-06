import pandas as pd

# read altmetric data and clean
alt_df = pd.read_csv('../data/raw/altmetric.csv')
alt_df['Journal/Collection Title'].value_counts()
cols = ['DOI', 'Altmetric Attention Score']
alt_df = alt_df[cols]

# list of models
model_list = ['BERT', 'SciBERT', 'ROBERTA']

for model in model_list:
    # load embeddings data downloaded from google colab
    df = pd.read_parquet('../data/representations/' + model + '_all_embeddings.gzip')

    # merge embeddings with altmetric scores
    df = pd.merge(df, alt_df, how = 'left', on = 'DOI', sort = False)

    # clean up
    df.drop(labels = ['altmetric_score'], axis = 1, inplace = True)
    df.rename(columns={'Altmetric Attention Score': 'altmetric_score'}, inplace=True)

    # define sets of variables
    cls_embed = ['min_CLS', 'max_CLS', 'mean_CLS']
    last_embed = [x for x in df.columns if x.startswith('last')]
    secondlast_embed = [x for x in df.columns if x.startswith('second')]
    thirdlast_embed = [x for x in df.columns if x.startswith('third')]
    outcomes = ['DOI', 'citations', 'altmetric_score']

    # split into representations and save
    selection = outcomes + cls_embed
    df[selection].to_parquet('../data/representations/' + model + '_cls_embeddings.gzip', compression='gzip')

    selection = outcomes + last_embed
    df[selection].to_parquet('../data/representations/' + model + '_last_embeddings.gzip', compression='gzip')

    selection = outcomes + secondlast_embed
    df[selection].to_parquet('../data/representations/' + model + '_secondlast_embeddings.gzip', compression='gzip')

    selection = outcomes + thirdlast_embed
    df[selection].to_parquet('../data/representations/' + model + '_thirdlast_embeddings.gzip', compression='gzip')
    
    print(f'Splitting completed for {model} model')

