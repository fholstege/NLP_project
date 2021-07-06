import pandas as pd

# load csv after manually downloading of papers ('filled')
df = pd.read_csv('../data/raw/marketingscience_urls_filled.csv', sep=';')

# load xml data for each article
for i, row in df.iterrows():
    file_name = '../data/raw/marketing_science_xml/' + str(i+1) + '.tei.xml'
    with open(file_name, 'r', encoding="utf8") as f:
            df.loc[i, 'body'] = f.read()

# keep only articles (Type == NaN)
df['Type'].value_counts()
df = df.loc[df['Type'].isna(), 'DOI']
df.reset_index(drop = True, inplace = True)

# merge with scopus data








df = pd.read_parquet('../data/clean/all_journals_BERT_merged.gzip')

df = df[['DOI', 'year', 'title', 'abstract', 'journal', 'keywords']]

df.reset_index(drop = True, inplace = True)

df.to_parquet('../data/clean/abstracts.gzip', compression='gzip')