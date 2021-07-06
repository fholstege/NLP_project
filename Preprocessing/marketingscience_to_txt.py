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
df = df.loc[df['Type'].isna(), ['DOI', 'body']]
df.reset_index(drop = True, inplace = True)

# convert to DOI without URL prefix
base_url_informs = 'https://pubsonline-informs-org.eur.idm.oclc.org/doi/pdf/'
df['DOI'] = df['DOI'].str.replace(base_url_informs, '')

# merge with scopus data
scopus_df = pd.read_csv('../data/raw/marketingscience_WoS.csv')
scopus_df = pd.merge(scopus_df, df, how = 'left', on = 'DOI', sort = False)

# how many missing bodies?
print(f"There are {sum(scopus_df['body'].isna())} missing bodies")

# save scraped data
scopus_df.to_parquet("../data/scraped/marketingscience_data_lim.gzip", compression='gzip')








df = pd.read_parquet('../data/clean/all_journals_BERT_merged.gzip')

df = df[['DOI', 'year', 'title', 'abstract', 'journal', 'keywords']]

df.reset_index(drop = True, inplace = True)

df.to_parquet('../data/clean/abstracts.gzip', compression='gzip')