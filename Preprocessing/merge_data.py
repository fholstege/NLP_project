import pandas as pd
from helpfunctions_preprocessing import prepare_scopus

# turn scimago data into dataframe to extract fields
field_dict = {'Arts and Humanities' : 'data/raw/fields/arts_humanities.csv', 
              'Computer Science' : 'data/raw/fields/computer_science.csv', 
              'Economics / Econometrics / Finance' : 'data/raw/fields/econ_finance.csv', 
              'Social Sciences' : 'data/raw/fields/social_sciences.csv', 
              'Environmental Science' : 'data/raw/fields/environ_science.csv', 
              'Marketing' : 'data/raw/fields/marketing.csv', 
              'Neuroscience' : 'data/raw/fields/neuro.csv', 
              'Psychology' : 'data/raw/fields/psych.csv', 
              'Medicine' : 'data/raw/fields/medicine.csv', 
              'Mathematics' : 'data/raw/fields/mathematics.csv',
              'Accounting' : 'data/raw/fields/accounting.csv',
              'Mgmt. Information Systems' : 'data/raw/fields/mgmt_info_systems.csv',
              'Mgmt. Technology and Innovation' : 'data/raw/fields/mgmt_tech_inno.csv',
              'Mgmt. Science and OR' : 'data/raw/fields/mgmt_science_or.csv',
              'Org. Behavior and HR' : 'data/raw/fields/org_behav_hr.csv',
              'Strategy and Mgmt.' : 'data/raw/fields/strat_mgmt.csv',
              'Agricultural and Biological Sciences' : 'data/raw/fields/agricultural_biological.csv',
              'Multidisciplinary' : 'data/raw/fields/multidisciplinary.csv',
              'Business and Int. Mgmt.' : 'data/raw/fields/business_int_mgmt.csv',
              'Tourism, Leisure, Hospitality Mgmt.' : 'data/raw/fields/tourism_hospitality.csv'
              }

# initialise dataframe
all_journals = pd.DataFrame(columns = ['Rank', 'Title', 'Field', 'ID'])

for field, path in field_dict.items():
    # input data
    journals = pd.read_csv('../' + path, sep = ";")
    # clean frame
    journals = journals[['Rank', 'Title']]
    # create field info
    journals['Field'] = field
    # merge onto dataframe
    all_journals = all_journals.append(journals)
    
# create unique identifier
all_journals = all_journals.reset_index()
all_journals['ID'] = all_journals.index
    
# get entries that appear more than once
df_dupl = all_journals[all_journals['Title'].duplicated(keep = False)]

# sort by rank
df_dupl = df_dupl.sort_values(by = 'Rank')

# check which journals are rank the same for more than one field
test =  all_journals[all_journals[['Title', 'Rank']].duplicated(keep = False)]
test.sort_values(by = ['Title', 'Rank'])

# get duplicates to remove from data
df_dupl = df_dupl[df_dupl['Title'].duplicated(keep = 'first')]
    
# remove duplicate titles from other fields
cond = ~all_journals['ID'].isin(df_dupl['ID'])
all_journals = all_journals[cond]

# clean to merge onto corpus
field_data = all_journals[['Title', 'Field']]
field_data.columns = ['journal', 'field']

# read altmetric data and clean
alt_df = pd.read_csv('../data/raw/altmetric.csv')
alt_df['Journal/Collection Title'].value_counts()
cols = ['DOI', 'Altmetric Attention Score', 'News mentions', 'Blog mentions', 'Policy mentions', 'Twitter mentions', 'Facebook mentions', 'Reddit mentions', 'Wikipedia mentions', 'Number of Mendeley readers', 'Number of Dimensions citations']
alt_df = alt_df[cols]

# prepare scopus data

# list of journals 
journals = ['journalofmarketing',
            'journalofmarketingresearch',
            'journalofconsumerresearch',
            'journalofconsumerpsych',
            'journalacademyofmarketingscience']

# init dataframe
scopus_df = prepare_scopus('../data/raw/' + journals[0] + '_WoS.csv', field_data, field_dict)
if journals[0] in ['journalofmarketing', 'journalofmarketingresearch']:
    scopus_df.drop(labels = ['Abbreviated Source Title'], axis = 1, inplace = True)

# loop over remaining journals and concatenate dataframe
for journal in journals[1:len(journals)]:
    load_df = prepare_scopus('../data/raw/' + journal + '_WoS.csv', field_data, field_dict)

    # for first two journal also drop abbreviated source title
    if journal in ['journalofmarketing', 'journalofmarketingresearch']:
        load_df.drop(labels = ['Abbreviated Source Title'], axis = 1, inplace = True)
    
    # concatenate scopus data
    scopus_df = pd.concat([scopus_df, load_df]) 


# MERGE DATA

# loop over data types
data_types = ['_BERT', '_allWords', '_adject_nouns']

for data_type in data_types:

    # load text data
    text_df = pd.read_parquet('../data/clean/all_journals' + data_type + '.gzip')
    
    # merge scopus and text data
    intermediate = pd.merge(text_df, scopus_df, how = 'left', on = 'DOI', sort = False)

    # merge data and altmetric data
    result =  pd.merge(intermediate, alt_df, how = 'left', on = 'DOI', sort = False)
    
    # remove those with missings in scopus variables, should be none
    result.dropna(subset=['DOI', 'Year'], inplace = True)
    
    # select columns of interest
    select_cols = ['DOI', 'Year', 'title', 'title_lemmatized', 'abstract', 'abstract_lemmatized' ,'body', 'body_lemmatized', 'Source title', 'Cited by', 'Author Keywords', 'num_ref'] + list(result.columns[18:len(result.columns)])
    result = result[select_cols]
    
    result.rename(columns={"Year": "year", "Source title": "journal", "Cited by": "citations", 
                           "Author Keywords": "keywords", "Altmetric Attention Score": "altmetric_score", 
                           "News mentions": "news_mentions", "Blog mentions": "blog_mentions",
                           "Policy mentions": "policy_mentions", "Twitter mentions": "twitter_mentions",
                           "Facebook mentions": "fb_mentions", "Reddit mentions": "reddit_mentions",
                           "Wikipedia mentions": "wiki_mentions", "Number of Mendeley readers": "mendeley_readers", "Number of Dimensions citations": "dimensions_citations"},
                  inplace = True)
    
    if data_type == '_BERT':
        result.drop(labels = ['body_lemmatized', 'abstract_lemmatized', 'title_lemmatized'], 
                     axis = 1, inplace = True)

    # make year and citations an integer
    result['year'] = result['year'].astype(int)
    result['citations'] = result['citations'].astype(int)
    
    # remove potential duplicated DOIs (6 duplicates)
    result = result.sort_values(by = 'year', ascending = False)
    result = result.loc[~result['DOI'].duplicated()]
    
    # save result
    result.to_parquet('../data/clean/all_journals' + data_type + '_merged.gzip', compression = 'gzip')
    print(f'Merged data saved for data type {data_type}')