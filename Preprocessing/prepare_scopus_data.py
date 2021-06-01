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
              'Mgmt. Information Systems ' : 'data/raw/fields/mgmt_info_systems.csv',
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



# loop over journals and construct merge dataframes
file_list = ['journalofmarketing',
             'journalofmarketingresearch',
             'journalofconsumerresearch',
             'journalofconsumerpsych',
             'journalacademyofmarketingscience']

for f in file_list:
    # load text data
    text_df = pd.read_parquet('../data/clean/' + f + '_words.gzip')
    
    # prepare relevant scopus data
    scopus_df = prepare_scopus('../data/raw/' + f + '_WoS.csv', field_data, field_dict)
    
    # for first two journal also drop abbreviated source title
    if f in ['journalofmarketing', 'journalofmarketingresearch']:
        scopus_df.drop(labels = ['Abbreviated Source Title'], axis = 1, inplace = True)
    
    # merge scopus and text data
    intermediate = pd.merge(text_df, scopus_df, how = 'left', on = 'DOI', sort = False)
    
    # merge data and altmetric data
    result =  pd.merge(intermediate, alt_df, how = 'left', on = 'DOI', sort = False)
    
    # check how many could not be merged
    # sum(result['refs_found'].isna())
    
    # check for missings in body of text
    result = result.loc[result['body'] != '']
    result = result.loc[result['body'] != 'na']
    
    # save result
    result.to_parquet('../data/clean/' + f + '_merged.gzip', compression = 'gzip')
    print(f'Merged data saved for {f}')