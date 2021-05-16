import pandas as pd
from helpfunctions_preprocessing import prepare_scopus

# turn scimago data into dataframe to extract fields
field_dict = {'Marketing' : 'data/raw/marketing_journals.csv', 
          'AI' :  'data/raw/ai_journals.csv', 
          'Statistics' : 'data/raw/statistics_journals.csv', 
          'Sociology / Pol.Sc.' : 'data/raw/socio_political_journals.csv', 
          'Mgmt. Science / OR' : 'data/raw/mgmtscience_operationsres_journals.csv', 
          'Economics / Econometrics' : 'data/raw/econ_econometrics_journals.csv',
          'Psychology' : 'data/raw/psych_journals.csv',
          'Medicine' : 'data/raw/medicine_journals.csv',
          'Computer Science' : 'data/raw/cs_journals.csv',
          'Behavioral Neuroscience' : 'data/raw/behav_neuro_journals.csv',
          'Org. Behavior / HR Management' : 'data/raw/hr_org_behav_journals.csv',
          'Mgmt of Tech. and Innovation' : 'data/raw/mgmt_tech_inno_journals.csv',
          'Business / Int. Management' : 'data/raw/business_int_mgmt_journals.csv',
          'Mgmt Information Systems' : 'data/raw/mgmt_info_systems_journals.csv',
          'Strategy and Management' : 'data/raw/strategy_mgmt_journals.csv',
          'Social Sciences' : 'data/raw/social_sciences_journals.csv'}

# initialise dataframe
all_journals = pd.DataFrame(columns = ['Rank', 'Title', 'Field', 'ID'])

for field, path in field_dict.items():
    # input data
    journals = pd.read_csv(path, sep = ";")
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

# get duplicates to remove from data
df_dupl = df_dupl[df_dupl['Title'].duplicated(keep = 'first')]
    
# remove duplicate titles from other fields
cond = ~all_journals['ID'].isin(df_dupl['ID'])
all_journals = all_journals[cond]

# clean to merge onto corpus
field_data = all_journals[['Title', 'Field']]
field_data.columns = ['journal', 'field']


prepare_scopus('data/raw/journalofmarketing_WoS.csv', field_data, field_dict)


#TODO: merge it with the clean text 

    
# # merge scopus data onto scraped data
# result = pd.merge(scraped_data, scopus_data, how = 'left', on = 'DOI', sort = False)
    
# # remove redundant unnamed column (old index)
# result.drop(labels = ['Unnamed: 0'], axis = 1, inplace = True)

# # save data
# result.to_parquet('data/cleaned/' + output_name + '.gzip', compression='gzip')
    
# merge_data('data/scraped/journalofmarketing.gzip', 'data/raw/journalofmarketing_WoS.csv', 'jom_merged')