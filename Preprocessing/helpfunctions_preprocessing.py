# -*- coding: utf-8 -*-
import pandas as pd
import time
import re
import numpy as np

def create_list_urls(list_DOI, base_url):
    """
    

    Parameters
    ----------
    list_DOI : list
        Takes list of the DOI's for a set of articles that needs to be scraped from the SAGE website.

    Returns
    -------
    list_urls_sage : TYPE
        list of urls from the SAGE website to be scraped.

    """
    
    list_urls_sage = [base_url + DOI for DOI in list_DOI]

    return list_urls_sage



def check_items_from_soup_search(ReturnObj):
    
    if len(ReturnObj) == 0:
        return 'NA'
    else:
        return ReturnObj[0].text



def create_dataset_publisher(list_csv_locations, cookies, base_url, scrape_func):
    """

    """
    
    # column names for df to be filled
    col_df = ['title', 'DOI' , 'body', 'author_notes', 'abstract', 'keywords', 'acknowledge']
    
    # list of dfs with data from SAGE
    list_data_per_journal = []
    
    # save dict with stats on what was missed
    dict_missed = {'total_missed': 0,
                   'indexes_missed': []}
    
    # go through each csv
    for csv_location in list_csv_locations:
        
        # get DOI list
        df_WoS = pd.read_csv(csv_location)
        list_DOI = list(df_WoS['DOI'])
        
        # turn to urls, create df to be filled
        urls = create_list_urls(list_DOI, base_url)
        df_data = pd.DataFrame(index = range(1,len(urls)+1), columns = col_df)
        
        # go through each url and pull the data
        for index_url in range(0,len(urls)):
                    
            # extract data from url
            url = urls[index_url]
            
            # return list with items for dataframe
            result_scrape = scrape_func(url, cookies) 
            
            # check the output
            if  result_scrape == "NA":
                dict_missed['total_missed'] += 1
                dict_missed['indexes_missed'].append(index_url)
                
                continue 
            
            # put data in row and add to df
            row_entry = result_scrape
            df_data.iloc[index_url] = row_entry
            
            print("{0} / {1} -- {2} NAs" .format(index_url+1, len(urls), dict_missed['total_missed']))
            
            # sleep to make sure we are not recognized as DoS attack
            time.sleep(5)
        # add the df to list
        list_data_per_journal.append(df_data)
    
        print("Number missed: {0}/{1}".format(dict_missed['total_missed'], len(urls)))
    
    # return list of df
    return list_data_per_journal
        

def prepare_scopus(data_path, field_data, field_dict):
    """
    Function that pulls that cleans the scopus data and combines it with data on journal fields from scimago.

    Parameters
    ----------
    data_path : str
        Path to the scopus data csv.
    
    field_data : DataFrame
        Data on journal names and associated research fields.
        
    field_dict : Dictionary
        Data on journal names and associated research fields for easy looping.

    Returns
    -------
    output_df : DataFrame
        Cleaned scopus data and additional variables on the percentage of citations coming from the specified research fields. References that could not be extracted from the reference list or belong to journals that are not found on scimago, books, or newspaper articles, etc. cannot be accounted for.

    """
    # references
    scopus_data = pd.read_csv(data_path)
    
    # drop unnecessary variables
    scopus_data.drop(labels = ['Volume', 'Issue', 'Art. No.', 'Page start', 'Page end',
                  'ISBN', 'CODEN', 'PubMed ID', 'Language of Original Document', 'Page count',
                  'Document Type', 'Publication Stage',
                  'Open Access', 'Source', 'EID', 'Link', 'Affiliations', 'Index Keywords',
                  'Funding Text 2', 'Editors', 'Publisher', 'ISSN', 'Authors with affiliations',
                  'Author(s) ID', 'Correspondence Address', 'Funding Text 1', 'Funding Details'], axis = 1, inplace = True)
    
    # sort by year
    scopus_data = scopus_data.sort_values(by = 'Year')
    
    # get rid of potential duplicates
    scopus_data = scopus_data[~scopus_data['DOI'].duplicated(keep = 'first')]

    # get total number of references for each article
    scopus_data['num_ref'] = scopus_data['References'].apply(lambda x: str(x).count(";") + 1)
    
    # treat missing number of citations as 0
    scopus_data['Cited by'] = scopus_data['Cited by'].fillna(0)

    # get references as columns
    references = scopus_data['References'].str.split(pat = ";", expand = True)

    # clean references
    for col in range(references.shape[1]):
        references[col] = references[col].str.extract(r"\d{4}\)([a-zA-Z0-9_ :]*),")
        references[col] = references[col].str.strip()
    
    # rename columns
    references = references.add_prefix("ref_")

    # make all lower case before comparison
    references = references.applymap(lambda s:s.lower() if type(s) == str else s)
    field_data['journal'] = field_data['journal'].str.lower()

    # remove one duplicate that occurs when putting all journals in lower case
    field_data = field_data[~field_data.duplicated()]

    # get DOIs
    references['DOI'] = scopus_data['DOI']

    # replace journals by their respective fields
    # melt into long format
    references = references.melt(id_vars = 'DOI', var_name = 'reference', value_name = 'journal')

    # prepare join columns
    references['journal'] = references['journal'].astype(str).str.strip()
    field_data['journal'] = field_data['journal'].astype(str).str.strip()

    # left join field data
    result = pd.merge(references, field_data, how = 'left', on = 'journal', validate = 'many_to_one')

    # clean journal column, code nan as NaN
    result['journal'] = result['journal'].replace('nan', np.nan) 

    # those not in fields of interest, none, NaN, empty, are NA

    # get back into correct format
    result = result.pivot(index = 'DOI', columns = 'reference', values = 'field')

    # count fields per journal
    result = result.apply(pd.Series.value_counts, axis = 1).fillna(0)

    # add count of references found
    result['refs_found'] = result.sum(axis = 1)

    # merge onto scopus data
    output_df = scopus_data.merge(result, how = 'left', on = 'DOI', validate = 'many_to_one')
    output_df['refs_not_found'] = output_df['num_ref'] - output_df['refs_found']

    # get citations in percentages
    for field in list(field_dict):
        output_df[field] = output_df[field]/output_df['refs_found']
        
    return output_df