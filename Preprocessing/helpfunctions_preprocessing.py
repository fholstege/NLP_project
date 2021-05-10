# -*- coding: utf-8 -*-
import pandas as pd
import time


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
    col_df = ['title', 'DOI' , 'body', 'author_notes', 'abstract', 'keywords', 'acknowledge', 'ref_list', 'fngroup']
    
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
        
        