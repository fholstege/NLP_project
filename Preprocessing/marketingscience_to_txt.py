import pandas as pd
import subprocess
import os
from tqdm import tqdm

# load csv after manually downloading of papers ('filled')
df = pd.read_csv('../data/raw/marketingscience_urls_filled.csv', sep=';')

# get list of all downloaded pdfs
pdf_list = os.listdir('../data/raw/marketing_science_pdf')

# convert all pdfs to txt
for pdf_file in tqdm(pdf_list):
    
    # get number of file
    no_article = os.path.splitext(pdf_file)[0]
    
    # convert to txt and save in new folder
    cmd = f'pdftotext -f 2 ../data/raw/marketing_science_pdf/{no_article}.pdf ../data/raw/marketing_science_txt/{no_article}.txt'
    subprocess.call(cmd, shell=True)

# keep only articles (Type == NaN)
df = df.loc[df['Type'].isna(), 'DOI']

# save scraped DOIs
df.to_csv('../data/raw/marketingscience_scraped_DOI.csv')
