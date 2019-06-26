#Author: Chetan Singh

#Import libraries
from bs4 import BeautifulSoup
from pathlib import Path
import pandas as pd
import numpy as np
import os
import platform

#Set path of the directory
path = ('html_files/')
base = Path(path)

#Set total count of files for progress indicator
total_file_len = len([x for x in base.iterdir()])

docs = []
for count, filename in enumerate(base.iterdir()):    
    #Initialize beautiful soup
    with open (filename, 'r') as file:
        contents = file.read()
        soup = BeautifulSoup(contents, 'lxml')
        
        #Fetch Main Text
        if soup.findAll("txt_te"):
            tags = soup.findAll("txt_te")
            
            main_text = []
            for para in tags:
                main_text.extend(((para.text).lower().replace(',',' ').replace('.', ' ')).split())
        else:
            tags0 = soup.findAll('p', {'class': 'doc-ti'})
            tags = soup.findAll('p', {'class': 'normal'})
            tags.extend(tags0)
            
            main_text = []
            for para in tags:
                main_text.extend(((para.text).lower().replace(',',' ').replace('.', ' ')).split())
        
        #Fetch Classes
        cls = soup.findAll("div", {"id": "PPClass_Contents"})
        if cls:
            classes = [x for x in (((cls[0].text).strip()).split('\n')) if x != '' and x != ',' and x != ', ' and x != ' ']
            ctg = ['Subject matter: ', 'Directory code: ']
            try:
                classes = [x.lower().replace(' ', '_') for x in classes[1:classes.index(ctg[0])]]
            except:
                classes = [x.lower().replace(' ', '_') for x in classes[1:classes.index(ctg[1])]]
        
        #Append text and classes together in multi-D list
        if main_text and classes:
            docs.append([' '.join(main_text), ', '.join(classes)])
            
    #Display progress %
    if (platform.system().lower() == 'windows'):
        os.system('cls')
    else:
        os.system("printf '\\033c'")
    print("Progress: {:.2f}".format(count/total_file_len * 100), "%")

#Save the output in a dataframe and write output to .csv
df = pd.DataFrame(docs)
df.to_csv('final_scraped.csv', header = None, index = None)
