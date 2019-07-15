# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 13:50:50 2019

@author: renoult
"""

#-----------------------------------------------------------------------------#
# PREPARE DATASET FOR GENDER CLASSIFICATION OF CFD
#-----------------------------------------------------------------------------#


#LIBRARY
import os 
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

#READ TAB 
df = pd.read_excel('D:/CFD Version 2.0.3/CFD 2.0.3 Norming Data and Codebook.xlsx', sheetname='CFD 2.0.3 Norming Data') 
df = df.drop([0,1,2])
df.columns = df.iloc[0]
df = df.reindex(df.index.drop(3))

#SELECT 
df = df[['Target' , 'Gender', 'Feminine']]



def list_of_pict(dirName):
    """Get the list of all files in directory tree at given path"""
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(dirName):
        #listOfFiles.append([os.path.join(file) for file in filenames])
        for file in filenames:
            if file.endswith('.jpg'):
                listOfFiles.append(dirpath + '/' + file)
    return(listOfFiles)
    
    
#IMAGES - GET FULLPATH OF ALL IMAGES
DIR  = 'D:/CFD Version 2.0.3/CFD 2.0.3 Images/'
full_path = list_of_pict(DIR)
#Get name of indiv
indiv = [i.split('/')[3] for i in full_path ]


#IMAGES - GET FINAL TABS
d = {'Target':indiv,'full_path':full_path}
df_1 = pd.DataFrame(d, columns=['Target','full_path'])
tags_pict = pd.merge(df_1, df, on='Target')
cwd = 'D:/stage_sonia_2/human_gender/'
#or cwd = os.getcwd()
tags_pict.to_csv( cwd + 'datas/pict_metadatas.csv')
