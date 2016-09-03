# -*- coding: utf-8 -*-
"""
get_uci_data_matrices.py

"""

### imports
import urllib2
from bs4 import BeautifulSoup
import os

CLASSIFICATION_MATRICES_URL = 'http://archive.ics.uci.edu/ml/datasets.html?\
format=mat&task=cla&sort=taskUp&view=table'

def get_data_folder_links(lnk_lst):                     
    
    data_folder_link_list = []
          
    for lnk in lnk_lst:
        conn_ = urllib2.urlopen('http://archive.ics.uci.edu/ml/' + lnk) 
        chld_lnk_lst = BeautifulSoup(conn_).find_all('a')
        for chld_lnk in chld_lnk_lst:
            chld_lnk = chld_lnk.get('href')
            if chld_lnk.startswith('../machine-learning-databases'):
                data_fldr_lnk = str(chld_lnk).replace('..', '')
                data_fldr_lnk = 'http://archive.ics.uci.edu/ml' + data_fldr_lnk
                data_folder_link_list.append(data_fldr_lnk)
                
                # data folder link is first link
                # that startswith('../machine-learning-databases')
                break 
            
    # return sorted, deduped list of dataset links         
    return sorted(list(set(data_folder_link_list)))

def get_data(lnk_lst, out_fldr):
    
    for lnk in lnk_lst:
        conn_ = urllib2.urlopen(lnk) 
        chld_lnk_lst = BeautifulSoup(conn_).find_all('a')
        for chld_lnk in chld_lnk_lst:
            chld_lnk = chld_lnk.get('href')
            if chld_lnk.endswith('.data'):

                print 'Downloading ' + lnk + chld_lnk + '... '

                conn_ = urllib2.urlopen(lnk + chld_lnk)
                mtrx = connection.read()
                mtrx_fname = out_fldr + os.sep + chld_lnk
                with open(mtrx_fname, 'w+') as mxtrf:
                    for line in table.split('\n'):
                        line = line.strip()
                        if line != '':
  
def main():

    ### created sorted, deduped list of potential datasets 
    ### associated with classification tasks
    conn_ = urllib2.urlopen(CLASSIFICATION_MATRICES_URL)
    cla_lnk_lst = BeautifulSoup(conn_).find_all('a')
    cla_lnk_lst = sorted(list(set([l.get('href') for l in cla_lnk_lst if\
                                   l.get('href').startswith('datasets/')])))

    data_folder_link_list = get_data_folder_links(cla_lnk_lst)
    get_data(data_folder_link_list)

if __name__ == '__main__':
    main()
