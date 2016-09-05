# -*- coding: utf-8 -*-
"""
UCIDataMatrixFetcher.py

There are many types of data sets available from UCI:
https://archive.ics.uci.edu/ml/datasets.html.

The most readily usable are data matrices stored as plain text.

UCI also provides the facility to search available data by type and task.
Structured URLs are associated with each search.

This class uses a structured URL to determine the data sets associated with a
given task, determines which of these data sets are stored as data matrices,
determines the data folder location for these data matrices, and downloads the
data matrices if they are available as plain text.

UCI_CLA_DATA_MATRIX_BLACK_LIST: List of plain text data matrices that should
    not be downloaded for classification tasks b/c of problematic formats.

UCI_REG_DATA_MATRIX_BLACK_LIST: List of plain text data matrices that should
    not be downloaded for regression tasks b/c of problematic formats.

"""

### imports
import os
import sys
import time
from urllib2 import urlopen, HTTPError
from bs4 import BeautifulSoup

### global constants
DIV_BAR = '==================================================================='

UCI_CLA_DATA_MATRIX_BLACK_LIST = ['audiology.data',
                                  'badges.data',
                                  'diagnosis.data',
                                  'poker-hand-testing.data',
                                  'promoters.data',
                                  'Reaction%20Network%20(Undirected).data',
                                  'Relation%20Network%20(Directed).data',
                                  'secom.data',
                                  'secom_labels.data',
                                  'splice.data',
                                  'segmentation.data',
                                  'synthetic_control.data',
                                  'trains-original.data',
                                  'wdbc.data',
                                  'wpbc.data']

UCI_REG_DATA_MATRIX_BLACK_LIST = ['auto-mpg.data',
                                  'breast-cancer-wisconsin.data',
                                  'communities.data',
                                  'o-ring-erosion-only.data',
                                  'o-ring-erosion-or-blowby.data',
                                  'Relation%20Network%20(Directed).data',
                                  'Reaction%20Network%20(Undirected).data',
                                  'wdbc.data',
                                  'wpbc.data']

class UCIDataMatrixFetcher(object):

    """ Instantiated and called from gen_results.py; can be run as a
    standalone script as well.

    """

    def __init__(self):
        """ null constructor """
        pass

    @staticmethod
    def fetch_task_matrix_url_list(tsk_prfx):

        """ Uses structured URL from UCI to created sorted, deduped list of
        potential datasets associated with the task specified by tsk_prfx.

        Args:
            tsk_prfx: 'cla' for classification or 'reg' for regression.

        Raises:
            HTTPError: UCI repo url for either classification or regression
                task data matrices is unavailable.

        Returns:
            tsk_mtrx_urllst: Sorted, deduped list of potential datasets.

        """

        tic = time.time()
        print DIV_BAR
        print 'Fetching list of potential data matrices for ' + tsk_prfx +\
              ' task from UCI ...'

        tsk_mtrx_urllst = []
        url = 'http://archive.ics.uci.edu/ml/datasets.html?format=mat&task=' +\
              tsk_prfx + '&sort=taskUp&view=table'

        try:
            conn_ = urlopen(url)
        except HTTPError as err_:
            print 'UCI url for %s task data matrices unavailable!' % tsk_prfx
            print err_.code
            sys.exit(-1)

        tsk_mtrx_urllst = BeautifulSoup(conn_).find_all('a')
        tsk_mtrx_urllst = sorted(list(set(\
                            [l.get('href') for l in tsk_mtrx_urllst\
                            if l.get('href').startswith('datasets/')]\
                            )))

        print 'List fetched in %.2f s.' % (time.time()-tic)

        return tsk_mtrx_urllst

    @staticmethod
    def fetch_data_folder_links_list(lnk_lst, tsk_prfx):

        """ Uses list entries in lnk_lst to determine folders from which the
        potential data matrices for the task specified by tsk_prfx can be
        downloaded.

        Args:
            lnk_lst: List of potential data matrices.
            tsk_prfx: 'cla' for classification or 'reg' for regression.

        Raises:
            HTTPError: UCI repo url for a given data set is unavailable.

        Returns:
            A sorted, deduped list of potential data folders.

        """

        tic = time.time()
        print DIV_BAR
        print 'Fetching list of potential data folder links for ' + tsk_prfx +\
              ' task from UCI ...'

        data_folder_link_list = []

        for lnk in lnk_lst:

            data_url = 'http://archive.ics.uci.edu/ml/' + lnk
            try:
                conn_ = urlopen(data_url)
            except HTTPError as err_:
                print 'UCI data set url %s unavailable!' % data_url
                print err_.code
                sys.exit(-1)

            chld_lnk_lst = BeautifulSoup(conn_).find_all('a')
            for chld_lnk in chld_lnk_lst:
                chld_lnk = chld_lnk.get('href')
                if chld_lnk.startswith('../machine-learning-databases'):
                    data_fldr_lnk = str(chld_lnk).replace('..', '')
                    data_fldr_lnk = 'http://archive.ics.uci.edu/ml' +\
                                    data_fldr_lnk
                    data_folder_link_list.append(data_fldr_lnk)
                    # data folder link is first link
                    # that startswith('../machine-learning-databases')
                    break

        print 'List fetched in %.2f s.' % (time.time()-tic)

        # return sorted, deduped list of dataset links
        return sorted(list(set(data_folder_link_list)))

    @staticmethod
    def fetch_data(lnk_lst, tsk_prfx):

        """ Connects to data folders in lnk_lst and downloads plain text data
        matrices when available.

        Args:
            lnk_lst: List of potential folders with plain text data matrices.
            tsk_prfx: 'cla' for classification or 'reg' for regression.

        Raises:
            HTTPError: UCI repo url for a given data folder or data set is
                unavailable.

        """

        tic = time.time()
        print DIV_BAR
        print 'Fetching data matrices for ' + tsk_prfx + ' task from UCI ...'

        blck_lst_dct = {'cla':UCI_CLA_DATA_MATRIX_BLACK_LIST,
                        'reg':UCI_REG_DATA_MATRIX_BLACK_LIST}

        for lnk in lnk_lst:

            try:
                conn_ = urlopen(lnk)
            except HTTPError as err_:
                print 'UCI data folder url %s unavailable!' % lnk
                print err_.code
                sys.exit(-1)

            chld_lnk_lst = BeautifulSoup(conn_).find_all('a')
            for chld_lnk in chld_lnk_lst:
                chld_lnk = chld_lnk.get('href')
                if chld_lnk.endswith('.data') and chld_lnk not in\
                    blck_lst_dct[tsk_prfx]:
                    print 'Downloading ' + lnk + chld_lnk + ' ...'

                    try:
                        conn_ = urlopen(lnk + chld_lnk)
                    except HTTPError as err_:
                        print 'UCI data set url %s unavailable!' %\
                            (lnk + chld_lnk)
                        print err_.code
                        sys.exit(-1)

                    mtrx = conn_.read()
                    out_fldr = 'data' + os.sep + tsk_prfx
                    mtrx_fname = out_fldr + os.sep + chld_lnk
                    with open(mtrx_fname, 'w+') as mxtrf:
                        for line in mtrx.split('\n'):
                            line = line.strip()
                            if line != '':
                                # standardize missing values
                                line = line.replace('?', 'NA')
                                line = line.replace(' -', ' NA')
                                line = line.replace(',*', ',NA')
                                mxtrf.write(line + '\n')

        print 'Data matrices fetched %.2f s.' % (time.time()-tic)

def main():

    """ For standalone execution or testing purposes.
        For classification (cla) and regression (reg) tasks,
        fetch_task_matrix_url_list() determines data sets stored as matrices,
        fetch_data_folder_links_list() determines data folder urls,
        fetch_data() downloads data matrices stored as plain text.

    """

    uci_fetcher = UCIDataMatrixFetcher()

    for tsk_prfx in ['cla', 'reg']:
        tsk_mtrx_url_lst = uci_fetcher.fetch_task_matrix_url_list(tsk_prfx)
        data_folder_link_list = uci_fetcher.fetch_data_folder_links_list(
            tsk_mtrx_url_lst,
            tsk_prfx
        )
        uci_fetcher.fetch_data(data_folder_link_list, tsk_prfx)

if __name__ == '__main__':
    main()
