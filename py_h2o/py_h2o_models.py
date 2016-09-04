# -*- coding: utf-8 -*-
"""
py_h2o_models.py

Will contain models from h2o.ai package using Python interface

Contributors will add data to appropriate folder and (concise, neat) modeling
scripts to the run_models function in this file. All (concise, neat) modeling
scripts must create a list with the following elements:

* model name
* model description
* data name
* data description
* performance

This resulting list should be appended to the models list at the end of
each contributor's model. The results in the models list will be described
automatically with a table and plot in the main repo results.md file.

Run the gen_results.py script after adding (and testing) models to this file to
regenerate results.md with the new, contributed model's results.

"""

### imports
import inspect
import os
import re
import time

### global constants
DIV_BAR = '==================================================================='
SEED = 12345

def run_cla_models():

    """ Contributors add h2o.ai classification models in this function.

    Returns:
        models: List of all contributed modeling scripts results.

    """

    ### list to contain all individual model results
    models = []

    import h2o # install h2o: http://www.h2o.ai/download/h2o/choose
    h2o.init() # it may be necessary to start h2o outside of this script

    cla_dat_dir = (os.sep).join(['..', 'data', 'cla'])
    d_file_list = sorted([cla_dat_dir + os.sep + d_file for d_file in
                          os.listdir(cla_dat_dir) if d_file.endswith('.data')],
                         key=str.lower)

    for i, d_file in enumerate(d_file_list):

        tic = time.time()
        print DIV_BAR
        print 'Modeling %s - Classification Task: (%d/%d) ...' %\
            (d_file, i+1, len(d_file_list))

        # import matrix
        d_frame = h2o.import_file(d_file)

        ### last column is usually target, but ...
        ### first column can be target, id, or data
        ### use simple rules below to determine
        col1_y_matrices = ['CNAE-9.data',
                           'letter-recognition.data',
                           'meta.data',
                           'parkinsons.data',
                           'wine.data']

        id_col_name = ''
        y_name = ''

        if d_file.split(os.sep)[-1] in col1_y_matrices:
            y_name = d_frame.names[0]
        else:
            y_name = d_frame.names[-1]
            col1_name = d_frame.names[0]
            # check col1 for high cardinality
            if d_frame.types[col1_name] not in ['real', 'time']:
                col1_nlevels = d_frame[col1_name].asfactor().nlevels()[0]
                if float(col1_nlevels)/float(d_frame.nrow) > 0.9:
                    id_col_name = col1_name
            # check if col1 is date
            if id_col_name == '' and d_frame.types[col1_name] == 'time':
                id_col_name = col1_name

        ### specifiy modeling roles
        print 'Target: ' + y_name + ' ...'
        if id_col_name != '':
            print 'Column 1 treated as date or row ID: ' + id_col_name + ' ...'
        x_names = [n for n in d_frame.names if n not in [y_name, id_col_name]]

        ### 70/30 partition into train and valid frames
        tr_frame, v_frame = d_frame.split_frame([0.7], seed=SEED)
        del d_frame

        ### contributors add h2o regression models below ##################
        # use tr_frame for training/CV
        # use v_frame for validation, report assessment measures on v_frame
        # use y_name for target
        # use x_names for inputs

        ### simple classification models using defaults on raw data



        #######################################################################

        del tr_frame, v_frame

        print '%s modeled in %.2f s.' % (d_file, time.time()-tic)

    ### placeholder example

    example_model_results = ['model name', 'model description', 'data name',
                             'data description', 0.0, 0.0]

    models.append(example_model_results)

    h2o.cluster().shutdown()

    return models

def run_reg_models():

    """ Contributors add h2o.ai regression models in this function.

    Returns:
        models: List of all contributed modeling scripts results.

    """

    ### list to contain all individual model results
    models = []

    import h2o # install h2o: http://www.h2o.ai/download/h2o/choose
    h2o.init() # it may be necessary to start h2o outside of this script

    reg_dat_dir = (os.sep).join(['..', 'data', 'reg'])
    d_file_list = sorted([reg_dat_dir + os.sep + d_file for d_file in
                          os.listdir(reg_dat_dir) if d_file.endswith('.data')],
                         key=str.lower)

    for i, d_file in enumerate(d_file_list):

        tic = time.time()
        print DIV_BAR
        print 'Modeling %s - Regression Task: (%d/%d) ...' %\
            (d_file, i+1, len(d_file_list))

        # import matrix
        d_frame = h2o.import_file(d_file)

        ### last column is target, but ...
        ### first column can be id
        ### use simple rules below to determine
        id_col_name = ''
        y_name = ''
        y_name = d_frame.names[-1]
        col1_name = d_frame.names[0]
        # check col1 for high cardinality
        if d_frame.types[col1_name] not in ['real', 'time']:
            col1_nlevels = d_frame[col1_name].asfactor().nlevels()[0]
            if float(col1_nlevels)/float(d_frame.nrow) > 0.9:
                id_col_name = col1_name
        # check if col1 is date
        if id_col_name == '' and d_frame.types[col1_name] == 'time':
            id_col_name = col1_name

        ### specifiy modeling roles
        print 'Target: ' + y_name + ' ...'
        if id_col_name != '':
            print 'Column 1 treated as date or row ID: ' + id_col_name + ' ...'
        x_names = [n for n in d_frame.names if n not in [y_name, id_col_name]]

        ### 70/30 partition into train and valid frames
        tr_frame, v_frame = d_frame.split_frame([0.7], seed=SEED)
        del d_frame

        ### contributors add h2o regression models below ##################
        # use tr_frame for training/CV
        # use v_frame for validation, report assessment measures on v_frame
        # use y_name for target
        # use x_names for inputs

        ### simple regression models using defaults on raw data



        #######################################################################

        del tr_frame, v_frame

        print '%s modeled in %.2f s.' % (d_file, time.time()-tic)

    ### placeholder example

    example_model_results = ['model name', 'model description', 'data name',
                             'data description', 0.0, 0.0]

    models.append(example_model_results)

    h2o.cluster().shutdown()

    return models

def gen_table_md(models, section_header, table_header_list, out_txt_fname):

    """ Generates markdown table containing results in output markdown file.

    Args:
        models: List of individual model results.
        section_header: Table title, second level header.
        table_header_list: Names of attributes in table as a list of strings.
        out_txt_fname: Determined name of output markdown file.

    """

    # conditional delete/open markdown file
    if os.path.exists(out_txt_fname):
        os.remove(out_txt_fname)
    out = open(out_txt_fname, 'wb')

    # write section header markdown
    section_header = '## ' + section_header
    out.write(section_header + '\n')

    # write table header markdown
    num_table_attrs = len(table_header_list)
    out.write(' | '.join(table_header_list) + '\n')
    out.write(' | '.join(['---' for _ in range(0, num_table_attrs)]) + '\n')

    # write model attributes
    for model in models:
        out.write(' | '.join([str(attr) for attr in model]))

    out.close()

def main():

    """ Determines output markdown filename from current filename.
        Executes contributed model scripts in run_*_models().
        Generates results markdown file in gen_table_md().

    """

    # file-specific constants
    section_header = 'Python H20.ai Models'
    table_header_list = ['Model Name', 'Model Description', 'Data Name',
                         'Data Description', 'Performance Metric 1',
                         'Performance Metric 2']

    # determine output markdown filename from current filename
    current_path = re.split(r'[\\/]', inspect.getfile(inspect.currentframe()))
    current_fname_prefix = current_path[-1].split('.')[0]
    out_txt_fname = current_fname_prefix + '.txt'

    # run benchmark models
    models = []
    models.append(run_cla_models())
    models.append(run_reg_models())

    # generate markdown
    gen_table_md(models, section_header, table_header_list, out_txt_fname)

if __name__ == '__main__':
    main()
