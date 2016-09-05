# -*- coding: utf-8 -*-
"""
py_h2o_models.py

Will contain models from h2o.ai package using Python interface

Contributors will add data to appropriate folder and (concise, neat) modeling
scripts to the a function in this file. All (concise, neat) modeling scripts
must be placed in a function that returns a list with the following elements:

* model name
* model description
* data name
* data description
* performance

This resulting list should be appended to the models list. The results in the
models list will be described automatically with a table and plot(s) in the
main repo results.md file.

Run the gen_results.py script after adding models to this file (and testing
them) to regenerate results.md with the new, contributed model's results.

"""

### imports
import inspect
import os
import re
import time

### global constants
DIV_BAR = '==================================================================='
SEED = 12345

### contributors add model functions here #####################################

def example_model(frames, y_name, x_names):

    """ Placeholder example model function.

    Args:
        frames[0], h2o training frame.
        frames[1], h2o validation frame.
        y_name: Target name.
        x_names: List of input names.

    Returns:
        List of modeling results, for example
        ['model name', 'model description', 'data name', 'data description',
         0.0, 0.0]

    """

    tr_frame = frames[0]
    v_frame = frames[1]

    ### impute numeric
    ### categorical 'NA' treated as valid level
    tr_frame.impute(method='mean')
    v_frame.impute(method='mean')

    ### stdize
    tr_frame = h2o_stdize(tr_frame, y_name, x_names)
    v_frame = h2o_stdize(v_frame, y_name, x_names)

    example_model_results = ['model name', 'model description', 'data name',
                             'data description', 0.0, 0.0]

    return example_model_results

###############################################################################

def h2o_stdize(frame, y_name, x_names):

    """ Conditionally standardizes numeric columns in an h2o dataframe.

    Args:
        frame: Frame containing columns specified by y_name and x_names.
        y_name: Name of target column.
        x_names: List of input columns.

    Returns:
        H2o dataframe with appropriate columns standardized.

    """

    ### conditionally stdize target
    if not frame[y_name].isfactor():
        frame[y_name] = frame[y_name].scale()

    ### determine numeric inputs
    numeric_x_names_bool_list = frame[x_names].isnumeric()
    numeric_x_names = [name for i, name in enumerate(x_names) if\
                       numeric_x_names_bool_list[i]]

    # stdize numeric inputs
    frame[numeric_x_names] = frame[numeric_x_names].scale()

    return frame

def h2o_check_col_cardinality(frame, col_name):

    """ Checks the cardinality of a colummn with name col_name in an h2o
    dataframe frame.

    Args:
        frame: H2o dataframe containing column col_name.
        col_name: Integer or enum column in h2o dataframe frame.

    Returns:
       If cardinality is high, returns the name of the column, else ''.

    """

    # if number of unique levels / number of rows is above id_threshold
    # or if the column is a time stamp
    # the column will be considered high cardinality and dropped
    id_threshold = 0.9

    if frame.types[col_name] not in ['real', 'time']:
        col_nlevels = frame[col_name].asfactor().nlevels()[0]
        if float(col_nlevels)/float(frame.nrow) > id_threshold:
            return col_name
        else:
            return str('')
    elif frame.types[col_name] == 'time':
        return col_name
    else:
        return str('')

def run_cla_models():

    """ Loops through data matrices for classification task.

    Returns:
        models: List of all contributed modeling scripts results.

    """

    ### list to contain all contributed model results
    models = []

    import h2o # install h2o: http://www.h2o.ai/download/h2o/choose
    h2o.init() # it may be necessary to start h2o outside of this script
               # you can use max_mem_size_GB here to increase available memory
               # ex: h2o.init(max_mem_size_GB=<int>)
    h2o.cluster().show_status()

    ### specify classification task dir and data matrices
    cla_dat_dir = (os.sep).join(['..', 'data', 'cla'])
    d_file_list = sorted([cla_dat_dir + os.sep + d_file for d_file in
                          os.listdir(cla_dat_dir) if d_file.endswith('.data')],
                         key=str.lower)

    ### loop through data matrices in dir
    for i, d_file in enumerate(d_file_list):

        tic = time.time()
        print DIV_BAR
        print 'Modeling %s - Classification Task: (%d/%d) ...' %\
            (d_file, i+1, len(d_file_list))

        # import current data matrix
        d_frame = h2o.import_file(d_file)

        ### last column is usually target, but ...
        ### first column can be target, id, or date
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
            # check col1 cardinality
            id_col_name = h2o_check_col_cardinality(d_frame, col1_name)

        ### specifiy modeling roles
        d_frame[y_name] = d_frame[y_name].asfactor()
        print 'Target: ' + y_name + ' ...'
        if id_col_name != '':
            print 'Column 1 treated as date or row ID: ' + id_col_name + ' ...'
        x_names = [name for name in d_frame.names if name not in\
                  [y_name, id_col_name]]

        ### 70/30 partition into train and valid frames
        d_frames = d_frame.split_frame([0.7], seed=SEED)
        del d_frame

        ### call model functions
        models.append(example_model(d_frames, y_name, x_names))

        del d_frames

        print '%s modeled in %.2f s.' % (d_file, time.time()-tic)

    h2o.cluster().shutdown()

    return models

def run_reg_models():

    """ Loops through data matrices for regression task.

    Returns:
        models: List of all contributed modeling scripts results.

    """

    ### list to contain all individual model results
    models = []

    import h2o # install h2o: http://www.h2o.ai/download/h2o/choose
    h2o.init() # it may be necessary to start h2o outside of this script
               # you can use max_mem_size_GB to increase available memory
               # ex: h2o.init(max_mem_size_GB=<int>)
    h2o.cluster().show_status()

    ### specify regression task dir and data matrices
    reg_dat_dir = (os.sep).join(['..', 'data', 'reg'])
    d_file_list = sorted([reg_dat_dir + os.sep + d_file for d_file in
                          os.listdir(reg_dat_dir) if d_file.endswith('.data')],
                         key=str.lower)

    ### loop through data matrices in dir
    for i, d_file in enumerate(d_file_list):

        tic = time.time()
        print DIV_BAR
        print 'Modeling %s - Regression Task: (%d/%d) ...' %\
            (d_file, i+1, len(d_file_list))

        # import current data matrix
        d_frame = h2o.import_file(d_file)

        ### last column is target, but ...
        ### first column can be id
        ### use simple rules below to determine
        id_col_name = ''
        y_name = d_frame.names[-1]
        col1_name = d_frame.names[0]
        # check col1 cardinality
        id_col_name = h2o_check_col_cardinality(d_frame, col1_name)

        ### specifiy modeling roles
        d_frame[y_name] = d_frame[y_name].asnumeric()
        print 'Target: ' + y_name + ' ...'
        if id_col_name != '':
            print 'Column 1 treated as date or row ID: ' + id_col_name + ' ...'
        x_names = [name for name in d_frame.names if name not in\
                   [y_name, id_col_name]]

        ### 70/30 partition into train and valid frames
        frames = d_frame.split_frame([0.7], seed=SEED)
        del d_frame

        ### call model functions
        models.append(example_model(frames, y_name, x_names))

        del frames

        print '%s modeled in %.2f s.' % (d_file, time.time()-tic)

    h2o.cluster().shutdown()

    return models

def gen_table_md(models, section_header, table_header_list, out_txt_fname,
                 write_mode):

    """ Generates markdown table containing results in output markdown file.

    Args:
        models: List of individual model results.
        section_header: Table title, second level header.
        table_header_list: Names of attributes in table as a list of strings.
        out_txt_fname: Determined name of output markdown file.

    """

    # conditional delete/open markdown file
    out = open(out_txt_fname, write_mode)

    # write section header markdown
    section_header = '## ' + section_header
    out.write(section_header + '\n')

    # write table header markdown
    num_table_attrs = len(table_header_list)
    out.write(' | '.join(table_header_list) + '\n')
    out.write(' | '.join(['---' for _ in range(0, num_table_attrs)]) + '\n')

    # write model attributes
    for model in models:
        out.write(' | '.join([str(attr) for attr in model]) + '\n')

    # seperate classification and regression models
    if write_mode == 'w+':
        out.write('\n')

    out.close()

def main():

    """ Determines output markdown filename from current filename.
        Executes contributed model scripts in run_*_models().
        Generates results markdown file in gen_table_md().

    """

    ### determine output markdown filename from current filename
    current_path = re.split(r'[\\/]', inspect.getfile(inspect.currentframe()))
    current_fname_prefix = current_path[-1].split('.')[0]
    out_txt_fname = current_fname_prefix + '.txt'

    ### run benchmark models
    ### extends prevents another level of nesting
    cla_models = run_cla_models()
    reg_models = run_reg_models()

    ### generate markdown
    cla_section_header = 'Python H20.ai Classification Models'
    table_header_list = ['Model Name', 'Model Description', 'Data Name',
                         'Data Description', 'Performance Metric 1',
                         'Performance Metric 2']
    gen_table_md(cla_models, cla_section_header, table_header_list,
                 out_txt_fname, 'w+')

    reg_section_header = 'Python H20.ai Regression Models'
    table_header_list = ['Model Name', 'Model Description', 'Data Name',
                         'Data Description', 'Performance Metric 1',
                         'Performance Metric 2']
    gen_table_md(reg_models, reg_section_header, table_header_list,
                 out_txt_fname, 'a+')

if __name__ == '__main__':
    main()
