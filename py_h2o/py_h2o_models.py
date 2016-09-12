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
* N
* p
* relavent performance metrics

This resulting list should be appended to the models list. The results in the
models list will be described automatically with a table (and plot(s)?) in the
main repo results.md file.

Run the gen_results.py script after adding models to this file (and testing
them) to regenerate results.md with the new, contributed model's results.

"""

### imports
import inspect
import os
import re
import time
import h2o # install h2o: http://www.h2o.ai/download/h2o/choose
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.grid.grid_search import H2OGridSearch

### global constants
DIV_BAR = '==================================================================='
SEED = 12345

# TODO: Make code less terrible
# TODO: Linear, logistic regression
# TODO: Naive Bayes?
# TODO: Fix colors, borders in scatter plots
# TODO: Fix bottom border in bar plots 
# TODO: Add graph of error metrics by model type

### contributors add model functions here #####################################

# But probably not yet ... this is still a work in progress.

###############################################################################

def cla_randomsearch_gbm(frames, y_name, x_names, dname):

    """ H2o GBM with parameter tuning

    Args:
        frames[0], h2o training frame.
        frames[1], h2o validation frame.
        y_name: Target name.
        x_names: List of input names.
        dname: Name of data file.

    Returns:
        List of modeling results:

    """

    ### basic descriptors
    mname = 'GBM'
    mdesc = 'GBM w/ random hyperparameter search'

    ### assign partitions
    tr_frame, v_frame = frames[0], frames[1]

    ### impute numeric
    ### categorical 'NA' treated as valid level
    tr_frame.impute(method='mean')
    v_frame.impute(method='mean')

    ### stdize
    tr_frame = h2o_stdize(tr_frame, y_name, x_names)
    v_frame = h2o_stdize(v_frame, y_name, x_names)

    ### define random grid search parameters
    ntrees_opt = range(0, 100, 1)
    max_depth_opt = range(0, 20, 1)
    sample_rate_opt = [s/float(10) for s in range(1, 11)]
    col_sample_rate_opt = [s/float(10) for s in range(1, 11)]

    hyper_parameters = {"ntrees":ntrees_opt,
                        "max_depth":max_depth_opt,
                        "sample_rate":sample_rate_opt,
                        "col_sample_rate":col_sample_rate_opt}

    search_criteria = {"strategy":"RandomDiscrete",
                       "max_models":20,
                       "max_runtime_secs":600,
                       "seed":SEED}

    gsearch = H2OGridSearch(H2OGradientBoostingEstimator,
                            hyper_params=hyper_parameters,
                            search_criteria=search_criteria)

    ### execute training w/ grid search
    gsearch.train(x=x_names,
                  y=y_name,
                  training_frame=tr_frame,
                  validation_frame=v_frame
                 )

    ### retrieve best model
    bst_logl_model_id = gsearch.sort_by('logloss(valid=True)', False)\
                                ['Model Id'][0]
    bst_logl_model = h2o.get_model(bst_logl_model_id)

    ### collect validation error measures (w/ reasonable precision)
    logloss = '%.3f' % bst_logl_model.logloss(valid=True)
    rmse = '%.3f' % bst_logl_model.rmse(valid=True)
    if tr_frame[y_name].nlevels()[0] > 2: # multinomial
        acc = '%.3f' % bst_logl_model.confusion_matrix(v_frame)['Error'][-1]
    else: # binary
        acc = '%.3f' % bst_logl_model.accuracy(thresholds=[0.5],
                                               valid=True)[0][-1]

    return [mname, mdesc, dname, tr_frame.nrow, len(x_names), logloss, rmse,\
            acc]

def cla_earlystop_rf(frames, y_name, x_names, dname):

    """ H2o RF with early stopping

    Args:
        frames[0], h2o training frame.
        frames[1], h2o validation frame.
        y_name: Target name.
        x_names: List of input names.
        dname: Name of data file.

    Returns:
        List of modeling results:

    """

    ### basic descriptors
    mname = 'RF'
    mdesc = 'RF w/ early stopping'

    ### assign partitions
    tr_frame, v_frame = frames[0], frames[1]

    ### impute numeric
    ### categorical 'NA' treated as valid level
    tr_frame.impute(method='mean')
    v_frame.impute(method='mean')

    ### stdize
    tr_frame = h2o_stdize(tr_frame, y_name, x_names)
    v_frame = h2o_stdize(v_frame, y_name, x_names)

    ### specify rf model
    rf_model = H2ORandomForestEstimator(
        ntrees=200,
        max_depth=30,
        stopping_rounds=2,
        stopping_tolerance=0.01,
        score_each_iteration=True,
        seed=SEED
       )
        
    ### train rf model
    rf_model.train(x=x_names,
                   y=y_name,
                   training_frame=tr_frame,
                   validation_frame=v_frame
                  )

    ### collect validation error measures (w/ reasonable precision)
    logloss = '%.3f' % rf_model.logloss(valid=True)
    rmse = '%.3f' % rf_model.rmse(valid=True)
    if tr_frame[y_name].nlevels()[0] > 2: # multinomial
        acc = '%.3f' % rf_model.confusion_matrix(v_frame)['Error'][-1]
    else: # binary
        acc = '%.3f' % rf_model.accuracy(thresholds=[0.5],
                                               valid=True)[0][-1]

    return [mname, mdesc, dname, tr_frame.nrow, len(x_names), logloss, rmse,\
            acc]

def cla_randomsearch_nn(frames, y_name, x_names, dname):

    """ H2o neural network with parameter tuning

    Args:
        frames[0], h2o training frame.
        frames[1], h2o validation frame.
        y_name: Target name.
        x_names: List of input names.
        dname: Name of data file.

    Returns:
        List of modeling results:

    """

    ### basic descriptors
    mname = 'NN'
    mdesc = 'NN w/ random hyperparameter search'

    ### assign partitions
    tr_frame, v_frame = frames[0], frames[1]

    ### impute numeric
    ### categorical 'NA' treated as valid level
    tr_frame.impute(method='mean')
    v_frame.impute(method='mean')

    ### stdize
    tr_frame = h2o_stdize(tr_frame, y_name, x_names)
    v_frame = h2o_stdize(v_frame, y_name, x_names)

    ### define random grid search parameters
    hidden_opt = [[17, 32], [8, 19], [32, 16, 8], [100], [10, 10, 10, 10]]
    l1_opt = [s/1e6 for s in range(1, 1001)]
    l2_opt = [s/1e4 for s in range(1, 101)]
    input_dropout_ratio_opt = [s/1e2 for s in range(1, 21)]    

    hyper_parameters = {"hidden":hidden_opt,
                        "l1":l1_opt,
                        "l2":l2_opt,
                        "input_dropout_ratio":input_dropout_ratio_opt}

    search_criteria = {"strategy":"RandomDiscrete",
                       "max_models":20,
                       "max_runtime_secs":600,
                       "seed":SEED}

    gsearch = H2OGridSearch(H2ODeepLearningEstimator,
                            hyper_params=hyper_parameters,
                            search_criteria=search_criteria)

    ### execute training w/ grid search
    gsearch.train(x=x_names,
                  y=y_name,
                  training_frame=tr_frame,
                  validation_frame=v_frame
                 )

    ### retrieve best model
    bst_logl_model_id = gsearch.sort_by('logloss(valid=True)', False)\
                                ['Model Id'][0]
    bst_logl_model = h2o.get_model(bst_logl_model_id)

    ### collect validation error measures (w/ reasonable precision)
    logloss = '%.3f' % bst_logl_model.logloss(valid=True)
    rmse = '%.3f' % bst_logl_model.rmse(valid=True)
    if tr_frame[y_name].nlevels()[0] > 2: # multinomial
        acc = '%.3f' % bst_logl_model.confusion_matrix(v_frame)['Error'][-1]
    else: # binary
        acc = '%.3f' % bst_logl_model.accuracy(thresholds=[0.5],
                                               valid=True)[0][-1]

    return [mname, mdesc, dname, tr_frame.nrow, len(x_names), logloss, rmse,\
            acc]

def reg_randomsearch_gbm(frames, y_name, x_names, dname):

    """ H2o GBM with parameter tuning

    Args:
        frames[0], h2o training frame.
        frames[1], h2o validation frame.
        y_name: Target name.
        x_names: List of input names.
        dname: Name of data file.

    Returns:
        List of modeling results:

    """

    ### basic descriptors
    mname = 'GBM'
    mdesc = 'GBM w/ random hyperparameter search'

    ### assign partitions
    tr_frame, v_frame = frames[0], frames[1]

    ### impute numeric
    ### categorical 'NA' treated as valid level
    tr_frame.impute(method='mean')
    v_frame.impute(method='mean')

    ### stdize
    tr_frame = h2o_stdize(tr_frame, y_name, x_names)
    v_frame = h2o_stdize(v_frame, y_name, x_names)

    ### define random grid search parameters
    ntrees_opt = range(0, 100, 1)
    max_depth_opt = range(0, 20, 1)
    sample_rate_opt = [s/float(10) for s in range(1, 11)]
    col_sample_rate_opt = [s/float(10) for s in range(1, 11)]

    hyper_parameters = {"ntrees":ntrees_opt,
                        "max_depth":max_depth_opt,
                        "sample_rate":sample_rate_opt,
                        "col_sample_rate":col_sample_rate_opt}

    search_criteria = {"strategy":"RandomDiscrete",
                       "max_models":20,
                       "max_runtime_secs":600,
                       "seed":SEED}

    gsearch = H2OGridSearch(H2OGradientBoostingEstimator,
                            hyper_params=hyper_parameters,
                            search_criteria=search_criteria)

    ### execute training w/ grid search
    gsearch.train(x=x_names,
                  y=y_name,
                  training_frame=tr_frame,
                  validation_frame=v_frame
                 )

    ### retrieve best model
    bst_rd_model_id = gsearch.sort_by('residual_deviance(valid=True)', False)\
                         ['Model Id'][0]
    bst_rd_model = h2o.get_model(bst_rd_model_id)

    ### collect validation error measures (w/ reasonable precision)
    mrd = '%.3f' % bst_rd_model.mean_residual_deviance(valid=True)
    rmse = '%.3f' % bst_rd_model.rmse(valid=True)
    r2_ = '%.3f' % bst_rd_model.r2(valid=True)

    return [mname, mdesc, dname, tr_frame.nrow, len(x_names), mrd, rmse, r2_]

def reg_earlystop_rf(frames, y_name, x_names, dname):

    """ H2o RF with early stopping

    Args:
        frames[0], h2o training frame.
        frames[1], h2o validation frame.
        y_name: Target name.
        x_names: List of input names.
        dname: Name of data file.

    Returns:
        List of modeling results:

    """

    ### basic descriptors
    mname = 'RF'
    mdesc = 'RF w/ early stopping'

    ### assign partitions
    tr_frame, v_frame = frames[0], frames[1]

    ### impute numeric
    ### categorical 'NA' treated as valid level
    tr_frame.impute(method='mean')
    v_frame.impute(method='mean')

    ### stdize
    tr_frame = h2o_stdize(tr_frame, y_name, x_names)
    v_frame = h2o_stdize(v_frame, y_name, x_names)

    ### specify rf model
    rf_model = H2ORandomForestEstimator(
        ntrees=200,
        max_depth=30,
        stopping_rounds=2,
        stopping_tolerance=0.01,
        score_each_iteration=True,
        seed=SEED
       )
        
    ### train rf model
    rf_model.train(x=x_names,
                   y=y_name,
                   training_frame=tr_frame,
                   validation_frame=v_frame
                  )
                  

    ### collect validation error measures (w/ reasonable precision)
    mrd = '%.3f' % rf_model.mean_residual_deviance(valid=True)
    rmse = '%.3f' % rf_model.rmse(valid=True)
    r2_ = '%.3f' % rf_model.r2(valid=True)

    return [mname, mdesc, dname, tr_frame.nrow, len(x_names), mrd, rmse, r2_]
    
def reg_randomsearch_nn(frames, y_name, x_names, dname):

    """ H2o neural network with parameter tuning

    Args:
        frames[0], h2o training frame.
        frames[1], h2o validation frame.
        y_name: Target name.
        x_names: List of input names.
        dname: Name of data file.

    Returns:
        List of modeling results:

    """
    ### basic descriptors
    mname = 'NN'
    mdesc = 'NN w/ random hyperparameter search'

    ### assign partitions
    tr_frame, v_frame = frames[0], frames[1]

    ### impute numeric
    ### categorical 'NA' treated as valid level
    tr_frame.impute(method='mean')
    v_frame.impute(method='mean')

    ### stdize
    tr_frame = h2o_stdize(tr_frame, y_name, x_names)
    v_frame = h2o_stdize(v_frame, y_name, x_names)

    ### define random grid search parameters
    hidden_opt = [[17, 32], [8, 19], [32, 16, 8], [100], [10, 10, 10, 10]]
    l1_opt = [s/1e6 for s in range(1, 1001)]
    l2_opt = [s/1e4 for s in range(1, 101)]
    input_dropout_ratio_opt = [s/1e2 for s in range(1, 21)]    

    hyper_parameters = {"hidden":hidden_opt,
                        "l1":l1_opt,
                        "l2":l2_opt,
                        "input_dropout_ratio":input_dropout_ratio_opt}

    search_criteria = {"strategy":"RandomDiscrete",
                       "max_models":20,
                       "max_runtime_secs":600,
                       "seed":SEED}

    gsearch = H2OGridSearch(H2ODeepLearningEstimator,
                            hyper_params=hyper_parameters,
                            search_criteria=search_criteria)

    ### execute training w/ grid search
    gsearch.train(x=x_names,
                  y=y_name,
                  training_frame=tr_frame,
                  validation_frame=v_frame
                 )
                 
    ### retrieve best model
    bst_rd_model_id = gsearch.sort_by('residual_deviance(valid=True)', False)\
                         ['Model Id'][0]
    bst_rd_model = h2o.get_model(bst_rd_model_id)

    ### collect validation error measures (w/ reasonable precision)
    mrd = '%.3f' % bst_rd_model.mean_residual_deviance(valid=True)
    rmse = '%.3f' % bst_rd_model.rmse(valid=True)
    r2_ = '%.3f' % bst_rd_model.r2(valid=True)

    return [mname, mdesc, dname, tr_frame.nrow, len(x_names), mrd, rmse, r2_]

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
    if len(numeric_x_names) > 0:
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

        dname = d_file.split(os.sep)[-1]
        if dname in col1_y_matrices:
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
        frames = d_frame.split_frame([0.7], seed=SEED)
        del d_frame

        ### call model functions
        try:
            models.append(cla_randomsearch_gbm(frames, y_name, x_names,
                                               dname))
            models.append(cla_earlystop_rf(frames, y_name, x_names,
                                              dname))                                               
            models.append(cla_randomsearch_nn(frames, y_name, x_names,
                                              dname))  
        except ValueError:
            print 'Warning: Model training failure.'

        del frames

        print '%s modeled in %.2f s.' % (d_file, time.time()-tic)

    return models

def run_reg_models():

    """ Loops through data matrices for regression task.

    Returns:
        models: List of all contributed modeling scripts results.

    """

    ### list to contain all individual model results
    models = []

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
        try:
            models.append(reg_randomsearch_gbm(frames, y_name, x_names,
                                               d_file.split(os.sep)[-1]))
            models.append(reg_earlystop_rf(frames, y_name, x_names,
                                           d_file.split(os.sep)[-1]))                                                
            models.append(reg_randomsearch_nn(frames, y_name, x_names,
                                              d_file.split(os.sep)[-1]))
        except ValueError:
            print 'Warning: Model training failure.'

        del frames

        print '%s modeled in %.2f s.' % (d_file, time.time()-tic)

    return models

def gen_table_md(models, section_header, out_txt_fname, write_mode):

    """ Generates markdown table containing results in output markdown file.

    Args:
        models: pandas df of individual model results.
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
    num_table_attrs = len(models.columns)
    out.write(' | '.join(models.columns) + '\n')
    out.write(' | '.join(['---' for _ in range(0, num_table_attrs)]) + '\n')

    # write model attributes
    for i in models.index:
        out.write(' | '.join([str(j) for j in list(models.loc[i,\
                                                              models.columns])\
                             ]) + '\n')

    out.write('\n')
    out.close()

def gen_plot_md(section_header, out_txt_fname, current_path, out_plot_fname,\
                write_mode):
    
    # conditional delete/open markdown file
    out = open(out_txt_fname, write_mode)
    
    # write section header markdown
    section_header = '## ' + section_header
    out.write(section_header + '\n')
    
    # reference image
    out_plot_fname = out_plot_fname + '.png'
    git_sub_dir = current_path[-2]
    out.write('![alt text](' + str(git_sub_dir + '/' + out_plot_fname) + ')\n')
        
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

    h2o.init(max_mem_size_GB=12) # it may be necessary to start h2o outside of this script
               # you can use max_mem_size_GB to increase available memory
               # ex: h2o.init(max_mem_size_GB=<int>)
    h2o.cluster().show_status()

    ### run benchmark models
    cla_models = run_cla_models()
    reg_models = run_reg_models()

    h2o.cluster().shutdown()

    ### create pandas dfs
    table_header_list = ['Model Name', 'Model Description', 'Data Name',
                         'N', 'p', 'Logloss', 'RMSE', 'Accuracy']
    cla_models_df = pd.DataFrame(cla_models, columns=table_header_list)    
    
    cla_sum_df = cla_models_df.drop(['Model Description', 'Data Name','N',
                                       'p'], axis=1)
    cla_sum_df = cla_sum_df.apply(pd.to_numeric, errors='ignore')\
                           .groupby('Model Name', as_index=False).mean()                                  

    table_header_list = ['Model Name', 'Model Description', 'Data Name',
                         'N', 'p', 'Mean Residual Deviance', 'RMSE', 'R2']
    reg_models_df = pd.DataFrame(reg_models, columns=table_header_list)    

    reg_sum_df = reg_models_df.drop(['Model Description', 'Data Name','N',\
                                       'p',], axis=1)

    reg_sum_df = reg_sum_df.apply(pd.to_numeric, errors='ignore')\
                           .groupby('Model Name', as_index=False).mean()   

    ### create pandas/matplotlib figures

    ### cla bar
    bar_plot_df = cla_models_df.drop(['Model Description', 'N', 'p',\
                                      'Logloss', 'RMSE'], axis=1)
    bar_plot_df = bar_plot_df.sort_values(by=['Data Name', 'Accuracy']\
                                          , ascending=[True, False])\
                                          .groupby('Data Name').head(1)
                                   
    bar_plot_df['Count'] = 0
    bar_plot_df = bar_plot_df.drop(['Data Name', 'Accuracy'], axis=1)\
                                  .groupby('Model Name', as_index=False)\
                                  .count()
                                  
    bar_plot = bar_plot_df.plot.bar(x='Model Name', y='Count',\
                                    color=['r', 'b', 'g'], legend=False)
    bar_plot.set_ylabel('Count')
    fig = bar_plot.get_figure()
    fig.savefig('cla_bar_plot')

    ### cla scatter
    scatter_plot_df = cla_models_df.drop(['Model Description', 'Logloss',\
                                          'RMSE'], axis=1)
    scatter_plot_df = scatter_plot_df.sort_values(by=['Data Name', 'Accuracy']\
                                                  , ascending=[True, False])\
                                                  .groupby('Data Name').head(1)
                                            
    scatter_plot_df = scatter_plot_df.drop(['Data Name', 'Accuracy'], axis=1)   
    
    groups = scatter_plot_df.groupby('Model Name')
    fig, ax = plt.subplots()
    ax.margins(0.05)
    xlim_ = scatter_plot_df['N'].max()
    ylim_ = scatter_plot_df['p'].max()
    for name, group in groups:
        ax.plot(group.N, group.p, marker='o', linestyle='', ms=10, label=name)
        plt.xlim(0, xlim_)
        plt.ylim(0, ylim_)
    ax.legend(loc=4)
    fig.savefig('cla_scatter_plot')

    ### reg bar
    bar_plot_df = reg_models_df.drop(['Model Description', 'N', 'p',\
                                      'Mean Residual Deviance', 'RMSE'],\
                                      axis=1)
    bar_plot_df = bar_plot_df.sort_values(by=['Data Name', 'R2']\
                                          , ascending=[True, False])\
                                          .groupby('Data Name').head(1)
                                   
    bar_plot_df['Count'] = 0
    bar_plot_df = bar_plot_df.drop(['Data Name', 'R2'], axis=1)\
                                  .groupby('Model Name', as_index=False)\
                                  .count()
                                  
    bar_plot = bar_plot_df.plot.bar(x='Model Name', y='Count',\
                                    color=['r', 'b', 'g'], legend=False)
    bar_plot.set_ylabel('Count')
    fig = bar_plot.get_figure()
    fig.savefig('reg_bar_plot')

    ### reg scatter 
    scatter_plot_df = reg_models_df.drop(['Model Description',\
                                          'Mean Residual Deviance', 'RMSE'],\
                                          axis=1)
    scatter_plot_df = scatter_plot_df.sort_values(by=['Data Name', 'R2']\
                                                  , ascending=[True, False])\
                                                  .groupby('Data Name').head(1)
                                            
    scatter_plot_df = scatter_plot_df.drop(['Data Name', 'R2'], axis=1)   
    
    groups = scatter_plot_df.groupby('Model Name')
    fig, ax = plt.subplots()
    ax.margins(0.05)
    xlim_ = scatter_plot_df['N'].max()
    ylim_ = scatter_plot_df['p'].max()
    for name, group in groups:
        ax.plot(group.N, group.p, marker='o', linestyle='', ms=12, label=name)
        plt.xlim(0, xlim_)
        plt.ylim(0, ylim_)
    ax.legend(loc=4)
    fig.savefig('reg_scatter_plot')

                                       
    ### generate markdown
    gen_table_md(cla_models_df, 
                 'Python H20.ai Classification Models', 
                 out_txt_fname, 
                 'w+')

    gen_table_md(cla_sum_df, 
                 'Python H20.ai Classification Models Summary by Model', 
                 out_txt_fname, 
                 'a+')    

    gen_plot_md('Count of Best Models', 
                out_txt_fname, 
                current_path, 
                'cla_bar_plot',
                'a+')
                
    gen_plot_md('Best Models by N and p', 
                out_txt_fname, 
                current_path, 
                'cla_scatter_plot',
                'a+')

    gen_table_md(reg_models_df, 
                 'Python H20.ai Regression Models', 
                 out_txt_fname, 
                 'a+')

    gen_table_md(reg_sum_df, 
                 'Python H20.ai Regression Models Summary by Model', 
                 out_txt_fname, 
                 'a+')  

    gen_plot_md('Count of Best Models', 
                out_txt_fname, 
                current_path, 
                'reg_bar_plot',
                'a+')
                
    gen_plot_md('Best Models by N and p', 
                out_txt_fname, 
                current_path, 
                'reg_scatter_plot',
                'a+')


if __name__ == '__main__':
    main()
