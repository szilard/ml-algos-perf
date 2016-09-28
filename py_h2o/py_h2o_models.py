# -*- coding: utf-8 -*-
"""
py_h2o_models.py

Contain models from h2o.ai package using Python interface

Contributors may add data to appropriate folder and modeling scripts as a
function in this file. All modeling scripts must be placed in a function that
returns a list with the following elements:

* model name
* model description
* data name
* N
* p
* relavent performance metrics

This resulting list should be appended to the models list in the appropriate
function:

- run_cla_models for classification tasks
- run_reg_models for regression

The results in the models list will be described automatically with a table and
plots in the main repo results.md file.

Run the gen_results.py script after adding models to this file (and testing
them) to regenerate results.md with the new, contributed model's results. This
run could take several hours.

# TODO: Report missingness and plot error v. missingness by type
# TODO: Linear, logistic regression, Naive Bayes
# TODO: Understand why some models are performing terribly on some data sets

"""

### imports
import inspect
import os
import re
import time
import h2o
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.grid.grid_search import H2OGridSearch

### global constants
DIV_BAR = '==================================================================='
SEED = 12345

### contributors add model functions here #####################################
# remember to append results to models list in:
# run_cla_models for classification tasks
# run_reg_models for regression



###############################################################################

def h2o_randomsearch_gbm(frames, y_name, x_names, dname):

    """ H2o GBM with parameter tuning.

    Args:
        frames[0], H2o training frame.
        frames[1], h2o validation frame.
        y_name: Target name.
        x_names: List of input names.
        dname: Name of data file.

    Returns:
        List of modeling results:

    """

    ### assign partitions
    tr_frame, v_frame = frames[0], frames[1]

    ### impute numeric
    ### categorical 'NA' treated as valid level
    tr_frame.impute(method='mean')
    v_frame.impute(method='mean')

    ### stdize
    tr_frame = h2o_stdize(tr_frame, y_name, x_names)
    v_frame = h2o_stdize(v_frame, y_name, x_names)

    ### define random grid search parameters and criteria
    hyper_parameters = {"ntrees":range(0, 100, 1),
                        "max_depth":range(0, 20, 1),
                        "sample_rate":[s/float(10) for s in range(1, 11)],
                        "col_sample_rate":[s/float(10) for s in range(1, 11)]}

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
                  validation_frame=v_frame)

    ### collect error measures
    if tr_frame[y_name].isfactor()[0]:
        measures = h2o_cla_err_measures(gsearch, tr_frame, v_frame, y_name)
    else:
        measures = h2o_reg_err_measures(gsearch)

    ### return appropriate list
    return ['GBM', 'GBM w/ random hyperparameter search', dname,\
            tr_frame.nrow, len(x_names), measures[0], measures[1], measures[2]]

def h2o_earlystop_rf(frames, y_name, x_names, dname):

    """ H2o RF with early stopping.

    Args:
        frames[0], H2o training frame.
        frames[1], H2o validation frame.
        y_name: Target name.
        x_names: List of input names.
        dname: Name of data file.

    Returns:
        List of modeling results:

    """

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
        seed=SEED)

    ### train rf model
    rf_model.train(x=x_names,
                   y=y_name,
                   training_frame=tr_frame,
                   validation_frame=v_frame)

    ### collect error measures
    if tr_frame[y_name].isfactor()[0]:
        measures = h2o_cla_err_measures(rf_model, tr_frame, v_frame, y_name)
    else:
        measures = h2o_reg_err_measures(rf_model)

    ### return appropriate list
    return ['RF', 'RF w/ early stopping', dname, tr_frame.nrow, len(x_names),
            measures[0], measures[1], measures[2]]

def h2o_randomsearch_nn(frames, y_name, x_names, dname):

    """ H2o neural network with parameter tuning.

    Args:
        frames[0], H2o training frame.
        frames[1], H2o validation frame.
        y_name: Target name.
        x_names: List of input names.
        dname: Name of data file.

    Returns:
        List of modeling results:

    """
    ### assign partitions
    tr_frame, v_frame = frames[0], frames[1]

    ### impute numeric
    ### categorical 'NA' treated as valid level
    tr_frame.impute(method='mean')
    v_frame.impute(method='mean')

    ### stdize
    tr_frame = h2o_stdize(tr_frame, y_name, x_names)
    v_frame = h2o_stdize(v_frame, y_name, x_names)

    ### define random grid search parameters and criteria
    hyper_parameters = {"hidden":[[17, 32], [8, 19], [32, 16, 8], [100],\
                                  [10, 10, 10, 10]],
                        "l1":[s/1e6 for s in range(1, 1001)],
                        "l2":[s/1e4 for s in range(1, 101)],
                        "input_dropout_ratio":[s/1e2 for s in range(1, 21)]}

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
                  validation_frame=v_frame)

    ### collect error measures
    if tr_frame[y_name].isfactor()[0]:
        measures = h2o_cla_err_measures(gsearch, tr_frame, v_frame, y_name)
    else:
        measures = h2o_reg_err_measures(gsearch)

    ### return appropriate list
    return ['NN', 'NN w/ random hyperparameter search', dname, tr_frame.nrow,
            len(x_names), measures[0], measures[1], measures[2]]

def h2o_cla_err_measures(model, tr_frame, v_frame, y_name):

    """ Collects classification error measures from a trained h2o classifier.

    Args:
        model: Trained h2o model object.
        tr_frame: H2o training frame.
        v_frame: H2o validation frame.
        y_name: Target name.

    Returns:
        A list containing logloss, rmse, and accuracy as strings.
    """

    ### conditionall retrieve best model from grid/random search
    try:
        bst_model_id = model.sort_by('logloss(valid=True)', False)\
                                    ['Model Id'][0]
        model = h2o.get_model(bst_model_id)
    except AttributeError:
        pass

    ### collect validation error measures (w/ reasonable precision)
    logloss = '%.3f' % model.logloss(valid=True)
    rmse = '%.3f' % model.rmse(valid=True)
    if tr_frame[y_name].nlevels()[0] > 2: # multinomial
        acc = '%.3f' % model.confusion_matrix(v_frame)['Error'][-1]
    else: # binary
        acc = '%.3f' % model.accuracy(thresholds=[0.5], valid=True)[0][-1]

    return [logloss, rmse, acc]

def h2o_reg_err_measures(model):

    """ Collects regression error measures from a trained h2o predictor.

    Args:
        model: Trained h2o model object.

    Returns:
        A list containing mean residual deviation, rmse, and r2 as
        strings.

    """

    ### conditionall retrieve best model from grid/random search
    try:
        bst_model_id = model.sort_by('residual_deviance(valid=True)', False)\
                         ['Model Id'][0]
        model = h2o.get_model(bst_model_id)
    except AttributeError:
        pass

    ### collect validation error measures (w/ reasonable precision)
    mrd = '%.3f' % model.mean_residual_deviance(valid=True)
    rmse = '%.3f' % model.rmse(valid=True)
    r2_ = '%.3f' % model.r2(valid=True)

    return [mrd, rmse, r2_]

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
            models.append(h2o_randomsearch_gbm(frames, y_name, x_names, dname))
            models.append(h2o_earlystop_rf(frames, y_name, x_names, dname))
            models.append(h2o_randomsearch_nn(frames, y_name, x_names, dname))
        except ValueError:
            print 'Warning: model training failure.'

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

        ### set target to numeric
        d_frame[y_name] = d_frame[y_name].asnumeric()

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
            models.append(h2o_randomsearch_gbm(frames, y_name, x_names,
                                               d_file.split(os.sep)[-1]))
            models.append(h2o_earlystop_rf(frames, y_name, x_names,
                                           d_file.split(os.sep)[-1]))
            models.append(h2o_randomsearch_nn(frames, y_name, x_names,
                                              d_file.split(os.sep)[-1]))
        except ValueError:
            print 'Warning: Model training failure.'

        del frames

        print '%s modeled in %.2f s.' % (d_file, time.time()-tic)

    return models

def pd_summarize_model_df(all_models_df):

    """ Summarizes pandas df of all model results of from a task (cla/reg).

    Args:
        all_models_df: Pandas df of all model results.

    """

    return all_models_df.drop(['Model Description',\
                               'Data Name', 'N', 'p'], axis=1)\
                        .apply(pd.to_numeric, errors='ignore')\
                        .groupby('Model Name', as_index=False)\
                        .mean()

def pd_bar_chart(drop_list, by_list, all_models_df, out_png_name):

    """ Generates bar chart from pandas df representing the number of times
    each model type is the best model for a data set.

    Args:
        drop_list: Columns to be dropped during first summarization.
        by_list: Columns by which to sort an intermediate summarized table,
                 later dropped.
        all_models_df: Pandas df of all model results.
        out_png_name: Name for generated plot, no '.png' necessary, must match
                      name in markdown results file.

    """

    ### drop superfluous cols
    ### find best model for each data set
    bar_plot_df = all_models_df.drop(drop_list, axis=1)
    bar_plot_df = bar_plot_df.sort_values(by=by_list\
                                          , ascending=[True, False])\
                                          .groupby('Data Name').head(1)

    ### count number of times model type is best model
    bar_plot_df['Count'] = 0
    bar_plot_df = bar_plot_df.drop(by_list, axis=1)\
                                  .groupby('Model Name', as_index=False)\
                                  .count()

    ### generate plot
    ### uniform color for each model in all plots
    bar_plot = bar_plot_df.plot.bar(x='', y='Count',\
                                    color=['r', 'b', 'g'], legend=False)
    bar_plot.set_ylabel('Count')
    fig = bar_plot.get_figure()
    fig.savefig(out_png_name)

def pd_scatter_plot_np(drop_list, by_list, all_models_df, out_png_name):

    """ Generates a scatter plot where the best model for a data set is
    plotted at (N, p) by model type.

    Args:
        drop_list: Columns to be dropped during first summarization.
        by_list: Columns by which to sort an intermediate summarized table,
                 later dropped.
        all_models_df: Pandas df of all model results.
        out_png_name: Name for generated plot, no '.png' necessary, must match
                      name in markdown results file.

    """

    ### drop superfluous cols
    ### find best model for each data set
    scatter_plot_df = all_models_df.drop(drop_list, axis=1)
    scatter_plot_df = scatter_plot_df.sort_values(by=by_list,\
                                                  ascending=[True, False])\
                                                  .groupby('Data Name').head(1)
    scatter_plot_df = scatter_plot_df.drop(by_list, axis=1)

    ### plot best model for each data set at (N, p)
    groups = scatter_plot_df.groupby('Model Name')
    fig, ax_ = plt.subplots()
    ms_ = 3 # markersize
    plt.xlabel('N')
    plt.ylabel('p')
    xlim_ = scatter_plot_df['N'].max() + (2 * ms_)
    ylim_ = scatter_plot_df['p'].max() + (2 * ms_)

    ### plot groups with appropriate color
    color_list = ['r', 'b', 'g'] # uniform color for each model in all plots
    c_idx = 0
    for name, group in groups:
        ax_.plot(group.N, group.p, marker='o', ms=ms_, label=name,
                 linestyle='', color=color_list[c_idx])
        plt.xlim(0 - (5 * ms_), xlim_)
        plt.ylim(0 - (2 * ms_), ylim_)
        c_idx += 1

    ax_.legend(loc=4)
    fig.savefig(out_png_name)

def pd_scatter_plot_measures(drop_list, by_list, all_models_df, out_png_name):

    """ Generates a scatter plot where the each model is plotted at
    (error measure 1, error measure 2) by model type.

    Args:
        drop_list: Columns to be dropped during first summarization.
        by_list: Columns by which to sort an intermediate summarized table,
                 later dropped.
        all_models_df: Pandas df of all model results.
        out_png_name: Name for generated plot, no '.png' necessary, must match
                      name in markdown results file.

    """

    ### coerce df cols to numeric
    ### to keep reported precision reasonable
    ### error measures originally reported as formatted strings
    all_models_df = all_models_df.apply(pd.to_numeric, errors='ignore')

    ### one regression task creates huge errors, should investigate,
    ### but for now ...
    if 'Accuracy' not in by_list:
        all_models_df = all_models_df[all_models_df['RMSE'] < 10]

    ### drop superfluous cols
    ### find best model for each data set
    scatter_plot_df = all_models_df.drop(drop_list, axis=1)
    scatter_plot_df = scatter_plot_df.sort_values(by=by_list,\
                                                  ascending=[True, False])\
                                                  .groupby('Data Name').head(1)
    scatter_plot_df = scatter_plot_df.drop(by_list, axis=1)
    ### plot best model for each data set at (measure 1, measure 2)
    groups = scatter_plot_df.groupby('Model Name')
    fig, ax_ = plt.subplots()
    ms_ = 3 # markersize

    plt.xlabel('RMSE')
    xlim_ = scatter_plot_df['RMSE'].max() + (0.5 * ms_)
    if 'Accuracy' in by_list:
        ylim_ = scatter_plot_df['Logloss'].max() + (0.5 * ms_)
        plt.ylabel('Logloss')
    else:
        ylim_ = scatter_plot_df['Mean Residual Deviance'].max() + (0.5 * ms_)
        plt.ylabel('Mean Residual Deviance')

    ### plot groups with appropriate color
    color_list = ['r', 'b', 'g']  # uniform color for each model in all plots
    c_idx = 0
    for name, group in groups:
        if 'Accuracy' in by_list:
            ax_.plot(group.RMSE, group.Logloss, marker='o', ms=ms_,\
                     linestyle='', label=name, color=color_list[c_idx])
        else:
            group = group.rename(columns={'Mean Residual Deviance':'MRD'})
            ax_.plot(group.RMSE, group.MRD, marker='o', ms=ms_,\
                     linestyle='', label=name, color=color_list[c_idx])
        plt.xlim(0, xlim_)
        plt.ylim(0, ylim_)
        c_idx += 1

    ax_.legend(loc=4)
    fig.savefig(out_png_name)

def gen_table_md(models, section_header, out_txt_fname, write_mode='a+'):

    """ Generates markdown table containing results in output markdown file.

    Args:
        models: pandas df of individual model results.
        section_header: Table title, second level header.
        table_header_list: Names of attributes in table as a list of strings.
        out_txt_fname: Determined name of output markdown file.
        write_mode: w+ or a+, starting a new file or appending, respectively.

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

def gen_plot_md(section_header, out_txt_fname, out_plot_fname,
                write_mode='a+'):

    """ Generates placeholder mark down for plot *.png files.

    Args:
        section_header: Table title, second level header.
        out_txt_name: Determined name of output markdown file.
        out_plot_fname: Filename for plot to be written to markdown file.
        write_mode: w+ or a+, starting a new file or appending, respectively.

    """

    # conditional delete/open markdown file
    out = open(out_txt_fname, write_mode)

    # write section header markdown
    section_header = '## ' + section_header
    out.write(section_header + '\n')

    # reference image
    out_plot_fname = out_plot_fname + '.png'
    out.write('![alt text](py_h2o/' + str(out_plot_fname) + ')\n')

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

    # it may be necessary to start h2o outside of this script
    # you can use max_mem_size_GB to increase available memory
    # ex: h2o.init(max_mem_size_GB=<int>)
    h2o.init()

    h2o.cluster().show_status()

    ### run benchmark models
    cla_models = run_cla_models()
    reg_models = run_reg_models()

    h2o.cluster().shutdown()

    ### create pandas dfs from results lists
    cla_all_models_df = pd.DataFrame(cla_models,
                                     columns=['Model Name',\
                                     'Model Description', 'Data Name', 'N',\
                                     'p', 'Logloss', 'RMSE', 'Accuracy'])
    cla_all_models_df.to_csv('cla_models.csv') # save for external analysis
    cla_summary_df = pd_summarize_model_df(cla_all_models_df)

    reg_all_models_df = pd.DataFrame(reg_models,
                                     columns=['Model Name',\
                                     'Model Description', 'Data Name', 'N',\
                                     'p', 'Mean Residual Deviance', 'RMSE',\
                                     'R2'])
    reg_all_models_df.to_csv('reg_models.csv') # save for external analysis
    reg_summary_df = pd_summarize_model_df(reg_all_models_df)

    ### create pandas/matplotlib figures
    matplotlib.style.use('ggplot')

    ### bar charts
    pd_bar_chart(['Model Description', 'N', 'p', 'Logloss', 'RMSE'],
                 ['Data Name', 'Accuracy'],
                 cla_all_models_df,
                 'classification_bar_plot')

    pd_bar_chart(['Model Description', 'N', 'p', 'Mean Residual Deviance',\
                  'RMSE'],
                 ['Data Name', 'R2'],
                 reg_all_models_df,
                 'regression_bar_plot')

    ### scatter plots
    pd_scatter_plot_np(['Model Description', 'Logloss', 'RMSE'],
                       ['Data Name', 'Accuracy'],
                       cla_all_models_df,
                       'classification_scatter_plot_Np')

    pd_scatter_plot_measures(['Model Description', 'N', 'p'],
                             ['Data Name', 'Accuracy'],
                             cla_all_models_df,
                             'classification_scatter_plot_measures')

    pd_scatter_plot_np(['Model Description', 'Mean Residual Deviance',\
                        'RMSE'],
                       ['Data Name', 'R2'],
                       reg_all_models_df,
                       'regression_scatter_plot_Np')

    pd_scatter_plot_measures(['Model Description', 'N', 'p'],
                             ['Data Name', 'R2'],
                             reg_all_models_df,
                             'regression_scatter_plot_measures')

    ### generate markdown
    write_mode_ = 'w+' # erase file in first call
    for task_desc in ['classification', 'regression']:

        task_desc_df_dict = {'classification': [cla_all_models_df,
                                                cla_summary_df],
                             'regression': [reg_all_models_df,
                                            reg_summary_df]}

        gen_table_md(task_desc_df_dict[task_desc][0],
                     'Python H20.ai ' + task_desc.capitalize() + ' Models',
                     out_txt_fname,
                     write_mode=write_mode_)

        gen_table_md(task_desc_df_dict[task_desc][1],
                     'Python H20.ai ' + task_desc.capitalize() +\
                     ' Models Summary by Model',
                     out_txt_fname)

        gen_plot_md('Count of Best Models',
                    out_txt_fname,
                    task_desc + '_bar_plot')

        gen_plot_md('Best Models by N and p',
                    out_txt_fname,
                    task_desc + '_scatter_plot_Np')

        gen_plot_md('Best Model Error Measures',
                    out_txt_fname,
                    task_desc + '_scatter_plot_measures')

        # append to file in other calls
        write_mode_ = 'a+'

if __name__ == '__main__':
    main()
