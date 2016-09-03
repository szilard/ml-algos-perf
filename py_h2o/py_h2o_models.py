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

def run_models():

    """ Contributors add h2o.ai models in this function.

    Returns:
        models: list of all contributed modeling scripts results.

    """

    ### list to contain all individual model results
    models = []

    ### placeholder example
    example_model_results = ['model name', 'model description', 'data name',
                             'data description', 0.0, 0.0]
    models.append(example_model_results)

    ### contributors add models below #########################################



    ###########################################################################

    return models

def gen_table_md(models, section_header, table_header_list, out_txt_fname):

    """ Generates markdown table containing results in output markdown file.

    Args:
        models: list of individual model results
        section_header: table title, second level header
        table_header_list: names of attributes in table as a list of strings
        out_txt_fname: determined name of output markdown file

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
        Executes contributed model scripts in run_models().
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
    models = run_models()

    # generate markdown
    gen_table_md(models, section_header, table_header_list, out_txt_fname)

if __name__ == '__main__':
    main()
