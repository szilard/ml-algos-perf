
## Performance of Machine Learning Algorithms

This repo is intended as a *playground* for experimentation with various machine 
learning algorithms (primarily supervised learning) in order to understand their performance characteristics
as a function of the attributes of the datasets used for training such as size, structure, sparsity,
type of variables (numeric/categorical), special types of data (e.g. images, text),
complexity, signal/noise ratio etc.

Despite the no-free lunch theorem, there are a handful of supervised learning algorithms that are 
performing the best (using various accuracy measures) on a wide range of datasets encountered in
practice - see e.g. [academic research](https://www.cs.cornell.edu/~caruana/ctp/ct.papers/caruana.icml06.pdf)
or the results of many Kaggle competitions. These algorithms are random forests, gradient boosting machine,
support vector machines and deep learning neural networks - and furthermore,
combining them into ensembles provides usually some additional increase in accuracy. 
The goal of this project is to understand the performance characteristics of these methods.

This repo/research will start with 2 main tasks:

1. Make a broad plan for the subject of study of this research in terms of algorithms, implementations, tuning
strategies, data attributes to be varied, datasets, accuracy measures etc.

2. Make a "minimal viable product" that is study the behavior of a limited set of algos with
a limited number of implementations and 
limited tuning on *simulated data* with changing a limited variety of characteristics 
(e.g. dataset size, signal/noise ratio etc.)
using a limited number of accuracy measures (e.g. AUC). The work has started 
[here](simul).

Then we'll try to tackle the various directions in #1 above.



