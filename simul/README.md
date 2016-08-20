
## Performance of Machine Learning Algorithms 

### A "Minimal Viable Product"

We study here the behavior of a limited set of algos with a limited number of implementations and 
limited tuning on *simulated data* with changing a limited variety of characteristics 
(e.g. dataset size, signal/noise ratio etc.) using a limited number of accuracy measures (e.g. AUC). 

We generate data inspired by [Hastie etal](http://statweb.stanford.edu/~tibs/ElemStatLearn/) Example 10.2
that is `p` Gaussian features (but folded to `x_i>0`). We generate `n` examples which are classified
whether they are inside or ouside the sphere that divides the number of points equally.

R code is [here](hastie10-2.R). We ran logistic regression (glmnet) and GBM (xgboost) and obtained...



