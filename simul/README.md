
## Performance of Machine Learning Algorithms 

### A "Minimal Viable Product"

We study here the behavior of a limited set of algos (with
a limited number of open source implementations) and using 
limited tuning on *simulated data* with changing a limited variety of characteristics 
(e.g. dataset size, signal/noise ratio etc.)
using a limited number of accuracy measures (e.g. AUC). 

We generate data inspired by [Hastie etal](http://statweb.stanford.edu/~tibs/ElemStatLearn/) Example 10.2:
`n` examples  of `p` Gaussian features (but folded to `x_i>0`) which are classified
whether they are inside or outside the sphere that divides the number of points roughly equally.

R code is [here](hastie10-2.R). We ran logistic regression (glmnet) and GBM (xgboost) and obtained
AUC 0.9689 and 0.9813, respectively. Next it's time to change `n`, `p` and signal/noise characteristics
and see how the AUC of these 2 algos performs...




