# Deep Knowledge Tracing On Skills With Limited Data

## Abstract
Deep Knowledge Tracing (DKT), along with other machine
learning approaches, are biased toward data they have seen during the
training  step.  Thus  for  problems  where  we  have  few  amount  of  data
for a certain class, the models will tend to give good results on classes
where  there  are  many  examples,  and  poor  results  on  those  with  few
examples. This problem generally occurs when the classes to predict are
imbalanced and this is frequent in educational data where for example,
there  are  skills  that  are  very  difficult  or  very  easy  to  master.  There
will be less data on students that correctly answered questions related
to difficult knowledge and that incorrectly answered knowledge easy to
master.  In  that  case,  DKT  is  unable  to  correctly  predict  the  current
student’s  knowledge  on  those  skills.  In  order  to  improve  DKT  in  that
sense, we penalized the model using a cost-sensitive technique. In other
words, we have augmented the loss function with the same loss where
we have masked certain skills. We also included in the DKT, a Bayesian
Network (built from domain experts) by using the attention mechanism.
The resulting model is able to accurately track knowledge of students in
Logic-Muse Intelligent Tutoring System (ITS), compare to the BKT or
the original DKT
