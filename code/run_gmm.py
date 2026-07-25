#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pycave
from pycave.clustering import KMeans
import numpy as np
import torch
import sys
np.Inf = np.inf

all_npy = sys.argv[1]
print(all_npy)
k =  int(sys.argv[2])
print("k,", k)

out_dir =  (sys.argv[3])
gmm_out =  (sys.argv[4])

X_list =[]
for sname in all_npy.split(","):
    X_list = X_list+[np.load(sname)]

X = np.concatenate(X_list)

print("X.shape",X.shape)

from  pycave.bayes import GaussianMixture
import random
#import matplotlib.pyplot as plt
import pathlib
npy_dir= out_dir



sample_size = [x.shape[0] for x in X_list]
print(sample_size)

k_list = [k]
estimator_gmms = [GaussianMixture(x, trainer_params=dict(accelerator='cpu', devices=1,
                                                       enable_progress_bar = True),
                                init_strategy = "kmeans++", batch_size = 15000000) for x in k_list]
for estimator_gmm in estimator_gmms:
    estimator_gmm.fit(torch.from_numpy(X))
cluster_id_gmms = [estimator_gmm.predict(torch.from_numpy(X)) for estimator_gmm in estimator_gmms]
## Score of samples
#     gmm13_scores_list = [estimator_gmm.score_samples(X) for estimator_gmm in estimator_gmms]
#     pi_x_list = [estimator_gmm.predict_proba(X) for estimator_gmm in estimator_gmms]

for list_i,k in enumerate(k_list): 
    start_i = 0
    #sample_score_gmm = list()
    sample_cluster_gmm = list()
    #pi_x_cluster_gmm = list()
    out_dir = npy_dir+"gmm"+str(k)+"_trained/"
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    estimator_gmms[list_i].save(out_dir+"model_fit/")
    for s_size, out_file in zip(sample_size, gmm_out.split(",")):
        #sample_score_gmm.append(gmm13_scores_list[list_i].numpy()[0,][start_i:(start_i+s_size)])
        sample_cluster_gmm.append( cluster_id_gmms[list_i][start_i:(start_i+s_size)])
        #pi_x_cluster_gmm.append( pi_x_list[list_i][start_i:(start_i+s_size),:])
        start_i = start_i+s_size

    for c_id, out_file in zip(sample_cluster_gmm, gmm_out.split(",")):
        np.savetxt(out_file, c_id.numpy().astype(int), fmt="%s")
       # np.savetxt(out_dir+f_name+'_noMeans_+'t_name+'_gmm_score.txt',score_gmm)



