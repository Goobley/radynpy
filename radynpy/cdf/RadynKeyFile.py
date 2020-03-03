import os
import cdflib
import pickle
from radynpy.cdf.auxtypes import Val, Array
import numpy as np

cdfFile = '/data/crisp/RadynGrid/radyn_out.val3c_d3_1.0e11_t20s_10kev_fp'

res = {}
cdf = cdflib.CDF(cdfFile)
for k in cdf.cdf_info()['zVariables']:
    var = cdf.varget(k)

    if len(var.shape) == 0:
        # When the shape is () we have a 0-d ndarray in cdf[k][...]. 
        # The only way to get the single value is with .item()
        res[k] = Val(var.item())
    else:
        res[k] = Array(var.shape)
# Add ntime, because it's a useful value
res['ntime'] = Val(cdf.varget('time').shape[0])
# And max number of atomic levels
res['maxatomlevels'] = Val(cdf.varget('nk').max())
cdf.close()

        
with open('RadynKeySizes.pickle', 'wb') as p:
    pickle.dump(res, p)
