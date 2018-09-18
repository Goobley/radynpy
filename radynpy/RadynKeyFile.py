import os
os.environ['CDF_LIB'] = '/usr/local/cdf/lib'
from spacepy import pycdf
import pickle
from radynpy.AuxTypes import Val, Array

cdfFile = '/local1/scratch/RadynGrid/radyn_out.val3c_d3_1.0e11_t20s_10kev_fp'

res = {}
with pycdf.CDF(cdfFile) as cdf:
    for k in cdf:
        if len(cdf[k].shape) == 0:
            # When the shape is () we have a 0-d ndarray in cdf[k][...]. 
            # The only way to get the single value is with .item()
            res[k] = Val(cdf[k][...].item())
        else:
            res[k] = Array(cdf[k][...].shape)
    # Add ntime, because it's a useful value
    res['ntime'] = Val(cdf['time'].shape[0])
    # And max number of atomic levels
    res['maxatomlevels'] = Val(cdf['nk'][...].max())

        
with open('RadynKeySizes.pickle', 'wb') as p:
    pickle.dump(res, p)
