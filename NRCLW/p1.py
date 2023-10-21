import numpy as np
import matplotlib.pyplot as plt
import pickle
from transitspectroscopy import spectroscopy as tspec
from poetss import poetss
import os
import time

# This file is to compute FWHM for all integrations for all visits

visit = 'NRCLW'
p1 = os.getcwd() + '/RateInts/Corr_' + visit
pout = os.getcwd() + '/NRCLW/Outputs/' + visit

## Segment!!!
segs = []
for i in range(6):
    if i < 9:
        segs.append('00' + str(i+1))
    else:
        segs.append('0' + str(i+1))

fwhm, super_fwhm = {}, {}
for seg in range(len(segs)):
    t1 = time.time()
    print('>>>> --- --- --- --- --- --- --- --- --- ---')
    print('>>>> --- Working on Segment ' + str(segs[seg]))
    print('>>>> --- --- --- --- --- --- --- --- --- ---')

    # Loading the data
    print('>>>> --- Loading the dataset...')
    corrected_data = np.load(p1 + '/Corrected_data_seg' + segs[seg] + '.npy')
    mask_bcr = np.load(p1 + '/Mask_bcr_seg' + segs[seg] + '.npy')
    print('>>>> --- Done!!')

    # Loading the trace positions
    
    print('>>>> --- Finding trace positions...')
    xstart, xend = 25, 1600
    cent_cub1 = poetss.find_trace_cof(clean_cube=corrected_data[:,4:,xstart:xend], margin=5)
    median_trace, dx1 = poetss.fit_multi_trace(cent_mat=cent_cub1, deg=2, clip=3)
    xpos = np.arange(xstart, xend, 1)
    print('>>>> --- Done!!')
    
    # And, computing the FWHM
    print('>>>> --- Finding the FWHM...')
    fwhm[segs[seg]], super_fwhm[segs[seg]] = tspec.trace_fwhm(tso=corrected_data[:,4:,:], x=xpos, y=median_trace, distance_from_trace=10)
    print('>>>> --- Done!!')

    t2 = time.time()
    print('>>>> --- Total time taken: {:.4f} minutes'.format((t2-t1)/60))
    print('>>>> --- --- --- --- --- --- --- --- --- ---')
    print('>>>> ---  Segment ' + str(segs[seg]) + ' analysis completed!!')
    print('>>>> --- --- --- --- --- --- --- --- --- ---')


pickle.dump(fwhm, open(pout + '/fwhm_' + visit + '.pkl','wb'))
pickle.dump(super_fwhm, open(pout + '/super_fwhm_' + visit + '.pkl','wb'))