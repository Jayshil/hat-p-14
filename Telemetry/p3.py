import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import mad_std
from glob import glob
import os

# This file is to compare telemetry data with white-light lightcurves

visit = 'NRCSW'

lst = ['INRC_FA_SC4_VDD_V', 'INRC_FA_SC0_CDRN_V', \
       'INRC_FA_SC4_DRN_V', 'INRC_FA_SC2_VDDA_V', 'INRC_FA_SC0_DRN_V']

# Loading the white-light lightcurve data
tim1, fl1, fle1 = np.loadtxt(os.getcwd() + '/NRCSW/Outputs/' + visit + '/Photometry_' + visit + '_photutils.dat', usecols=(0,1,2), unpack=True)
tim1 = tim1 + 2400000.5

for i in range(len(lst)):
    # Loading the telemetry data
    fsa = glob(os.getcwd() + '/Telemetry/Data/' + visit + '/' + lst[i] + '*')[0]
    sa1 = np.genfromtxt(fsa, delimiter=',')
    tim_sa1, val_sa = sa1[1:,1], sa1[1:,2]
    tim_sa1 = tim_sa1 + 2400000.5
    tim_sa, val_sa = tim_sa1[tim_sa1>=np.min(tim1)], val_sa[tim_sa1>=np.min(tim1)]
    med_sa, std_sa = np.nanmedian(val_sa), mad_std(val_sa)

    # And plotting them
    fig, axs = plt.subplots(2, 1, figsize=(16/1.5, 9/1.5), sharex=True)

    # Top panel (Science data)
    axs[0].errorbar(tim1, fl1, fmt='.', c='cornflowerblue', alpha=0.5, zorder=30)
    if visit == 'NRCSW':
       ## First jump
       axs[0].axvline(tim1[493], color='k', zorder=20, alpha=0.7)
       axs[0].axvline(tim1[606], color='k', zorder=20, alpha=0.7)
    axs[0].set_title('Short-wave lightcurve')

    # Bottom panel
    axs[1].errorbar(tim_sa, val_sa, c='cornflowerblue', zorder=30)
    if visit == 'NRCSW':
       ## First jump
       axs[1].axvline(tim1[493], color='k', zorder=20, alpha=0.7)
       axs[1].axvline(tim1[606], color='k', zorder=20, alpha=0.7)
    #axs[1].set_ylim([med_sa-7*std_sa, med_sa+7*std_sa])
    axs[1].set_title(lst[i])

    axs[1].set_xlim([np.min(tim1), np.max(tim1)])
    plt.show()