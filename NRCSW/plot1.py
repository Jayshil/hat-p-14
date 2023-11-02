import numpy as np
import matplotlib.pyplot as plt
import utils
import os

# This file is to visualise principal components of SW photometry data
# However, I am overplotting some binning scheme to better visualise the trends

visit = 'NRCSW'
pin = os.getcwd() + '/NRCSW/Outputs/' + visit

# Loading the principal components
pcs = np.load(pin + '/PCs.npy')
tim = np.loadtxt(pin + '/Photometry_' + visit + '_photutils.dat', usecols=0, unpack=True)
tim = tim + 2400000.5

## Location of jump events
jump1, jump2 = 2459701.941132727, 2459701.9777086237
hgmove = (2459701.9236214 + 2459701.9237162) / 2

for i in range(1):
    # Binning scheme
    tbin, flbin, flebin, _ = utils.lcbin(time=tim, flux=pcs[i,:], binwidth=0.0005)
    tbin2, flbin2, flebin2, _ = utils.lcbin(time=tim, flux=pcs[i,:], binwidth=0.005)

    med, std = np.nanmedian(pcs[i,:]), np.nanstd(pcs[i,:])

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(16/1.5, 9/1.5))
    axs.errorbar(tim, pcs[i,:], fmt='.', c='cornflowerblue', alpha=0.25)
    axs.errorbar(tbin, flbin, yerr=flebin, fmt='.', c='gray', alpha=0.7, zorder=50)
    axs.errorbar(tbin2, flbin2, yerr=flebin2, fmt='o', c='navy', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=150)
    axs.set_ylim([med-3*std, med+3*std])
    axs.set_xlim([np.min(tim), np.max(tim)])

    axs.axvline(jump1, color='k', ls='--', zorder=5, label='Jump 1')
    axs.axvline(jump2, color='k', ls='--', zorder=5, label='Jump 2')
    axs.axvline(hgmove, color='k', ls=':', zorder=5, label='High-gain antenna move')

    axs.legend(loc='best')

    axs.set_xlabel('Time (BJD)', fontsize=14)
    axs.set_ylabel('PC' + str(i+1), fontsize=14)
    plt.show()

fig, axs = plt.subplots(figsize=(15/1.75, 5/1.75))

axs.plot(tim, pcs[0,:], c='cornflowerblue', zorder=10)
axs.axvline(jump1, color='k', ls='--', zorder=5, label='Jump 1')
axs.axvline(jump2, color='k', ls='--', zorder=5, label='Jump 2')
axs.axvline(hgmove, color='k', ls=':', zorder=5, label='High-gain antenna move')

axs.set_xlabel('Time (BJD)', fontsize=18)
axs.set_ylabel('1st PC', fontsize=18)

plt.setp(axs.get_xticklabels(), fontsize=16)
plt.setp(axs.get_yticklabels(), fontsize=16)

plt.tight_layout()
plt.savefig(pin + '/Figures_PCA/PC1_1442.pdf')
#plt.show()