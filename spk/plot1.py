import numpy as np
import matplotlib.pyplot as plt
import os
import utils

visit = 'NRCSW'
pout = os.getcwd() + '/spk/Outputs/' + visit

## Location of jump events
jump1, jump2 = 2459701.941132727, 2459701.9777086237
hgmove = (2459701.9236214 + 2459701.9237162) / 2

# Loading the data
tim, fbin, fbinerr = np.loadtxt(pout + '/Guide_star_binned_' + visit + '.dat', usecols=(0,1,2), unpack=True)
## Binning them (for plotting purposes)
tbin3, fg_flbin3, fg_flebin3, _ = utils.lcbin(time=tim, flux=fbin, binwidth=0.0005)
tbin4, fg_flbin4, fg_flebin4, _ = utils.lcbin(time=tim, flux=fbin, binwidth=0.005)

# And plotting
fig, ax = plt.subplots(figsize=(16/1.5,9/1.5))
ax.errorbar(tim, fbin, fmt='.', c='cornflowerblue', alpha=0.25, zorder=10)
ax.errorbar(tbin3, fg_flbin3, yerr=fg_flebin3, fmt='.', c='gray', alpha=0.7, zorder=50)
ax.errorbar(tbin4, fg_flbin4, yerr=fg_flebin4, fmt='o', c='navy', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=150)

ax.axvline(jump1, color='k', ls='--', zorder=20, alpha=0.7)
ax.axvline(jump2, color='k', ls='--', zorder=20, alpha=0.7)
ax.axvline(hgmove, color='k', ls=':', zorder=20, alpha=0.7)

plt.setp(ax.get_xticklabels(), fontsize=16)
plt.setp(ax.get_yticklabels(), fontsize=16)

ax.set_title('FGS Guide Star Flux (COM/NIRCam 1442)', fontsize=20)
ax.set_ylabel('Relative Flux', fontsize=18)
ax.set_xlabel('Time (BJD)', fontsize=18)
ax.set_xlim(np.min(tim), np.max(tim))

#plt.show()
plt.savefig(os.getcwd() + '/spk/Figures/Binned_fl_1442.pdf')