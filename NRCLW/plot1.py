import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import utils
import astropy.units as u
from astropy.timeseries import LombScargle

# This file is to visualise how FWHM varies with time

visit = 'NRCLW'
pin = os.getcwd() + '/NRCLW/Outputs/' + visit
pin2 = os.getcwd() + '/RateInts/Corr_' + visit

## Segment!!!
segs = []
for i in range(6):
    if i < 9:
        segs.append('00' + str(i+1))
    else:
        segs.append('0' + str(i+1))

fwhm_all = pickle.load(open(pin + '/fwhm_' + visit + '.pkl', 'rb'))
fwhm = fwhm_all[segs[0]]
super_fwhm_all = pickle.load(open(pin + '/super_fwhm_' + visit + '.pkl', 'rb'))
super_fwhm = super_fwhm_all[segs[0]]
times = np.load(pin2 + '/Times_bjd_seg001.npy')

for i in range(len(segs)-1):
    # Saving fwhm
    fwhm = np.vstack((fwhm, fwhm_all[segs[i+1]]))
    # Saving super fwhm
    super_fwhm = np.hstack((super_fwhm, super_fwhm_all[segs[i+1]]))
    # Saving time
    time_seg = np.load(pin2 + '/Times_bjd_seg' + segs[i+1] + '.npy')
    times = np.hstack((times, time_seg))

med_fwhm_along_time = np.nanmedian(fwhm, axis=0)
norm_fwhm = fwhm / med_fwhm_along_time[None,:]

med_fwhm = np.nanmedian(norm_fwhm, axis=1)
times = times + 2400000.5# - 2459702.

# For FWHM
tbin, fbin, febin, _ = utils.lcbin(time=times, flux=med_fwhm, binwidth=0.005)
tbin2, fbin2, febin2, _ = utils.lcbin(time=times, flux=med_fwhm, binwidth=0.05)

# For super FWHM
tbin_sf, fbin_sf, flebin_sf, _ = utils.lcbin(time=times, flux=super_fwhm, binwidth=0.005)
tbin_sf2, fbin_sf2, flebin_sf2, _ = utils.lcbin(time=times, flux=super_fwhm, binwidth=0.05)

# For Full model
fig, ax = plt.subplots(figsize=(16/1.5,9/1.5))

# Top panel
ax.errorbar(times, med_fwhm, fmt='-', c='cornflowerblue', alpha=0.5)
ax.errorbar(tbin, fbin, yerr=febin, fmt='.', c='gray', alpha=0.7, zorder=50)
ax.errorbar(tbin2, fbin2, yerr=febin2, fmt='o', c='navy', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=150)
ax.set_ylabel('FWHM', fontsize=14)
ax.set_xlabel('Time (BJD)', fontsize=14)
ax.set_xlim(np.min(times), np.max(times))
#ax.set_ylim(0.9990,1.0015)
plt.setp(ax.get_xticklabels(), fontsize=12)
plt.setp(ax.get_yticklabels(), fontsize=12)
ax.set_title('FWHM for Visit ' + str(visit), fontsize=15)
#plt.show()
plt.savefig(pin + '/Figures/Full_fwhm_' + visit + '.png', dpi=500)


# Finding PSD of the FWHM
def find_PSD(time, data):
    tim1 = time*24*60*60*u.second

    min_frequency = 1. / (np.ptp(tim1))   # inverse clock-time of a column
    max_frequency = (1. / (tim1[1]-tim1[0])) * 0.5
    freq = np.linspace(min_frequency, max_frequency, 100000)

    comps = 1e6*data*u.dimensionless_unscaled
    power = LombScargle(tim1,comps).power(frequency=freq)

    freq1 = freq[np.argsort(power.value)[-1]]
    per2 = 1/freq1
    per3 = per2.to(u.day)
    max_power_loc_min = per3.to(u.min)

    return freq, power, max_power_loc_min

def freq2min(x):
    x = 1/x * u.s
    return x.to(u.min).value

def min2freq(x):
    x = x*u.min
    x = 1/x
    return x.to(u.s**-1).value

freq, power, max_power = find_PSD(times, med_fwhm)
print('>>> --- Maximum power at {:.4f}'.format(max_power))

# Making the plot:
fig, axs = plt.subplots(figsize=(16/1.5, 9/1.5))

axs.plot(freq.value, power.value, color='orangered', alpha=0.9, label='PC 4')
#axs.axvline(freq1.value, ls='--', c='darkgreen')
axs.set_xscale('log')
axs.set_yscale('log')

axs.legend(loc='best')

axs.set_xlabel(r'Frequency [Hz]')
axs.set_ylabel(r'Power [ppm$^2$ Hz$^{-1}$] + Offset')
#axs.set_title('PSD of the PCA component' + str(nComp) + ' (Visit ' + visit[-1] + ')')

secax = axs.secondary_xaxis('top', functions=(freq2min, min2freq))
secax.set_xlabel(r'Time [min]')

axs.set_xlim([np.min(freq.value), np.max(freq.value)])
#plt.show()
plt.savefig(pin + '/Figures/Fwhm_PSD_' + visit + '.png', dpi=500)

# Super FWHM

# For Full model
fig, ax = plt.subplots(figsize=(16/1.5,9/1.5))

# Top panel
ax.errorbar(times, super_fwhm, fmt='.', c='cornflowerblue', alpha=0.5)
ax.errorbar(tbin_sf, fbin_sf, yerr=flebin_sf, fmt='.', c='gray', alpha=0.7, zorder=50)
ax.errorbar(tbin_sf2, fbin_sf2, yerr=flebin_sf2, fmt='o', c='navy', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=150)
ax.set_ylabel('FWHM', fontsize=14)
ax.set_xlabel('Time (BJD)', fontsize=14)
ax.set_xlim(np.min(times), np.max(times))
#ax.set_ylim(0.9990,1.0015)
plt.setp(ax.get_xticklabels(), fontsize=12)
plt.setp(ax.get_yticklabels(), fontsize=12)
ax.set_title('Super FWHM for Visit ' + str(visit), fontsize=15)
#plt.show()
plt.savefig(pin + '/Figures/Full_super_fwhm_' + visit + '.png', dpi=500)

f1 = open(pin + '/fwhm_full_' + visit + '.dat', 'w')
f1.write('# Median FWHM\t Super FWHM\n')
for i in range(len(med_fwhm)):
    f1.write(str(med_fwhm[i]) + '\t' + str(super_fwhm[i]) + '\n')
f1.close()