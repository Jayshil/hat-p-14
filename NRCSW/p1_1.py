import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from photutils.aperture import CircularAperture
from photutils.aperture import aperture_photometry as aphot
import matplotlib.gridspec as gd
from tqdm import tqdm
import os
from path import Path
from exotoolbox.utils import tdur
from utils import lcbin

# This file is to find aperture photmetry of the mirror segments

visit = 'NRCSW'

pin = os.getcwd() + '/RateInts/Corr_' + visit
pout = os.getcwd() + '/NRCSW/Outputs/' + visit + '/Mirror'

segs = []
for i in range(6):
    if i < 9:
        segs.append('00' + str(i+1))
    else:
        segs.append('0' + str(i+1))

nseg, nint = np.random.choice(segs), np.random.randint(0,100)
aprad, cen_r, cen_c = 10., 90., 568.     # Aperture radius, and position of centroids *in case of __circular__ aperture*

example_data = np.load(pin + '/Corrected_data_seg' + nseg + '.npy')
example_data1 = example_data[nint,:,:]
fig, ax = plt.subplots(figsize=(15,5))
im = ax.imshow(example_data1)#, cmap='plasma')
Aperture = Wedge((cen_c, cen_r), aprad, 0, 360, color='cyan', fill=False)
ax.add_patch(Aperture)
ax.set_title('Example data with aperture for Segment ' + nseg + ', Int ' + str(nint))
plt.savefig(pout + '/Mir_Ap_aprad=' + str(aprad) + '_cenr=' + str(cen_r) + '_cenc=' + str(cen_c) + '.png')
#plt.show()

# Arrays to store the data products
tim_all, fl_all, fle_all = np.array([]), np.array([]), np.array([])

for i in range(len(segs)):
    fphoto = Path(pout + '/Mir_Ap_aprad=' + str(aprad) + '_cenr=' + str(cen_r) + '_cenc=' + str(cen_c) + '.dat')
    if fphoto.exists():
        print('>>>> --- Looks like the data already exists...')
        continue
    else:
        pass
    seg = segs[i]
    # Loading the data
    print('>>>> --- Working on Segment ' + str(seg))
    # Loading the data
    corrected_data = np.load(pin + '/Corrected_data_seg' + seg + '.npy')
    corrected_errs = np.load(pin + '/Corrected_errors_seg' + seg + '.npy')
    times_bjd = np.load(pin + '/Times_bjd_seg' + seg + '.npy')
    # Saving times
    tim_all = np.hstack((tim_all, times_bjd))
    for integration in tqdm(range(corrected_data.shape[0])):
        # Simply computing the flux inside an aperture 
        circ_aper = CircularAperture((cen_c, cen_r), r=aprad)
        ap_phot = aphot(data=corrected_data[integration,:,:], apertures=circ_aper, error=corrected_errs[integration,:,:])
        ape_flx, ape_err = ap_phot['aperture_sum'][0], ap_phot['aperture_sum_err'][0]
        # Saving them
        fl_all, fle_all = np.hstack((fl_all, ape_flx)), np.hstack((fle_all, ape_err))

# Saving the whole dataset (or, loading it, if it already exists)
fphoto = Path(pout + '/Mir_Ap_aprad=' + str(aprad) + '_cenr=' + str(cen_r) + '_cenc=' + str(cen_c) + '.dat')
if fphoto.exists():
    tim_all, fl_all, fle_all = np.loadtxt(pout + '/Mir_Ap_aprad=' + str(aprad) + '_cenr=' + str(cen_r) + '_cenc=' + str(cen_c) + '.dat', usecols=(0,1,2), unpack=True)
else:    
    fname = open(fphoto, 'w')
    for i in range(len(tim_all)):
        fname.write(str(tim_all[i]) + '\t' + str(fl_all[i]) + '\t' + str(fle_all[i]) + '\n')
    fname.close()

tim_all = tim_all + 2400000.5

tbin, flbin, flebin, _ = lcbin(time=tim_all, flux=fl_all/np.median(fl_all), binwidth=0.0005)
tbin2, flbin2, flebin2, _ = lcbin(time=tim_all, flux=fl_all/np.median(fl_all), binwidth=0.005)

# For Full model
fig = plt.figure(figsize=(16/1.5,9/1.5))
gs = gd.GridSpec(2,1, height_ratios=[2,5])

# Top panel
axtop = plt.subplot(gs[0])
im = axtop.imshow(example_data1, aspect='equal')#, cmap='plasma')
Aperture = Wedge((cen_c, cen_r), aprad, 0, 360, color='cyan', fill=False)
axtop.add_patch(Aperture)
axtop.set_title('Example data with aperture for Segment ' + nseg + ', Int ' + str(nint), fontsize=15)
plt.setp(axtop.get_xticklabels(), fontsize=12)
plt.setp(axtop.get_yticklabels(), fontsize=12)

# Bottom panel
ax = plt.subplot(gs[1])
ax.errorbar(tim_all, fl_all/np.median(fl_all), fmt='.', c='cornflowerblue', alpha=0.25)
ax.errorbar(tbin, flbin, yerr=flebin, fmt='.', c='gray', alpha=0.7, zorder=50)
ax.errorbar(tbin2, flbin2, yerr=flebin2, fmt='o', c='navy', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=150)
ax.set_ylabel('Relative Flux', fontsize=14)
ax.set_xlabel('Time (BJD)', fontsize=14)
ax.set_xlim(np.min(tim_all), np.max(tim_all))
#ax.set_ylim(0.9990,1.0015)
plt.setp(ax.get_xticklabels(), fontsize=12)
plt.setp(ax.get_yticklabels(), fontsize=12)
ax.set_title('Photometry for Visit ' + str(visit[-1]) + ', Aprad: ' + str(aprad) + ', Centroids: (' + str(cen_r) + ', ' + str(cen_c) + ')', fontsize=15)
plt.tight_layout()
#plt.show()
plt.savefig(pout + '/Mir_photo_aprad=' + str(aprad) + '_cenr=' + str(cen_r) + '_cenc=' + str(cen_c) + '.png', dpi=500)