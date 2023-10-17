import numpy as np
import matplotlib.pyplot as plt
import juliet
import os
import utils as utl
import matplotlib.gridspec as gd
import multiprocessing
multiprocessing.set_start_method('fork')

# This file is to analyse photometric light curves for SW channel

visit = 'NRCSW'
catwoman = False

# Input/Output folder
pin = os.getcwd() + '/NRCSW/Outputs/' + visit
pout = os.getcwd() + '/NRCSW/Analysis/PhotometryLC'

# Data files
tim, fl, fle = {}, {}, {}
lin_pars = {}

tim_all, fl_all, fle_all = np.loadtxt(pin + '/Photometry_' + visit + '_photutils.dat', usecols=(0,1,2), unpack=True)
cenr_all, cenc_all, bkg_all = np.loadtxt(pin + '/Photometry_' + visit + '_photutils.dat', usecols=(3,4,5), unpack=True)

# And the final lightcurve
tim9, fl9, fle9 = tim_all, fl_all, fle_all
tim9 = tim9 + 2400000.5

# Removing Nan values
tim7, fl7, fle7 = tim9[~np.isnan(fl9)], fl9[~np.isnan(fl9)], fle9[~np.isnan(fl9)]
cenr7, cenc7 = cenr_all[~np.isnan(fl9)], cenc_all[~np.isnan(fl9)]

# Outlier removal
msk2 = utl.outlier_removal(tim7, fl7, fle7, clip=5, msk1=True, verbose=False)
tim7, fl7, fle7 = tim7[msk2], fl7[msk2], fle7[msk2]
cenr7, cenc7 = cenr7[msk2], cenc7[msk2]

# Outlier removal from centroids
## From row centroids
msk3 = utl.outlier_removal_ycen(cenr7, clip=5., verbose=False)
tim7, fl7, fle7 = tim7[msk3], fl7[msk3], fle7[msk3]
cenr7, cenc7 = cenr7[msk3], cenc7[msk3]
## From column centroids
msk4 = utl.outlier_removal_ycen(cenc7, clip=5., verbose=False)
tim7, fl7, fle7 = tim7[msk4], fl7[msk4], fle7[msk4]
cenr7, cenc7 = cenr7[msk4], cenc7[msk4]

# Trimming first 45 integrations
#tim7, fl7, fle7 = tim7[45:], fl7[45:], fle7[45:]
#cenr7, cenc7 = cenr7[45:], cenc7[45:]

# Note down how many points actually got removed in this process
print('>>>> --- Total number of points removed: ', len(tim_all) - len(tim7))
print('>>>> --- Total per cenr of points removed: {:.4f} %'.format(100 * (len(tim_all) - len(tim7)) / len(tim_all)))

# Saving them!
tim[visit], fl[visit], fle[visit] = tim7, fl7/np.median(fl7), fle7/np.median(fl7)

# Linear regressors
lins = np.vstack([(tim7-np.median(tim7))/np.std(tim7), (cenr7-np.median(cenr7))/np.std(cenr7),\
                    (cenc7-np.median(cenc7))/np.std(cenc7)])
lin_pars[visit] = np.transpose(lins)


# Some planetary parameters
per, per_err = 4.62766172, 0.00000046                       # Ivshina & Winn 2022
bjd0, bjd0_err = 2457138.21636, 0.00018                     # Ivshina & Winn 2022
ar, ar_err = 8.90, 0.36                                     # Stassun et al. 2017
inc, inc_err = 83.50, 0.30                                  # Bonomo et al. 2017
bb, bb_err = 0.9072, (0.0057+0.0051)/2                      # Fukui et al. 2016
ecc, omega = 0.1074, 106.1                                  # Bonomo et al. 2017

cycle = round((tim[visit][0]-bjd0)/per)
tc1 = np.random.normal(bjd0, bjd0_err, 100000) + (cycle*np.random.normal(per, per_err, 100000))

## Priors
### Planetary parameters
par_P = ['P_p1', 't0_p1', 'b_p1', 'q1_' + visit, 'q2_' + visit, 'ecc_p1', 'omega_p1', 'a_p1']
dist_P = ['fixed', 'normal', 'truncatednormal', 'uniform', 'uniform', 'fixed', 'fixed', 'normal']
hyper_P = [per, [np.median(tc1), np.std(tc1)], [bb, 3*bb_err, 0., 1.], [0., 1.], [0., 1.], ecc, omega, [ar, 3*ar_err]]
if not catwoman:
    par_P = par_P + ['p_p1_' + visit]
    dist_P = dist_P + ['uniform']
    hyper_P = hyper_P + [[0., 1.]]
else:
    par_P = par_P + ['p1_p1_' + visit, 'p2_p1_' + visit, 'phi_p1']
    dist_P = dist_P + ['uniform', 'uniform', 'fixed']
    hyper_P = hyper_P + [[0., 1.], [0., 1.], 90.]

### Instrumental and linear parameters
par_lin, dist_lin, hyper_lin = [], [], []
## Instrumental parameters
par_ins = ['mdilution_' + visit, 'mflux_' + visit, 'sigma_w_' + visit]
dist_ins = ['fixed', 'normal', 'loguniform']
hyper_ins = [1.0, [0., 0.1], [0.1, 10000.]]
## Linear parameters
for j in range(lin_pars[visit].shape[1]):
    par_lin.append('theta' + str(j) + '_' + visit)
    dist_lin.append('uniform')
    hyper_lin.append([-1., 1.])


# Total priors
par_tot = par_P + par_ins + par_lin
dist_tot = dist_P + dist_ins + dist_lin
hyper_tot = hyper_P + hyper_ins + hyper_lin

priors = juliet.utils.generate_priors(par_tot, dist_tot, hyper_tot)

## And fitting
dataset = juliet.load(priors=priors, t_lc=tim, y_lc=fl, yerr_lc=fle, linear_regressors_lc=lin_pars, #GP_regressors_lc=gp_pars
                      out_folder=pout)
res = dataset.fit(sampler = 'dynesty', nthreads=8)


# Some plots
model = res.lc.evaluate(visit)

# Binned datapoints
tbin, flbin, flebin, _ = utl.lcbin(time=tim[visit], flux=fl[visit], binwidth=0.003)

# Let's make sure that it works:
fig = plt.figure(figsize=(16,9))
gs = gd.GridSpec(2,1, height_ratios=[2,1])

# Top panel
ax1 = plt.subplot(gs[0])
ax1.errorbar(tim[visit], fl[visit], yerr=fle[visit], fmt='.', alpha=0.3)
ax1.errorbar(tbin, flbin, yerr=flebin, fmt='o', color='red', zorder=10)
ax1.plot(tim[visit], model, c='k', zorder=100)
ax1.set_ylabel('Relative Flux')
ax1.set_xlim(np.min(tim[visit]), np.max(tim[visit]))
ax1.xaxis.set_major_formatter(plt.NullFormatter())

# Bottom panel
ax2 = plt.subplot(gs[1])
ax2.errorbar(tim[visit], (fl[visit]-model)*1e6, yerr=fle[visit]*1e6, fmt='.', alpha=0.3)
ax2.axhline(y=0.0, c='black', ls='--', zorder=100)
ax2.set_ylabel('Residuals (ppm)')
ax2.set_xlabel('Time (BJD)')
ax2.set_xlim(np.min(tim[visit]), np.max(tim[visit]))

plt.savefig(pout + '/full_model_' + visit + '.png')

residuals = fl[visit]-model
rms, stderr, binsz = utl.computeRMS(residuals, binstep=1)
normfactor = 1e-6

plt.figure(figsize=(8,6))
plt.plot(binsz, rms / normfactor, color='black', lw=1.5,
                label='Fit RMS', zorder=3)
plt.plot(binsz, stderr / normfactor, color='red', ls='-', lw=2,
                label=r'Std. Err. ($1/\sqrt{N}$)', zorder=1)
plt.xlim(0.95, binsz[-1] * 2)
plt.ylim(stderr[-1] / normfactor / 2., stderr[0] / normfactor * 2.)
plt.xlabel("Bin Size (N frames)", fontsize=14)
plt.ylabel("RMS (ppm)", fontsize=14)
plt.xscale('log')
plt.yscale('log')
plt.legend(loc='best')
plt.savefig(pout + '/alan_deviation_' + visit + '.png')

utl.corner_plot(pout, False)