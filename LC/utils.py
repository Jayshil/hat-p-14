import numpy as np
import matplotlib.pyplot as plt
import juliet
from astropy.stats import SigmaClip, mad_std
import re
import corner
from glob import glob
import pickle
from poetss import poetss
import batman

def outlier_removal(tims, flx, flxe, clip=5, msk1=True, verbose=True):
    # Let's first mask transits and occultations
    if msk1==True:
        per, T0 = 4.62766172, 2457138.21636  # From Ivshina & Winn 2022
        t14 = 0.9*(2.22/24)                  # From Simpson et al. 2011
        phs_t = juliet.utils.get_phases(tims, per, T0)
        phs_e = juliet.utils.get_phases(tims, per, (T0+(per/2)))

        mask = np.where((np.abs(phs_e*per) >= t14)&(np.abs(phs_t*per) >= t14))[0]
        tim7, fl7, fle7 = tims[mask], flx[mask], flxe[mask]
    else:
        tim7, fl7, fle7 = tims, flx, flxe

    # Sigma clipping
    sc = SigmaClip(sigma_upper=clip, sigma_lower=clip, stdfunc=mad_std, maxiters=None)
    msk1 = sc(fl7).mask

    tim_outliers = tim7[msk1]

    ## Removing outliers from the data
    msk2 = np.ones(len(tims), dtype=bool)
    for i in range(len(tim_outliers)):
        msk2[np.where(tims == tim_outliers[i])[0]] = False
    if verbose:
        print('---- Total number of points removed: ', len(msk2) - np.sum(msk2))
        print('---- Total per cent of point removed: {:.4f} %'.format((len(msk2) - np.sum(msk2))*100/len(msk2)))
    return msk2

def outlier_removal_ycen(ycen, clip=3.5, verbose=True):
    sc = SigmaClip(sigma_upper=clip, sigma_lower=clip, stdfunc=mad_std, maxiters=None)
    msk1 = sc(ycen).mask

    loc_out = ycen[msk1]

    msk2 = np.ones(len(ycen), dtype=bool)
    for i in range(len(loc_out)):
        msk2[np.where(ycen == loc_out[i])[0]] = False
    if verbose:
        print('---- Total number of points removed: ', len(msk2) - np.sum(msk2))
        print('---- Total per cent of point removed: {:.4f} %'.format((len(msk2) - np.sum(msk2))*100/len(msk2)))
    return msk2

def white_light_by_sum(lc, lc_err):
    white_lc = np.sum(lc, axis=1)
    white_err = np.sqrt(np.sum(lc_err**2, axis=1))
    return white_lc, white_err

#------------------------------------------------------------------------------------------
#-------------------------------Natural Sorting--------------------------------------------
#------------------------------------------------------------------------------------------
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def computeRMS(data, maxnbins=None, binstep=1, isrmserr=False):
    """Compute the root-mean-squared and standard error for various bin sizes.
    Parameters: This function is taken from the code `Eureka` -- please cite them!!
    ----------
    data : ndarray
        The residuals after fitting.
    maxnbins : int; optional
        The maximum number of bins. Use None to default to 10 points per bin.
    binstep : int; optional
        Bin step size. Defaults to 1.
    isrmserr : bool
        True if return rmserr, else False. Defaults to False.
    Returns
    -------
    rms : ndarray
        The RMS for each bin size.
    stderr : ndarray
        The standard error for each bin size.
    binsz : ndarray
        The different bin sizes.
    rmserr : ndarray; optional
        The uncertainty in the RMS. Only returned if isrmserr==True.
    Notes
    -----
    History:
    - December 29-30, 2021 Taylor Bell
        Moved code to separate file, added documentation.
    """
    data = np.ma.masked_invalid(np.ma.copy(data))
    
    # bin data into multiple bin sizes
    npts = data.size
    if maxnbins is None:
        maxnbins = npts / 10.
    binsz = np.arange(1, maxnbins + binstep, step=binstep, dtype=int)
    nbins = np.zeros(binsz.size, dtype=int)
    rms = np.zeros(binsz.size)
    rmserr = np.zeros(binsz.size)
    for i in range(binsz.size):
        nbins[i] = int(np.floor(data.size / binsz[i]))
        bindata = np.ma.zeros(nbins[i], dtype=float)
        # bin data
        # ADDED INTEGER CONVERSION, mh 01/21/12
        for j in range(nbins[i]):
            bindata[j] = np.ma.mean(data[j * binsz[i]:(j + 1) * binsz[i]])
        # get rms
        rms[i] = np.sqrt(np.ma.mean(bindata ** 2))
        rmserr[i] = rms[i] / np.sqrt(2. * nbins[i])
    # expected for white noise (WINN 2008, PONT 2006)
    stderr = (np.ma.std(data) / np.sqrt(binsz)) * np.sqrt(nbins / (nbins - 1.))
    if isrmserr is True:
        return rms, stderr, binsz, rmserr
    else:
        return rms, stderr, binsz

def col_spec(lc, ch_nos):
    """Given a number of total number of channels, this function gives an array containing
    start and end column for each channel"""
    if ch_nos != 1:
        col_in_1_ch = round(lc.shape[1]/ch_nos)
        col_st = np.arange(0, lc.shape[1]-col_in_1_ch, col_in_1_ch, dtype=int)
        col_end = np.arange(0+col_in_1_ch, lc.shape[1], col_in_1_ch, dtype=int)
    else:
        col_st, col_end = np.array([0]), np.array([lc.shape[1]])
    
    if col_end[-1] != lc.shape[1]:
        col_st = np.hstack((col_st, col_end[-1]))
        col_end = np.hstack((col_end, lc.shape[1]))
    
    return col_st, col_end

def spectral_lc(lc, lc_err, wav, ch_nos):
    """Given lc data cube, lc error data cube, wavelength calibration array and number of channels
    this function generate spectral lightcurves along with wavelengths"""
    # Columns
    if ch_nos != lc.shape[1]:
        col_st, col_end = col_spec(lc, ch_nos)
        # Creating spectral lc array
        spec_lc, spec_err_lc = np.zeros((lc.shape[0], len(col_st))), np.zeros((lc.shape[0], len(col_st)))
        wavs, wav_bin_size = np.zeros(len(col_st)), np.zeros(len(col_st))
        for i in range(len(col_st)):
            spec_lc[:,i], spec_err_lc[:,i] = poetss.white_light(lc[:,col_st[i]:col_end[i]], \
                                                                lc_err[:,col_st[i]:col_end[i]])
            if col_end[i] != lc.shape[1]:
                wavs[i] = (wav[col_st[i]] + wav[col_end[i]])/2
                wav_bin_size[i] = np.abs(wav[col_st[i]] - wav[col_end[i]])
            else:
                wavs[i] = (wav[col_st[i]] + wav[col_end[i]-1])/2
                wav_bin_size[i] = np.abs(wav[col_st[i]] - wav[col_end[i]-1])
    else:
        print('>>>> --- Working at the native resolution of the instrument...')
        spec_lc, spec_err_lc = lc, lc_err
        wavs, wav_bin_size = wav, np.append(np.diff(wav), np.diff(wav)[-1])
    return spec_lc, spec_err_lc, wavs, wav_bin_size


def lcbin(time, flux, binwidth=0.06859, nmin=4, time0=None,
        robust=False, tmid=False):
    """
    This code is taken from the code `pycheops`
    Calculate average flux and error in time bins of equal width.
    The default bin width is equivalent to one CHEOPS orbit in units of days.
    To avoid binning data on either side of the gaps in the light curve due to
    the CHEOPS orbit, the algorithm searches for the largest gap in the data
    shorter than binwidth and places the bin edges so that they fall at the
    centre of this gap. This behaviour can be avoided by setting a value for
    the parameter time0.
    The time values for the output bins can be either the average time value
    of the input points or, if tmid is True, the centre of the time bin.
    If robust is True, the output bin values are the median of the flux values
    of the bin and the standard error is estimated from their mean absolute
    deviation. Otherwise, the mean and standard deviation are used.
    The output values are as follows.
    * t_bin - average time of binned data points or centre of time bin.
    * f_bin - mean or median of the input flux values.
    * e_bin - standard error of flux points in the bin.
    * n_bin - number of flux points in the bin.
    :param time: time
    :param flux: flux (or other quantity to be time-binned)
    :param binwidth:  bin width in the same units as time
    :param nmin: minimum number of points for output bins
    :param time0: time value at the lower edge of one bin
    :param robust: use median and robust estimate of standard deviation
    :param tmid: return centre of time bins instead of mean time value
    :returns: t_bin, f_bin, e_bin, n_bin
    """
    if time0 is None:
        tgap = (time[1:]+time[:-1])/2
        gap = time[1:]-time[:-1]
        j = gap < binwidth
        gap = gap[j]
        tgap = tgap[j]
        time0 = tgap[np.argmax(gap)]
        time0 = time0 - binwidth*np.ceil((time0-min(time))/binwidth)

    n = np.int(1+np.ceil(np.ptp(time)/binwidth))
    r = (time0,time0+n*binwidth)
    n_in_bin,bin_edges = np.histogram(time,bins=n,range=r)
    bin_indices = np.digitize(time,bin_edges)

    t_bin = np.zeros(n)
    f_bin = np.zeros(n)
    e_bin = np.zeros(n)
    n_bin = np.zeros(n, dtype=np.int)

    for i,n in enumerate(n_in_bin):
        if n >= nmin:
            j = bin_indices == i+1
            n_bin[i] = n
            if tmid:
                t_bin[i] = (bin_edges[i]+bin_edges[i+1])/2
            else:
                t_bin[i] = np.nanmean(time[j])
            if robust:
                f_bin[i] = np.nanmedian(flux[j])
                e_bin[i] = 1.25*np.nanmean(abs(flux[j] - f_bin[i]))/np.sqrt(n)
            else:
                f_bin[i] = np.nanmean(flux[j])
                e_bin[i] = np.std(flux[j])/np.sqrt(n-1)

    j = (n_bin >= nmin)
    return t_bin[j], f_bin[j], e_bin[j], n_bin[j]


def corner_plot(folder, planet_only=False):
    """
    This function will generate corner plots of posterios
    in a given folder
    -----------------------------------------------------
    Parameters:
    -----------
    folder : str
        Path of the folder where the .pkl file is located
    planet_only : bool
        Boolean on whether to make corner plot of only
        planetary parameters
        Default is False
    -----------
    return
    -----------
    corner plot : .pdf file
        stored inside folder directory
    """
    pcl = glob(folder + '/*.pkl')[0]
    post = pickle.load(open(pcl, 'rb'), encoding='latin1')
    p1 = post['posterior_samples']
    lst = []
    if not planet_only:
        for i in p1.keys():
            gg = i.split('_')
            if ('p1' in gg) or ('mflux' in gg) or ('sigma' in gg) or ('GP' in gg) or ('mdilution' in gg) or ('q1' in gg) or ('q2' in gg) or (gg[0][0:5] == 'theta'):
                lst.append(i)
    else:
        for i in p1.keys():
            gg = i.split('_')
            if 'p1' in gg or 'q1' in gg or 'q2' in gg:
                lst.append(i)
    if 't0' in lst[0].split('_'):
        t01 = np.floor(p1[lst[0]][0])
        cd = p1[lst[0]] - t01
        lst[0] = lst[0] + ' - ' + str(t01)
    elif 'fp' in lst[0].split('_'):
        cd = p1[lst[0]]*1e6
        lst[0] = lst[0] + ' (in ppm)'
    else:
        cd = p1[lst[0]]
    for i in range(len(lst)-1):
        if 't0' in lst[i+1].split('_'):
            t02 = np.floor(p1[lst[i+1]][0])
            cd1 = p1[lst[i+1]] - t02
            cd = np.vstack((cd, cd1))
            lst[i+1] = lst[i+1] + ' - ' + str(t02)
        elif 'fp' in lst[i+1].split('_'):
            cd = np.vstack((cd, p1[lst[i+1]]*1e6))
            lst[i+1] = lst[i+1] + ' (in ppm)'
        else:
            cd = np.vstack((cd, p1[lst[i+1]]))
    data = np.transpose(cd)
    value = np.median(data, axis=0)
    ndim = len(lst)
    fig = corner.corner(data, labels=lst)
    axes = np.array(fig.axes).reshape((ndim, ndim))

    for i in range(ndim):
        ax = axes[i,i]
        ax.axvline(value[i], color = 'r')

    for yi in range(ndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.axvline(value[xi], color = 'r')
            ax.axhline(value[yi], color = 'r')
            ax.plot(value[xi], value[yi], 'sr')

    fig.savefig(folder + "/corner.png")
    plt.close(fig)

def transit_model(times, per, tc, rp1, ar1, bb1, q1, q2):
    u1, u2 = juliet.utils.reverse_ld_coeffs('quadratic', q1, q2)
    params = batman.TransitParams()
    params.t0 = tc            
    params.per = per
    params.rp = rp1
    params.a = ar1
    params.inc = np.rad2deg(np.arccos(bb1/ar1))
    params.ecc = 0.1074
    params.w = 106.1
    params.u = [u1, u2]
    params.limb_dark = "quadratic"
    m1 = batman.TransitModel(params, times)
    flux1 = m1.light_curve(params)
    return flux1