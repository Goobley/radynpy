import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import pchip_interpolate, interp1d, griddata
from skimage.exposure import equalize_adapthist, equalize_hist
from palettable.colorbrewer.sequential import Blues_9, PuBu_9
from colour import Color
from matplotlib.ticker import MaxNLocator, LogLocator, ScalarFormatter
import warnings
from radynpy.matsplotlib import xt_calc

def contrib_fn_vars(cdf, kr, tStep=0, yRange=[-0.08, 2.5], vRange=[300.0, -300.0], mu=-1,
               heatPerParticle=False, wingOffset=None,
               dvMouseover=False, withBackgroundOpacity=True, colors={},
               tightenLayout=True, printMiscTopData=True, stdoutInfo=False,
               opctabPath=None):
    '''
    Computest the common line formation variables.

    Parameters
    ----------
    cdf : LazyRadynData or RadynData
        The RadynData object containing the data to compute the contribution function from.
        Due to the number of variables required, it is easier to use a LazyRadynData, if possible.
    kr : int
        The transition index, see `index_convention` and `var_info` on the RadynData object for
        more information about what this means.
    tStep : int, optional
        The time index at which to compute the contribution function see 't' in the `index_convention`
        of RadynData for more information. (default: 0)
    yRange : list of float, optional
        The height range to compute the values over, in Mm. (default: [-0.08, 2.5])
    vRange : list of float, optional
        The velocity range to compute the values (converted to a wavelength range around the transition core). (default: [-300, 300])
    mu : int, optional
        Index of the mu ray to be used. Default is the closest to the normal to the atmosphere. (default: -1)
    withBackgroundOpacity : bool, optional
        Include the background opacity at this wavelength in the calculation. (default: True)
    stdoutInfo : bool, optional
        Print progress information to stdout. (default: False)
    opctabPath : str, optional
        Path to non-standard opctab.dat if needed. (default: False)
    '''
    with warnings.catch_warnings():
        # There are a few log10(0) in here which naturally raise annoying RuntimeWarnings
        # They're not a real problem though, so we'll ignore them for this function
        warnings.simplefilter('ignore', RuntimeWarning)
        # This function works in ~ 2 parts, computation, and then lots and lots of plotting
        # debug info
        if stdoutInfo:
            print('=============================')
            print('atom: %s' % cdf.atomid[cdf.ielrad[kr]-1].upper())
            print('transition: %d -> %d' % (cdf.jrad[kr], cdf.irad[kr]))
            print('angle: %f degrees' % np.rad2deg(np.arccos(cdf.zmu[mu])))
            print('=============================')

        nDep = cdf.tg1.shape[1]
        iTrans = cdf.irad[kr] - 1
        jTrans = cdf.jrad[kr] - 1
        iel = cdf.ielrad[kr] - 1

        # Line intensity data
        x_ny, tauq_ny = xt_calc(cdf, tStep, iel, kr, withBackgroundOpacity=withBackgroundOpacity, opctabPath=opctabPath)
        dtau = np.zeros((nDep, cdf.nq[kr]))
        dtau[1:,:] = tauq_ny[1:nDep,:] - tauq_ny[:nDep-1,:]

        # Source function from level pops in CDF file
        sll = cdf.a[kr] * cdf.n1[tStep, :, jTrans, iel] \
                / (cdf.bij[kr] * cdf.n1[tStep, :, iTrans, iel] - cdf.bji[kr] * cdf.n1[tStep, :, jTrans, iel])

        nq = cdf.nq[kr]
        wavelength = cdf.alamb[kr] / (cdf.q[:nq, kr] * cdf.qnorm * 1e5 / cdf.cc + 1)
        # POSITIVE velocity is blueshift, towards shorter wavelength
        # NEGATIVE velocity is downflow, redshift, towards longer wavelength
        dVel = cdf.q[:nq, kr] * cdf.qnorm # frequency in km/s
        vMax = np.abs(vRange).max()
        wlIdxs = np.argwhere(np.abs(dVel) < vMax).ravel()
        nFreq = len(wlIdxs)

        # some comments courtesy of PJAS!
        # x_ny -> opacity (z1 = X*mu)
        # tauq_ny -> tau (z2 = tau * exp(-tau / mu))
        # z3 = 1 / tau
        # z4 = S (source fn)
        ny0 = wlIdxs[0]

        z1Local = x_ny[:, wlIdxs] / cdf.zmu[mu]
        z2 = tauq_ny[:, wlIdxs] * np.exp(-tauq_ny[:, wlIdxs] / cdf.zmu[mu])
        z3 = 1.0 / tauq_ny[:, wlIdxs]
        z4 = np.tile(sll, (nFreq, 1)).T
        y = (cdf.z1[tStep, :nDep-1] + cdf.z1[tStep, 1:nDep]) * 0.5e-8
        y = np.insert(y, 0, 2 * y[0] - y[1])

        # Find the tau=1 line using a cubic hermite spline interpolation that avoids overshoot
        tau1 = np.zeros(nFreq)
        for ny in range(nFreq):
            tau1[ny] = pchip_interpolate(np.log10(tauq_ny[1:, ny+ny0]), y[1:], 0.0)

        # Build a silly 4D matrix to contain the data... this would probably just as well be
        # a list of 4 matrices given the way we use it
        zTot = np.zeros((nFreq, nDep, 2, 2))
        zTot[:, :, 0, 0] = (z1Local * z3).T # Xv / tau
        zTot[:, :, 0, 1] = z2.T # tau * exp(-tau)
        zTot[:, :, 1, 0] = z4.T # Sv, uniform across wavelength
        zTot[:, :, 1, 1] = (z1Local * z2 *z3 *z4).T # contrib fn

        x = dVel[wlIdxs] # x-axis shrunk to plotting range
        # Find the indices for the range we want to plot
        iwy = np.argwhere((y > np.min(yRange)) & (y < np.max(yRange))).flatten()
        iwx = np.argwhere((x > np.min(vRange)) & (x < np.max(vRange))).flatten()
        vRange = [np.max(x[iwx]), np.min(x[iwx])] # invert x axis

        # Find the deltav / frequency indices for the core and wing indices,
        # where the wingOffset is specified in Angstrom from the line core.
        coreIdx = np.argmin(np.abs(dVel[wlIdxs]))
        if wingOffset is None:
            wingIdx = np.argmin(np.abs(dVel[wlIdxs] - vMax / 2))
        else:
            wingIdx = np.argmin(np.abs(wavelength[wlIdxs] - cdf.alamb[kr] - wingOffset))

        # Print wing choice if debugging in stdout
        if stdoutInfo:
            print('Wing index: %d' % wingIdx)
            print('Wing wavelength: %f Angstrom' % (wavelength[wlIdxs[wingIdx]] - cdf.alamb[kr]))


        # returns the radiation temperature for arrays of specific intensity and wavelength (Angstrom)
        def radiation_temperature(intens, lamb):
            c = 2.99792458e10
            h = 6.626176e-27
            k = 1.380662e-16
            l = lamb*1e-8
            tRad = h * c / k / l / np.log(2.0 * h * c / intens / (l**3) + 1.0)
            return tRad


        xEdges = 0.5 * (x[iwx][:-1] + x[iwx][1:])
        xEdges = np.insert(xEdges, 0, x[iwx][0])
        xEdges = np.insert(xEdges, -1, x[iwx][-1])

        yEdges = 0.5 * (y[iwy][:-1] + y[iwy][1:])
        yEdges = np.insert(yEdges, 0, y[iwy][0])
        yEdges = np.insert(yEdges, -1, y[iwy][-1])



        lineProfile = np.copy(cdf.outint[tStep, 1:cdf.nq[kr]+1, mu, kr])
        lineProfile -= lineProfile.min()
        lineProfile /= lineProfile.max()
        lineProfile *= (y[iwy][0] - y[iwy][-1])
        lineProfile += y[iwy][-1]




        out = {'atomId': cdf.atomid[0][iel],
                   'kr': kr,
                   'iel': iel,
                   'levels': [jTrans, iTrans],
                   'emissivity': zTot[np.ix_(iwx, iwy)][:,:, 0, 0],
                   'opacity': zTot[np.ix_(iwx, iwy)][:,:, 0, 1],
                   'contFn': zTot[np.ix_(iwx, iwy)][:,:, 1, 1],
                   'tau': tauq_ny,
                   'tau1': tau1,
                   'dVel': dVel,
                   'xEdges': xEdges,
                   'yEdges': yEdges,
                   'y': y,
                   'wavelength':wavelength,
                   'lineProfile':lineProfile,
                   'iwy':iwy,
                   'iwx':iwx
                  }
        return out

