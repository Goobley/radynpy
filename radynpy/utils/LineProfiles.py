import numpy as np
from scipy import constants
import sys

def profile(cdf, kr, t1 = 0, t2 = 0):

    '''
    A function to return the line intensity, flux, and wavelength
    of a spectral line transition, from a RADYN simulation

    Parameters
    __________
    cdf : float
          the radyn readout
    kr  : int
          the index of the line transition
    t1  : float
          the first time at which compute the integrated intensity
          default = 0s
    t2  : float
          the first time at which compute the integrated intensity
          default = final time
    out_dict: a dictionary containing the intensity, flux, and wavelength
              in units of [erg/s/cm^2/sr/A], [erg/s/cm^2/A], and [A]

    Graham Kerr, Oct 2020
    '''

    ########################################################################
    # Some preliminary set up, constants, and calculation of common terms
    ########################################################################

    if cdf.cont[kr] != 0:
        raise ValueError('The transition you entered is not a line transition')

    if t1 !=0:
        tind1 = np.abs(cdf.time-t1).argmin()
        t1 = cdf.time[tind1]
    else:
        tind1 = 0
    if t2 !=0:
        tind2 = np.abs(cdf.time-t1).argmin()
    else:
        tind2 = len(cdf.time)-1
        # t2 = cdf.time[tind2]



    pi = constants.pi
    mq = np.nanmax(cdf.nq)
    # nt = cdf.time.shape[0]
    nt = len(cdf.time[tind1:tind2+1])
    ii = cdf.q[0:cdf.nq[kr],kr]
    wavel = np.zeros(mq,dtype=np.float64)
    wavel[0:cdf.nq[kr]]=cdf.alamb[kr]/(cdf.q[0:cdf.nq[kr],kr]*cdf.qnorm*1e5/cdf.cc+1)
    wavel = np.flip(wavel[0:cdf.nq[kr]])
    line_int = np.zeros([cdf.nmu,mq,nt],dtype=np.float64)
    line_flux = np.zeros([mq,nt],dtype=np.float64)

    ########################################################################
    # Extract the line intensity, then sum up the contribution to the line flux
    ########################################################################

    for ind in range(1,cdf.nq[kr]+1):
        for tind in range(nt):
            line_int[:,cdf.nq[kr]-ind,tind] = cdf.outint[tind,ind,:,kr]*cdf.cc*1e8/wavel[ind-1]**2.0
            line_flux[cdf.nq[kr]-ind,tind] = 2*pi*np.sum(cdf.outint[tind,ind,:,kr]*cdf.wmu*cdf.zmu)*cdf.cc*1e8/wavel[ind-1]**2.0

    rest_wave = cdf.alamb[kr]

    t1 = cdf.time[tind1]
    t2 = cdf.time[tind2]
    times = cdf.time[tind1:tind2+1]

    ########################################################################
    # Output the results
    ########################################################################

    out_dict  = {'rest_wave':rest_wave,
                 'wavelength':wavel,
                 'line_int':line_int,
                 'line_flux':line_flux,
                 't1':t1, 't2':t2,
                 'tind1':tind1,'tind2':tind2,
                 'times':times,
                 'nq':cdf.nq[kr],
                 'Units':'int in [erg/s/cm^2/sr/A], flux in [erg/s/cm^2/A], wavelength in [A], t1,2 in[s]'}


    return out_dict



def lcurve(cdf, kr, w1 = 0, w2 = 0, t1 = 0, t2 = 0):

    '''
    A function to return the lightcurves of intensity and flux, for
    a spectral line transition, from a RADYN simulation.

    Calls LineProfiles.profile to compute the intensity
    and flux

    Parameters
    __________
    cdf : float
          the radyn readout
    kr  : int
          the index of the line transition
    w1  : float
          the wavelength to start integrating from
          default = first wavelength in wavelength array
    w2  : float
          the wavelength to integrate to
          default = final wavelength in wavelength array
    t1  : float
          the first time at which compute the integrated intensity
          default = 0s
    t2  : float
          the first time at which compute the integrated intensity
          default = final time
    out_dict : a dictionary holding the lightcurves of intensity,
               flux, and the time array, in units of
               [erg/s/cm^2/sr], [erg/s/cm^2], [s]

    Graham Kerr, Oct 2020

    TO DO:
            * add in option to subtract continuum to return only
              lightcurves of the line-only intensity or flux
    '''

    ########################################################################
    # Some preliminary set up, constants, and calculation of common terms
    ########################################################################

    if cdf.cont[kr] != 0:
        raise ValueError('The transition you entered is not a line transition')

    if t1 !=0:
        tind1 = np.abs(cdf.time-t1).argmin()
        t1 = cdf.time[tind1]
    else:
        tind1 = 0
    if t2 !=0:
        tind2 = np.abs(cdf.time-t1).argmin()
        t2 = cdf.time[tind2]
    else:
        tind2 = len(cdf.time)-1

    line = profile(cdf, kr, t1 = t1, t2 = t2)

    if w1 !=0:
        wind1 = np.abs(line['wavelength']-w1).argmin()
        w1 = line['wavelength'][wind1]
    else:
        wind1 = 0
        # w1 = line['wavelength'][wind1]
    if w2 !=0:
        wind2 = np.abs(line['wavelength']-w2).argmin()
        w2 = line['wavelength'][wind2]
    else:
        wind2 = cdf.nq[kr]-1
        # w2 = line['wavelength'][wind2]


    ########################################################################
    # Integrate the line intensity and flux over wavelength
    ########################################################################


    lcurve_int = np.zeros([line['line_int'].shape[0],
                           line['line_int'].shape[2]], dtype=np.float64)
    lcurve_flux = np.zeros([line['line_flux'].shape[1]], dtype=np.float64)

    for tind in range(line['line_int'].shape[2]):
        lcurve_flux[tind] = np.trapz(line['line_flux'][wind1:wind2+1,tind],line['wavelength'][wind1:wind2+1])
        for muind in range(line['line_int'].shape[0]):
            lcurve_int[muind, tind] = np.trapz(line['line_int'][muind,wind1:wind2+1,tind],line['wavelength'][wind1:wind2+1])

    times = cdf.time[tind1:tind2+1]

    ########################################################################
    # Output the results
    ########################################################################

    t1 = cdf.time[tind1]
    t2 = cdf.time[tind2]
    w1 = line['wavelength'][wind1]
    w2 = line['wavelength'][wind2]


    out_dict = {'lcurve_flux':lcurve_flux,
                'lcurve_int':lcurve_int,
                'w1':w1, 'w2':w2,
                'wind1':wind1,
                'wind2':wind2,
                't1':t1, 't2':t2,
                'tind1':tind1,'tind2':tind2,
                'times':times,
                'Units':'int in [erg/s/cm^2/sr], flux in [erg/s/cm^2], wavelength in [A], t1,2 in[s]'}



    return out_dict
