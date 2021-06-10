import numpy as np
from scipy import constants
# import sys
from radynpy.utils.Utils import tradiation as tr
from radynpy.utils.Utils import tradiation_flux as tr_flux

def profile(cdf, kr, t1 = 0, t2 = 0, trad=False):

    '''
    A function to return the line intensity, flux, and wavelength
    of a spectral line transition, from a RADYN simulation

    Parameters
    __________
    cdf : the radyn cdf object
    kr  : int
          the index of the line transition
    t1  : float
          the first time at which to extract the line profile
          default = 0s
    t2  : float
          the final time at which to extract the line profile
          default = final time
          The output is a time series between t1-->t2
    trad : boolean
           set to True to output the line profiles in units of radiation temperature
           default = false
    

    OUTPUTS:  out_dict: A dictionary containing the intensity, flux, and wavelength
                        in units of [erg/s/cm^2/sr/A] (or K), [erg/s/cm^2/A], and [A]
                        Dimensions are: 
                            Intensity [time, viewing angle, wavelength]
                            Flux [time, wavelength]
                        Also includes ancillary information such as the
                        nq, the number of wavelength points for the line
                        of interest.              

    Orig Written: Graham Kerr, Oct 2020
    
    Modifications: Graham Kerr, March 2021 :
                   restructing of arrays to have the convention of time
                   being the first dimension. 
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
    line_int = np.zeros([nt,cdf.nmu,mq],dtype=np.float64)
    line_flux = np.zeros([nt,mq],dtype=np.float64)

    ########################################################################
    # Extract the line intensity, then sum up the contribution to the line flux
    ########################################################################

    if trad == True:
        for ind in range(1,cdf.nq[kr]+1):
            for tind in range(nt):
                for muind in range(cdf.nmu):
                    line_int[tind,muind,cdf.nq[kr]-ind] = tr(cdf.outint[tind,ind,muind,kr], wavel[ind-1])
                line_flux[tind,cdf.nq[kr]-ind] = tr_flux( (2*pi*np.sum(cdf.outint[tind,ind,:,kr]*cdf.wmu*cdf.zmu)), wavel[ind-1])
    elif trad == False:
        for ind in range(1,cdf.nq[kr]+1):
            for tind in range(nt):
                line_int[tind,:,cdf.nq[kr]-ind] = cdf.outint[tind,ind,:,kr]*cdf.cc*1e8/wavel[ind-1]**2.0
                line_flux[tind,cdf.nq[kr]-ind] = 2*pi*np.sum(cdf.outint[tind,ind,:,kr]*cdf.wmu*cdf.zmu)*cdf.cc*1e8/wavel[ind-1]**2.0

    rest_wave = cdf.alamb[kr]

    t1 = cdf.time[tind1]
    t2 = cdf.time[tind2]
    times = cdf.time[tind1:tind2+1]

    ########################################################################
    # Output the results
    ########################################################################
   
    if trad == False:

        out_dict  = {'rest_wave':rest_wave,
                 'wavelength':wavel,
                 'line_int':line_int,
                 'line_flux':line_flux,
                 't1':t1, 't2':t2,
                 'tind1':tind1,'tind2':tind2,
                 'times':times,
                 'nq':cdf.nq[kr],
                 'Units':'int in [erg/s/cm^2/sr/A], flux in [erg/s/cm^2/A], wavelength in [A], t1,2 in[s]'}
    
    elif trad == True:

        out_dict  = {'rest_wave':rest_wave,
                 'wavelength':wavel,
                 'line_int':line_int,
                 'line_flux':line_flux,
                 't1':t1, 't2':t2,
                 'tind1':tind1,'tind2':tind2,
                 'times':times,
                 'nq':cdf.nq[kr],
                 'Units':'int in [K], flux in [erg/s/cm^2/A], wavelength in [A], t1,2 in[s]'}
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
          The output is a time series from t1 --> t2

    OUTPUTS:  out_dict: a dictionary holding the lightcurves of intensity,
                        flux, and the time array, in units of
                        [erg/s/cm^2/sr], [erg/s/cm^2], [s]
                        Dimensions are: 
                            Intensity [time, viewing angle]
                            Flux [time] 
                        Also includes ancillary information such as the
                        wavelengths and indices over which the integtation 
                        was performed.

    Orig Written: Graham Kerr, Oct 2020
    
    Modifications: Graham Kerr, March 2021 :
                   restructing of arrays to have the convention of time
                   being the first dimension. 

    TO DO:    Add in option to subtract continuum to return only
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
                           line['line_int'].shape[1]], dtype=np.float64)
    lcurve_flux = np.zeros([line['line_flux'].shape[0]], dtype=np.float64)

    for tind in range(line['line_int'].shape[0]):
        lcurve_flux[tind] = np.trapz(line['line_flux'][tind,wind1:wind2+1],line['wavelength'][wind1:wind2+1])
        for muind in range(line['line_int'].shape[1]):
            lcurve_int[tind,muind] = np.trapz(line['line_int'][tind,muind,wind1:wind2+1],line['wavelength'][wind1:wind2+1])

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
