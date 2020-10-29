import numpy as np
from radynpy.utils import gaunt_bf, planck_fn, prfhbf_rad, gaunt_factor, hminus_pops, transp_scat, wnstark2, poph2
from radynpy import matsplotlib as rd


def contin_contrib_fn(cdf, isteps= [0], wavels_ang = [6690.00], mu_ind = -1,
                      wstandard = 5000.00,
                      include_metals = True, include_helium = True, 
                      include_scatt = True, include_h2 = False, 
                      include_lz = False, basic_out = False, 
                      full_out =  False):

    '''
    Calculates the contribution function to continuum at wavelengths = wavel_ang.
    Based on an routines by Adam Kowalski and Graham Kerr.
    Parameters
    __________
        cdf :
            The RadynData object containing the data with which to compute
            the contribution function.
        isteps : int, optional
             The time indices to compute (default = 0). 
        wavels_ang : float, optional
              The wavelengths in angstroms at which to compute the contribution fn 
              (default = 6690.00). Note that these must be included in opctab.dat].
        mu_ind : int, optional
              Index of the mu ray to be used. Default is the closest to the normal to 
              the atmosphere. (default: -1).
        include_metals: bool, optional
              Set to include background LTE opacities from metal continuum (defaul = True).
              If doing this then you can only analyse a wavelength that is used by RADYN 
              to compute the background opacity (within .5A)... though really you should
              restrict yourself to that anyway.
        include_helium : bool, optional
               Set to include the NUV b-f continua for helium using the NLTE population 
               densities of helium (default = True).
               Also adds LTE opacities for higher levels of Helium II, which
               have edges at 3647, 5698, and 8205. Just used prhbf with zz=2.0. 
        include_scatt : bool, optional
               Set to include scattering terms in the opacity and emissivity 
               (default = True).
        include_h2 : bool, optioinal
               Calculates H2 population in LTE (as is done in RADYN)
               and then adds Rayleigh scattering of H2, important mainly for M dwarf 
               atmospheres (default = False). 
        include_lz : bool, optional
                Calculation Landau-Zener opacity and emissivity
                longward of the Balmer jump (default = False).
                **** Consult Adam Kowalski about this part! ****
        wstandard : float, optional
               The 'standard' wavelength for use with calculating the scattering emissivity
               (default = 5000.00 angstroms)
        basic_out : bool, optional
               Select to only output a portion of the typical code output (default = False)
        full_out : bool, optional
               Select to only output a comprehensive set of variables (default = False)
       
    
    **** Note that Adam's version includes some additional terms for  Mg II wing intenstity
         and irradiance for H2 scattering (I think mainly useful/necessary for M Dwarf atmos). 
         Will check with him if these should be included here, but they are not presently. 
    
    **** Where opacity and emissivity are not computed at the same time, the emissivities 
         are computed later in the routine using the Planck fn
    **** Requires subroutines mostly (all?) held in radynpy/utils.py
             - gaunt_factor
             - prfhbf_rad
             - gaunt_bf
             - planck_fn
             - hminus_pops
             - transp_scat
             - wnstark2
             - poph2
    Graham Kerr, Feb 2020
    ******** add in he II
    ******** add in H2
    '''

    ########################################################################
    # Some preliminary set up, constants, and calculation of common terms
    ########################################################################

    if include_helium ==  True:
       print('>>> Including Helium') 
    if include_scatt ==  True:
       print('>>> Including Scattering') 
    if include_metals ==  True:
       print('>>> Including Metals') 
    if include_h2 ==  True:
       print('>>> Including Rayleigh scattering from H2') 
    if include_lz ==  True:
       print('>>> Including Landau-Zener')
    if (basic_out == True) & (full_out == True): 
       print('... You asked for both basic output and full output... FULL OUTPUT is provided')
    

    # Turn the inputs into numpy arrays
    wavels_ang = np.array(wavels_ang)

    # Add 5000A to the wavelength list (if scattering is to be included) 
    # as the 'standard' opacity is required, if it is not there already 
    #if include_scatt == True:
    wind = np.where(wavels_ang == wstandard)
    if np.array(wind).size == 0:
        print('>>> Adding wstandard = '+str(wstandard)+' to wavelength list')
        wavels_ang = np.append(wavels_ang,[wstandard])

    # Sort the wavelength array to be in ascending order
    wavels_ang = np.sort(wavels_ang)

    tsteps = cdf.time[isteps]

    # Size of arrays
    num_times = len(isteps)
    num_waves = len(wavels_ang)


    # The viewing angle
    mu = cdf.zmu[mu_ind]
    
    # Hydrogen is element index 0
    iel = 0
    ielhe = 2
    # Arrays to hold various variables later, and reduce other variables
    totnhi = np.sum(cdf.n1[:,:,0:cdf.nk[iel]-1,iel],axis = 2) # Density of neutral hydrogen
    toth = np.sum(cdf.n1[:,:,0:cdf.nk[iel],iel],axis = 2) # Density of total hydrogen
    nh = cdf.n1[:,:,0:cdf.nk[iel],iel] # The hydrogen population densities
    nhe = cdf.n1[:,:,0:cdf.nk[ielhe],ielhe] # The hydrogen population densities

    # Some physical constants
    cc_ang = cdf.cc*1.0e8     # Speed of light in angstroms
    hc_k = cc_ang * cdf.hh / cdf.bk    
    nu = cc_ang / wavels_ang   # The frequencies being studied
    zz = 1.0                   # The nuclear charge
    phot_const = 2.815e29      # The constant from photoionisation cross sections

     
    # The B-F Gaunt fns for each wavelength
    gauntbf1 = gaunt_bf(wavels_ang, n=1.)
    gauntbf2 = gaunt_bf(wavels_ang, n=2.)
    gauntbf3 = gaunt_bf(wavels_ang, n=3.)
    gauntbf4 = gaunt_bf(wavels_ang, n=4.)
    gauntbf5 = gaunt_bf(wavels_ang, n=5.)

    # Free-free gaunt factor (from Gray pf 149)
    gff = np.zeros([num_waves,cdf.ndep,num_times], dtype=float)
    for k in range(num_times):
        gff[:,:,k] = [1.0 + 3.3512 * x**(-0.3333333) * (6.95e-9 * x * cdf.tg1[isteps[k],:] + 0.5) for x in wavels_ang]



    # Ratio of protons in nlte to ratio of protons assuming lte, for 
    # use in the calculation of the nlte b-f opacity 
    # NOTE:  n* in mihalis is b_c * nstar[*,level]
    b_c = cdf.n1[:,:,cdf.nk[iel]-1,iel] / cdf.nstar[:,:,cdf.nk[iel]-1,iel]
    b_c_he = cdf.n1[:,:,5,2] / cdf.nstar[:,:,5,2]


    # Terms used in the stimulated emission correction required for 
    # nlte opacity (the negative term in alpha_hbf_nlte)
    stim_term = np.zeros([num_waves,cdf.ndep,num_times], dtype=float)
    stim_term_2 = np.zeros([num_waves,cdf.ndep,num_times], dtype=float)
    for k in range(num_times):
        stim_term[:,:,k] = [1e0 - np.exp( hc_k / (x * cdf.tg1[isteps[k],:]) *(-1e0) ) for x in wavels_ang]
        stim_term_2[:,:,k] = [np.exp( hc_k / (x * cdf.tg1[isteps[k],:]) * (-1e0) ) for x in wavels_ang]

    # The constant terms used in the calc of emissivity.
    jcoeff = np.zeros([len(nu),cdf.ndep,num_times], dtype=float)
    for k in range(num_times):
        jcoeff[:,:,k] = [2e0 * cdf.hh * x**3.0 / cdf.cc**2.0 * np.exp(-1.0 * cdf.hh * x / (cdf.bk * cdf.tg1[isteps[k],:])) for x in nu]


    # Photoionisation cross-sections [Mihalis 1978, eq 4-114]
    phot_crss_1 = phot_const / (1.**5.) * (cc_ang/wavels_ang)**(-3.0) * gauntbf1
        
    phot_crss_2 = phot_const / (2.**5.) * (cc_ang/wavels_ang)**(-3.0) * gauntbf2
    
    phot_crss_3 = phot_const / (3.**5.) * (cc_ang/wavels_ang)**(-3.0) * gauntbf3

    phot_crss_4 = phot_const / (4.**5.) * (cc_ang/wavels_ang)**(-3.0) * gauntbf4

    phot_crss_5 = phot_const / (5.**5.) * (cc_ang/wavels_ang)**(-3.0) * gauntbf5


    # Planck Fn (erg/s/cm2/sr/Ang)
    SourceBp = np.zeros([num_waves, cdf.ndep, num_times], dtype = np.float)
    for i in range(cdf.ndep):
        SourceBp[:,i,:] = planck_fn(wavels_ang, tg=cdf.tg1[isteps,i]) 
   
   # Assume departure coefficient for H minus is 1 because 
   # RADYN essentially uses LTE value
    bhmin = 1e0

   ########################################################################
   # COMPUTE THE OPACITIES AND EMISSIVITIES            
   ########################################################################
   
   ####
   # Hydrogren Free-Free
   ####
    alpha_hff = np.zeros([num_waves, cdf.ndep, num_times], dtype = float)
    for i in range(num_waves):
        for j in range(cdf.ndep):
            for k in range(num_times):
                alpha_hff[i,j,k] = (3.69e8 * gff[i,j,k] * (cc_ang/wavels_ang[i])**(-3.0) * 
                   cdf.tg1[isteps[k],j]**(-0.5) * cdf.ne1[isteps[k],j] * 
                   cdf.n1[isteps[k],j,cdf.nk[iel]-1,iel] * stim_term[i,j,k])
                
   ####
   # Hydrogen bound-free (RADYN Detailed transitions)
   ####

    # Calculates the nlte b-f opactity [Mihalis 1978, eq 7-1]  
    # Calculates the nlte b-f sponataneous thermal emission [Mihalis 1978, eq 7-2] 
    # -- Does not have higher order terms (see below)

    alpha_hbf_nlte = np.zeros([num_waves, cdf.ndep, num_times], dtype = float)      
    jbf_hbf_nlte = np.zeros([num_waves, cdf.ndep, num_times], dtype = float)

    # Lyman continuum
    winds = (np.where(wavels_ang <= 911.0))[0] 
    if len(winds) != 0:  
        for i in range(len(winds)):
            for k in range(num_times):
                alpha_hbf_nlte[winds[i],:,k] =  ( 
                    phot_crss_1[winds[i]] * ( cdf.n1[isteps[k],:,0,iel] - b_c[isteps[k],:] * cdf.nstar[isteps[k],:,0,iel] * stim_term_2[winds[i],:,k] ) + 
                    phot_crss_2[winds[i]] * ( cdf.n1[isteps[k],:,1,iel] - b_c[isteps[k],:] * cdf.nstar[isteps[k],:,1,iel] * stim_term_2[winds[i],:,k] ) + 
                    phot_crss_3[winds[i]] * ( cdf.n1[isteps[k],:,2,iel] - b_c[isteps[k],:] * cdf.nstar[isteps[k],:,2,iel] * stim_term_2[winds[i],:,k] ) + 
                    phot_crss_4[winds[i]] * ( cdf.n1[isteps[k],:,3,iel] - b_c[isteps[k],:] * cdf.nstar[isteps[k],:,3,iel] * stim_term_2[winds[i],:,k] ) + 
                    phot_crss_5[winds[i]] * ( cdf.n1[isteps[k],:,4,iel] - b_c[isteps[k],:] * cdf.nstar[isteps[k],:,4,iel] * stim_term_2[winds[i],:,k] ) 
                                                ) 

                jbf_hbf_nlte[winds[i],:,k] = ( 
                    jcoeff[winds[i],:,k] * ( 
                        phot_crss_1[winds[i]] * cdf.nstar[isteps[k],:,0,iel] * b_c[isteps[k],:] +
                        phot_crss_2[winds[i]] * cdf.nstar[isteps[k],:,1,iel] * b_c[isteps[k],:] +
                        phot_crss_3[winds[i]] * cdf.nstar[isteps[k],:,2,iel] * b_c[isteps[k],:] +
                        phot_crss_4[winds[i]] * cdf.nstar[isteps[k],:,3,iel] * b_c[isteps[k],:] +
                        phot_crss_5[winds[i]] * cdf.nstar[isteps[k],:,4,iel] * b_c[isteps[k],:] )
                        * (cc_ang)/(wavels_ang[winds[i]])**2.0 
                                             ) 

    # Balmer continuum
    winds = (np.where((wavels_ang <= 3646.91) & (wavels_ang > 911.70 )))[0]
    if len(winds) != 0:  
        for i in range(len(winds)):
            for k in range(num_times):
                alpha_hbf_nlte[winds[i],:,k] =  ( 
                    phot_crss_2[winds[i]] * ( cdf.n1[isteps[k],:,1,iel] - b_c[isteps[k],:] * cdf.nstar[isteps[k],:,1,iel] * stim_term_2[winds[i],:,k] ) + 
                    phot_crss_3[winds[i]] * ( cdf.n1[isteps[k],:,2,iel] - b_c[isteps[k],:] * cdf.nstar[isteps[k],:,2,iel] * stim_term_2[winds[i],:,k] ) + 
                    phot_crss_4[winds[i]] * ( cdf.n1[isteps[k],:,3,iel] - b_c[isteps[k],:] * cdf.nstar[isteps[k],:,3,iel] * stim_term_2[winds[i],:,k] ) + 
                    phot_crss_5[winds[i]] * ( cdf.n1[isteps[k],:,4,iel] - b_c[isteps[k],:] * cdf.nstar[isteps[k],:,4,iel] * stim_term_2[winds[i],:,k] ) 
                                                )
            
                jbf_hbf_nlte[winds[i],:,k] = ( 
                    jcoeff[winds[i],:,k] * ( 
                        phot_crss_2[winds[i]] * cdf.nstar[isteps[k],:,1,iel] * b_c[isteps[k],:] +
                        phot_crss_3[winds[i]] * cdf.nstar[isteps[k],:,2,iel] * b_c[isteps[k],:] +
                        phot_crss_4[winds[i]] * cdf.nstar[isteps[k],:,3,iel] * b_c[isteps[k],:] +
                        phot_crss_5[winds[i]] * cdf.nstar[isteps[k],:,4,iel] * b_c[isteps[k],:] )
                        * (cc_ang)/(wavels_ang[winds[i]])**2.0 
                                              )  

    # Paschen continuum
    winds = (np.where((wavels_ang <= 8205.71) & (wavels_ang > 3646.91 )))[0]
    if len(winds) != 0:
        for i in range(len(winds)):
            for k in range(num_times):

                ### THIS IS IN PROGRESS
                if include_lz == True:
                     #  n = 3d
                    nistar = 2e0
                    lamxx = wavels_ang[winds[i]]
                    nstar_wn = (1e0 / nistar**2 - 6.6260755e-27 * 2.9979e18/(lamxx * 13.598434005136e0 * 1.6021772e-12))**(-0.5)
                    wn_total = wnstark2(nstar_wn,cdf.ne1[isteps[k],:],cdf.n1[isteps[k],:,0,0],cdf.tg1[isteps[k],:])                      
                    DissFrac = 1e0 - wn_total/1e0
                else:
                    DissFrac = np.zeros([cdf.ndep],dtype=float)
                #DissFrac = np.zeros([cdf.ndep],dtype=float)

                alpha_hbf_nlte[winds[i],:,k] =  ( 
                    phot_crss_3[winds[i]] * ( cdf.n1[isteps[k],:,2,iel] - b_c[isteps[k],:] * cdf.nstar[isteps[k],:,2,iel] * stim_term_2[winds[i],:,k] ) + 
                    phot_crss_4[winds[i]] * ( cdf.n1[isteps[k],:,3,iel] - b_c[isteps[k],:] * cdf.nstar[isteps[k],:,3,iel] * stim_term_2[winds[i],:,k] ) + 
                    phot_crss_5[winds[i]] * ( cdf.n1[isteps[k],:,4,iel] - b_c[isteps[k],:] * cdf.nstar[isteps[k],:,4,iel] * stim_term_2[winds[i],:,k] ) +
                    2.815e29/(2.**5) *  (cc_ang/wavels_ang[winds[i]])**(-3)  * gauntbf2[winds[i]]  * (cdf.n1[isteps[k],:,1,0] - b_c[isteps[k],:] * cdf.nstar[isteps[k],:,1,0] * stim_term_2[winds[i],:,k]) * DissFrac
                                                )
            
                jbf_hbf_nlte[winds[i],:,k] = ( 
                     jcoeff[winds[i],:,k] * ( 
                         phot_crss_3[winds[i]] * cdf.nstar[isteps[k],:,2,iel] * b_c[isteps[k],:] +
                         phot_crss_4[winds[i]] * cdf.nstar[isteps[k],:,3,iel] * b_c[isteps[k],:] +
                         phot_crss_5[winds[i]] * cdf.nstar[isteps[k],:,4,iel] * b_c[isteps[k],:] +
                         2.815e29/(2.**5)*(cc_ang/wavels_ang[winds[i]])**(-3.) * cdf.nstar[isteps[k],:,1,0] * gauntbf2[winds[i]] * b_c[isteps[k],:] * DissFrac)
                         * (cc_ang)/(wavels_ang[winds[i]])**2.0 
                    
                                              )         
    # Brackett continuum
    winds = (np.where((wavels_ang <= 14580.1) & (wavels_ang > 8205.71 )))[0]
    if len(winds) != 0:  
        for i in range(len(winds)):
            for k in range(num_times):
                alpha_hbf_nlte[winds[i],:,k] =  ( 
                    phot_crss_4[winds[i]] * ( cdf.n1[isteps[k],:,3,iel] - b_c[isteps[k],:] * cdf.nstar[isteps[k],:,3,iel] * stim_term_2[winds[i],:,k] ) + 
                    phot_crss_5[winds[i]] * ( cdf.n1[isteps[k],:,4,iel] - b_c[isteps[k],:] * cdf.nstar[isteps[k],:,4,iel] * stim_term_2[winds[i],:,k] ) 
                                                )
            
            jbf_hbf_nlte[winds[i],:,k] = ( 
                 jcoeff[winds[i],:,k] * ( 
                     phot_crss_4[winds[i]] * cdf.nstar[isteps[k],:,3,iel] * b_c[isteps[k],:] +
                     phot_crss_5[winds[i]] * cdf.nstar[isteps[k],:,4,iel] * b_c[isteps[k],:] )
                     * (cc_ang)/(wavels_ang[winds[i]])**2.0 
                          ) 

    # Pfund continuum
    winds = (np.where(wavels_ang > 14580.1))[0]
    if len(winds) != 0:  
        for i in range(len(winds)):
            for k in range(num_times):
                alpha_hbf_nlte[winds[i],:,k] =  ( 
                    phot_crss_5[winds[i]] * ( cdf.n1[isteps[k],:,4,iel] - b_c[isteps[k],:] * cdf.nstar[isteps[k],:,4,iel] * stim_term_2[winds[i],:,k] ) 
                                                )
            
                jbf_hbf_nlte[winds[i],:,k] = ( 
                     jcoeff[winds[i],:,k] * ( 
                         phot_crss_5[winds[i]] * cdf.nstar[isteps[k],:,4,iel] * b_c[isteps[k],:] )
                         * (cc_ang)/(wavels_ang[winds[i]])**2.0 
                                              ) 

   ####
   # Hydrogen bound-free (upper level transitions)
   ####

    # Calculates the H f-b opacity from levels higher than those 
    # treated in detail in RADYN. Based on the internal routines
    # used by RADYN
    alpha_hbf_upper = np.zeros([num_waves, cdf.ndep, num_times], dtype = float)      
    upplevs = np.array([6,7,8])
    for n in range(len(upplevs)): #Fortran numbering, so include levels n = 5, 6, 7

        crhbf = prfhbf_rad(wavels_ang, zz, upplevs[n])

        for k in range(num_times):
            hnukt = 157896.e0 / (upplevs[n] * upplevs[n] * cdf.tg1[isteps[k],:])
            const = (
                      4.1416e-16 * cdf.ne1[isteps[k],:] * nh[isteps[k],:,5] /
                      (cdf.tg1[isteps[k],:] * np.sqrt(cdf.tg1[isteps[k],:]))
                    )

            pop = upplevs[n] * upplevs[n] * const * np.exp(hnukt)
             
            for i in range(num_waves):
                if crhbf[i] != 0:
                    alpha_hbf_upper[i,:,k] = (
                          alpha_hbf_upper[i,:,k] + crhbf[i] * pop * (1.0 - stim_term_2[i,:,k])
                                                 )
           

   ####
   # H minus bound-free
   ####
    # Computes the H- bf opacity.
    # The values absorption b-f coefficients for H- were obtained from 
    # Geltman 1962, Aph 136
    # They are in units of 10^-17 cm^2.

      
    hminbfw = np.array([0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 
               500.0, 550.0, 600.0, 650.0, 700.0, 750.0, 800.0, 850.0, 900.0, 950.0, 
               1000.0, 1050.0, 1100.0, 1150.0, 1200.0, 1250.0, 1300.0, 1350.0,  
               1400.0, 1450.0, 1500.0, 1550.0, 1600.0, 1641.9]) * 10e0 #To get wavelengths in angstroms
    # hminbfw = hminbfw * 10.
    hminbfsigma = np.array([ 0.0,  0.15, 0.33, 0.57, 0.85, 1.17, 1.52, 1.89, 2.23, 2.55, 2.84,
                    3.11, 3.35, 3.56, 3.71, 3.83, 3.92, 3.95, 3.93, 3.85, 3.73, 3.58,
                    3.38, 3.14, 2.85, 2.54, 2.20, 1.83, 1.46, 1.06, 0.71, 0.40, 0.17, 0.0])
  
    # Interpolate to get the values we require on our wavelength grid
    hminbfsigma_interp = np.interp(wavels_ang, hminbfw, hminbfsigma)

    # The density of H- and the opacity were obtained using equations 12 & 13 
    # of Vernazza et al. 1976, ApJS 30
    #nhmin = np.zeros([cdf.ndep,num_times],dtype=float)
    nhmin = hminus_pops(cdf.tg1[isteps,:],cdf.ne1[isteps,:],nh[isteps,:,:], bhmin=bhmin)
    alpha_hmbf = np.zeros([num_waves,cdf.ndep,num_times],dtype=float)
    for k in range(num_times):
        # nhmin[:,k] = (
        #          1.0354e-16 * bhmin * cdf.ne1[isteps[k],:] * totnhi[isteps[k],:] * 
        #          cdf.tg1[isteps[k],:]**(-1.5) * np.exp(8762e0/cdf.tg1[isteps[k],:])
        #              )
        for i in range(num_waves):
            alpha_hmbf[i,:,k] = (
                    nhmin[:,k] * 1e-17 * hminbfsigma_interp[i] * 
                    (1e0 - 1e0/bhmin * np.exp(hc_k/wavels_ang[i]/cdf.tg1[isteps[k],:] * (-1e0)))
                                )


   ####
   # H minus free-free 
   ####
    # Calculate the free-free contribution from H-
    # Expressions from gray p. 149-150. Note misprint of first
    # term of f0 in gray 
    # Based on polynomial fits to Stilley and Callaway
    winds = (np.where((wavels_ang < 91130.) & (wavels_ang > 3038. )))[0]
    
    x1 = np.log10(wavels_ang[winds])
    x2 = x1*x1
    x3 = x2*x1
    f0 = -31.63602+0.48735*x1+0.296585*x2-0.0193562*x3
    f1 = 15.3126-9.33651*x1+2.000242*x2-0.1422568*x3
    f2 = -2.6117+3.22259*x1-1.082785*x2+0.1072635*x3

    alpha_hmff = np.zeros([num_waves,cdf.ndep,num_times],dtype=float)

    for k in range(num_times):

        thlg = np.log10(5040./cdf.tg1[isteps[k],:])
        thlg2 = thlg*thlg
    
        for i in range(len(winds)):
            alpha_hmff[winds[i],:,k] = cdf.ne1[isteps[k],:] * cdf.bk * cdf.tg1[isteps[k],:] * 10.0**(f0[i]+f1[i]*thlg+f2[i]*thlg2) * totnhi[isteps[k],:]

    # The following alternative code is the updated Gray pg136 (2nd ed)
    # Based on polynomial fits to Bell and Berrington instead.
    # BUT RH uses Stilley and Callaway, as does RADYN internally!
    # winds = (np.where((wavels_ang < 113900.) & (wavels_ang > 2600. )))[0]
 
    # x1 = np.log10(wavels_ang)
    # x2 = x1*x1
    # x3 = x2*x1
    # x4 = x3*x1
    # f0 = -2.2763-1.6850*x1+0.76661*x2-0.053346*x3
    # f1 = 15.2827-9.2846*x1+1.99381*x2-0.142631*x3
    # f2 = -197.789+190.266*x1-67.9775*x2+10.6913*x3-0.625151*x4
    
    # alpha_hmff = np.zeros([num_waves,cdf.ndep,num_times],dtype=float)

    # for k in range(num_times):

    #     thlg = np.log10(5040./cdf.tg1[isteps[k],:])
    #     thlg2 = thlg*thlg
        
    #     for i in range(len(winds)):
    #         alpha_hmff[winds[i],:,k] = 1e-26*cdf.ne1[isteps[k],:] * cdf.bk * cdf.tg1[isteps[k],;] * 10.0**(f0[i]+f1[i]*thlg+f2[i]*thlg2) * totnhi[isteps[k],:]


   ####
   # Scattering Terms
   ####
    # Rayleigh scattering opacity
    w2 = 1./(wavels_ang * wavels_ang)
    w4 = w2*w2
    scatrh = np.zeros([num_waves,cdf.ndep,num_times],dtype=float)
    for k in range(num_times):
        for i in range(num_waves):
            scatrh[i,:,k] = w4[i]*(5.799e-13+w2[i]*(1.422e-6+w2[i]*2.784))*nh[isteps[k],:,0]

    # Thomson scattering opacity
    scatne = np.zeros([num_waves,cdf.ndep,num_times],dtype=float)
    for k in range(num_times):
        scatne[:,:,k] = 6.655e-25 * cdf.ne1[isteps[k],:]


   ####
   # Helium
   ####
    # Include Helium recombination edges in NLTE
    alpha_hebf65 = np.zeros([num_waves,cdf.ndep,num_times],dtype=float)
    jbf_hebf65 = np.zeros([num_waves,cdf.ndep,num_times],dtype=float)
    alpha_hebf64 = np.zeros([num_waves,cdf.ndep,num_times],dtype=float)
    jbf_hebf64 = np.zeros([num_waves,cdf.ndep,num_times],dtype=float)
    alpha_hebf63 = np.zeros([num_waves,cdf.ndep,num_times],dtype=float)
    jbf_hebf63 = np.zeros([num_waves,cdf.ndep,num_times],dtype=float)
    alpha_hebf62 = np.zeros([num_waves,cdf.ndep,num_times],dtype=float)
    jbf_hebf62 = np.zeros([num_waves,cdf.ndep,num_times],dtype=float)
    alpha_heiibf_upper = np.zeros([num_waves,cdf.ndep,num_times],dtype=float)
    if (include_helium == True):
        ielhe = 2
        winds = (np.where((wavels_ang < 3664.41e0)))[0]
        if len(winds) != 0:  
            for i in range(len(winds)):
                for k in range(num_times):
                    alpha_hebf65[winds[i],:,k] =  (
                                    1.38e-17 * (cc_ang/3664.41e0)**3 / 
                                    (cc_ang/wavels_ang[winds[i]])**3 * 
                                    (cdf.n1[isteps[k],:,4,ielhe] - b_c_he[isteps[k],:] * 
                                        cdf.nstar[isteps[k],:,4,ielhe] * stim_term_2[winds[i],:,k])
                                                  ) 
                    jbf_hebf65[winds[i],:,k] = (
                                    jcoeff[winds[i],:,k] * 1.38e-17 * (cc_ang/3664.41e0)**3 / 
                                    (cc_ang/wavels_ang[winds[i]])**3 *  
                                    b_c_he[isteps[k],:] * cdf.nstar[isteps[k],:,4,ielhe] * (cc_ang)/(wavels_ang[winds[i]])**2
                                               )
        winds = (np.where((wavels_ang < 3408.60e0)))[0]
        if len(winds) != 0:  
            for i in range(len(winds)):
                for k in range(num_times):
                    alpha_hebf64[winds[i],:,k] =  (
                                    1.38e-17 * (cc_ang/3408.60e0)**3 / 
                                    (cc_ang/wavels_ang[winds[i]])**3 * 
                                    (cdf.n1[isteps[k],:,3,ielhe] - b_c_he[isteps[k],:] * 
                                        cdf.nstar[isteps[k],:,3,ielhe] * stim_term_2[winds[i],:,k])
                                                  ) 
                    jbf_hebf64[winds[i],:,k] = (
                                    jcoeff[winds[i],:,k] * 1.63e-17 * (cc_ang/3408.60e0)**3 / 
                                    (cc_ang/wavels_ang[winds[i]])**3 *  
                                    b_c_he[isteps[k],:] * cdf.nstar[isteps[k],:,3,ielhe] * (cc_ang)/(wavels_ang[winds[i]])**2
                                               )
        winds = (np.where((wavels_ang < 3110.70e0)))[0]
        if len(winds) != 0:  
            for i in range(len(winds)):
                for k in range(num_times):
                    alpha_hebf63[winds[i],:,k] =  (
                                    9.24e-18 * (cc_ang/3110.70e0)**3 / 
                                    (cc_ang/wavels_ang[winds[i]])**3 * 
                                    (cdf.n1[isteps[k],:,2,ielhe] - b_c_he[isteps[k],:] * 
                                        cdf.nstar[isteps[k],:,2,ielhe] * stim_term_2[winds[i],:,k])
                                                  ) 
                    jbf_hebf63[winds[i],:,k] = (
                                    jcoeff[winds[i],:,k] * 9.24e-18 * (cc_ang/3110.70e0)**3 / 
                                    (cc_ang/wavels_ang[winds[i]])**3 *  
                                    b_c_he[isteps[k],:] * cdf.nstar[isteps[k],:,2,ielhe] * (cc_ang)/(wavels_ang[winds[i]])**2
                                               )
        winds = (np.where((wavels_ang < 2592.80)))[0]
        if len(winds) != 0:  
            for i in range(len(winds)):
                for k in range(num_times):
                    alpha_hebf62[winds[i],:,k] =  (
                                    5.50e-18 * (cc_ang/2592.80e0)**3 / 
                                    (cc_ang/wavels_ang[winds[i]])**3 * 
                                    (cdf.n1[isteps[k],:,1,ielhe] - b_c_he[isteps[k],:] * 
                                        cdf.nstar[isteps[k],:,1,ielhe] * stim_term_2[winds[i],:,k])
                                                  ) 
                    jbf_hebf62[winds[i],:,k] = (
                                    jcoeff[winds[i],:,k] * 5.50e-18 * (cc_ang/2592.80e0)**3 / 
                                    (cc_ang/wavels_ang[winds[i]])**3 *  
                                    b_c_he[isteps[k],:] * cdf.nstar[isteps[k],:,1,ielhe] * (cc_ang)/(wavels_ang[winds[i]])**2
          
                                               )
    
    # Include Helium II higher levels LTE
    winds = ( np.where((wavels_ang > 2052.00) & (wavels_ang < 3647.00)) )[0]
    upplevs = np.array([4,5,6,7,8])
    if len(winds) != 0: 
        zz_he=2e0 
        for n in range(len(upplevs)): #Fortran numbering, so include levels n = 5, 6, 7
            crhbf = prfhbf_rad(wavels_ang[winds], zz_he, upplevs[n])
            for k in range(num_times):
                hnukt = 157896.e0 / (upplevs[n] * upplevs[n] * cdf.tg1[isteps[k],:])
                const = (
                        4.1416e-16 * cdf.ne1[isteps[k],:] * nhe[isteps[k],:,8] /
                        (cdf.tg1[isteps[k],:] * np.sqrt(cdf.tg1[isteps[k],:]))
                        )
                pop = upplevs[n] * upplevs[n] * const * np.exp(hnukt)
                for i in range(len(winds)):
                    if crhbf[i] != 0:
                       alpha_heiibf_upper[winds[i],:,k] = (
                                alpha_heiibf_upper[winds[i],:,k] + crhbf[i] * pop * (1.0 - stim_term_2[i,:,k])
                                                          )
    winds = ( np.where((wavels_ang > 3647.00)) )[0]
    upplevs = np.array([5,6,7,8])
    if len(winds) != 0: 
        zz_he=2e0 
        for n in range(len(upplevs)): #Fortran numbering, so include levels n = 5, 6, 7
            crhbf = prfhbf_rad(wavels_ang[winds], zz_he, upplevs[n])
            for k in range(num_times):
                hnukt = 157896.e0 / (upplevs[n] * upplevs[n] * cdf.tg1[isteps[k],:])
                const = (
                        4.1416e-16 * cdf.ne1[isteps[k],:] * nhe[isteps[k],:,8] /
                        (cdf.tg1[isteps[k],:] * np.sqrt(cdf.tg1[isteps[k],:]))
                        )
                pop = upplevs[n] * upplevs[n] * const * np.exp(hnukt)
                for i in range(len(winds)):
                    if crhbf[i] != 0:
                       alpha_heiibf_upper[winds[i],:,k] = (
                                alpha_heiibf_upper[winds[i],:,k] + crhbf[i] * pop * (1.0 - stim_term_2[i,:,k])
                                                          )


   ####
   # Background metals
   ####
    # Include the contribution from background metals using the opctab.dat tables used
    # in the actual RADYN simulation. 
    xcontm_tmp = np.zeros([num_waves,cdf.ndep,num_times],dtype=float)
    xcontm = np.zeros([num_waves,cdf.ndep,num_times],dtype=float)
    if include_metals == True:
        metal_file = rd.OpcFile(path='./opctab.dat')
        roptab_tmp = metal_file.roptab(cdf.tg1[isteps[0],:],cdf.ne1[isteps[0],:],4)
        iw = ([np.where(np.abs(x - roptab_tmp['wavel']) < 0.5) for x in wavels_ang])
        for i in range(num_waves):
            if np.array(iw[i]).size == 0:
                print(' >>> ERROR <<<')
                print(' ... i = '+str(i))
                print(' ... wavels_ang = '+str(wavels_ang[i]))
                print(' ... no wavelength in range in opctab.dat')
                ### SHOULD ADD SOMETHING TO END THE PROGRAM HERE

        for i in range(num_waves):
            for k in range(num_times):
                # opacity from metals per h,  opacity in cm^2/cm^3
                roptab_tmp2 = metal_file.roptab(cdf.tg1[isteps[k],:],cdf.ne1[isteps[k],:],(iw)[i][0][0]+4)
                xcontm_tmp = roptab_tmp2['v']
                xcontm[i,:,k] = xcontm_tmp*toth[isteps[k],:] 
        
   ####
   # H2 scattering
   #### 
    # Include the contribution from Rayleigh scattering of H2, as is done in RADYN.
    # The populations are computed as in RADYN's source code. 
    # This is mainly for use with M Dwarf stars, with cooler photospheres than the 
    # usual quiet Solar atmospheres. 
    # Have not yet included illumination boundary condition to mimic neighbouring 
    # loops.
    scatrh2 = np.zeros([num_waves,cdf.ndep,num_times],dtype=float)
    nh2_pop = np.zeros([len(cdf.time),cdf.ndep],dtype=float)
    if (include_h2 == True): 
        arh2 = np.array([ 8.779e+01, 1.323e+06, 2.245e+10 ])

        # Rayleigh scattering, from RH's hydrogen.c
        # sigma in units of Mb, 1.0E-22 m^2
        lamrh2 = ( np.array([ 121.57, 130.00, 140.00, 150.00, 160.00, 170.00, 185.46, 
                   186.27, 193.58, 199.05, 230.29, 237.91, 253.56, 275.36, 
                   296.81, 334.24, 404.77, 407.90, 435.96, 546.23, 632.80 ])
                 )
        sigmarh2 = ( np.array([ 2.35E-06, 1.22E-06, 6.80E-07, 4.24E-07, 2.84E-07, 2.00E-07, 1.25E-07, 
                   1.22E-07, 1.00E-07, 8.70E-08, 4.29E-08, 3.68E-08, 2.75E-08, 1.89E-08,  
                   1.36E-08, 8.11E-09, 3.60E-09, 3.48E-09, 2.64E-09, 1.04E-09, 5.69E-10 ])
                   )
      
        nh2_pop = poph2(cdf)
        for i in range(len(winds)):
            for k in range(num_times):
                winds = (np.where((wavels_ang[i] <= 6328.00) & (wavels_ang[i] >= 1215.70)))[0]
                if len(winds) == 0:
                    sigmarh2_cgs = np.interp(wavels_ang[i], lamrh2 * 10.0, sigmarh2) * 1e-22 * 1e4
                else: 
                   lambda2 = 1e0 / (wavels_ang[i]/10e0)**2
                   sigmarh2_cgs = (arh2[0] + (arh2[1] + arh2[2]*lambda2) * lambda2) * lambda2**2 * 1e-22 * 1e4
         
                scatrh2[i,:,k] = sigmarh2_cgs * nh2_pop[isteps[k],:]  


   ####
   # Total Opacity and Emissivity
   ####    
    # Total Opacity
    # The sum of:
    #       1) H b-f nlte opacity
    #       2) H b-f upper levels
    #       3) H f-f opacity
    #       4) H- b-f opacity
    #       5) H- f-f opacity
    alpha_tot = (
                alpha_hbf_nlte + alpha_hbf_upper + alpha_hff + 
                alpha_hmbf + alpha_hmff)

    #       6) Metal background opacity
    if (include_metals == True):
        alpha_tot = alpha_tot + xcontm

    #       7) Thomson scattering
    #       8) Rayleigh scattering
    #if (include_scatt == True):  
    alpha_tot = alpha_tot + scatne  + scatrh
   
    #       9) Helium contribution
    if (include_helium == True): 
        alpha_tot = alpha_tot + alpha_hebf62 + alpha_hebf63 + alpha_hebf64  + alpha_hebf65 + alpha_heiibf_upper
   
    #      10) H2 Rayleigh Scattering contribution
    if (include_h2 == True):
        alpha_tot = alpha_tot + scatrh2


    # Total Emissivity 
    # The sum of:
    #       1) H b-f nlte 
    #       2) H b-f upper levels
    #       3) H f-f 
    #       4) H- b-f 
    #       5) H- f-f 
    j_tot = jbf_hbf_nlte + (alpha_hff + alpha_hmbf + alpha_hmff + alpha_hbf_upper) * SourceBp 
    
    #       6) Metal background opacity
    if (include_metals == True):
        j_tot = j_tot + xcontm*SourceBp

    #       7) Thomson scattering
    #       8) Rayleigh scattering
    jlam = np.zeros([num_waves,cdf.ndep,num_times],dtype=float)
    if (include_scatt == True):  
        # Contribution of scattering to total opacity
        epsilon = (alpha_tot - scatrh - scatne) / alpha_tot

        tau_standard = cdf.tau[isteps,:]

        # The opacity at wstandard... this is why wstandard is added to the wavelength 
        # list if it is not always there. 
        # calc_x_oldopc,timet[istep],5000d0,zmu[mu_ind],nh2_popt,alpha_standard;,read_oldopctab=0
        wind = (np.where(wavels_ang == wstandard))[0]
        alpha_standard = alpha_tot[wind,:,:]
        
        # Ratio of the opacity to the opacity at wstandard
        x_ratio = alpha_tot / alpha_standard

        BP_nu = np.zeros([num_waves,cdf.ndep,num_times],dtype=float)
        for i in range(num_waves):
            BP_nu[i,:,:] = SourceBp[i,:,:] / cc_ang * wavels_ang[i]**2.0
        jnu = transp_scat(tau_standard, x_ratio, (epsilon * BP_nu), 1.-epsilon)

        for i in range(num_waves):
            for k in range(num_times):
                jlam[i,:,k] = jnu[i,:,k] * cc_ang / wavels_ang[i]**2. # average intensity J_lambda

        j_tot = j_tot + jlam * (scatne + scatrh) 

    #       9) Helium contribution
    if (include_helium == True): 
        j_tot = j_tot + jbf_hebf62 + jbf_hebf63 + jbf_hebf64 + jbf_hebf65 + alpha_heiibf_upper*SourceBp
    
    #      10) H2 Rayleigh Scattering contribution
    if (include_h2 == True):
        j_tot = j_tot + jlam*scatrh2    



   ########################################################################
   # COMPUTE THE CONTRIBUTION FNS            
   ########################################################################
 
    xx = alpha_tot
    tauq = np.zeros([num_waves,cdf.ndep,num_times],dtype=float)
    #jq = np.zeros([num_waves,cdf.ndep,num_times],dtype=float)
    tauq[:,0,:] = xx[:,0,:]*1e-9

    # calculate the optical depth: tauq
    # optical depth at interface depth k averages the opacity at k and
    # the previous k (the average within the grid cell interfaced by k and k-1)
    for j in range(1, cdf.ndep): 
        for k in range(num_times):
            tauq[:,j,k] = tauq[:,j-1,k]+0.5*(xx[:,j,k]+xx[:,j-1,k])*(cdf.z1[isteps[k],j]-cdf.z1[isteps[k],j-1])*(-1.0) 
            #jq[:,j,k] = jq[:,j-1,k]+0.5*(j_tot[:,j,k]+j_tot[:,j-1,k])


    # z (and z_ci) is not the same as z1t; the ndep-th value of z is 
    # the average of the (ndep-1)-th and ndep-th value of z1t
    z_ci = np.zeros([cdf.ndep,num_times],dtype=float)
    for k in range(num_times):
        z_ci[1:,k] = (cdf.z1[isteps[k],0:cdf.ndep-1]+cdf.z1[isteps[k],1:cdf.ndep])*0.5
        z_ci[0,k] =  2*z_ci[0,k]-z_ci[1,k]    # extrapolate at top of loop
    # z_ci now ndep elements big, with the first element between the average of the 0th value of z1t and some extrapolated
    #  value higher up.
   
    dzt = np.zeros([num_waves,cdf.ndep,len(cdf.time)],dtype=float)
    for k in range(num_times):
        dzt[:,1:cdf.ndep,k] = cdf.z1[k,0:cdf.ndep-1] - cdf.z1[k,1:cdf.ndep]
    dzt_ci = dzt[:,:,isteps]
    # now jtot and alpha are calculated at the interface depths, and z_ci which
    # is at grid centers, and dzt_ci, is centered on the interface points
    

    # ci at depth ndep is the emissivity at depth ndep attenuated 
    # by the average optical depth within grid bordered by interfaces
    # ndep-1 and ndep. 
    # the contribution function within grid cell bordered by interfaces
    # ndep-1 and ndep is the emissivity from 
    # the interface ndep exponentially attenuated by the optical depth
    # within this grid cell.  
    # So, ci corresponds to z_ci (interface midpoints), and dzt_ci should be 
    # the width of the grid cell (=dzt) 
    ci = j_tot * np.exp(tauq/mu*(-1e0))/mu

    ci_prime = np.zeros([num_waves,cdf.ndep,num_times],dtype=float)
    for i in range(num_waves):
        for k in range(num_times):
            ci_prime[i,:,k] = np.cumsum(dzt_ci[i,:,k] * ci[i,:,k])/np.sum(dzt_ci[i,:,k] * ci[i,:,k]) 



    tau_ci = tauq 
    srcfunc = j_tot/xx 
    for i in range(num_waves):
        srcfunc[i,:,:] = srcfunc[i,:,:] * wavels_ang[i]**2 / cc_ang

    logm = np.log10(cdf.cmass1[isteps,:])
    dlogm_tmp = np.abs(logm[:,0:cdf.ndep-1]-logm[:,1:cdf.ndep])
    dlogm = np.zeros([num_times,cdf.ndep],dtype=float)
    for i in range(num_times):
        dlogm[i,:] = np.append([-5],dlogm_tmp[i,:])
    ci_dlogm = ci * dzt_ci 
    for k in range(num_times):
        ci_dlogm[:,:,k] = ci_dlogm[:,:,k] / dlogm[k,:]
    
    
    cont_cool = j_tot - jlam * alpha_tot
   

    if basic_out == True: 
        print('>>> Basic output only')
        out = {'alpha_tot':alpha_tot, 
              'j_tot':j_tot, 'wavels_ang':wavels_ang, 'isteps':isteps, 'tsteps':tsteps}
    elif full_out == True:
        out = {
              'SourceBp':SourceBp, 
              'alpha_hbf_nlte':alpha_hbf_nlte, 'alpha_hbf_upper':alpha_hbf_upper, 
              'alpha_hff':alpha_hff, 'alpha_hmbf':alpha_hmbf, 'alpha_hmff':alpha_hmff,
              'xcontm': xcontm, 'scatne':scatne, 'scatrh':scatrh, 'scatrh2':scatrh,
              'alpha_hebf62':alpha_hebf62, 'alpha_hebf63':alpha_hebf63, 'alpha_hebf64':alpha_hebf64, 
              'alpha_hebf65':alpha_hebf65, 'alpha_heiibf_upper':alpha_heiibf_upper,
              'alpha_tot':alpha_tot, 'nh2_pop':nh2_pop, 'nhmin':nhmin,
              'jbf_hbf_nlte':jbf_hbf_nlte, 'j_hff':alpha_hff*SourceBp, 'j_hmbf':alpha_hmbf*SourceBp, 
              'j_hmff':alpha_hmff*SourceBp, 'j_hbf_upper':alpha_hbf_upper*SourceBp, 
              'j_metals':xcontm*SourceBp, 'j_ne':jlam*scatne, 'j_rh':jlam*scatrh, 'j_rh2':jlam*scatrh2,
              'jbf_hebf62':jbf_hebf62, 'jbf_hebf63':jbf_hebf63, 'jbf_hebf64':jbf_hebf64, 
              'jbf_hebf65':jbf_hebf65, 'j_heiibf_upper':alpha_heiibf_upper*SourceBp,
              'j_tot':j_tot, 'jlam':jlam, 
              'tauq':tauq, 'z_ci':z_ci, 'dzt_ci':dzt_ci, 'ci':ci, 'tau_ci':tau_ci, 'mu':mu, 
              'srcfunc':srcfunc, 
              'ci_dlogm':ci_dlogm, 'logm':logm, 'dlogm':dlogm, 'cont_cool':cont_cool,
              'wavels_ang':wavels_ang, 'isteps':isteps, 'tsteps':tsteps,
              'gauntbf1':gauntbf1, 'gauntbf2':gauntbf2,'gauntbf3':gauntbf3,'gauntbf4':gauntbf4,
              'gauntbf5':gauntbf5, 'gff':gff,
              'phot_crss_1':phot_crss_1,'phot_crss_2':phot_crss_2,'phot_crss_3':phot_crss_3,
              'phot_crss_4':phot_crss_4,'phot_crss_5':phot_crss_5, 
              'b_c':b_c, 'b_c_he':b_c_he, 'stim_term':stim_term, 'stim_term2':stim_term2,
              'mu':mu}
    else:
        out = {
              'SourceBp':SourceBp, 
              'alpha_hbf_nlte':alpha_hbf_nlte, 'alpha_hbf_upper':alpha_hbf_upper, 
              'alpha_hff':alpha_hff, 'alpha_hmbf':alpha_hmbf, 'alpha_hmff':alpha_hmff,
              'xcontm': xcontm, 'scatne':scatne, 'scatrh':scatrh, 'scatrh2':scatrh,
              'alpha_hebf62':alpha_hebf62, 'alpha_hebf63':alpha_hebf63, 'alpha_hebf64':alpha_hebf64, 
              'alpha_hebf65':alpha_hebf65, 'alpha_heiibf_upper':alpha_heiibf_upper,
              'alpha_tot':alpha_tot, 'nh2_pop':nh2_pop,'nhmin':nhmin,
              'jbf_hbf_nlte':jbf_hbf_nlte, 'j_hff':alpha_hff*SourceBp, 'j_hmbf':alpha_hmbf*SourceBp, 
              'j_hmff':alpha_hmff*SourceBp, 'j_hbf_upper':alpha_hbf_upper*SourceBp, 
              'j_metals':xcontm*SourceBp, 'j_ne':jlam*scatne, 'j_rh':jlam*scatrh, 'j_rh2':jlam*scatrh2,
              'jbf_hebf62':jbf_hebf62, 'jbf_hebf63':jbf_hebf63, 'jbf_hebf64':jbf_hebf64, 
              'jbf_hebf65':jbf_hebf65, 'j_heiibf_upper':alpha_heiibf_upper*SourceBp,
              'j_tot':j_tot, 'jlam':jlam, 
              'tauq':tauq, 'z_ci':z_ci, 'dzt_ci':dzt_ci, 'ci':ci, 'tau_ci':tau_ci, 'mu':mu, 
              'srcfunc':srcfunc, 
              'ci_dlogm':ci_dlogm, 'logm':logm, 'dlogm':dlogm, 'cont_cool':cont_cool,
              'wavels_ang':wavels_ang, 'isteps':isteps, 'tsteps':tsteps, 'mu':mu
              }

    return out