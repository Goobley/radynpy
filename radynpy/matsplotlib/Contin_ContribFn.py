import numpy as np
from radynpy.utils import gaunt_bf, planck_fn, prfhbf_rad, gaunt_factor, hminus_pops, transp_scat


def contin_contrib_fn(cdf, isteps= [0], wavels_ang = [6690.00], mu_ind = -1,
                      include_metals = True, include_helium = True, 
                      include_scatt = True, wstandard = 5000.00):

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
        include_helium : bool, optional
               Set to include the NUV b-f continua for  helium using the NLTE population 
               densities of helium (default = True).
        scatt : bool, optional
               Set to include scattering terms in the emissivity (default = True).
        wstandard : float, optional
               The 'standard' wavelength for use with calculating the scattering emissivity
               (defaul = 5000.00)
        
    
    Note that Adam's version includes some additional terms for Landau-Zener, Mg II wing intenstity
    and H2 (I think mainly useful/necessary for M Dwarf atmos.). Will check with him if these should 
    be included here, but they are presently 


    This uses the routines in opacity.py, in radynpy.matsplotlib to read opcatb.dat 

    **** I have not yet included some stuff from the IDL version - LZ, Mg II wings. That will 
         done later

    **** Requires subroutines mostly (all?) held in radynpy/utils.py
             - gaunt_factor
             - prfhbf_rad
             - gaunt_bf
             - planck_fn
             - hminus_pops

    Graham Kerr, Feb 2020

    '''

    ########################################################################
    # Some preliminary set up and constants
    ########################################################################

    if include_helium ==  True:
       print('>>> Including Helium') 
    if include_scatt ==  True:
       print('>>> Including Scattering') 
    if include_metals ==  True:
       print('>>> Including Metals') 


    # Turn the inputs into numpy arrays
    wavels_ang = np.array(wavels_ang)

    # Add 5000A to the wavelength list (if scattering is to be included) 
    # as the 'standard' opacity is required, if it is not there already 
    if include_scatt == True:
        wind = np.where(wavels_ang == wstandard)
        if len(wind) == 0:
            print('adding wstandard')
            print(wstandard)
            wavels_ang = np.append(wavels_ang,[wstandard])

    # Sort the wavelength array to be in ascending order
    wavels_ang = np.sort(wavels_ang)

    # Size of arrays
    num_times = len(isteps)
    num_waves = len(wavels_ang)

    # The viewing angle
    mu = cdf.zmu[mu_ind]
    
    # Hydrogen is element index 0
    iel = 0

    # Arrays to hold various variables later, and reduce other variables
    xconth = np.zeros(cdf.ndep, dtype = float) #  metal background opacity 
    totnhi = np.sum(cdf.n1[:,:,0:cdf.nk[iel]-1,iel],axis = 2) # Density of neutral hydrogen
    nh = cdf.n1[:,:,0:cdf.nk[iel],iel] # The hydrogen population densities

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
    for i in range(num_times):
        gff[:,:,i] = [1.0 + 3.3512 * x**(-0.3333333) * (6.95e-9 * x * cdf.tg1[isteps[i],:] + 0.5) for x in wavels_ang]



    # Ratio of protons in nlte to ratio of protons assuming lte, for 
    # use in the calculation of the nlte b-f opacity 
    # NOTE:  n* in mihalis is b_c * nstar[*,level]
    b_c = cdf.n1[:,:,cdf.nk[iel]-1,iel] / cdf.nstar[:,:,cdf.nk[iel]-1,iel]
    b_c_he = cdf.n1[:,:,5,2] / cdf.nstar[:,:,5,2]


    # Terms used in the stimulated emission correction required for 
    # nlte opacity (the negative term in alpha_hbf_nlte)
    stim_term = np.zeros([num_waves,cdf.ndep,num_times], dtype=float)
    stim_term_2 = np.zeros([num_waves,cdf.ndep,num_times], dtype=float)
    for i in range(num_times):
        stim_term[:,:,i] = [1e0 - np.exp( hc_k / (x * cdf.tg1[isteps[i],:]) *(-1e0) ) for x in wavels_ang]
        stim_term_2[:,:,i] = [np.exp( hc_k / (x * cdf.tg1[isteps[i],:]) * (-1e0) ) for x in wavels_ang]

    # The constant terms used in the calc of emissivity.
    jcoeff = np.zeros([len(nu),cdf.ndep,num_times], dtype=float)
    for i in range(num_times):
        jcoeff[:,:,i] = [2e0 * cdf.hh * x**3.0 / cdf.cc**2.0 * np.exp(-1.0 * cdf.hh * x / (cdf.bk * cdf.tg1[isteps[i],:])) for x in nu]


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
   # COMPUTE THE OPACITIES AND EMISSIVITIES     ############          
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
                alpha_hbf_nlte[winds[i],:,k] =  ( 
                    phot_crss_3[winds[i]] * ( cdf.n1[isteps[k],:,2,iel] - b_c[isteps[k],:] * cdf.nstar[isteps[k],:,2,iel] * stim_term_2[winds[i],:,k] ) + 
                    phot_crss_4[winds[i]] * ( cdf.n1[isteps[k],:,3,iel] - b_c[isteps[k],:] * cdf.nstar[isteps[k],:,3,iel] * stim_term_2[winds[i],:,k] ) + 
                    phot_crss_5[winds[i]] * ( cdf.n1[isteps[k],:,4,iel] - b_c[isteps[k],:] * cdf.nstar[isteps[k],:,4,iel] * stim_term_2[winds[i],:,k] ) 
                                                )
            
                jbf_hbf_nlte[winds[i],:,k] = ( 
                     jcoeff[winds[i],:,k] * ( 
                         phot_crss_3[winds[i]] * cdf.nstar[isteps[k],:,2,iel] * b_c[isteps[k],:] +
                         phot_crss_4[winds[i]] * cdf.nstar[isteps[k],:,3,iel] * b_c[isteps[k],:] +
                         phot_crss_5[winds[i]] * cdf.nstar[isteps[k],:,4,iel] * b_c[isteps[k],:] )
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
   # H- free-free 
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

    # Include Helium recombination edges
    alpha_hebf65 = np.zeros([num_waves,cdf.ndep,num_times],dtype=float)
    jbf_hebf65 = np.zeros([num_waves,cdf.ndep,num_times],dtype=float)
    alpha_hebf64 = np.zeros([num_waves,cdf.ndep,num_times],dtype=float)
    jbf_hebf64 = np.zeros([num_waves,cdf.ndep,num_times],dtype=float)
    alpha_hebf63 = np.zeros([num_waves,cdf.ndep,num_times],dtype=float)
    jbf_hebf63 = np.zeros([num_waves,cdf.ndep,num_times],dtype=float)
    alpha_hebf62 = np.zeros([num_waves,cdf.ndep,num_times],dtype=float)
    jbf_hebf62 = np.zeros([num_waves,cdf.ndep,num_times],dtype=float)
    if (include_helium == True):
        ielhe = 2
        winds = (np.where((wavels_ang < 3664.41e0)))[0]
        if len(winds) != 0:  
            for i in range(len(winds)):
                for k in range(num_times):
                    alpha_hebf64[winds[i],:,k] =  (
                                    1.38e-17 * (cc_ang/3664.41e0)**3 / 
                                    (cc_ang/wavels_ang[winds[i]])**3 * 
                                    (cdf.n1[isteps[i],:,4,ielhe] - b_c_he[isteps[i],:] * 
                                        cdf.nstar[isteps[i],:,4,ielhe] * stim_term_2[winds[i],:,k])
                                                  ) 
                    jbf_hebf64[winds[i],:,k] = (
                                    jcoeff[winds[i],:,k] * 1.38e-17 * (cc_ang/3664.41e0)**3 / 
                                    (cc_ang/wavels_ang[winds[i]])**3 *  
                                    b_c_he[isteps[i],:] * cdf.nstar[isteps[i],:,4,ielhe] * (cc_ang)/(wavels_ang[winds[i]])**2
                                               )
        winds = (np.where((wavels_ang < 3408.60e0)))[0]
        if len(winds) != 0:  
            for i in range(len(winds)):
                for k in range(num_times):
                    alpha_hebf64[winds[i],:,k] =  (
                                    1.38e-17 * (cc_ang/3408.60e0)**3 / 
                                    (cc_ang/wavels_ang[winds[i]])**3 * 
                                    (cdf.n1[isteps[i],:,3,ielhe] - b_c_he[isteps[i],:] * 
                                        cdf.nstar[isteps[i],:,3,ielhe] * stim_term_2[winds[i],:,k])
                                                  ) 
                    jbf_hebf64[winds[i],:,k] = (
                                    jcoeff[winds[i],:,k] * 1.63e-17 * (cc_ang/3408.60e0)**3 / 
                                    (cc_ang/wavels_ang[winds[i]])**3 *  
                                    b_c_he[isteps[i],:] * cdf.nstar[isteps[i],:,3,ielhe] * (cc_ang)/(wavels_ang[winds[i]])**2
                                               )
        winds = (np.where((wavels_ang < 3110.70e0)))[0]
        if len(winds) != 0:  
            for i in range(len(winds)):
                for k in range(num_times):
                    alpha_hebf63[winds[i],:,k] =  (
                                    9.24e-18 * (cc_ang/3110.70e0)**3 / 
                                    (cc_ang/wavels_ang[winds[i]])**3 * 
                                    (cdf.n1[isteps[i],:,2,ielhe] - b_c_he[isteps[i],:] * 
                                        cdf.nstar[isteps[i],:,2,ielhe] * stim_term_2[winds[i],:,k])
                                                  ) 
                    jbf_hebf63[winds[i],:,k] = (
                                    jcoeff[winds[i],:,k] * 9.24e-18 * (cc_ang/3110.70e0)**3 / 
                                    (cc_ang/wavels_ang[winds[i]])**3 *  
                                    b_c_he[isteps[i],:] * cdf.nstar[isteps[i],:,2,ielhe] * (cc_ang)/(wavels_ang[winds[i]])**2
                                               )
        winds = (np.where((wavels_ang < 2592.80)))[0]
        if len(winds) != 0:  
            for i in range(len(winds)):
                for k in range(num_times):
                    alpha_hebf62[winds[i],:,k] =  (
                                    5.50e-18 * (cc_ang/2592.80e0)**3 / 
                                    (cc_ang/wavels_ang[winds[i]])**3 * 
                                    (cdf.n1[isteps[i],:,1,ielhe] - b_c_he[isteps[i],:] * 
                                        cdf.nstar[isteps[i],:,1,ielhe] * stim_term_2[winds[i],:,k])
                                                  ) 
                    jbf_hebf63[winds[i],:,k] = (
                                    jcoeff[winds[i],:,k] * 5.50e-18 * (cc_ang/2592.80e0)**3 / 
                                    (cc_ang/wavels_ang[winds[i]])**3 *  
                                    b_c_he[isteps[i],:] * cdf.nstar[isteps[i],:,1,ielhe] * (cc_ang)/(wavels_ang[winds[i]])**2
                                               )
    # Total Opacities
    # The sum of:
    #       1) H b-f nlte opacity
    #       2) H b-f upper levels
    #       3) H f-f opacity
    #       4) H- b-f opacity
    #       5) H- f-f opacity
    #       6) Metal background opacity
    #       7) Thomson scattering
    #       8) Rayleigh scattering
    #       9) Helium contributions
    alpha_tot = (
                alpha_hbf_nlte + alpha_hbf_upper + alpha_hff + 
                alpha_hmbf + alpha_hmff)# + 
                #xcontm
                #)

    if (include_scatt == True):  
        alpha_tot = alpha_tot + scatne  + scatrh
    if (include_helium == True): 
        alpha_tot = alpha_tot + alpha_hebf62 + alpha_hebf63 + alpha_hebf64  + alpha_hebf65 

    # Total Emissivity 
    #j_tot = jbf_hbf_nlte + (alpha_hff + alpha_hmbf + alpha_hmff + alpha_hbf_upper + xcontm) * sourceBp 
    j_tot = jbf_hbf_nlte + (alpha_hff + alpha_hmbf + alpha_hmff + alpha_hbf_upper) * SourceBp 
         
    # Add scattering to emissivity   
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
        print(jnu.shape)
        #jlam = jnu * cc_ang / wavels_ang**2. # average intensity J_lambda

    #     j_tot = j_tot + jlam *  (scatne + scatrh + scatrh2) ;+ (1.5e6 - 1.3e5) * scatrh2 ; last term is to emulate NUV radiation from above

    #     ;print,jlam
    #     geom_fact = 1. / (4d * !pi / (0.08 * 2.23))

    #     if mean_int_extra gt 0 then  j_tot = j_tot + (mean_int_extra - 1.3e5) * scatrh2 *geom_fact
    #     j_scat =  jlam *  (scatne + scatrh + scatrh2)
    #     j_scat_h2 = jlam * scatrh2
    #     j_thoms = jlam * scatne
    #     if mean_int_extra gt 0 then j_thoms = j_thoms + (mean_int_extra - 1.3e5) * scatrh2 * geom_fact
    #     if mean_int_extra gt 0 then j_scat = j_scat + (mean_int_extra - 1.3e5) * scatrh2 * geom_fact

    # endif else begin

    #   jlam = 0d

    # endelse
    

    out = {'gauntbf1':gauntbf1, 'gauntbf2':gauntbf2, 'gff':gff, 'SourceBp':SourceBp,
            'alpha_hff':alpha_hff, 'b_c':b_c, 'phot_crss_1':phot_crss_1, 
            'alpha_hbf_nlte':alpha_hbf_nlte, 'jbf_hbf_nlte':jbf_hbf_nlte,
            'alpha_hbf_upper':alpha_hbf_upper, 'nhmin':nhmin, 'alpha_hmbf':alpha_hmbf,
            'alpha_hmff':alpha_hmff, 'scatrh':scatrh, 'scatne':scatne, 'alpha_tot':alpha_tot}

    return out

