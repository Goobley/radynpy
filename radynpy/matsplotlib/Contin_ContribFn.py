import numpy as np
from radynpy.utils import gaunt_bf, planck_fn


def contin_contrib_fn(cdf, isteps= [0], wavels_ang = [6690.00], mu_ind = -1,
                      include_metals = True, include_helium = True, 
                      scatt = True):

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
        
    
    Note that Adam's version includes some additional terms for Landau-Zener, Mg II wing intenstity
    and H2 (I think mainly useful/necessary for M Dwarf atmos.). Will check with him if these should 
    be included here. 

    There may also be redundant routines here that are already defined elsewhere in radynpy. Will
    catch these eventually.

    This uses the routines in opacity.py, in radynpy.matsplotlib to read opcatb.dat   
    '''

    ########################################################################
    # Some preliminary set up and constants
    ########################################################################

    

    # Turn the inputs into numpy arrays
    wavels_ang = np.array(wavels_ang)

    # Size of arrays
    num_times = len(isteps)
    num_waves = len(wavels_ang)

    # The viewing angle
    mu = cdf.zmu[mu_ind]
    
    # Hydrogen is element index 0
    iel = 0

    # Arrays to hold various variables later, and reduce other variables
    xconth = np.zeros(cdf.ndep, dtype = float) #  metal background opacity 
    totnhi = np.sum(cdf.n1[:,:,0:cdf.nk[iel]-2,iel],axis = 2) # Density of neutral hydrogen
    nh = cdf.n1[:,:,0:cdf.nk[iel]-1,iel] # The hydrogen population densities

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



        

    out = {'gauntbf1':gauntbf1, 'gauntbf2':gauntbf2, 'gff':gff, 'SourceBp':SourceBp,
            'alpha_hff':alpha_hff, 'b_c':b_c, 'phot_crss_1':phot_crss_1, 
            'alpha_hbf_nlte':alpha_hbf_nlte, 'jbf_hbf_nlte':jbf_hbf_nlte}

    return out

