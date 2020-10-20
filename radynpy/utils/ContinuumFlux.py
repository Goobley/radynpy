import numpy as np

def continuum_flux(cdf, perhz = False, non_zero = True):

    '''
    A function to return continuum intensity 
    and flux
    Parameters
    __________
    cdf : float
          the radyn readout
    perhx : bool, optional
            Set to output in /Hz rather than /Angstrom
            (default = False)
    non_zero : bool, optional
            Set to remove points with zero intensity 
            (default = True)
   
  
    Graham Kerr, Feb 2nd March 2020
    '''

    ########################################################################
    # Some preliminary set up, constants, and calculation of common terms
    ########################################################################

    # Measure the wavelengths:
	#	-Here frq are the continua wavelengths [nu, krc]
	# 	 where krc is the bound-free transition. 
	#	 nu = 0 is the continua edge
	#	 1:nq[krc] are the frequency points
	#	-Finds the non-zero values, sorts them, and 
	#    extracts the unique elements
    clambda = cdf.cc / cdf.frq[np.where(cdf.frq != 0)] * 1e8  
    clambda = np.sort(clambda)	
    clambda = np.unique(clambda)

    # Number of unique wavelength points and timesteps
    num_waves = (clambda.shape)[0]
    num_times = (cdf.time.shape)[0]
	

    # Arrays to hold the continua flux
    cflux = np.zeros([num_waves, num_times],dtype = float)
    cint = np.zeros([num_waves, cdf.nmu, num_times],dtype=float)
    cfsig = cflux

	# Max number of frequency values
    nfrq = (cdf.frq.shape)[0]

    # Should be added to cdf reader. For now hardcode here, but make sure
    # they are the correct values for your simulation (though I doubt they 
    # ever change)
    wmu = np.array([0.11846344252809000,0.23931433524967999,0.28444444444444000,0.23931433524967999,0.11846344252809000])


    ########################################################################
    #  Extract the continuum flux
    ########################################################################

    for i in range(num_waves):

		# Find the correct indices for wavelength point and transition number
		# for each wavelength in clambda
        qind = (np.where(np.abs(cdf.cc/cdf.frq*1.0e8-clambda[i])/clambda[i] < 1e-5))[0]
        ind = (np.where(np.abs(cdf.cc/cdf.frq*1.0e8-clambda[i])/clambda[i] < 1e-5))[1]

		#kt=where(ktrans-1 eq ind)
        for l in range(num_times): 
			
            cf = np.zeros([len(ind)],dtype=float)
            ci = np.zeros([len(ind),cdf.nmu],dtype=float)

            for j in range(len(ind)):
			
                kt = np.where(cdf.ktrans-1 == ind[j])

				# wmu = gaussian quad. weights and zmu=gaussian quad points
			    # flux is 2pi*integral(I_nu_mu*dmu)
                cf[j] = np.sum( cdf.outint[l, qind[j], :, kt] * cdf.zmu * wmu )
                ci[j,:] = cdf.outint[l, qind[j], :, kt]
   
            cflux[i,l]= 2 * np.pi * np.mean(cf)
            cint[i,:,l] = np.mean(ci,axis=0)

            if perhz==False:
                cflux[i,l] = cflux[i,l]*cdf.cc*1e8/clambda[i]**2.0
                cint[i,:,l] = cint[i,:,l]*cdf.cc*1e8/clambda[i]**2.0
                units = 'erg cm^-2 s^-1 sr^-1 Ang^-1'
            else:
                units = 'erg cm^-2 s^-1 sr^-1 Hz^-1'


            if (len(cf) >= 2): 
                cfsig[i,l]= 2*cdf.pi*np.std(cf)
				
    if non_zero == True:
        
        ind = (np.where(cflux[:,0] != 0.0))[0]
        clambda = clambda[ind]
        cflux = cflux[ind,:]
        cint = cint[ind,:,:]

        



    out = {'clambda':clambda, 'cflux':cflux, 'cint':cint, 'cfsig':cfsig, 'units':units}

    return out