import numpy as np
from scipy import special

# https://www.harrisgeospatial.com/docs/VOIGT.html
# https://www.uio.no/studier/emner/matnat/astro/AST4310/h15/undervisningsmateriale/ssa_sample.pdf
def voigt_H(a, v):
    z = (v + 1j * a)
    return special.wofz(z).real

def gaunt_factor(n, qf):
    # ;+
    # ;   gaunt(nn,qf)
    # ;
    # ;            computes the bound-free gaunt factors for
    # ;            hydrogen n=quantum number of level considered 
    # ;            x=reciprocal wavelength in inverse mincrons 
    # ;            qf = frequency 
    # ;
    # ;-
    x = qf/2.99793e14
    if n == 1:
        gaunt = 1.2302628 + x*( -2.9094219e-3+x*(7.3993579e-6-8.7356966e-9*x) )+ (12.803223/x-5.5759888)/x
    elif n == 2:
        gaunt = 1.1595421+x*(-2.0735860e-3+2.7033384e-6*x)+ ( -1.2709045+(-2.0244141/x+2.1325684)/x )/x
    elif n == 3: 
        gaunt = 1.1450949+x*(-1.9366592e-3+2.3572356e-6*x)+ ( -0.55936432+(-0.23387146/x+0.52471924)/x )/x
    elif n == 4: 
        gaunt = 1.1306695+ x*( -1.3482273e-3+x*(-4.6949424e-6+2.3548636e-8*x) )+ ( -0.31190730+(0.19683564-5.4418565e-2/x)/x )/x
    elif n == 5: 
        gaunt = 1.1190904+ x*( -1.0401085e-3+x*(-6.9943488e-6+2.8496742e-8*x) )+ ( -0.16051018+(5.5545091e-2-8.9182854e-3/x)/x )/x
    elif n == 6: 
        gaunt = 1.1168376+ x*( -8.9466573e-4+x*(-8.8393133e-6+3.4696768e-8*x) )+ ( -0.13075417+(4.1921183e-2-5.5303574e-3/x)/x )/x
    elif n == 7: 
        gaunt = 1.1128632+ x*( -7.4833260e-4+x*(-1.0244504e-5+3.8595771e-8*x) )+ ( -9.5441161e-2+(2.3350812e-2-2.2752881e-3/x)/x )/x
    elif n == 8: 
        gaunt = 1.1093137+ x*( -6.2619148e-4+x*(-1.1342068e-5+4.1477731e-8*x) )+ ( -7.1010560e-2+(1.3298411e-2-9.7200274e-4/x)/x )/x
    elif n == 9: 
        gaunt = 1.1078717+ x*( -5.4837392e-4+x*(-1.2157943e-5+4.3796716e-8*x) )+ ( -5.6046560e-2+(8.5139736e-3-4.9576163e-4/x)/x )/x 
    elif n == 10: 
        gaunt = 1.1052734+x*( -4.4341570e-4+x*(-1.3235905e-5+4.7003140e-8*x) )+( -4.7326370e-2+(6.1516856e-3-2.9467046e-4/x)/x )/x
    else: 
        gaunt = 1.0

    return gaunt

def gaunt_bf(wvls = [], n = 1, Z = 1, *args):

    '''
    Calculates the bound-free Gaunt fn for a given 
    wavelength and quantum number.

    M. J. Seaton (1960), Rep. Prog. Phys. 23, 313

    This is the same calculation that is used in
    RH 

    Parameters
    __________

    wvls : float
            the wavelength(s)
    n : int, optional
        Principal quantum number (default = 1) 
    Z : int, optional
        charge (default = 1)


    Graham Kerr, Feb 17th 2020

    '''
    gbf = []
    for i in range(len(wvls)):
        x    = (6.626e-27 * 3e18) / wvls[i] / (2.1799e-11 * Z**2)
        x3   = x**(0.33333333e0)
        nsqx = 1e0 / (n**2 * x)     

        gbf.append(1e0 + 0.1728*x3 * (1e0 - 2e0*nsqx) - 0.0496*(x3)**2 * (1e0 - (1e0 - nsqx)*0.66666667*nsqx))

     
    return gbf

def hydrogen_bf_profile(wl, z, i):
    # ;+
    # ;   prfhbf(wl,z,i)
    # ;
    # ;           absorption crossection profile for hydrogenic ion
    # ;
    # ;
    # ;           wl - wavelength
    # ;           z  - nuclear charge
    # ;           i  - level from which absorption takes place, FORTRAN numbering
    # ;
    # ;-
    # ;
    prfhbf = 0.0
    wl0 = 911.7535278/(z*z)*i*i
    if wl > wl0: 
        return 0.0
    
    frq = 2.9979e18/wl
    g = gaunt_factor(i,frq)
    pr0 = 1.04476e-14*z*z*z*z
    a5 = float(i)**5
    wm = wl*1.0e-4
    wm3 = wm*wm*wm
    prfhbf = pr0*wm3*g/a5

    return prfhbf

def hydrogen_absorption(xlamb, icont, temp, ne, nh): #xconth, xconth_lte):
    # ;+
    # ;   abshyd,xlamb,icont,temp,nne,nh,xconth,xconth_lte
    # ;
    # ;            gives total true absorption coefficient for all 
    # ;            hydrogen-combinations in cm**2 per cm**3
    # ;            icont .ge. 0 will exclude hydrogen b-f continuum from level
    # ;            icont
    # ;            based on abshyd.f
    # ;
    # ;-
    cc=2.99792e+10
    bk=1.38066e-16
    ee=1.60219e-12
    hh=6.62618e-27

    xnu=cc*1.0e8/xlamb
    ex=np.exp(-1.438786e8/temp/xlamb)      # stimulated emission correction
    ndep = len(temp)
#     xconth = np.zeros(ndep)
    totnhi = nh[:,:5].sum(axis=1) # neutral hydrogen
#     ;
#     ;  free-free contribution from hydrogen minus ion
#     ;  expressions from gray p. 149-150. note misprint of first
#     ;  term of f0 in gray 
#     ;                   
    if xlamb > 3038.0 and xlamb < 91130.0:
        x1=np.log10(xlamb)
        x2=x1*x1
        x3=x2*x1
        f0=-31.63602+0.48735*x1+0.296585*x2-0.0193562*x3
        f1=15.3126-9.33651*x1+2.000242*x2-0.1422568*x3
        f2=-2.6117+3.22259*x1-1.082785*x2+0.1072635*x3
        thlg=np.log10(5040./temp)
        thlg2=thlg*thlg
        abhmff=ne*bk*temp*10.0**(f0+f1*thlg+f2*thlg2)*totnhi
    else:
        abhmff=np.zeros(ndep)
#     ; 
#     ;  bound-free contribution from hydrogen minus ion
#     ;  expressions from gray p. 147-149
#     ;
    if xlamb > 1500.0 and xlamb < 20000.0:
        if xlamb < 5250.0:
            f0=-16.20450
            f1=0.17280e-3
            f2=0.39422e-7
            f3=0.51345e-11
        elif xlamb < 11250.0:
            f0=-16.40383
            f1=0.61356e-6
            f2=-0.11095e-7
            f3=0.44965e-11
        elif xlamb < 20000.0:
            f0=-15.95015
            f1=-0.36067e-3
            f2=0.86108e-7
            f3=-0.90741e-11
        crhmbf=f0+f1*(xlamb-8500.)+f2*(xlamb-8500.)**2+f3*(xlamb-8500.)**3
        crhmbf=10.0**crhmbf
        abhmbf=1.03526e-16*ne*crhmbf/temp**1.5* np.exp(0.754*ee/bk/temp)*totnhi*(1.0-ex)
    else:
        abhmbf=np.zeros(ndep)
#     ;
#     ;  free-free contribution from hydrogen
#     ;  expressions from gray p. 146-147
#     ;                          
    gff=1.0+3.3512*xlamb**(-0.3333333)*(6.95e-9*xlamb*temp+0.5)
    abhff=3.69e8/np.sqrt(temp)*ne/xnu/xnu/xnu*nh[:,5]*(1.0-ex)*gff
#     ;
#     ;  bound-free contribution from hydrogen bound-free, high levels
#     ;  opacity from levels above nk(iel)-1 up to level 8 are included
#     ;  assuming lte populations relative to the continuum
#     ;
    zz=1.0               
    abhbf=np.zeros(ndep)
    for i in range(6,9): #; FORTRAN level numbering
        crhbf=hydrogen_bf_profile(xlamb,zz,i)
        if crhbf > 0.0:
            hnukt=157896./(i*i*temp)
            const=4.1416e-16*ne*nh[:,5]/(temp*np.sqrt(temp))
            pop=i*i*const*np.exp(hnukt)
            abhbf=abhbf+crhbf*pop*(1.0-ex)
    xconth_lte = abhmbf+abhmff+abhff+abhbf
#     ;
#     ;  bound-free contribution from hydrogen, explicit levels
#     ;                          
    zz=1.0               
    for i in range(1,6):
        if i != icont: 
            crhbf=hydrogen_bf_profile(xlamb,zz,i)
            if crhbf > 0.0:
                hnukt=157896./(i*i*temp)
                const=4.1416e-16*ne*nh[:,5]/(temp*np.sqrt(temp))
                pop=i*i*const*np.exp(hnukt)
                abhbf=abhbf+crhbf*(nh[:,i-1]-pop*ex)
                xconth_lte=xconth_lte+crhbf*pop*(1.0-ex)

    xconth=abhmbf+abhmff+abhbf+abhff
    return xconth, xconth_lte

def planck_fn(wvls = [], tg = [], *args):

    '''
    Calculates the Planck fn in units of 
    [ergs /s / cm^2 / sr / ang] given wavelength
    in angstrom and temperature in Kelvin. 

    e.g. to produce the Planck Fn at 5800K 
    from 2000 to 10000 A every 1 angstrom:
        wvls = np.arange(2000,10001,1,dtype=float)
        bb = planck_fn(wvls, tg=5800.00)

    Parameters
    __________

    wvls : float
           the wavelength(s)
    tg : float
         gas temperature 
   
    Graham Kerr, Feb 18th 2020

    '''

    # Convert to np array in case it is in input 
    # as a regular list
    wvls = np.array(wvls)
    tg = np.array(tg)

    # Convert to wavelength in cm
    w = wvls / 1.0e8 

    # Constants appropriate to cgs units.
    c1 =  3.7417749e-05     # =2*!DPI*h*c*c       
    c2 =  1.4387687e0       # =h*c/k

    bbflux = np.zeros([len(wvls),len(tg)],dtype=float)
    for i in range(len(tg)):
        bbflux[:,i] =  ([c1 / ( x**5 * ( np.exp( c2/tg[i]/x )-1.e0 ) ) for x in w]) 

    bbint = bbflux*1.0e-8/np.pi

    return bbint


def prfhbf_rad(wvls = [], Z = 1, n=6, *args):

    '''
    A function to return the absorption 
    cross section fr a hydrogenic ion 

    *** Originaly from abshyd.pro subroutine
        in RADYN, and uses Fortran indexing.
        Using this code to ensure that we 
        compute the upper level contribution
        to the opacity in the same way that 
        RADYN does internally.

    Parameters
    __________

    wvls : float
            the wavelength(s)
    n : int, optional
        The level from which absorption
        takes place. FORTRAN NUMBERING, legacy
        from RADYN (default = 6, i.e. level 5)
    Z : int, optional
        charge (default = 1)


    Graham Kerr, Feb 19th 2020

    '''
    wvls = np.array(wvls)

    prfhbf = np.zeros([len(wvls)],dtype=float)

    wl0 = 911.7535278/(Z*Z)*n*n

    for i in range(len(wvls)):

        if (wvls[i] > wl0):    
          
            prfhbf[i] = 0.0
        
        else: 
    
            frq = 2.9979e18/wvls[i]

            gau = gaunt_factor(n,frq)
            
            pr0 = 1.04476e-14*Z*Z*Z*Z
    
            a5 = n**5
            a5 = np.float(a5)

            wm = wvls[i]*1.0e-4

            wm3 = wm*wm*wm

            prfhbf[i] = pr0*wm3*gau/a5

     
    return prfhbf

def hminus_pops(tg1, ne1, nh, bhmin=1.0, *args):

    '''
    A function to return the H minus populations

    Parameters
    __________

    tg1 : float
          the temperature stratification 
    nh  : float
          the population densities of hydrogen 
    bhmin : float, optional
            the departure coefficient from LTE
            for H minus (default = 0).

    Make sure the tg, and nh have dimensions of 
    [timesteps, height], and [timesteps, height, level]

   

    The density of H minus from  
    Vernazza et al. 1976, ApJS 30

    Graham Kerr, Feb 19th 2020

    '''
    ndep = (tg1.shape)[1]
    num_times = (tg1.shape)[0]
    num_levs = (nh.shape)[2]

    nhmin = np.zeros([ndep,num_times],dtype=float)
    totnhi = np.sum(nh[:,:,0:num_levs-1],axis = 2) # Density of neutral hydrogen

    for k in range(num_times):
        nhmin[:,k] = (
                1.0354e-16 * bhmin * ne1[k,:] * totnhi[k,:] * 
                 tg1[k,:]**(-1.5) * np.exp(8762e0/tg1[k,:])
                     )

    return nhmin

def transp_scat(tau=[],x=[],sc=[],scat=[],
                *args):

    '''
    A function to return the average intensity J at some
    wavelength  

    Parameters
    __________
    
    tau : float 
          the optical depth at some standard wavelength 
          (usually 5000A)
    x : float
        the ratio of total opacity to the opacity at some
        standard wavelength
    sc : float
         epsilon B
    scat: float
         1 - epsilon 

    

    Graham Kerr, Feb 20th 2020

    '''
    ntimes = (tau.shape)[0]
    ndep = (tau.shape)[1]
    nwave = (x.shape)[0]


    nmu=3
    wmu=np.array([0.277778,0.444444,0.277778], dtype=float)
    xmu=np.array([0.112702,0.500000,0.887298], dtype=float)

    itran=0

    sp1=np.zeros([nmu,nmu,nwave,ndep,ntimes],dtype=float)
    sp2=np.zeros([nmu,nmu,nwave,ndep,ntimes],dtype=float)
    sp3=np.zeros([nmu,nmu,nwave,ndep,ntimes],dtype=float)
    a1=np.zeros([nwave,ndep,ntimes],dtype=float)
    c1=np.zeros([nwave,ndep,ntimes],dtype=float)
    p=np.zeros([nmu,nwave,ndep,ntimes],dtype=float)
    iplus=np.zeros([nmu,nmu,nwave,ndep,ntimes],dtype=float)
    iminus=np.zeros([nwave,ndep,ntimes],dtype=float)
    pms=np.zeros([nwave,ndep,ntimes],dtype=float)
    tauq=np.zeros([nwave,ndep,ntimes],dtype=float)
    dtauq=np.zeros([nwave,ndep,ntimes],dtype=float)
    jnu=np.zeros([nwave,ndep,ntimes],dtype=float)

    #
    # k=1: upper boundary
    #
    cmu=0.5/xmu[0]
    for k in range(ntimes):
        for i in range(nwave):
            dtauq[i,1,k]=((x[i,0,k]+x[i,1,k])*(tau[k,1]-tau[k,0]))*cmu
            a1[i,0,k]=1./dtauq[i,1,k]
            t = tau[k,0]*x[i,0,k]*2.0*cmu
            tauq[i,0,k]=t
            dtauq[i,0,k]=t

            #
            # calculate dtauq
            #
            dtauq[i,1:ndep-1,k]=(x[i,1:ndep-1,k]+x[i,0:ndep-2,k])*(tau[k,1:ndep-1]-tau[k,0:ndep-2])*cmu
         
            #  
            # calculate tauq
            #
            for j in range(ndep): 
                tauq[i,j,k]=tauq[i,k-1,k]+dtauq[i,j,k]

            #
            #  calculate a1 and c1
            #
            a1[i,1:ndep-2,k] = 2./(dtauq[i,1:ndep-2,k]+dtauq[i,2:ndep-1,k])/dtauq[i,1:ndep-2,k]
            c1[i,1:ndep-2,k] = 2./(dtauq[i,1:ndep-2,k]+dtauq[i,2:ndep-1,k])/dtauq[i,2:ndep-1,k]

            #  call formal solver
            #
            #  calculate tridiagonal coefficients
            #
            for mu in range(nmu):
                xmu1=xmu[mu]/xmu[0]
                xmu2=xmu1*xmu1

                #
                # k=1: upper boundary
                #
                t=tauq[i,0,k]/xmu1
                if (t < 0.01):
                    ex1=t*(1.-t*(0.5-t*(0.1666667-t*0.041666667)))
                elif(t < 20.):
                    ex1=1.-np.exp(-t) 
                else:
                    ex1=1.
  
                ex = 1.-ex1
                sp1[mu,mu,i,0,k] = 0.
                sp2[mu,mu,i,0,k] = 1.0+2.0*xmu1*a1[i,0,k]
                sp3[mu,mu,i,0,k] = -2.0*xmu2*a1[i,0,k]*a1[i,0,k]
                fact = 1.0+2.0*xmu1*a1[i,0,k]*ex1
                sp2[mu,mu,i,0,k] = sp2[mu,mu,i,0,k]/fact
                sp3[mu,mu,i,0,k] = sp3[mu,mu,i,0,k]/fact
                p[mu,i,0,k]=sc[i,0,k]

                #
                # interior
                #
                sp1[mu,mu,i,1:ndep-2,k]=-xmu2*a1[i,1:ndep-2,k]
                sp2[mu,mu,i,1:ndep-2,k]=1.0
                sp3[mu,mu,i,1:ndep-2,k]=-xmu2*c1[i,1:ndep-2,k]
                p[mu,i,1:ndep-2,k]=sc[i,1:ndep-2,k]

                #
                # k=ndep: lower boundary
                #
                sp1[mu,mu,i,ndep-1,k]=-1.0                        
                sp2[mu,mu,i,ndep-1,k]=dtauq[i,ndep-1,k]/xmu1+0.5*dtauq[i,ndep-1,k]**2/xmu2
                sp3[mu,mu,i,ndep-1,k]=0.0
                sk=(dtauq[i,ndep-1,k]/xmu1+0.5*dtauq[i,ndep-1,k]**2/xmu2)+1.0
                p[mu,i,ndep-1,k]=sc[i,ndep-1,k]*sk - sc[i,ndep-2,k]

                # 
                # non-diagonal elements
                #
                for mu2 in range(nmu): 
                    sp2[mu,mu2,i,0,k] = sp2[mu,mu2,i,0,k]-scat[i,0,k]*wmu[mu2]
                    sp2[mu,mu2,i,1:ndep-2,k] = sp2[mu,mu2,i,1:ndep-2,k]-scat[i,1:ndep-2,k]*wmu[mu2]
                    sp2[mu,mu2,i,ndep-1,k] = ( sp2[mu,mu2,i,ndep-1,k]-scat[i,ndep-1,k]*wmu[mu2]*sk + 
                                               scat[i,ndep-2,k]*wmu[mu2])
                    sp1[mu,mu2,i,ndep-1,k]= sp1[mu,mu2,i,ndep-1,k]+scat[i,ndep-2,k]*wmu[mu2]

            #
            # eliminate subdiagonal
            #
            for j in range(0,ndep-1): 
                sp1p = sp1[:,:,i,j+1,k]
                sp2k = sp2[:,:,i,j,k]
                sp3k = sp3[:,:,i,j,k]
                f = -sp1p#invert(sp2k-sp3k)
                p[:,i,j+1,k] = p[:,i,j+1,k]+np.matmul(f,(p[:,i,j,k]))
                sp2[:,:,i,j+1,k] = sp2[:,:,i,j+1,k]+np.matmul(f,sp2k)
                sp2[:,:,i,j,k] = sp2[:,:,i,j,k] - sp3[:,:,i,j,k]

            sp2[:,:,i,ndep-1,k]=sp2[:,:,i,ndep-1,k]-sp3[:,:,i,ndep-1,k]

            #
            # backsubstitute
            #
            p[:,i,ndep-1,k] = np.matmul(np.linalg.inv(sp2[:,:,i,ndep-1,k]),p[:,i,ndep-1,k])
            for j in range(ndep-2, -1, -1):
                p[:,i,j,k]= np.matmul( 
                                   np.linalg.inv(sp2[:,:,i,j,k]), np.matmul(p[:,i,j,k]-sp3[:,:,i,j,k],p[:,i,j+1,k]) )


            
            for mu in range(nmu):
                jnu[i,:,k] = jnu[i,:,k] + wmu[mu]*p[mu,i,:,k]


    return jnu