import numpy as np
from scipy import special
import warnings

def voigt_H(a, v):
    '''
    Voigt H function -- same as is used in Lightweaver.
    '''
    z = (v + 1j * a)
    return special.wofz(z).real

def gaunt_bf_h(n, qf):
    # ;+
    # ;   gaunt(nn,qf)
    # ;
    # ;            computes the bound-free gaunt factors for
    # ;            hydrogen n=quantum number of level considered 
    # ;            x=reciprocal wavelength in inverse mincrons 
    # ;            qf = frequency 
    # ;
    # ;-
    # TODO(cmo): Replace with more general one underneath
    warnings.warn('This function is to be replaced with the more general gaunt_bf', DeprecationWarning)
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

def gaunt_bf(wvls, n = 1, Z = 1):

    '''
    Calculates the bound-free Gaunt fn for a given wavelength and quantum
    number.
    
    M. J. Seaton (1960), Rep. Prog. Phys. 23, 313
    
    This is the same calculation that is used in RH.

    Parameters
    ----------

    wvls : `array_like` (1D)
        The wavelengths (Angstrom).
    n : `int`, optional
        Principal quantum number (default = 1).
    Z : `int`, optional
        Charge (default = 1).

    
    Returns
    -------

    gbf : `array`
        The bound-free Gaunt factors.


    Graham Kerr, Feb 17th 2020

    '''

    wvls = np.asarray(wvls)

    x    = (6.626e-27 * 3e18) / wvls / (2.1799e-11 * Z**2)
    x3   = x**(1.0/3.0)
    nsqx = 1.0 / (n**2 * x)     
    gbf = 1.0 + 0.1728*x3 * (1.0 - 2.0*nsqx) - 0.0496*(x3)**2 * (1.0 - (1.0 - nsqx)*(2.0/3.0)*nsqx)

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
    g = gaunt_bf_h(i,frq)
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

def planck_fn(wvls, tg):

    '''
    Calculates the Planck fn in units of [erg / s / cm^2 / sr / AA] given
    wavelength in angstrom and temperature in Kelvin.
    
    e.g. to produce the Planck Fn at 5800K from 2000 to 10000 A every 1
    angstrom:
    ```
        wvls = np.arange(2000,10001,1,dtype=float)
        bb = planck_fn(wvls, tg=5800.00)
    ```

    Parameters
    ----------

    wvls : `array_like` (1D)
        The wavelengths (Angstrom).
    tg : `array_like` (1D)
        Gas temperature (K).

    Returns
    -------

    bb : `array` (2D)
        Black-body intensity as a squeezed 2D array of shape [len(wvls), len(tg)].
   
    Graham Kerr, Feb 18th 2020

    '''

    # Convert to np array in case it is in input 
    # as a regular list
    wvls = np.atleast_1d(np.asarray(wvls))
    tg = np.atleast_1d(np.array(tg))

    # Convert to wavelength in cm
    w = wvls / 1.0e8 

    # Constants appropriate to cgs units.
    c1 =  3.7417749e-05     # =2*pi*h*c**2       
    c2 =  1.4387687e0       # =h*c/k

    bbflux = c1 / (w[:, None]**5 * (np.exp(c2 / tg[None,  :] / w[:, None]) - 1.0))
    bbflux *= 1e-8 / np.pi

    return bbflux.squeeze()
