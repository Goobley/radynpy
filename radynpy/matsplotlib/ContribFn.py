import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import pchip_interpolate, interp1d, griddata
from skimage.exposure import equalize_adapthist, equalize_hist
from palettable.colorbrewer.sequential import Blues_9, PuBu_9
from colour import Color
from matplotlib.ticker import MaxNLocator, LogLocator, ScalarFormatter
import warnings
from radynpy.matsplotlib import xt_calc

def contrib_fn(cdf, kr, tStep=0, yRange=[-0.08, 2.5], vRange=[300.0, -300.0], mu=-1, 
               heatPerParticle=False, wingOffset=None,
               dvMouseover=False, withBackgroundOpacity=True, colors={}, 
               tightenLayout=True, printMiscTopData=True, stdoutInfo=False, returnData=False,
               opctabPath=None):
    '''
    Plots the contribution function of a transition. Based on the original 4
    panel Mats Carlsson plot, but using the 6 panel layout of Paulo Simoes.
    Gratefully developed from the IDL scripts of M. Carlsson, G. Kerr, and P. Simoes.
    This function produces a plot in matplotlib, that will then need to be shown or printed.

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
        The height range to plot over, in Mm. (default: [-0.08, 2.5])
    vRange : list of float, optional
        The velocity range to plot over (converted to a wavelength range around the transition core). (default: [-300, 300])
    mu : int, optional
        Index of the mu ray to be used. Default is the closest to the normal to the atmosphere. (default: -1)
    heatPerParticle : bool, optional
        Display heating per particle or per unit volume. (default: False)
    wingOffset : float, optional
        The offset of the slice to take through the wing in Angstrom from the line core.
    dvMouseover : bool, optional
        Display the mouseover x-values in units of velocity (default: False, angstrom are used).
    withBackgroundOpacity : bool, optional
        Include the background opacity at this wavelength in the calculation. (default: True)
    colors : dict[str, str], optional
        Dictionary of line names and the hex code to color them.
        Keys and default values:
            'tau': Color('OrangeRed').hex,
            'vz': Color('Chocolate').hex,
            'sv': Color('OrangeRed').hex,
            'bb': Color('LightSeaGreen').hex,
            'pop1': Blues_9.mpl_colors[3],
            'pop2': Blues_9.mpl_colors[6],
            'ne': Color('Coral').hex,
            'line': Color('LightSeaGreen').hex,
            'temp': Color('OrangeRed').hex,
            'cf': Blues_9.mpl_colors[6],
            'heat': Color('Salmon').hex,
            'core': Blues_9.mpl_colors[6],
            'wing': Blues_9.mpl_colors[3]
    tightenLayout : bool, optional
        Tighten the whitespace around the plots. (default: True)
    printMiscTopData : bool, optional
        Add data describing the plot to the top of the figure (line, frequency, timestep...). (default: True) 
    stdoutInfo : bool, optional
        Print progress information to stdout. (default: False)
    returnData : bool, optional
        Return a dictionary of the data computed to produce the plots. (default: False)
    opctab : str, optional
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
        outMu = cdf.outint[tStep,:,mu,:]
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

        # Plotting code starts here

        # returns the radiation temperature for arrays of specific intensity and wavelength (Angstrom)
        def radiation_temperature(intens, lamb):
            c = 2.99792458e10
            h = 6.626176e-27
            k = 1.380662e-16
            l = lamb*1e-8
            tRad = h * c / k / l / np.log(2.0 * h * c / intens / (l**3) + 1.0)
            return tRad

        # Set up our default colours, non-hideous
        defaultColors = {
            'tau': Color('OrangeRed').hex,
            'vz': Color('Chocolate').hex,
            'sv': Color('OrangeRed').hex,
            'bb': Color('LightSeaGreen').hex,
            'pop1': Blues_9.mpl_colors[3],
            'pop2': Blues_9.mpl_colors[6],
            'ne': Color('Coral').hex,
            'line': Color('LightSeaGreen').hex,
            'temp': Color('OrangeRed').hex,
            'cf': Blues_9.mpl_colors[6],
            'heat': Color('Salmon').hex,
            'core': Blues_9.mpl_colors[6],
            'wing': Blues_9.mpl_colors[3],
        }

        choose_color = lambda col: colors[col] if col in colors else defaultColors[col]

        tauColor = choose_color('tau')
        vzColor = choose_color('vz')
        svColor = choose_color('sv') 
        bbColor = choose_color('bb')
        pop1Color = choose_color('pop1')
        pop2Color = choose_color('pop2')
        neColor = choose_color('ne')
        lineColor = choose_color('line')
        tempColor = choose_color('temp')
        cfColor = choose_color('cf') 
        heatColor = choose_color('heat')
        coreColor = choose_color('core')
        wingColor = choose_color('wing')

        # Code to add the wavelength axis to plots, in addition to the delta-vel axis
        def add_wl_axis(ax, label=False, numbers=False):
            wlAx = ax.twiny()
            lambdaRange = -np.array(ax.get_xlim())*cdf.alamb[kr] / cdf.cc * 1e5
#             wlAx.set_xlim(lambdaRange[-1], lambdaRange[0])
            wlAx.set_xlim(lambdaRange)
            wlAx.tick_params('x', direction='in')
            if label:
#                 wlAx.set_xlabel(r'$\Delta\lambda$ [$\AA$]')
                wlAx.set_xlabel(r'$\lambda-\lambda_0$ [$\AA$]')
            else:
                wlAx.tick_params('x', labeltop=numbers)
            if dvMouseover:
                axDupe = ax.figure.add_axes(ax.get_position(True), sharex=ax, sharey=ax, frameon=False)
                axDupe.xaxis.set_visible(False)
                axDupe.yaxis.set_visible(False)

        # add the vertical lines for the core and wings to the image plots
        def add_core_wing_lines(ax):
            ax.axvline(dVel[wlIdxs[coreIdx]], c=coreColor)
            ax.axvline(dVel[wlIdxs[wingIdx]], c=wingColor)

        # Add a customised legend, with no box, or line segment, but the text with the line's colour
        def add_legend(ax, loc='lower right'):
            leg = ax.legend(loc=loc, handlelength=0, frameon=False, labelspacing=0.0)
            for line, text in zip(leg.get_lines(), leg.get_texts()):
                text.set_color(line.get_color())

        # Add a simple black label to the upper left corner of an image plot
        def add_image_label(ax, label):
            ax.annotate(label, xy=(0.04, 0.89), xycoords=ax.transAxes)

        # make figure
        fig, ax = plt.subplots(3,2, figsize=(8, 10), sharey=True, constrained_layout=True)

        ## First image plot: emissivity
        ix = 0
        iy = 0
        z = np.copy(zTot[np.ix_(iwx, iwy)][:,:,ix, iy])
        z = np.transpose(z, (1,0))
        zi = np.log10(z)
        ziMax = np.max(zi[np.isfinite(zi)])
        ziMin = np.min(zi[np.isfinite(zi)])
        zi = np.clip(zi, ziMin, ziMax)

        # Use histogram equalisation to improve contrast - thanks Paulo!
        zi = equalize_hist(zi)

        # These plots are like images on irregular "pixel" centres, pcolormesh can handle this though, 
        # if we work out where the edges are
        xEdges = 0.5 * (x[iwx][:-1] + x[iwx][1:])
        xEdges = np.insert(xEdges, 0, x[iwx][0]) 
        xEdges = np.insert(xEdges, -1, x[iwx][-1])

        yEdges = 0.5 * (y[iwy][:-1] + y[iwy][1:])
        yEdges = np.insert(yEdges, 0, y[iwy][0])
        yEdges = np.insert(yEdges, -1, y[iwy][-1])

        ax[0,0].pcolormesh(xEdges, yEdges, zi, cmap=PuBu_9.mpl_colormap)

        # Link all the subplot axes and invert the x-axis for readability
        ax[0,0].invert_xaxis()
        ax[0,0].get_shared_x_axes().join(ax[0,0], ax[1,0])
        ax[0,0].get_shared_x_axes().join(ax[0,0], ax[2,0])
        
        # Save xlim as it can be clobbered by plotting some of the lines 
        lim = ax[0,0].get_xlim()
        
        # Add the fluid velocity and tau=1 line
        ax[0,0].plot(cdf.vz1[tStep][iwy] * 1e-5, y[iwy], c=vzColor, ls='--', label=r'v$_\mathrm{z}$')
        ax[0,0].plot(x, tau1, c=tauColor, label=r'$\tau_\nu$=1')

        
        # Add the extra info
        # We now add the wl axis at the end of the image plots, since there are so many 
        # things that seem to want to mess with the axis scale, and after plotting everything 
        # we can set it for good, then add these extra axes, as they are computed relative 
        # to the axis limits at the time of creation
#         add_wl_axis(ax[0,0], label=True)
        add_legend(ax[0,0], loc='upper right')
        add_core_wing_lines(ax[0,0])

        # Label and turn ticks in for space
        ax[0,0].set_xlabel(r'$\Delta$v [km s$^{-1}$]')
        ax[0,0].set_ylabel('Height (Mm)')
        ax[0,0].tick_params('both', direction='in')
        add_image_label(ax[0,0], r'$\chi_\nu$ / $\tau_\nu$')

        # Second image plot: opacity
        ix = 0
        iy = 1
        z = np.copy(zTot[np.ix_(iwx, iwy)][:,:,ix, iy])
        z = np.transpose(z, (1,0))
        zi = z
        ziMax = np.max(zi[np.isfinite(zi)])
        ziMin = np.min(zi[np.isfinite(zi)])
        zi = np.clip(zi, ziMin, ziMax)

        # Same again
        ax[1,0].pcolormesh(xEdges, yEdges, zi, cmap=PuBu_9.mpl_colormap)
        ax[1,0].plot(cdf.vz1[tStep][iwy] * 1e-5, y[iwy], c=vzColor, ls='--')
        ax[1,0].plot(x, tau1, c=tauColor)
#         add_wl_axis(ax[1,0], label=True)
        add_core_wing_lines(ax[1,0])

        ax[1,0].set_xlabel(r'$\Delta$v [km s$^{-1}$]')
        ax[1,0].set_ylabel('Height (Mm)')
        ax[1,0].tick_params('both', direction='in')
        add_image_label(ax[1,0], r'$\tau_\nu$ exp($-\tau_\nu$)')

        # Third image plot: contribution functions and line profile
        ix = 1
        iy = 1
        z = np.copy(zTot[np.ix_(iwx, iwy)][:,:,ix, iy])
        z = np.transpose(z, (1,0))
        zi = np.log10(z)
        ziMax = np.max(zi[np.isfinite(zi)])
        ziMin = np.min(zi[np.isfinite(zi)])
        zi = np.clip(zi, ziMin, ziMax)

        # Same again
        zi = equalize_hist(zi)
        ax[2,0].pcolormesh(xEdges, yEdges, zi, cmap=PuBu_9.mpl_colormap)
        ax[2,0].plot(cdf.vz1[tStep][iwy] * 1e-5, y[iwy], c=vzColor, ls='--')
        ax[2,0].plot(x, tau1, c=tauColor)
        # Adjust line profile to fill plot
        lineProfile = cdf.outint[tStep, 1:cdf.nq[kr]+1, mu, kr]
        lineProfile -= lineProfile.min()
        lineProfile /= lineProfile.max()
        lineProfile *= (y[iwy][0] - y[iwy][-1])
        lineProfile += y[iwy][-1]
        ax[2,0].plot(dVel, lineProfile, c=lineColor, label='Line Profile')
        # axes and labelling
#         add_wl_axis(ax[2,0], label=True)
        add_core_wing_lines(ax[2,0])
        ax[2,0].set_xlabel(r'$\Delta$v [km s$^{-1}$]')
        ax[2,0].set_ylabel('Height (Mm)')
        ax[2,0].tick_params('both', direction='in')
        add_image_label(ax[2,0], r'C$_\mathrm{I}$')
        add_legend(ax[2,0], loc='upper right')
        
        # restore xlim from ages ago and add the wavelength axes
        ax[0,0].set_xlim(lim)
        add_wl_axis(ax[0,0], label=True)
        add_wl_axis(ax[1,0], numbers=True)
        add_wl_axis(ax[2,0], numbers=True)

        # The "normal" (line) plots auto-adjust the y range, so we want to preserve it and reapply it at the end
        yLim = ax[0,0].get_ylim()

        # Plot 4: Level populations
        pop1 = np.log10(cdf.n1[tStep, iwy, iTrans, iel])
        pop2 = np.log10(cdf.n1[tStep, iwy, jTrans, iel])
        ne = np.log10(cdf.ne1[tStep, iwy])
        if iel == 0:
            def make_label(i):
                ionI = 'I' * cdf.ion[i, iel]
                ionI = r'$_\mathrm{' + ionI + '}$'
                return 'H ' + ionI + ' ' + cdf.label[i, iel].strip()
        else:
            def make_label(i):
                l = cdf.label[i, iel].strip().split(' ')
                l[0] = l[0][0].upper() + l[0][1:]
                l[1] = r'$_\mathrm{' + l[1].upper() + '}$'
                return ' '.join(l)
        labelI = make_label(iTrans)
        labelJ = make_label(jTrans)
            
        popRange = (lambda v: [np.min(v), np.max(v)])(np.stack((pop1, pop2)))
        ax[0,1].plot(pop1, y[iwy], c=pop1Color, label=labelI)
        ax[0,1].plot(pop2, y[iwy], c=pop2Color, label=labelJ)
        # dummy plot for legend
        ax[0,1].plot(pop1[1], y[iwy][1], c=neColor, label='electron')
        neAx = ax[0,1].twiny()
        neAx.plot(ne, y[iwy], c=neColor)
        ax[0,1].tick_params('both', direction='in')
        neAx.tick_params('x', direction='in')
        ax[0,1].set_xlabel(r'log$_{10}$ ion density [cm$^{-3}$]')
        neAx.set_xlabel(r'log$_{10}$ electron density [cm$^{-3}$]')
        add_legend(ax[0,1], loc='upper right')

        # Plot 5: Radiation Temperatures, and Core and Wing opacities
        # Source function
        sv0 = sll
        # Black-body function
        # bb0 = blackbody_nu(Q(cdf.alamb[kr], 'Angstrom'), cdf.tg1[tStep, iwy])
        # bbTemp = np.log10(radiation_temperature(bb0.value, cdf.alamb[kr]))
        # The radiation temperautre of a blackbody is simply its temperature, 
        # by definition, so we don't need the blackbody stuff
        bbTemp = np.log10(cdf.tg1[tStep])
        svTemp = np.log10(radiation_temperature(sv0, cdf.alamb[kr]))

        ax[1,1].plot(bbTemp[iwy], y[iwy], c=bbColor, label='T')
        ax[1,1].plot(svTemp[iwy], y[iwy], c=svColor, label=r'T(S$_\nu$)')
        # dummy plots for legend
        ax[1,1].plot(bbTemp[iwy][1], 0, c=coreColor, label=r'$\tau$ core')
        ax[1,1].plot(bbTemp[iwy][1], 0, c=wingColor, label=r'$\tau$ wing')
        ax[1,1].tick_params('both', direction='in')

        tauAx = ax[1,1].twiny()
        tauAx.semilogx(tauq_ny[iwy, wlIdxs[coreIdx]], y[iwy], c=coreColor)
        tauAx.semilogx(tauq_ny[iwy, wlIdxs[wingIdx]], y[iwy], c=wingColor)
        tauAx.set_xlim([0.5e-5, 200])
        tauAx.tick_params('x', which='both', direction='in')
        tauAx.xaxis.set_major_locator(LogLocator(numticks=4))
        ax[1,1].set_xlabel(r'log$_{10}$ Temperature [K]')
        tauAx.set_xlabel(r'$\tau$')
        add_legend(ax[1,1])

        # Plot 6: Heating, and Core and Wing Contribution Functions 
        heat = cdf.bheat1[tStep, iwy]
        if heatPerParticle:
            # Only using hydrogen density
            heat /= cdf.n1[tStep,:,:6,0].sum(axis=1)[iwy]
        contFn = np.copy(z)
        contFn = np.log10(contFn / np.max(contFn))
        contFnRange = np.maximum([np.nanmin(contFn), np.nanmax(contFn)], np.nanmax(contFn) - 10.2) 

        ax[2,1].plot(contFn[:,coreIdx], y[iwy], c=coreColor, label=r'Core C$_\mathrm{I}$')
        ax[2,1].plot(contFn[:,wingIdx], y[iwy], c=wingColor, label=r'Wing C$_\mathrm{i}$')
        # dummy plot for legend
        ax[2,1].plot(0, 0, c=heatColor, label='Heat')
        ax[2,1].set_xlim(contFnRange)


        # We have to scale heat manually since matplotlib doesn't realise it needs put the offset 
        # (e.g. x1e-8) at the top, and there's no way to move it
        heatAx = ax[2,1].twiny()
        if 1e-2 <= np.max(heat) <= 1e3:
            # Don't need to scale
            heatAx.plot(heat, y[iwy], c=heatColor)
            offsetStr = ''
        else:
            power = np.floor(np.log10(np.max(heat)))
            scale = 10**power
            heat /= scale
            heatAx.plot(heat, y[iwy], c=heatColor)
            offsetStr = '10$^\mathrm{%d}$ ' % power

        ax[2,1].tick_params('both', direction='in')
        heatAx.tick_params('x', direction='in')
        ax[2,1].set_xlabel(r'log$_{10}$ C$_\mathrm{I}$ (normalised)')

        if heatPerParticle:
            heatAx.set_xlabel(r'Heating per H particle [%serg s$^{-1}$]' % offsetStr)
        else:
            heatAx.set_xlabel(r'Heating [%serg s$^{-1}$ cm$^{-3}$]' % offsetStr)

        add_legend(ax[2,1])

        # Add the timestep, viewing angle and Core Wavelength to the top of the plot
        if printMiscTopData:
            timeVal = cdf.time[tStep]
            muVal = cdf.zmu[mu]
            lineCoreVal = cdf.alamb[kr]
            ax[0,0].annotate(r't = %.1f s  $\mu$ = %.4f''\n'r'Line Core: %.1f$\AA$' % (timeVal, muVal, lineCoreVal),
                             xy=(0.0,1.1), xycoords=('axes fraction', 'axes fraction'), ha='left', va='bottom')



        # reapply clobbered yLim
        ax[0,0].set_ylim(yLim)
        # Tighten the layout to maximise graph space
        if tightenLayout:
            fig.set_constrained_layout_pads(h_pad=20.0/72.0, w_pad=0.01,
                                            hspace=-0.18, wspace=0.01)
            
        if returnData:
            out = {'atomId': cdf.atomid[iel], 
                   'kr': kr,
                   'iel': iel,
                   'levels': [jTrans, iTrans], 
                   'labels': [labelI, labelJ],
                   'emissivity': zTot[np.ix_(iwx, iwy)][:,:, 0, 0],
                   'opacity': zTot[np.ix_(iwx, iwy)][:,:, 0, 1],
                   'contFn': zTot[np.ix_(iwx, iwy)][:,:, 1, 1],
                   'tau1': tau1,
                   'dVel': dVel,
                   'xEdges': xEdges,
                   'yEdges': yEdges,
                   'y': y
                  }
            return out


