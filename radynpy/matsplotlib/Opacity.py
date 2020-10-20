import numpy as np
from radynpy.utils import voigt_H, gaunt_bf_h, hydrogen_absorption, hydrogen_bf_profile
import struct
import os
here = os.path.dirname(os.path.abspath(__file__)) + '/'

# Almost everything in here is a verbatim translation of the Radyn IDL routines
# supplied by M. Carlsson, many improvements could be made for pythonic-ness

class OpcFile:
    def __init__(self, path=None):
        if path is None: 
            path = here+'opctab.dat'
        mtPts = 39
        xc = np.zeros((4, mtPts, 20))
        with open(path, 'rb') as opc:
            opc.seek(0, 2)
            fileSize = opc.tell()
            opc.seek(0)

            sumh = np.fromfile(opc, dtype='float64', count=1).item()
            grph = np.fromfile(opc, dtype='float64', count=1).item()
            xv = np.fromfile(opc, dtype='float64', count=mtPts)
            yv = np.fromfile(opc, dtype='float64', count=20)
            nelsr = np.fromfile(opc, dtype='int32', count=1).item()

            # text is just padding here, it's empty
            text = ''.join([s.decode('UTF-8') for s in struct.unpack('s'*(15*80), opc.read(15*80))])
            cel = ''.join([s.decode('UTF-8') for s in struct.unpack('s'*(2 * nelsr), opc.read(2 * nelsr))])
            cel = [cel[i:i+2] for i in range(0,len(cel),2)]

            abund = np.fromfile(opc, dtype='float64', count=nelsr)

            awi = np.fromfile(opc, dtype='float64', count=nelsr)

            loopmx = np.fromfile(opc, dtype='int32', count=1).item()

            xl = np.fromfile(opc, dtype='float64', count=loopmx)
            wavel = xl

            recNum = 1
            xcs = []
            while True:
                if fileSize - opc.tell() <= xc.shape[0] * xc.shape[1] * xc.shape[2] * 8:
                    break

                opc.seek(3120*8*recNum)

                record = np.fromfile(opc, dtype='float64', count=3120).reshape((xc.shape[2], xc.shape[1], xc.shape[0])).T
                xcs.append(record)
                recNum += 1
                
        self.sumh = sumh
        self.grph = grph
        self.xv = xv
        self.yv = yv
        self.nelsr = nelsr
        self.text = text
        self.cel = cel
        self.abund = abund
        self.awi = awi
        self.loopmx = loopmx
        self.wavel = wavel
        self.xcs = xcs
    
    def roptab(self, tg1, ne1, irec):

        kB = 1.380662e-16
        alge = 1.0 / np.log(10.0)

        nDep = tg1.shape[0]

        y  = np.zeros((nDep, 4))
        y1 = np.zeros((nDep, 4))
        y2 = np.zeros((nDep, 4))
        y12 = np.zeros((nDep, 4))
        v = np.zeros(nDep)
        dvdt = np.zeros(nDep)
        dvdne = np.zeros(nDep)
        xcs = self.xcs


        if irec == 4 or self.i1 is None:
            i1 = []
            i2 = []
            theta = 5040.0 / tg1
            pelg = np.log10(ne1 * kB * tg1)

            for k in range(nDep):
                leftIdx = np.searchsorted(self.xv, theta[k])
                if leftIdx == 0 or leftIdx == len(self.xv):
                    print('roptab: T outside range')
                i1.append(leftIdx)

                leftIdx = np.searchsorted(self.yv, pelg[k])
                if leftIdx == 0 or leftIdx == len(self.yv):
                    print('roptab: pe outside range')
                i2.append(leftIdx)

            self.i1 = np.array(i1) - 1
            self.i2 = np.array(i2) - 1
            i1 = self.i1
            i2 = self.i2
            dx1 = self.xv[i1+1] - self.xv[i1]
            dx2 = self.yv[i2+1] - self.yv[i2]
            self.dx1 = dx1
            self.dx2 = dx2
            self.theta = theta
            self.pelg = pelg
        else:
            i1 = self.i1
            i2 = self.i2
            dx1 = self.dx1
            dx2 = self.dx2
            pelg = self.pelg
            theta = self.theta


        # read record
        xc = xcs[irec-2]

        for k in range(nDep):
            y[k,0] = xc[0, i1[k],   i2[k]]
            y[k,1] = xc[0, i1[k]+1, i2[k]]
            y[k,2] = xc[0, i1[k]+1, i2[k]+1]
            y[k,3] = xc[0, i1[k],   i2[k]+1]

            y1[k,0] = xc[1, i1[k],   i2[k]]
            y1[k,1] = xc[1, i1[k]+1, i2[k]]
            y1[k,2] = xc[1, i1[k]+1, i2[k]+1]
            y1[k,3] = xc[1, i1[k],   i2[k]+1]

            y2[k,0] = xc[2, i1[k],   i2[k]]
            y2[k,1] = xc[2, i1[k]+1, i2[k]]
            y2[k,2] = xc[2, i1[k]+1, i2[k]+1]
            y2[k,3] = xc[2, i1[k],   i2[k]+1]

            y12[k,0] = xc[3, i1[k],   i2[k]]
            y12[k,1] = xc[3, i1[k]+1, i2[k]]
            y12[k,2] = xc[3, i1[k]+1, i2[k]+1]
            y12[k,3] = xc[3, i1[k],   i2[k]+1]

        # bicubic interpolation, looks horrible, but copied from original and works
        wt = np.zeros((16,16))
        wt[:,0] = [ 1.,0.,-3., 2.,0.,0., 0., 0.,-3., 0., 9.,-6., 2., 0.,-6., 4.]
        wt[:,1] = [ 0.,0., 0., 0.,0.,0., 0., 0., 3., 0.,-9., 6.,-2., 0., 6.,-4.]
        wt[:,2] = [ 0.,0., 0., 0.,0.,0., 0., 0., 0., 0., 9.,-6., 0., 0.,-6., 4.]
        wt[:,3] = [ 0.,0., 3.,-2.,0.,0., 0., 0., 0., 0.,-9., 6., 0., 0., 6.,-4.]
        wt[:,4] = [ 0.,0., 0., 0.,1.,0.,-3., 2.,-2., 0., 6.,-4., 1., 0.,-3., 2.]
        wt[:,5] = [ 0.,0., 0., 0.,0.,0., 0., 0.,-1., 0., 3.,-2., 1., 0.,-3., 2.]
        wt[:,6] = [ 0.,0., 0., 0.,0.,0., 0., 0., 0., 0.,-3., 2., 0., 0., 3.,-2.]
        wt[:,7] = [ 0.,0., 0., 0.,0.,0., 3.,-2., 0., 0.,-6., 4., 0., 0., 3.,-2.]
        wt[:,8] = [ 0.,1.,-2., 1.,0.,0., 0., 0., 0.,-3., 6.,-3., 0., 2.,-4., 2.]
        wt[:,9] = [ 0.,0., 0., 0.,0.,0., 0., 0., 0., 3.,-6., 3., 0.,-2., 4.,-2.]
        wt[:,10]= [ 0.,0., 0., 0.,0.,0., 0., 0., 0., 0.,-3., 3., 0., 0., 2.,-2.]
        wt[:,11]= [ 0.,0.,-1., 1.,0.,0., 0., 0., 0., 0., 3.,-3., 0., 0.,-2., 2.]
        wt[:,12]= [ 0.,0., 0., 0.,0.,1.,-2., 1., 0.,-2., 4.,-2., 0., 1.,-2., 1.]
        wt[:,13]= [ 0.,0., 0., 0.,0.,0., 0., 0., 0.,-1., 2.,-1., 0., 1.,-2., 1.]
        wt[:,14]= [ 0.,0., 0., 0.,0.,0., 0., 0., 0., 0., 1.,-1., 0., 0.,-1., 1.]
        wt[:,15]= [ 0.,0., 0., 0.,0.,0.,-1., 1., 0., 0., 2.,-2., 0., 0.,-1., 1.]

        d1 = dx1
        d2 = dx2

        # pack temporary x
        x = np.zeros((nDep, 16))
        for i in range(4):
            x[:,i] = y[:,i]
            x[:,i+4] = y1[:,i] * d1
            x[:,i+8] = y2[:,i] * d2
            x[:,i+12] = y12[:,i]*d1*d2

        rc = np.zeros((nDep, 4, 4))
        for i in range(4):
            for j in range(4):
                l = i*4 + j
                for m in range(16):
                    rc[:,i,j] += wt[l,m] * x[:, m]

        t = (theta - self.xv[i1]) / dx1
        u = (pelg - self.yv[i2]) / dx2

        for l in range(3, -1, -1):
            v = t*v + ((rc[:,l,3]*u + rc[:,l,2])*u + rc[:,l,1])*u + rc[:,l,0]
            dvdt = dvdt*u + (3.0 * rc[:,3,l]*t + 2.0*rc[:,2,l])*t + rc[:,1,l]
            dvdne = t*dvdne + (3.0*rc[:,l,3]*u + 2.0*rc[:,l,2])*u + rc[:,l,1]

        v = 10**v
        dvdne = dvdne/dx2 * v
        dvdt = -dvdt/dx1*v/alge*theta+dvdne

        return {'v': v, 'dvdt': dvdt, 'dvdne': dvdne, 'wavel': self.wavel}

class XtNyCalc:
    def __init__(self, cdf, t, iel, kr, dnyd, withBackgroundOpacity=True, opctabPath=None):
        self.cdf = cdf
        self.t = t
        self.iel = iel
        self.kr = kr
        self.dnyd = dnyd
        self.withBackgroundOpacity = withBackgroundOpacity
        self.i = cdf.irad[kr] - 1
        self.j = cdf.jrad[kr] - 1
        self.vel = cdf.vz1[t, :] *  1e-5 / cdf.qnorm
        self.nDep = cdf.tg1.shape[1]
        
        xlamb = cdf.alamb[kr]
        xlamb5 = 5000.0
        temp = cdf.tg1[t]
        ne = cdf.ne1[t]
        nh = cdf.n1[t,:,:,0]
        toth = cdf.n1[t,:,:,0].sum(axis=1)
        opcFile = OpcFile(path=opctabPath)
        opc = opcFile.roptab(temp, ne, 4)
        wavel = opc['wavel']

        xnorm5 = opc['v'] * toth
        xconth, xconth_lte = hydrogen_absorption(xlamb5, 0, temp, ne, nh)
        xnorm5 += xconth
        w2 = 1.0 / (5000.0*5000.0)
        w4 = w2**2
        scatrh = w4 * (5.799e-13+w2*(1.422e-6+w2*2.784))*nh[:,0]
        scatne = 6.655e-25 * ne
        xnorm5 = xnorm5 + scatrh + scatne
        xnormtCurrent = np.copy(xnorm5)

        iw = np.argwhere(np.abs(xlamb - wavel) < 0.5)
        if len(iw) == 0:
            raise ValueError('xlamb not found in opctab, wavelength='+repr(cdf.alamb[kr]))

        opcm = opcFile.roptab(temp, ne, iw[0].item()+4)
        xcontm = opcm['v'] * toth
        xconth, xconth_lte = hydrogen_absorption(xlamb, 0, temp, ne, nh)
        w2 = 1.0 / (xlamb*xlamb)
        w4 = w2**2
        scatrh = w4 * (5.799e-13+w2*(1.422e-6+w2*2.784))*nh[:,0]
        scatne = 6.655e-25 * ne
        chi = xcontm + xconth +scatrh + scatne
        xcont = chi / xnorm5
        self.xcont = xcont
        self.xnormtCurrent = xnormtCurrent
    
    def xt_nycalc(self, ny):
        cdf = self.cdf
        t = self.t
        kr = self.kr
        i = self.i
        j = self.j
        iel = self.iel
        vel = self.vel
        dnyd = self.dnyd

        if not cdf.cont[kr]:
            # The 1.0 was xmu -- holdover from mhd
            v = (cdf.q[ny, kr] - 1.0 * vel) / dnyd
            h = voigt_H(cdf.adamp[t, :, kr], v)
#             h = radyn_voigt(cdf.adamp[t, :, kr], v)
            phi = h / (dnyd * np.sqrt(np.pi))
            gijk = cdf.g[i, iel] / cdf.g[j, iel]
            hn3c2 = cdf.a[kr] / cdf.bji[kr]
            z = cdf.n1[t, :, i, iel] - gijk * cdf.n1[t, :, j, iel]
            alpha = cdf.bij[kr] * phi * cdf.hny4p
            xlamb = cdf.alamb[kr]
        else:
            raise NotImplementedError('Not Implemented for continua')

        x = z * alpha / self.xnormtCurrent 
        if self.withBackgroundOpacity:
            x += self.xcont
        tauq = np.zeros(self.nDep)
        tauq[0] = x[0] * 0.0 # tau[0] -- which should always be 0
        for k in range(1, self.nDep):
            tauq[k] = tauq[k-1] + 0.5 * (x[k] + x[k-1]) * (cdf.tau[t,k] - cdf.tau[t,k-1])
            
        return x, tauq, self.xnormtCurrent


def xt_calc(cdf, t, iel, kr, withBackgroundOpacity=True, opctabPath=None):
    nDep = cdf.tg1.shape[1]
    kB = 1.380662e-16
    nq0 = cdf.nq[kr]
    x_ny = np.zeros((nDep, nq0))
    tauq_ny = np.zeros((nDep, nq0))
    dnyd = np.sqrt(2 * kB * cdf.tg1[t, :] / cdf.awgt[iel]) * 1.0e-5 / cdf.qnorm
    dnyd = np.sqrt(dnyd**2 + (cdf.vturb * 1e-5 / cdf.qnorm)**2)

    xt = XtNyCalc(cdf, t, iel, kr, dnyd, withBackgroundOpacity=withBackgroundOpacity, opctabPath=opctabPath)
    for ny in range(nq0):
        x, tauq, xnorm = xt.xt_nycalc(ny)
        x_ny[:,ny] = x * xnorm
        tauq_ny[:,ny] = tauq
        
    return x_ny, tauq_ny