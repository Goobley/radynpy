import pickle
import re
from itertools import permutations
from collections import deque
from radynpy.AuxTypes import Val, Array, Unknown

with open('RadynKeySizes.pickle', 'rb') as p:
    keys = pickle.load(p)

# ntime and maxatomlevels added by cmo
# some corrections made where things are inconsistent with docs
# need to keep additions up to date or the next stage won't work
additions = ['ntime', 'maxatomlevels']
pdfDocStr = """Index convention:
i
total range [0:maxatomlevels-1]!!nper species level index [0:nk[iel]-1]!!n[j=upper level, i=lower level]
j
total range [0:maxatomlevels-1]!!nper species level index [0:nk[iel]-1]!!n[j=upper level, i=lower level]
iel
element index [0:nel-1]
kr
transition index [0:nrad-1]
krc
continuum transition index [0:nrad-nline-1]
kfx
fixed transition index [0:nrfix-1]
k
depth index [0:ndep-1]
mu
angle index [0:nmu-1]
nu
frequency index [0:nq[kr]-1] or [1:nq[kr]]
t
time index [0:ntime-1]
Variables that do not change with time:
nel
number of elements
nk[iel]
number of levels including continuum levels for element iel
nk1
number of levels in total over all elements
nrad
number of radiative transitions treated in detail
nline
number of radiative bound-bound transitions
nrfix
number of transitions with fixed rates
nq[kr]
number of frequencies
nmu
number of angles
ndep
number of depth points
ndepm
ndep-1
ntime
number of timesteps
maxatomlevels
maximum number of atomic levels used in any of for any species
atomid[iel]
4 character identification of atom.
abnd[iel]
atomic abundance, log scale with hydrogen=12
awgt[iel]
atomic weight. input in atomic units, converted to cgs
ev[i,iel]
energy above ground state. input in cm^-1, copnverted to eV
g[i,iel]
statistical weight of level
label[i,iel]
20 character identification of level
ion[i,iel]
ionization stage of level, 1=neutral
ielrad[kr]
ielrad[kr]-1 is element number for radiative transition kr
jrad[kr]
jrad[kr]-1 is upper level of radiative transition kr
irad[kr]
irad[kr]-1 is lower level of radiative transition kr
krad[i,j,iel]
krad[i,j,iel]-1=krad[j,i,iel]-1 is the kr index of the transition from level i to level j, element iel
ktrans[kr]
ktrans[kr]-1 is continuum transition nr for transition kr, krc=ktrans[kr]-1, krc=-1 for bb transitions 
cont[kr]
is 1 if the transition is bound-free and 0 for bound-bound transitions
f[kr]
oscillator strength
ga[kr]
radiative damping parameter
gw[kr]
van der waals damping parameter
gq[kr]
stark damping parameter
alamb[kr]
vacuum wavelength in angstrom
a[kr]
einstein a coefficient
bij[kr]
einstein b[i,j] coefficient
bji[kr]
einstein b[j,i] coefficient
ielfx[kfx]
ielfx[kfx]-1 is element for fixed transition kfx
jfx[kfx]
jfx[kfx]-1 is upper level of fixed transition kfx
ifx[kfx]
ifx[kfx]-1 is lower level of fixed transition kfx
ipho[kfx]
=1 continuum, =0 line
a0[kfx]
crossection at limit
trad[kfx]
brightness temperature for continua
itrad[kfx]
radiation temperature option. =1 rad.temp=temp,!!n=2 photospheric option, rad.temp=temp out to temp.lt.trad then temp=trad outwards!!n=3 chromospheric option, rad.temp=temp except when temp .gt. trad and temp increasing outwards. then rad.temp=trad
alfac[nu,krc]
photoionization crossection!!nin nu=0 the frequency for the edge is stored!!nin nu=1:nq[kr] the frequency points
qnorm
unit typical doppler width in km per second at line center
hn3c2
h*ny**3/c**2
q[nu,kr]
frequency variable, in units of a typical doppler width!!npositive q for increased frequency
qmax[kr]
maximum frequency, same units as q
q0[kr]
frequency within which quadrature points are distributed!!nlinearly instead of logarithmically
ind[kr]
=1 for one sided quadrature [symmetric profile]!!n=2 for two sided quadrature [asymmetric profile]
frq[nu,krc]
frequency in hz for continua!!nin nu=0 the frequency for the edge is stored!!nin nu=1:nq[kr] the frequency points
zmu[mu]
cosine of angle vs z-axis (xmu and ymu also exist but are not used except for 1D MHD)
atmoid
72 character identification of atmosphere used
dpid
72 character identification of depth-scale
dptype
=t depth scale is tauscale, =m depth scale is mass scale, see routine atmos
grav
gravitation acceleration
vturb[k]
microturbulence velocity
ee
electron charge
em
electron mass
hh
planck constant
cc
velocity of light
bk
boltzmann constant
amu
universal mass constant
hce
hh*cc/ee*1.e8, lambda[angstrom]=hce/energy[ev]
hc2
2*hh*cc*1.e24, 2*h*ny**3/c**2=hc2/lambda[angstrom]**3
hck
hh*cc/bk*1.e8, h*ny/kt=hck/lambda[angstrom]/t
ek
ee/bk
pi
pi
Variables that change with time:
z1[k,t]
interface heights
--zh[k,t]
--cell center heights
--dz[k,t]
--cell depths
vz1[k,t]
macroscopic velocity
d1[k,t]
density
tg1[k,t]
temperature
n1[k,i,iel,t]
population density in cm-3
ne1[k,t]
electron density
pg1[k,t]
gas pressure
fmj[k,t]
mass flux
gml[k,i,iel,t]
gains minus losses in rate equations
en1[k,t]
internal energy
coolt1[k,t]
total cooling
c[k,i,j,iel,t]
collisional transition rate
heat1[k,t]
mechanical energy needed to sustain initial atmosphere
eion1[k,t]
ionization+excitation energy
outint[nu,mu,kr,t]
monochromatic surface intensity in cgs units!!nin nu=0 the continuum intensity is stored!!nin nu=1:nq[kr] outint for the nu-points of the transition
cmass1[k,t]
column mass at interface
dnyd[k,iel,t]
doppler width in units of a typical doppler width
adamp[k,kr,t]
voigt damping parameter
z0[k,t]
interface height at previous timestep
vz0[k,t]
velocity at previous timestep
itime[t]
timestep number
time[t]
time
dtn[t]
timestep size from current to next time
dtnm[t]
timestep size from old to current time
iiter[t]
number of iterations needed for convergence
zm[k,t]
height for column masses of original gridpoints
nstar[k,i,iel,t]
lte population density
sl[k,kr,t]
line source function
bp[k,kr,t]
planck function
rij[k,kr,t]
radiative rate from i to j per ni atom
rji[k,kr,t]
radiative rate from j to i per nj atom
"""
docStrs = [x.replace('!!n', '\n') for x in pdfDocStr.splitlines() if not x.startswith('--')]
categories = {}
idx = 0
while True:
    s = docStrs[idx]
    # This line requires that the first entry in this docstring be a category header
    if s.endswith(':'):
        sanitised = s[:s.find(':')]
        categories[sanitised] = {}
        currentCat = categories[sanitised]
        idx += 1
        continue
    
    if s.endswith(']'):
        sanitisedName = s[:s.find('[')]
        val = Array(tuple(s[s.find('[')+1:s.find(']')].split(',')))
    else:
        sanitisedName = s
        val = Val(None)

    sanitisedName = s if not s.endswith(']') else s[:s.find('[')]
    currentCat[sanitisedName] = (val, docStrs[idx+1])

    idx += 2
    if idx >= len(docStrs):
        break

# print(categories)

def extract_idl_range(s):
    start = s.find('[')
    i = start + 1
    end = len(s)
    depth = 1
    while i < end:
        if s[i] == '[':
            depth += 1

        if s[i] == ']':
            depth -= 1
            if depth == 0:
                return s[start:i+1]

        i += 1

        


indices = categories['Index convention']
indexMax = {}
for k, v in indices.items():
    assert(type(v[0]) is Val)
    var = re.match('\[0:(.*)-1\]$', extract_idl_range(v[1])).group(1)
    if var.find('[') != -1:
        # Then there's an array index involved, and this data isn't in the pickle
        indexMax[k] = Val(Unknown())
    else:
        if re.match('^\w*$', var) is not None:
            indexMax[k] = keys[var]
        elif re.match('^(\w*)-(\w*)$', var):
            matches = re.match('^(\w*)-(\w*)$', var)
            var1 = matches.group(1)
            var2 = matches.group(2)
            indexMax[k] = Val(keys[var1].val - keys[var2].val)
# print(indexMax)
# print(indexMax['i'])
# Special case for the grid sims, probably all other modern ones, ignore the size of kfx
print('Ignoring kfx\'s size while generating the reader as it seems to be a relic')
indexMax['kfx'].val = Unknown()

def deque_rotator(deq, reverse=False):
    n = len(deq)
    v = -1 if reverse else 1
    for i in range(n):
        deq.rotate(v)
        yield list(deq)


noChange = categories['Variables that do not change with time']
change = categories['Variables that change with time']
def check_and_correct_indices(d):
    variableNotInFile = []
    for k, v in d.items():
        try:
            if type(v[0]) is Array:
                loadedArray = keys[k]
                if len(v[0].shape) == len(loadedArray.shape):
                    idxs = v[0].shape
                    correctIdxs = idxs

                    if not all(indexMax[idx].val == loadedArray.shape[j] for j, idx in enumerate(idxs)):
                        if any(type(indexMax[i].val) is Unknown for i in idxs):
                            if len(v[0].shape) == 1:
                                # It's only 1D --  has to be correct
                                break
                            # If the 'all' passes then it still was correct -- just had an unknown in a consistent location
                            if not all(indexMax[idx].val == loadedArray.shape[j] for j, idx in enumerate(idxs) if type(indexMax[idx]) is not Unknown):
                                # Test cyclic permutations. I don't think we need anything more than this
                                # The order of rotation is such that everything shifts 1 to the right
                                # This looked the most reasonable from looking at a few examples by eye
                                # So we just take the first succesful permutation
                                # if there aren't any, then we need to have a closer look
                                rot = deque_rotator(deque(idxs))
                                cyclic = [perm for perm in rot]
                                success = [all(indexMax[i].val == loadedArray.shape[j] for j, i in enumerate(c) if type(indexMax[i].val) is not Unknown) for c in cyclic]
                                if not any(success):
                                    print('Cyclic permutation (with unknown) failed for: %s' % k)
                                    exit()
                                for j, correct in enumerate(success):
                                    if correct:
                                        correctIdxs = tuple(cyclic[j])
                                        break
                        else:
                            # First test cyclic permutations -- Should be sufficient
                            deq = deque(idxs) 
                            rot = deque_rotator(deque(idxs))
                            cyclic = [perm for perm in rot]
                            success = [all(indexMax[i].val == loadedArray.shape[j] for j, i in enumerate(c)) for c in cyclic]
                            if not any(success):
                                print('Cyclic permutation failed for: %s' % k)
                                exit()
                            for j, correct in enumerate(success):
                                if correct:
                                    correctIdxs = tuple(cyclic[j])
                                    break

                    if v[0].shape != correctIdxs:
                        print('Changing indexing string of %s from %s to %s' % (k, v[0].shape.__repr__(), correctIdxs.__repr__()))
                        v[0].shape = correctIdxs
                else:
                    print('Expected number of dimensions different to CDF File: %s, %dD in Example File (%s), and %dD in Doc' 
                          % (k, len(loadedArray.shape), str(loadedArray), len(v[0].shape)))
                    exit()
        except KeyError:
            variableNotInFile.append(k)
            print('Need to compute the values for key %s in the generated class' % k)
    return variableNotInFile

calculateVar = []
calculateVar += check_and_correct_indices(noChange)
calculateVar += check_and_correct_indices(change)

def check_and_remove_nonexistent_vals(d):
    removeList = []
    for k, v in d.items():
        if type(v[0]) is Val:
            try:
                loaded = keys[k]
            except KeyError:
                print('Removing key %s as not present in sample data.\nIf this is an attribute then it will be loaded by the final stage automatically' % k)
                removeList.append(k)
    for k in removeList:
        del d[k]

check_and_remove_nonexistent_vals(noChange)
check_and_remove_nonexistent_vals(change)
            
def convert_to_help_dict(d):
    res = {}
    for k, v in d.items():
        if k in additions:
            continue
        if (type(v[0]) is Array):
            res[k] = (k+v[0].idl_repr(), v[1])
        else:
            res[k] = (k, v[1])
    return res

def convert_to_type_dict(d):
    res = {}
    for k, v in d.items():
        if k in additions:
            continue
        if (type(v[0]) is Array):
            res[k] = 'Array'
        else:
            res[k] = 'Val'
    return res

constVars = convert_to_help_dict(noChange)
timeVaryingVars = convert_to_help_dict(change)

c = categories['Index convention']
for k in c:
    c[k] = (c[k][0], c[k][1].replace('-1', ''))
c['nu'] = (None, 'frequency index [0:nq[kr]] or [1:nq[kr]+1]')

# To fix the strings in varinfo it looks like it's sufficient to just replace instances of nq[kr] wioth nq[kr]+1
def update_nu_indexing(d):
    for k in d:
        if 'nu=1:nq[kr]' in d[k]:
            d[k] = d[k].replace('nu=1:nq[kr]', 'nu=1:nq[kr]+1')
    return d

update_nu_indexing(constVars)
update_nu_indexing(timeVaryingVars)

helpInfo = []
helpInfo.append(categories['Index convention'])
helpInfo.append({'const': constVars, 'timeVary': timeVaryingVars})
typeDict = {**convert_to_type_dict(noChange), **convert_to_type_dict(change)}
helpInfo.append(typeDict)
helpInfo.append(calculateVar)
with open('RadynFormatHelp.pickle', 'wb') as p:
    pickle.dump(helpInfo, p)
