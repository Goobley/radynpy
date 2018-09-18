import pickle
import os
from spacepy import pycdf

fileLocation = os.path.dirname(os.path.abspath(__file__)) + '/'
with open(fileLocation + 'RadynFormatHelp.pickle', 'rb') as p:
    helpInfo = pickle.load(p)
typeDict = helpInfo[2]
allVars = {**helpInfo[1]['const'], **helpInfo[1]['timeVary']}


def maybe_lookup(d, k):
    try:
        return d[k]
    except KeyError:
        return None

def index_convention():
    for k, v in helpInfo[0].items():
        print(k)
        print(v[1], '\n')

def var_info(var):
    if type(var) is list:
        var_info_list(var)
    elif type(var) is str:
        if var == '*':
            var_info_all()
            return
        var_info_str(var)
    else:
        raise ValueError('Unknown type "%s" in argument "%s" to var_info' % (str(type(var)), str(var)))

def var_info_all():
    print('Constant in time:')
    for k, v in helpInfo[1]['const'].items():
        print(v[0])
        print('  ', v[1])
    print('\nTime-varying:')
    for k, v in helpInfo[1]['timeVary'].items():
        print(v[0])
        print('  ', v[1])

def var_info_str(var):
    val = maybe_lookup(helpInfo[1]['const'], var)
    if val is None:
        val = maybe_lookup(helpInfo[1]['timeVary'], var)
        timeNature = 'Time-varying'
        if val is None:
            raise ValueError('Unknown variable "%s"' % var)
    else:
        timeNature = 'Constant in time'

    print(val[0])
    print('  ', val[1].replace('\n', '\n     '))
    print('  ', timeNature)

def var_info_list(varList):
    assert(all(type(x) is str for x in varList))
    for v in varList:
        var_info_str(v)

class RadynData:
    def __getattr__(self, name):
        if name in self.notLoadedVars:
            raise AttributeError('Array "%s" was not requested to be loaded' % name)
        else:
            raise AttributeError('Unknown attribute "%s" requested' % name)
    
    def index_convention(self):
        return index_convention()

    def var_info(self, var):
        return var_info(var)

    def __repr__(self):
        s = 'RadynData: Variables loaded:\n'
        for k in self.__dict__.keys():
            if k != 'notLoadedVars':
                info = maybe_lookup(allVars, k)
                if info is None:
                    s += ('  %s\n' % k)
                else:
                    s += ('  %s: %s\n' % (info[0], info[1].replace('\n', '\n    ')))
        return s

class LazyRadynData:

    def __init__(self, cdfHandle):
        self.cdf = cdfHandle

    def __getattr__(self, name):
        if name in allVars:
            setattr(self, name, self.cdf[name][...])
            return getattr(self, name)
        else:
            raise AttributeError('Unknown attribute "%s" requested' % name)
    
    def index_convention(self):
        return index_convention()

    def var_info(self, var):
        return var_info(var)

    def __repr__(self):
        s = 'RadynData: Variables loaded:\n'
        for k in self.__dict__.keys():
            info = maybe_lookup(allVars, k)
            if info is None:
                s += ('  %s\n' % k)
            else:
                s += ('  %s: %s\n' % (info[0], info[1].replace('\n', '\n    ')))
        return s

    def close(self):
        if self.cdf is not None:
            self.cdf.close()
            self.cdf = None

    def __del__(self):
        self.close()

def load_vars(cdfPath, varList, parseFilenameParams=True):
    if type(varList) is str:
        if varList == '*':
            return load_vars(cdfPath, allVars.keys())
        else:
            varList = [varList]
    
    assert(all(type(x) is str for x in varList))
    for var in varList:
        if not var in allVars:
            raise ValueError('Non-existent Radyn CDF variable "%s" requested from load_vars' % var)

    with pycdf.CDF(cdfPath) as cdf:
        res = RadynData()
        notLoadedVars = []
        for var in allVars.keys():
            # Just load all the scalar values, they're small enough, and most of them are
            # important
            if typeDict[var] == 'Val':
                setattr(res, var, cdf[var][...].item())
            else:
                if var in varList:
                    # [...] takes all of the data in a ndarray type of arrangement
                    # Then we copy it to have it memory, rather than (I presume) the memmap'd way it loads the data
                    setattr(res, var, cdf[var][...].copy())
                else:
                    notLoadedVars.append(var)
        if len(cdf.attrs) > 0:
            for k in cdf.attrs:
                setattr(res, k, str(cdf.attrs[k]))
        res.notLoadedVars = notLoadedVars

    cdfFilename = os.path.basename(cdfPath)
    if cdfFilename != 'radyn_out.cdf':
        if parseFilenameParams and cdfFilename.startswith('radyn_out.'):
            p = cdfFilename[cdfFilename.find('.')+1:]
            params = p.split('_')
            res.filenameParams = params
            if len(params) != 6:
                raise ValueError('FilenameParams should contain 6 underscore seperated terms.\n'
                                'See FCHROMA simulation documentation for examples.\n'
                                'If you don\'t want to parse these then call with parseFilenameParams=False')
            res.startingModelAtmosphere = params[0]
            res.beamSpectralIndex = float(params[1][1:])
            res.totalBeamEnergy = float(params[2])
            res.beamPlulseType = params[3]
            res.cutoffEnergy = params[4]
            res.beamType = params[5]

    return res

def lazy_load_all(cdfPath, parseFilenameParams=True):
    cdf = pycdf.CDF(cdfPath)
    res = LazyRadynData(cdf)
    for var in allVars.keys():
        # Just load all the scalar values, they're small enough, and most of them are
        # important
        if typeDict[var] == 'Val':
            setattr(res, var, res.cdf[var][...].item())
    if len(res.cdf.attrs) > 0:
        for k in res.cdf.attrs:
            setattr(res, k, str(res.cdf.attrs[k]))

    cdfFilename = os.path.basename(cdfPath)
    if cdfFilename != 'radyn_out.cdf':
        if parseFilenameParams and cdfFilename.startswith('radyn_out.'):
            p = cdfFilename[cdfFilename.find('.')+1:]
            params = p.split('_')
            res.filenameParams = params
            if len(params) != 6:
                raise ValueError('FilenameParams should contain 6 underscore seperated terms.\n'
                                'See FCHROMA simulation documentation for examples.\n'
                                'If you don\'t want to parse these then call with parseFilenameParams=False')
            res.startingModelAtmosphere = params[0]
            res.beamSpectralIndex = float(params[1][1:])
            res.totalBeamEnergy = float(params[2])
            res.beamPlulseType = params[3]
            res.cutoffEnergy = params[4]
            res.beamType = params[5]

    return res

