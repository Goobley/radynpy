import pickle
import os
import numpy as np
import cdflib

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
    '''
    Container for the RADYN simulation data loaded from a CDF file. 
    All required variables must be specified at construction.
    The loaded variables are added as attributes to constructed instance.

    Parameters
    ----------
    cdfPath : str
        The complete path to the CDF file
    varList : list of str or str
        The variables to be loaded from the file. If '*' is passed then all will be loaded, 
        a single str will be the only variable loaded, and a list of strs will all be loaded.
    parseFilenameParams : bool, optional
        If the filename is in the F-CHROMA format then parse the heating parameters (default True)

    Attributes
    ----------
    varName : np.ndarray
        The variables that have been loaded
    '''

    def __init__(self, cdfPath, varList, parseFilenameParams=True):
        if type(varList) is str:
            if varList == '*':
                varList = allVars.keys()
            else:
                varList = [varList]
        
        assert(all(type(x) is str for x in varList))
        for var in varList:
            if not var in allVars:
                raise ValueError('Non-existent Radyn CDF variable "%s" requested from load_vars' % var)

        cdf = cdflib.CDF(cdfPath)
        notLoadedVars = []
        for var in allVars.keys():
            # Just load all the scalar values, they're small enough, and most of them are
            # important
            if typeDict[var] == 'Val':
                setattr(self, var, cdf.varget(var).item())
            else:
                if var in varList:
                    setattr(self, var, cdf.varget(var))
                else:
                    notLoadedVars.append(var)
        atts = cdf.globalattsget()
        if len(atts) > 0:
            for k in atts:
                setattr(self, k, str(atts[k]))
        self.notLoadedVars = notLoadedVars
        cdf.close()

        cdfFilename = os.path.basename(cdfPath)
        if cdfFilename != 'radyn_out.cdf':
            if parseFilenameParams and cdfFilename.startswith('radyn_out.'):
                p = cdfFilename[cdfFilename.find('.')+1:]
                params = p.split('_')
                self.filenameParams = params
                if len(params) != 6:
                    raise ValueError('FilenameParams should contain 6 underscore seperated terms.\n'
                                    'See FCHROMA simulation documentation for examples.\n'
                                    'If you don\'t want to parse these then call with parseFilenameParams=False')
                self.startingModelAtmosphere = params[0]
                self.beamSpectralIndex = float(params[1][1:])
                self.totalBeamEnergy = float(params[2])
                self.beamPlulseType = params[3]
                self.cutoffEnergy = params[4]
                self.beamType = params[5]

    def __getattr__(self, name):
        if name in self.notLoadedVars:
            raise AttributeError('Array "%s" was not requested to be loaded' % name)
        else:
            raise AttributeError('Unknown attribute "%s" requested' % name)
    
    def index_convention(self):
        '''
        Return a string explaining the RADYN index convention.
        '''
        return index_convention()

    def var_info(self, var):
        '''
        Return a string explaining the specified variable(s) and their axes.

        Parameters
        ----------
        var : str or list of str
            Specifies the variable to return information about. 
            '*' will return information on all variables, and a list of variables can also be requested. 
        '''
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
    '''
    Container for the RADYN simulation data lazily loaded from a CDF file. 
    Variables are loaded as required when used as atrtributes.
    The loaded variables are added as attributes to constructed instance.

    Parameters
    ----------
    cdfPath : str
        The complete path to the CDF file
    parseFilenameParams : bool, optional
        If the filename is in the F-CHROMA format then parse the heating parameters (default True)

    Attributes
    ----------
    varName : np.ndarray
        The variables that have been loaded
    '''
    def __init__(self, cdfPath, parseFilenameParams=True):
        self.cdf = cdflib.CDF(cdfPath)
        for var in allVars.keys():
            # Just load all the scalar values, they're small enough, and most of them are
            # important
            if typeDict[var] == 'Val':
                setattr(self, var, self.cdf.varget(var).item())
        atts = self.cdf.globalattsget()
        if len(atts) > 0:
            for k in atts:
                setattr(self, k, str(atts[k]))

        cdfFilename = os.path.basename(cdfPath)
        if cdfFilename != 'radyn_out.cdf':
            if parseFilenameParams and cdfFilename.startswith('radyn_out.'):
                p = cdfFilename[cdfFilename.find('.')+1:]
                params = p.split('_')
                self.filenameParams = params
                if len(params) != 6:
                    raise ValueError('FilenameParams should contain 6 underscore seperated terms.\n'
                                    'See FCHROMA simulation documentation for examples.\n'
                                    'If you don\'t want to parse these then call with parseFilenameParams=False')
                self.startingModelAtmosphere = params[0]
                self.beamSpectralIndex = float(params[1][1:])
                self.totalBeamEnergy = float(params[2])
                self.beamPlulseType = params[3]
                self.cutoffEnergy = params[4]
                self.beamType = params[5]

    def __getattr__(self, name):
        if name in allVars:
            var = self.cdf.varget(name)
            setattr(self, name, var)
            return getattr(self, name)
        else:
            raise AttributeError('Unknown attribute "%s" requested' % name)

    def load_var(self, name):
        '''
        Load a variable that may not be present in the help documentation,
        and hence not auto-loaded by __getattr__. Variable will be loaded
        into self.name.

        Raises : ValueError if variable with name not present in CDF.
        '''
        var = self.cdf.varget(name)
        setattr(self, name, var)
    
    def index_convention(self):
        '''
        Return a string explaining the RADYN index convention.
        '''
        return index_convention()

    def var_info(self, var):
        '''
        Return a string explaining the specified variable(s) and their axes.

        Parameters
        ----------
        var : str or list of str
            Specifies the variable to return information about. 
            '*' will return information on all variables, and a list of variables can also be requested. 
        '''
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
        '''
        Closes the CDF file, if it is open. Should only be called when the object is no longer required.
        '''
        if self.cdf is not None:
            self.cdf.close()
            self.cdf = None

    def __del__(self):
        self.close()



