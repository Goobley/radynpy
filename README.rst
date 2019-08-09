RadynPy
-------

IMPORTANT NOTE
--------------

This is an early development release of RadynPy, a nascent suite of tools to
allow analysis of Radyn simulations to be performed in Python. It contains a
minimal set of features to perform analysis of Radyn CDF files, and produce
contribution functions.

If you don't know what Radyn is, then this package is probably not aimed at you!

The easiest way to load a CDF file is to use ``radynpy.cdf.LazyRadynData``. An
object can be constructed by passing the path to a Radyn CDF file. Data is
loaded lazily as it is requested. The normal keys of this file can then be
accessed by ``.`` notation (e.g. data.tg1). To look at the indexing convention
and the data layout (which is different to IDL due to handling of the record
time axis), the functions ``index_convention``, and for information on a
specific variable, or set of variables, ``var_info`` can be used. ``var_info``
takes a string or list of strings for the variables of interest (this string
may be '*' to print information on all variables).

The class ``radynpy.cdf.RadynData`` also exists, it takes the path to a CDF
file, and a list of keys to be loaded (or '*' for everything). It performs
similarly to LazyRadynData, but a key that was not requested at construction
will not be loaded.

**NOTE:** By default both ``RadynData`` and ``LazyRadynData`` expect a filename in
the F-CHROMA style (i.e. with the heating parameters), if your filename does
not fit this style then both classes take a ``parseFilenameParams=False``
kwarg.

Contribution functions can be plotted using ``radynpy.matsplotlib.contib_fn``.
See the extensive docstring of this function for more information. Utilities
used in the calculation of the contribution function can be found in
``radynpy.matsplotlib.utils`` and ``radynpy.matsplotlib.opacity``. For the
most part these are very simple functions and classes that mirror the
counterparts in the Radyn software distribution produced by M. Carlsson.

