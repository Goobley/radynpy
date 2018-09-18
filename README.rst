RadynPy
-------

This is a very early development release of RadynPy, a nascent suite of tools to allow 
analysis of Radyn simulations to be performed in Python.

If you don't know what Radyn is, then this package is probably not aimed at you!

Currently the only class that is designed to be used is RadynData (and LazyRadynData), 
which can be used to load the `.cdf` files created by Radyn.

This module exposes the functions: `index_convention`, `var_info`,
`load_vars`, and `lazy_load_all`.

* Calling `index_convention` prints the indexing convention used in `var_info`.
* `var_info` takes a `str` or list of `str` and prints the meaning of the
  associated variables, along with their dimensionality in the indexing
  convention used. `var_info` can also take '*' as an argument, for which it
  will return the info on all variables.
* `load_vars` takes the path to a CDF file and a list of the required variables
  which are then loaded into an instance of `RadynData` and can now be accesed
  as `.varName` rather than needing to go via the strings.
  If the filename is in the FCHROMA format then the heating parameters will be
  parsed from the string. If the filename is simply `radyn_out.cdf` then the
  filename isn't scanned for parameters. If the filename is a different format
  then you may need to pass `parseFilenameParams=False` to `load_vars`.
  `load_vars` can also take '*' as an argument, for which it
  will load all variables and timesteps into memory.
* `lazy_load_all` loads all of the variables in the CDF in a lazy manner,
  meaning that they are only loaded into memory when called for the first time.
  This has the unfortunate side effect of needing to keep a handle to the CDF
  file open, so ideally the `close()` method should be called when you are
  done with the file, and not opening thousands (!) of CDF files
  simulataneously. For just a few files the call to `close()` isn't too
  important as the class attempts to clean up after itself as much as
  possible. This mode is very good for easy exploration of CDF files as the
  requested variables do not need to be known a priori,
  making it easier to use in the REPL, without thehuge memory requirements of
  loading all of the variables from a file. `lazy_load_all` behaves in the
  same way as `load_vars` with respect to parsing the filename parameters,
  and also accepts the `parseFilenameParams` keyword.

I expect that `lazy_load_all` will become the prevailing mode of use, but in
scripts where a large number of simulations (e.g. the grid) are being
processed then `load_vars` will probably work out slightly more efficient.


This documentation will be fleshed out more in time, this is but a very early alpha release.
