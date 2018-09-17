RadynPy
-------

This is a very early development release of RadynPy, a nascent suite of tools to allow 
analysis of Radyn simulations to be performed in Python.

If you don't know what Radyn is, then this package is probably not aimed at you!

Currently the only class that is designed to be used is RadynData, 
which can be used to load the `.cdf` files created by Radyn.

This module exposes the functions: `index_convention`, `var_info`, and `load_vars`,

* Calling `index_convention` prints the indexing convention used in `var_info`.
* `var_info` takes a `str` or list of `str` and prints the meaning of the associated variables, along with their dimensionality in the indexing convention used. 
* `load_vars` takes the path to a CDF file and a list of the required variables which are then loaded into an instance of `RadynData` and can now be accesed as `.varName` rather than needing to go via the strings.

This documentation will be fleshed out more in time, this is but a very early alpha release.
