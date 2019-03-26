from __future__ import absolute_import
import imp

name = "netNMFsc"
from .classes import netNMFMU
print('netNMFMU imported')
try:
	imp.find_module('tensorflow')
	found = True
except ImportError:
	found = False
if found: # only import if tensorflow is installed
	from .classes import netNMFGD
	print('netNMFGD imported')
else:
	print('not importing netNMFGD : tensorflow not installed')