import sys
import importlib.util
import pathlib

path = pathlib.Path(__file__).parent.parent.parent / 'src'

spec = importlib.util.spec_from_file_location('ensembles', str(path / 'ensembles.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

del sys
del importlib
del pathlib
del path

for name in filter(lambda name: not name.startswith('_'), dir(mod)):
    globals()[name] = getattr(mod, name)

__name__ = mod.__name__
__doc__ = mod.__doc__
__package__ = mod.__package__
if '__all__' in dir(mod):
    __all__ = mod.__all__

del spec
del mod
