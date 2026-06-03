"""
geom package: molecular geometry and geometry-space tooling.

    geom.Geometry, geom.Trajectory      (molecule.py)
    geom.Intc, geom.Intdef, geom.Cart2int (intc.py)
    geom.Optimizer, geom.OptResult      (optimize.py)
    geom.Descriptor, geom.Soap          (descriptor.py; needs ase + dscribe)
"""
from .intc import Intc, Intdef, Cart2int
from .molecule import Geometry, Trajectory
from .optimize import Optimizer, OptResult

# descriptor pulls heavy optional deps (ase, dscribe); guard the re-export
# so geom (and geom.molecule/intc/optimize) stay importable without them
try:
    from .descriptor import Descriptor, Soap
except ImportError:
    Descriptor = Soap = None

__all__ = ['Geometry', 'Trajectory', 'Intc', 'Intdef', 'Cart2int',
           'Optimizer', 'OptResult', 'Descriptor', 'Soap']
