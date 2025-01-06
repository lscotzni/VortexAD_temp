__version__ = '0.1.4'

from pathlib import Path
ROOT = Path(__file__).parents[0]
SAMPLE_GEOMETRY_PATH = ROOT / 'core' / 'geometry' / 'sample_meshes'
AIRFOIL_PATH = ROOT / 'core' / 'geometry' / 'sample_airfoils'

try:
    from VortexAD.core.panel_method.unsteady_panel_solver import unsteady_panel_solver
except:
    pass
from VortexAD.core.panel_method.steady_panel_solver import steady_panel_solver