__version__ = '0.1.4'


from VortexAD.core.vlm.vlm_solver import vlm_solver

from pathlib import Path
ROOT = Path(__file__).parents[0]
SAMPLE_GEOMETRY_PATH = ROOT / 'core' / 'geometry' / 'sample_meshes'
AIRFOIL_PATH = ROOT / 'core' / 'geometry' / 'sample_airfoils'