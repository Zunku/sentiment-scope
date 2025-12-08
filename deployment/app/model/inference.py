import pickle
from pathlib import Path
import utils.ml_utils as ml_utils
import utils.text_utils as text_utils
import utils.viz_utils as viz_utils

__version__ = '0.1.0'
# __file__ Ubicacion del archivo actual
# BASE_DIR Carpeta padre del archivo
BASE_DIR = Path(__file__).resolve(strict=True).parent
with open(f'{BASE_DIR}/sentiment-analysis-pipeline-{__version__}.pkl', "rb") as f:
    model = pickle.load(f)
    
