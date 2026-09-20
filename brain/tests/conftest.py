import pytest

from companion_brain.config import load_config
from companion_brain.data.download import have_data

CFG = load_config()
needs_data = pytest.mark.skipif(not have_data(CFG), reason="connectome data not downloaded (run: companion download)")
