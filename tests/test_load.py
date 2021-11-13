import numpy as np

from tests import binvox_rw_original
from binvox import Binvox


def test_loading():

    with open('chair.binvox', 'rb') as f:
        loaded_original = binvox_rw_original.read_as_3d_array(f)

    loaded_new = Binvox.read('chair.binvox', mode='dense')

    assert np.allclose(loaded_original.data, loaded_new.numpy())
