import numpy as np
from tests import binvox_rw_original
from binvox import Binvox
import filecmp


def test_loading():
    # Load using original python script from Daniel Maturana
    with open('chair.binvox', 'rb') as f:
        loaded_old = binvox_rw_original.read_as_3d_array(f)

    # Load with our library
    loaded_new = Binvox.read('chair.binvox', mode='dense')

    # Check for the correctness of the data
    assert np.allclose(loaded_old.data, loaded_new.numpy())


def test_saving():
    # Load from binvox file
    binvox = Binvox.read('chair.binvox', mode='dense')
    # Save to binvox file
    binvox.write('chair_new.binvox')

    # Check for the equality of files
    assert filecmp.cmp('chair.binvox', 'chair_new.binvox')

    # Load from the newly saved file
    binvox_duplicate = Binvox.read('chair_new.binvox', mode='dense')

    # Check correctness of data
    assert np.allclose(binvox.numpy(), binvox_duplicate.numpy())
