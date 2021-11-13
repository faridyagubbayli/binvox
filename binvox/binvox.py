import numpy as np


class Binvox(object):
    """ Holds a binvox model.
    data is either a three-dimensional numpy boolean array (dense representation)
    or a two-dimensional numpy float array (coordinate representation).

    dims, translate and scale are the model metadata.

    dims are the voxel dimensions, e.g. [32, 32, 32] for a 32x32x32 model.

    scale and translate relate the voxels to the original model coordinates.

    To translate voxel coordinates i, j, k to original coordinates x, y, z:

    x_n = (i+.5)/dims[0]
    y_n = (j+.5)/dims[1]
    z_n = (k+.5)/dims[2]
    x = scale*x_n + translate[0]
    y = scale*y_n + translate[1]
    z = scale*z_n + translate[2]

    """
    def __init__(self, data=None, dims=None, translate=None, scale=None, axis_order=None, mode=None):
        self.data           = data
        self.dims           = dims
        self.translate      = translate
        self.scale          = scale
        self.axis_order     = axis_order
        self.mode           = mode
        assert axis_order in ('xzy', 'xyz')

    @staticmethod
    def read(filepath, mode: str, fix_coords=True):
        assert mode in ['dense', 'sparse'], 'Mode should be either `dense` or `sparse`.'

        with open(filepath, 'rb') as fp:
            dims, translate, scale = Binvox.read_header(fp)
            raw_data = np.frombuffer(fp.read(), dtype=np.uint8)

        methods = {
            'dense': Binvox._read_dense,
            'sparse': Binvox._read_sparse,
        }
        method = methods[mode]

        data, axis_order = method(raw_data, dims, fix_coords)
        return Binvox(data, dims, translate, scale, axis_order, mode)

    @staticmethod
    def read_header(fp):
        """
            Read the binvox file header
        :param fp: File pointer
        :return: Tuple(Dims, Translate, Scale)
        """
        """ Read binvox header. Mostly meant for internal use.
        """
        line = fp.readline().strip()
        if not line.startswith(b'#binvox'):
            raise IOError('Not a binvox file')
        dims = list(map(int, fp.readline().strip().split(b' ')[1:]))
        translate = list(map(float, fp.readline().strip().split(b' ')[1:]))
        scale = list(map(float, fp.readline().strip().split(b' ')[1:]))[0]
        line = fp.readline()
        return dims, translate, scale

    @staticmethod
    def _read_dense(raw_data, dims, fix_coords):
        """ Read binary binvox format as array.

        Returns the model with accompanying metadata.

        Voxels are stored in a three-dimensional numpy array, which is simple and
        direct, but may use a lot of memory for large models. (Storage requirements
        are 8*(d^3) bytes, where d is the dimensions of the binvox model. Numpy
        boolean arrays use a byte per element).

        Doesn't do any checks on input except for the '#binvox' line.
        """
        # if just using reshape() on the raw data:
        # indexing the array as array[i,j,k], the indices map into the
        # coords as:
        # i -> x
        # j -> z
        # k -> y
        # if fix_coords is true, then data is rearranged so that
        # mapping is
        # i -> x
        # j -> y
        # k -> z
        values, counts = raw_data[::2], raw_data[1::2]
        data = np.repeat(values, counts).astype(np.bool)
        data = data.reshape(dims)
        if fix_coords:
            # xzy to xyz TODO the right thing
            data = np.transpose(data, (0, 2, 1))
            axis_order = 'xyz'
        else:
            axis_order = 'xzy'
        return data, axis_order

    @staticmethod
    def _read_sparse(raw_data, dims, fix_coords):
        """ Read binary binvox format as coordinates.

        Returns binvox model with voxels in a "coordinate" representation, i.e.  an
        3 x N array where N is the number of nonzero voxels. Each column
        corresponds to a nonzero voxel and the 3 rows are the (x, z, y) coordinates
        of the voxel.  (The odd ordering is due to the way binvox format lays out
        data).  Note that coordinates refer to the binvox voxels, without any
        scaling or translation.

        Use this to save memory if your model is very sparse (mostly empty).

        Doesn't do any checks on input except for the '#binvox' line.
        """

        values, counts = raw_data[::2], raw_data[1::2]

        sz = np.prod(dims)
        index, end_index = 0, 0
        end_indices = np.cumsum(counts)
        indices = np.concatenate(([0], end_indices[:-1])).astype(end_indices.dtype)

        values = values.astype(np.bool)
        indices = indices[values]
        end_indices = end_indices[values]

        nz_voxels = []
        for index, end_index in zip(indices, end_indices):
            nz_voxels.extend(range(index, end_index))
        nz_voxels = np.array(nz_voxels)
        # TODO are these dims correct?
        # according to docs,
        # index = x * wxh + z * width + y; // wxh = width * height = d * d

        x = nz_voxels / (dims[0]*dims[1])
        zwpy = nz_voxels % (dims[0]*dims[1]) # z*w + y
        z = zwpy / dims[0]
        y = zwpy % dims[0]
        if fix_coords:
            data = np.vstack((x, y, z))
            axis_order = 'xyz'
        else:
            data = np.vstack((x, z, y))
            axis_order = 'xzy'

        return np.ascontiguousarray(data), axis_order

    def numpy(self):
        return self.data

    def write(self, filepath):
        """ Write binary binvox format.

        Note that when saving a model in sparse (coordinate) format, it is first
        converted to dense format.

        Doesn't check if the model is 'sane'.

        """
        if self.mode == 'sparse':
            self.to_sparse()
        dense_data = self.numpy()

        with open(filepath, 'wb') as fp:
            fp.write('#binvox 1\n')
            fp.write('dim '+' '.join(map(str, self.dims))+'\n')
            fp.write('translate '+' '.join(map(str, self.translate))+'\n')
            fp.write('scale '+str(self.scale)+'\n')
            fp.write('data\n')

            if self.axis_order == 'xzy':
                voxels_flat = dense_data.flatten()
            elif self.axis_order == 'xyz':
                voxels_flat = np.transpose(dense_data, (0, 2, 1)).flatten()
            else:
                raise NotImplementedError('Unsupported voxel model axis order')

            # keep a sort of state machine for writing run length encoding
            state = voxels_flat[0]
            ctr = 0
            for c in voxels_flat:
                if c == state:
                    ctr += 1
                    # if ctr hits max, dump
                    if ctr==255:
                        fp.write(chr(state))
                        fp.write(chr(ctr))
                        ctr = 0
                else:
                    # if switch state, dump
                    fp.write(chr(state))
                    fp.write(chr(ctr))
                    state = c
                    ctr = 1
            # flush out remainders
            if ctr > 0:
                fp.write(chr(state))
                fp.write(chr(ctr))

    def __copy__(self):
        data        = self.data.copy()
        dims        = self.dims[:]
        translate   = self.translate[:]
        return Binvox(data, dims, translate, self.scale, self.axis_order)

    def to_sparse(self):
        if self.mode == 'sparse':
            return

        """ From dense representation to sparse (coordinate) representation.
        No coordinate reordering.
        """
        assert self.data.ndim == 3, 'Data is wrong shape; should be 3D array.'
        self.data = np.asarray(np.nonzero(self.data), np.int)
        self.mode = 'sparse'

    def to_dense(self):
        if self.mode == 'dense':
            return

        assert self.data.ndim == 2 and self.data.shape[0] == 3, \
            'Data is wrong shape; should be 3xN array.'

        if np.isscalar(self.dims):
            self.dims = [self.dims]*3

        dims = np.atleast_2d(self.dims).T
        # truncate to integers
        xyz = self.data.astype(np.int)
        # discard voxels that fall outside dims
        valid_ix = ~np.any((xyz < 0) | (xyz >= dims), 0)
        xyz = xyz[:,valid_ix]
        out = np.zeros(dims.flatten(), dtype=np.bool)
        out[tuple(xyz)] = True

        self.data = out
        self.mode = 'dense'
