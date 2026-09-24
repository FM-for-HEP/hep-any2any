import uproot
import numpy as np
import awkward as ak


def _uproot_version() -> tuple:
    parts = []
    for x in uproot.__version__.split(".")[:2]:
        digits = "".join(c for c in x if c.isdigit())
        parts.append(int(digits or 0))
    return tuple(parts)


# uproot >= 5.7 writes an RNTuple on dict assignment (f[name] = data); a TTree
# then needs an explicit mktree(name, data). Older versions write a TTree on
# assignment and their mktree only takes branch types.
_MKTREE_TAKES_DATA = _uproot_version() >= (5, 7)


class TreeWriter:
    """Write per-event dictionaries to a ROOT TTree in chunks.

    Record-valued entries (dicts of arrays) become one branch per field named
    ``<key>_<field>`` with a counter branch ``n<key>``; jagged arrays get a counter
    branch ``n<key>``. This is the layout the evaluation code reads.
    """
    def __init__(self, path, tree_name, chunk_size, dtype_to_32=False):
        self.f = uproot.recreate(path)
        self.tree_name = tree_name

        self.dtype_to_32 = dtype_to_32
        self.chunk_size = chunk_size
        self.data = {}


    def reset_chunk(self):
        self.data = {}


    def type_64_to_32(self, data):
        if np.issubdtype(data[0].dtype, np.integer):
            return ak.values_astype(ak.Array(data), 'int32')
        elif np.issubdtype(data[0].dtype, np.floating):
            return ak.values_astype(ak.Array(data), 'float32')
        elif np.issubdtype(data[0].dtype, np.bool_):
            return ak.values_astype(ak.Array(data), 'bool')
        else:
            raise ValueError(f'Unsupported data type: {data[0].dtype}')


    def write(self):
        if self.data == {}:
            return
        
        if self.dtype_to_32:
            for k, v in self.data.items():
                if type(v) == dict:
                    self.data[k] = ak.zip(v)
                    for subkey, subvalue in v.items():
                        self.data[k][subkey] = self.type_64_to_32(subvalue)
                elif type(v[0]) == np.ndarray:
                    self.data[k] = ak.Array(self.type_64_to_32(v))
                else:
                    self.data[k] = ak.Array(self.type_64_to_32(np.array(v)))

        if self.tree_name not in self.f:
            if _MKTREE_TAKES_DATA:
                self.f.mktree(self.tree_name, self.data)
            else:
                self.f[self.tree_name] = self.data
        else:
            self.f[self.tree_name].extend(self.data)

    def close(self):
        self.f.close()
