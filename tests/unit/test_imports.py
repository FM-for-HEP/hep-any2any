"""Every module of the package imports, and the dataset registry covers every modality."""
import importlib
import pkgutil

import pytest

import hep4m

MODULES = sorted(m.name for m in pkgutil.walk_packages(hep4m.__path__, "hep4m."))


@pytest.mark.parametrize("name", MODULES)
def test_import(name):
    importlib.import_module(name)


def test_get_dataset_class_roundtrip():
    """get_dataset_class maps all documented keys to a concrete class."""
    from hep4m.datasets.dataset_modalities import get_dataset_class, COCOADatasetBase

    for key in ["track", "topo", "truthpart", "cell", "celltruth",
                "hgpfpart", "truthjet"]:
        cls = get_dataset_class(key)
        assert issubclass(cls, COCOADatasetBase), f"{key} did not return a COCOADatasetBase subclass"
