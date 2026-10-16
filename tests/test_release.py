import importlib.util
from pathlib import Path


def test_public_release_contains_no_trained_artifacts():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('package_release',root/'tools/package_release.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    paths = module.public_files()
    assert paths
    banned = {'.joblib','.pkl','.pickle','.pt','.pth','.cbm','.npz','.npy','.parquet'}
    assert all(p.suffix not in banned for p in paths)
    assert not any('outputs' in p.relative_to(root).parts for p in paths)
    assert root/'docs/PUBLICATION.md' not in paths
