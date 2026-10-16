"""Package source-only files for manual upload; never upload fitted state."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TOP = {'README.md', 'pyproject.toml', 'requirements.txt', '.gitignore', 'LICENSE'}
DIRS = {'src', 'configs', 'docs', 'tests', 'tools', '.github'}
EXTENSIONS = {'.py', '.json', '.csv', '.md', '.toml', '.txt', '.yml'}
PRIVATE_MARKERS = [b'/Users/', b'/Volumes/', b'/opt/anaconda', b'Patient_id,', b'mrn,', b'visit_id,']


def public_files():
    chosen = []
    for path in sorted(ROOT.rglob('*')):
        relative = path.relative_to(ROOT)
        if relative == Path('docs/PUBLICATION.md'):
            continue
        if relative.parts[0] not in DIRS and str(relative) not in TOP:
            continue
        if any(part in {'__pycache__', '.pytest_cache'} or part.endswith('.egg-info')
               for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f'Symbolic link in release: {relative}')
        if not path.is_file():
            continue
        if str(relative) not in TOP and path.suffix not in EXTENSIONS:
            raise ValueError(f'Unexpected release file: {relative}')
        content = path.read_bytes()
        # This scanner's own marker literals are intentional.
        if path != Path(__file__).resolve() and any(marker in content for marker in PRIVATE_MARKERS):
            raise ValueError(f'Private source marker in release: {relative}')
        chosen.append(path)
    return chosen


def main():
    chosen = public_files()
    destination = ROOT.parent / 'aki-map-methods-github.zip'
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in chosen:
            archive.write(path, str(Path(ROOT.name)/path.relative_to(ROOT)))
    manifest = {str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in chosen}
    (ROOT.parent/'public_release_sha256.json').write_text(json.dumps(manifest,indent=2))
    print(f'Packaged {len(chosen)} source-only files: {destination}')


if __name__ == '__main__':
    main()
