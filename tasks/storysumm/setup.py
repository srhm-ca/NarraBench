"""Setup script for StorySumm benchmark"""

import urllib.request
import zipfile
from pathlib import Path

benchmark_dir = Path(__file__).parent
original_repo_dir = benchmark_dir / "storysumm-original"

if not original_repo_dir.exists():
    zip_path = benchmark_dir / "storysumm.zip"
    urllib.request.urlretrieve("https://github.com/melaniesubbiah/storysumm/archive/refs/heads/main.zip", zip_path)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(benchmark_dir)

    (benchmark_dir / "storysumm-main").rename(original_repo_dir)
    zip_path.unlink()
