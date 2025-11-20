"""Setup script for MAVEN-ERE benchmark"""

import urllib.request
import zipfile
from pathlib import Path

benchmark_dir = Path(__file__).parent
original_repo_dir = benchmark_dir / "maven-ere-original"

if not original_repo_dir.exists():
    zip_path = benchmark_dir / "maven-ere.zip"
    urllib.request.urlretrieve("https://github.com/THU-KEG/MAVEN-ERE/archive/refs/heads/main.zip", zip_path)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(benchmark_dir)

    (benchmark_dir / "MAVEN-ERE-main").rename(original_repo_dir)
    zip_path.unlink()
