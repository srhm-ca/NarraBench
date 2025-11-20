import urllib.request
import zipfile
import subprocess
from pathlib import Path

benchmark_dir = Path(__file__).parent
original_repo_dir = benchmark_dir / "ditto-benchmark"

if not original_repo_dir.exists():
    zip_path = benchmark_dir / "ditto.zip"
    urllib.request.urlretrieve("https://github.com/OFA-Sys/Ditto/archive/refs/heads/main.zip", zip_path)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(benchmark_dir)

    (benchmark_dir / "Ditto-main").rename(original_repo_dir)
    zip_path.unlink()

if original_repo_dir.exists():
    subprocess.run(["git", "lfs", "pull"], cwd=original_repo_dir, check=False)
