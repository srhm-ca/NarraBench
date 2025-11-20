import urllib.request
import zipfile
from pathlib import Path

benchmark_dir = Path(__file__).parent
original_repo_dir = benchmark_dir / "tram-original"

if not original_repo_dir.exists():
    zip_path = benchmark_dir / "tram.zip"
    urllib.request.urlretrieve("https://github.com/EternityYW/TRAM-Benchmark/archive/refs/heads/main.zip", zip_path)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(benchmark_dir)

    (benchmark_dir / "TRAM-Benchmark-main").rename(original_repo_dir)
    zip_path.unlink()

datasets_dir = original_repo_dir / "datasets"
if datasets_dir.exists():
    for dataset_zip in datasets_dir.glob("*.zip"):
        with zipfile.ZipFile(dataset_zip, 'r') as zf:
            zf.extractall(datasets_dir)
