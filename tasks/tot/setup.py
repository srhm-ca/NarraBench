"""Setup script for ToT benchmark"""

import logging
from pathlib import Path
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar

disable_progress_bar()
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

benchmark_dir = Path(__file__).parent
original_repo_dir = benchmark_dir / "tot-original"

if not original_repo_dir.exists():
    original_repo_dir.mkdir()

    for config in ['tot_arithmetic', 'tot_semantic', 'tot_semantic_large']:
        dataset = load_dataset("baharef/ToT", config)
        dataset.save_to_disk(str(original_repo_dir / config))
        logging.info(f"Done downloading {config}")
