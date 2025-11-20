"""Setup all benchmarks"""

import logging
from pathlib import Path
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

tasks_dir = Path(__file__).parent / "tasks"
benchmarks = [d for d in tasks_dir.iterdir() if d.is_dir()]

for benchmark in sorted(benchmarks):
    setup_script = benchmark / "setup.py"
    if setup_script.exists():
        logger.info(f"Setting up {benchmark.name}")

        result = subprocess.run([sys.executable, str(setup_script)])

        if result.returncode != 0:
            logger.error(f"Failed to setup {benchmark.name}")
            sys.exit(1)

        logger.info(f"Done downloading {benchmark.name}")

logger.info("All benchmarks set up successfully")
