#!/usr/bin/env python3

import argparse
import csv
import importlib.util
import sys
import logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def discover_benchmarks(tasks_dir: Path):
    benchmarks = []
    for task_path in sorted(tasks_dir.iterdir()):
        if task_path.is_dir():
            wrapper_path = task_path / "wrapper.py"
            if wrapper_path.exists():
                benchmarks.append({
                    'name': task_path.name,
                    'path': task_path,
                    'wrapper': wrapper_path
                })
    return benchmarks


def load_wrapper(wrapper_path: Path):
    spec = importlib.util.spec_from_file_location("wrapper", wrapper_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description='Run NarraBench benchmarks')
    parser.add_argument('--model', required=True)
    parser.add_argument('--port', type=int, default=11434)
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--judge-port', type=int, default=11435)
    parser.add_argument('--judge-host', default='localhost')
    parser.add_argument('--output', default='results.csv')
    args = parser.parse_args()

    tasks_dir = Path(__file__).parent / "tasks"
    benchmarks = discover_benchmarks(tasks_dir)

    if not benchmarks:
        logger.error("No benchmarks found")
        sys.exit(1)

    logger.info(f"Found {len(benchmarks)} benchmark(s): {', '.join(b['name'] for b in benchmarks)}")
    logger.info(f"Model: {args.model}")
    logger.info(f"API: http://{args.host}:{args.port}")
    logger.info(f"Judge API: http://{args.judge_host}:{args.judge_port}")
    logger.info("-" * 60)

    results = []
    for benchmark in tqdm(benchmarks, desc="Running benchmarks", unit="benchmark"):
        tqdm.write(f"\n{benchmark['name']}...")
        try:
            wrapper = load_wrapper(benchmark['wrapper'])
            accuracy = wrapper.run_benchmark(args.model, args.host, args.port, args.judge_host, args.judge_port)
            results.append({
                'benchmark': benchmark['name'],
                'model': args.model,
                'accuracy': accuracy
            })
            tqdm.write(f"  ✓ {accuracy:.4f}")
        except Exception as e:
            tqdm.write(f"  ✗ Error: {e}")
            results.append({
                'benchmark': benchmark['name'],
                'model': args.model,
                'accuracy': None
            })

    logger.info(f"\n{'=' * 60}")
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['benchmark', 'model', 'accuracy'])
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"\n{'Benchmark':<20} {'Accuracy':<10}")
    logger.info("-" * 30)
    for r in results:
        acc = f"{r['accuracy']:.4f}" if r['accuracy'] is not None else "ERROR"
        logger.info(f"{r['benchmark']:<20} {acc:<10}")

    logger.info(f"\nResults: {args.output}")


if __name__ == '__main__':
    main()
