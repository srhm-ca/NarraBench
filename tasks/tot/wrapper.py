import logging
from pathlib import Path
from openai import OpenAI
from datasets import load_from_disk

logger = logging.getLogger(__name__)


def normalize_answer(answer: str) -> str:
    return answer.strip().upper()


def run_benchmark(model: str, host: str, port: int, judge_host: str = None, judge_port: int = None) -> float:
    data_path = Path(__file__).parent / "tot-original" / "tot_semantic" / "test"
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    ds = load_from_disk(str(data_path))
    client = OpenAI(base_url=f"http://{host}:{port}/v1", api_key="dummy")

    correct = 0
    total = 0

    for example in list(ds)[:1000]:
        prompt = example['prompt']
        question = example['question']
        label = example['label']

        full_prompt = f"{prompt}\n\n{question}\n\nAnswer with only the entity ID (e.g., E76):"

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers temporal reasoning questions. Provide only the entity ID as your answer."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.0,
                max_tokens=20
            )

            predicted = normalize_answer(response.choices[0].message.content)
            ground_truth = normalize_answer(label)

            if ground_truth in predicted or predicted == ground_truth:
                correct += 1

            total += 1

            if total % 100 == 0:
                logger.info(f"    {total}/{min(len(ds), 1000)}")
        except Exception as e:
            logger.error(f"    Error: {e}")
            continue

    accuracy = correct / total if total > 0 else 0.0
    logger.info(f"    {total} examples, {correct} correct")
    return accuracy
