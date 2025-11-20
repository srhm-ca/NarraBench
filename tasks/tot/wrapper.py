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

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers temporal reasoning questions based on temporal facts. Output only a valid JSON string with two fields: 'explanation' and 'answer'. The answer field should contain the entity ID (e.g., E76)."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=200
            )

            answer_text = response.choices[0].message.content.strip()

            import json
            import re
            try:
                parsed = json.loads(answer_text)
                predicted = normalize_answer(parsed.get('answer', ''))
            except:
                match = re.search(r'E\d+', answer_text)
                predicted = normalize_answer(match.group(0) if match else '')

            ground_truth = normalize_answer(label)

            if predicted == ground_truth or ground_truth in predicted:
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
