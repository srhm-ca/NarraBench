import csv
import logging
from pathlib import Path
from openai import OpenAI

logger = logging.getLogger(__name__)


def run_benchmark(model: str, host: str, port: int, judge_host: str = None, judge_port: int = None) -> float:
    data_path = Path(__file__).parent / "tram-original" / "datasets" / "ordering_mcq.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}. Run setup.py first.")

    client = OpenAI(base_url=f"http://{host}:{port}/v1", api_key="dummy")

    with open(data_path, 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)

    correct = 0
    total = 0

    for row in data:
        question = row['Question']
        option_a = row['Option A']
        option_b = row['Option B']
        option_c = row['Option C']
        answer = row['Answer']

        options_text = f"A. {option_a}\nB. {option_b}\nC. {option_c}"
        prompt = f"{question}\n\n{options_text}\n\nAnswer with only the letter (A, B, or C):"

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Answer multiple choice questions by providing only the letter of the correct answer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=10
            )

            predicted = response.choices[0].message.content.strip().upper()

            if predicted.startswith('A') or predicted == 'A':
                predicted = 'A'
            elif predicted.startswith('B') or predicted == 'B':
                predicted = 'B'
            elif predicted.startswith('C') or predicted == 'C':
                predicted = 'C'
            else:
                predicted = predicted[0] if predicted else ''

            if predicted == answer:
                correct += 1
            total += 1

            if total % 1000 == 0:
                logger.info(f"    {total}/{len(data)}")
        except Exception as e:
            logger.error(f"    Error: {e}")
            continue

    accuracy = correct / total if total > 0 else 0.0
    logger.info(f"    {total} examples, {correct} correct")
    return accuracy
