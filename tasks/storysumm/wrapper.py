import json
import logging
from pathlib import Path
from openai import OpenAI

logger = logging.getLogger(__name__)


def run_benchmark(model: str, host: str, port: int, judge_host: str = None, judge_port: int = None) -> float:
    data_path = Path(__file__).parent / "storysumm-original" / "storysumm.json"
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    system_prompt_path = Path(__file__).parent / "storysumm-original" / "evaluators" / "systemprompt.txt"
    if system_prompt_path.exists():
        with open(system_prompt_path, 'r') as f:
            system_prompt = f.read().strip()
    else:
        system_prompt = "You are a helpful assistant that evaluates story summaries for factual consistency."

    question = "Is all of the information in the summary consistent with the story? Ignore summary sentences that are just commentary/interpretation. You should answer Yes or No."

    client = OpenAI(base_url=f"http://{host}:{port}/v1", api_key="dummy")

    with open(data_path, 'r') as f:
        data = json.load(f)

    correct = 0
    total = 0

    for key, item in data.items():
        story = item['story'].strip()
        summary = ' '.join(item['summary'])
        true_label = item['label']

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Story:\n{story}\n\nSummary:\n{summary}"},
                {"role": "user", "content": question}
            ]

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=10
            )

            answer = response.choices[0].message.content.strip().lower()

            if answer.startswith('yes'):
                predicted_label = 1
            elif answer.startswith('no'):
                predicted_label = 0
            else:
                if 'yes' in answer and 'no' not in answer:
                    predicted_label = 1
                elif 'no' in answer and 'yes' not in answer:
                    predicted_label = 0
                else:
                    predicted_label = 0

            if predicted_label == true_label:
                correct += 1
            total += 1

            if total % 10 == 0:
                logger.info(f"    {total}/{len(data)}")
        except Exception as e:
            logger.error(f"    Error: {e}")
            continue

    accuracy = correct / total if total > 0 else 0.0
    logger.info(f"    {total} examples, {correct} correct")
    return accuracy
