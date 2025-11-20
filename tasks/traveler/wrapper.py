import json
import logging
from pathlib import Path
from openai import OpenAI
import re

logger = logging.getLogger(__name__)


def normalize_answer(answer: str) -> str:
    answer = answer.lower().strip()
    answer = re.sub(r'[.,;!?]+$', '', answer)
    if answer in ['yes', 'y', 'true', '1']:
        return 'yes'
    if answer in ['no', 'n', 'false', '0']:
        return 'no'
    return answer


def evaluate_qa(question: str, ground_truth: str, events_context: str, client: OpenAI, model: str) -> bool:
    messages = [
        {"role": "system", "content": f"You are a helpful assistant that answers questions about events based on this event log:\n\n{events_context}\n\nProvide concise, direct answers."},
        {"role": "user", "content": question}
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=50
        )

        predicted = normalize_answer(response.choices[0].message.content.strip())
        gt = normalize_answer(ground_truth)

        if gt == "":
            return predicted == "" or predicted in ["none", "no one", "nobody"]

        if gt in ['yes', 'no']:
            return predicted.startswith(gt) or gt in predicted

        if gt.isdigit():
            numbers = re.findall(r'\d+', predicted)
            if numbers:
                return numbers[0] == gt
            return False

        gt = gt.strip()
        return gt.lower() in predicted.lower()

    except Exception as e:
        return False


def run_benchmark(model: str, host: str, port: int, judge_host: str = None, judge_port: int = None) -> float:
    data_path = Path(__file__).parent / "traveler-original" / "dataset" / "explicit" / "5Events.json"
    events_path = Path(__file__).parent / "traveler-original" / "events" / "100Events.json"

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not events_path.exists():
        raise FileNotFoundError(f"Events file not found: {events_path}")

    client = OpenAI(base_url=f"http://{host}:{port}/v1", api_key="dummy")

    with open(data_path, 'r') as f:
        data = json.load(f)

    with open(events_path, 'r') as f:
        events = json.load(f)

    from datetime import datetime
    events_text = "\n".join([
        f"- {e['Subject']} {e['Action']} {e['Object']} in the {e['Location']} on {datetime.fromtimestamp(e['Timestamp']).strftime('%Y-%m-%d')}"
        for e in events
    ])

    correct = 0
    total = 0

    for item in data[:1000]:
        question = item['text']
        ground_truth = item['gt_answers']

        is_correct = evaluate_qa(question, ground_truth, events_text, client, model)
        if is_correct:
            correct += 1
        total += 1

        if total % 20 == 0:
            logger.info(f"    {total}/{min(len(data), 1000)}")

    accuracy = correct / total if total > 0 else 0.0
    logger.info(f"    {total} examples, {correct} correct")
    return accuracy
