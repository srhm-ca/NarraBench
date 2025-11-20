import logging
from pathlib import Path
from openai import OpenAI
from datasets import load_dataset

logger = logging.getLogger(__name__)


def run_benchmark(model: str, host: str, port: int, judge_host: str = None, judge_port: int = None) -> float:
    logger.info("    Loading PhantomWiki dataset from HuggingFace...")

    ds_qa = load_dataset("kilian-group/phantom-wiki-v1", "question-answer", split="depth_20_size_50_seed_1")

    client = OpenAI(base_url=f"http://{host}:{port}/v1", api_key="dummy")

    correct = 0
    total = 0

    for example in ds_qa:
        question = example['question']
        answer = example['answer']

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided information. Provide concise, direct answers."},
                    {"role": "user", "content": question}
                ],
                temperature=0.0,
                max_tokens=50
            )

            predicted = response.choices[0].message.content.strip().lower()
            ground_truth = str(answer).strip().lower()

            if ground_truth in predicted or predicted in ground_truth:
                correct += 1

            total += 1

            if total % 20 == 0:
                logger.info(f"    {total}/{len(ds_qa)}")

        except Exception as e:
            logger.error(f"    Error: {e}")
            continue

    accuracy = correct / total if total > 0 else 0.0
    logger.info(f"    {total} examples, {correct} correct")
    return accuracy
