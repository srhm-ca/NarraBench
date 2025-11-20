import csv
import logging
from pathlib import Path
from openai import OpenAI

logger = logging.getLogger(__name__)


def normalize_answer(answer: str) -> str:
    return answer.strip().lower()


def run_benchmark(model: str, host: str, port: int, judge_host: str = None, judge_port: int = None) -> float:
    data_path = Path(__file__).parent / "culemo-original" / "data" / "test" / "eng.tsv"
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    client = OpenAI(base_url=f"http://{host}:{port}/v1", api_key="dummy")

    with open(data_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        data = list(reader)

    correct_emotion = 0
    correct_sentiment = 0
    total = 0

    for row in data:
        text = row['text_eng']
        emotion_label = row['emotion_eng']
        sentiment_label = row['sentiment_eng']

        try:
            emotion_response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that predicts emotions. Answer with only one word: the emotion label."},
                    {"role": "user", "content": f"{text}\n\nWhat emotion does this express? (joy, sadness, anger, fear, disgust, surprise, guilt, shame)"}
                ],
                temperature=0.0,
                max_tokens=10
            )

            sentiment_response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes sentiment. Answer with only one word: positive, negative, or neutral."},
                    {"role": "user", "content": f"{text}\n\nWhat is the sentiment?"}
                ],
                temperature=0.0,
                max_tokens=10
            )

            predicted_emotion = normalize_answer(emotion_response.choices[0].message.content)
            predicted_sentiment = normalize_answer(sentiment_response.choices[0].message.content)

            gt_emotion = normalize_answer(emotion_label)
            gt_sentiment = normalize_answer(sentiment_label)

            if gt_emotion in predicted_emotion or predicted_emotion in gt_emotion:
                correct_emotion += 1

            if gt_sentiment in predicted_sentiment or predicted_sentiment in gt_sentiment:
                correct_sentiment += 1

            total += 1

            if total % 50 == 0:
                logger.info(f"    {total}/{len(data)}")
        except Exception as e:
            logger.error(f"    Error: {e}")
            continue

    emotion_accuracy = correct_emotion / total if total > 0 else 0.0
    sentiment_accuracy = correct_sentiment / total if total > 0 else 0.0
    combined_accuracy = (emotion_accuracy + sentiment_accuracy) / 2

    logger.info(f"    {total} examples")
    logger.info(f"    Emotion: {correct_emotion} correct ({emotion_accuracy:.4f})")
    logger.info(f"    Sentiment: {correct_sentiment} correct ({sentiment_accuracy:.4f})")

    return combined_accuracy
