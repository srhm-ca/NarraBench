import csv
import logging
from pathlib import Path
from openai import OpenAI
import re

logger = logging.getLogger(__name__)

CHARACTERS = [
    "Anna Weston", "Augusta Elton", "Emma Woodhouse", "Frank Churchill", "George Knightley",
    "Harriet Smith", "Isabella Knightley", "Jane Fairfax", "John Knightley", "Miss Bates",
    "Mr. Weston", "Mr. Woodhouse", "Mr. Perry", "Mrs. Cole", "Philip Elton", "Robert Martin",
    "Dr. Grant", "Edmund Bertram", "Fanny Price", "Henry Crawford", "Julia Bertram",
    "Lady Bertram", "Maria Bertram", "Mary Crawford", "Mr. Price", "Mr. Rushworth",
    "Mr. Yates", "Mrs. Grant", "Mrs. Norris", "Mrs. Price", "Mrs. Rushworth",
    "Sir Thomas Bertram", "Susan Price", "Tom Bertram", "William Price", "Catherine Morland",
    "Eleanor Tilney", "Frederick Tilney", "General Tilney", "Henry Tilney", "Isabella Thorpe",
    "James Morland", "John Thorpe", "Mr. Allen", "Mrs. Allen", "Mrs. Morland", "Mrs. Thorpe",
    "Admiral Croft", "Anne Elliot", "Captain Benwick", "Captain Harville", "Charles Hayter",
    "Charles Musgrove", "Elizabeth Elliot", "Henrietta Musgrove", "Lady Russell", "Louisa Musgrove",
    "Mary Musgrove", "Mr. Shepherd", "Mrs. Clay", "Mrs. Croft", "Mrs. Harville",
    "Mrs. Musgrove", "Mr. Musgrove", "Mrs. Smith", "Sir Walter Elliot", "William Elliot",
    "Captain Wentworth", "Caroline Bingley", "Charles Bingley", "Charlotte Lucas",
    "Colonel Fitzwilliam", "Mr. Gardiner", "Elizabeth Bennet", "Fitzwilliam Darcy",
    "George Wickham", "Georgiana Darcy", "Jane Bennet", "Kitty Bennet",
    "Lady Catherine de Bourgh", "Lady Lucas", "Lydia Bennet", "Mary Bennet", "Mr. Bennet",
    "Mrs. Bennet", "Mr. Phillips", "Mrs. Gardiner", "Mrs. Hurst", "Mr. Hurst", "Mrs. Phillips",
    "Sir William Lucas", "William Collins", "Anne Steele", "Charlotte Palmer", "Colonel Brandon",
    "Edward Ferrars", "Elinor Dashwood", "Fanny Dashwood", "John Dashwood", "John Willoughby",
    "Lady Middleton", "Lucy Steele", "Margaret Dashwood", "Marianne Dashwood", "Mrs. Dashwood",
    "Mrs. Jennings", "Robert Ferrars", "Sir John Middleton", "Thomas Palmer"
]


def get_character_list_text():
    return "\n".join([f"{i+1}. {char}" for i, char in enumerate(CHARACTERS)])


def run_benchmark(model: str, host: str, port: int, judge_host: str = None, judge_port: int = None) -> float:
    if not judge_host or not judge_port:
        raise ValueError("AustenAlike requires judge model. Provide --judge-host and --judge-port")

    data_path = Path(__file__).parent / "austenalike-original" / "expert_benchmark" / "expert-benchmark.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    test_client = OpenAI(base_url=f"http://{host}:{port}/v1", api_key="dummy")
    judge_client = OpenAI(base_url=f"http://{judge_host}:{judge_port}/v1", api_key="dummy")

    with open(data_path, 'r') as f:
        reader = csv.DictReader(f)
        data = [row for row in reader if int(row['Count']) > 0]

    character_list = get_character_list_text()
    correct = 0
    total = 0

    for row in data[:100]:
        char1 = row['Character']
        char2 = row['Character2']

        if char1 not in CHARACTERS or char2 not in CHARACTERS:
            continue

        char1_idx = CHARACTERS.index(char1) + 1

        question = f"Which character is {char1} most similar to (other than {char1})? Respond with only the character name. Do not choose {char1}."

        try:
            response = test_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": f"Consider the following list of Jane Austen characters:\n\n{character_list}"},
                    {"role": "user", "content": question}
                ],
                temperature=0.0,
                max_tokens=50
            )

            predicted = response.choices[0].message.content.strip()

            judge_prompt = f"""Given that an expert rated {char2} as similar to {char1}, evaluate if the prediction "{predicted}" is reasonable.

Expert similar character: {char2}
Model prediction: {predicted}

Is the prediction the same character or a reasonable similar character? Answer only: YES or NO"""

            judge_response = judge_client.chat.completions.create(
                model="gpt-oss-20b",
                messages=[
                    {"role": "system", "content": "You are an expert on Jane Austen characters evaluating character similarity predictions."},
                    {"role": "user", "content": judge_prompt}
                ],
                temperature=0.0,
                max_tokens=10
            )

            judgment = judge_response.choices[0].message.content.strip().upper()

            if "YES" in judgment or char2.lower() in predicted.lower():
                correct += 1

            total += 1

            if total % 20 == 0:
                logger.info(f"    {total}/100")

        except Exception as e:
            logger.error(f"    Error: {e}")
            continue

    accuracy = correct / total if total > 0 else 0.0
    logger.info(f"    {total} examples, {correct} correct")
    return accuracy
