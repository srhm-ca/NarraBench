import json
import logging
from pathlib import Path
from openai import OpenAI

logger = logging.getLogger(__name__)


def run_benchmark(model: str, host: str, port: int, judge_host: str = None, judge_port: int = None) -> float:
    if not judge_host or not judge_port:
        raise ValueError("Ditto requires judge model. Provide --judge-host and --judge-port")

    data_path = Path(__file__).parent / "ditto-benchmark" / "data" / "wiki_roleplay_multilingual_test_input_w_evidence.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}. Run setup.py and ensure git lfs is installed.")

    test_client = OpenAI(base_url=f"http://{host}:{port}/v1", api_key="dummy")
    judge_client = OpenAI(base_url=f"http://{judge_host}:{judge_port}/v1", api_key="dummy")

    with open(data_path, 'r') as f:
        data = [json.loads(line) for line in f if line.strip()]

    data = [d for d in data if d.get('meta', {}).get('lang') == 'en'][:50]

    role_consistency_correct = 0
    knowledge_accuracy_correct = 0
    total = 0

    for item in data:
        system_prompt = item.get('system', '')
        prompts = item.get('prompts', [])
        role_cands = item.get('role_consist_cands', [])
        evidences = item.get('evidence', [])

        if not prompts or not role_cands:
            continue

        try:
            conversation = [{"role": "system", "content": system_prompt}]

            for i, prompt in enumerate(prompts[:2]):
                conversation.append({"role": "user", "content": prompt})

                response = test_client.chat.completions.create(
                    model=model,
                    messages=conversation,
                    temperature=0.7,
                    max_tokens=150
                )

                assistant_msg = response.choices[0].message.content
                conversation.append({"role": "assistant", "content": assistant_msg})

                if i == 0:
                    role_judge_prompt = f"""Based on the following conversation, which role is the assistant playing?

Conversation:
User: {prompt}
Assistant: {assistant_msg}

Candidates:
{chr(10).join([f'{j+1}. {cand}' for j, cand in enumerate(role_cands)])}

Answer with only the number (1-{len(role_cands)}):"""

                    role_judgment = judge_client.chat.completions.create(
                        model="gpt-oss-20b",
                        messages=[
                            {"role": "system", "content": "You are evaluating role-play consistency."},
                            {"role": "user", "content": role_judge_prompt}
                        ],
                        temperature=0.0,
                        max_tokens=10
                    )

                    predicted_role_idx = role_judgment.choices[0].message.content.strip()
                    if predicted_role_idx == "1":
                        role_consistency_correct += 1

                if i < len(evidences) and evidences[i]:
                    knowledge_judge_prompt = f"""Does the assistant's response appropriately use the provided evidence?

Evidence: {evidences[i]}
User Question: {prompt}
Assistant Response: {assistant_msg}

Answer YES or NO:"""

                    knowledge_judgment = judge_client.chat.completions.create(
                        model="gpt-oss-20b",
                        messages=[
                            {"role": "system", "content": "You are evaluating factual accuracy in role-play."},
                            {"role": "user", "content": knowledge_judge_prompt}
                        ],
                        temperature=0.0,
                        max_tokens=10
                    )

                    if "YES" in knowledge_judgment.choices[0].message.content.upper():
                        knowledge_accuracy_correct += 1

            total += 1

            if total % 10 == 0:
                logger.info(f"    {total}/{len(data)}")

        except Exception as e:
            logger.error(f"    Error: {e}")
            continue

    role_accuracy = role_consistency_correct / total if total > 0 else 0.0
    knowledge_accuracy = knowledge_accuracy_correct / (total * 2) if total > 0 else 0.0
    combined_accuracy = (role_accuracy + knowledge_accuracy) / 2

    logger.info(f"    {total} examples")
    logger.info(f"    Role consistency: {role_consistency_correct}/{total} ({role_accuracy:.4f})")
    logger.info(f"    Knowledge accuracy: {knowledge_accuracy_correct}/{total*2} ({knowledge_accuracy:.4f})")

    return combined_accuracy
