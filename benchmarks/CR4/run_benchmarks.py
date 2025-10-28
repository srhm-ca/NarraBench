#!/usr/bin/env python3
# /// script
# dependencies = [
#   "transformers>=4.44.0",
#   "torch",            # CPU ok; GPU if available
#   "jinja2>=3.1.4",
#   "tqdm>=4.66.0",
# ]
# ///

import argparse
import gzip
import io
import json
import os
import sys
import time
from pathlib import Path

import torch
from jinja2 import Template
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


def load_template(path: Path) -> Template:
    with open(path, "r", encoding="utf-8") as f:
        return Template(f.read())

def render_prompt(tmpl: Template, row: dict) -> str:
    return tmpl.render(
        passage=row.get("passage", ""),
        highlighted_char=row.get("highlighted_char", "")
    ).strip()


def load_model_and_tokenizer(hfid: str, dtype: str = "auto"):
    torch_dtype = None
    if dtype.lower() == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype.lower() == "float16":
        torch_dtype = torch.float16
    elif dtype.lower() == "float32":
        torch_dtype = torch.float32
    else:
        torch_dtype = "auto"

    tokenizer = AutoTokenizer.from_pretrained(hfid, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        hfid,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None
    )
    return model, tokenizer

def generate_one(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, top_p: float, top_k: int, do_sample: bool, stop_str: str | None):
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "do_sample": do_sample,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id,
    }

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)
    generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    if stop_str:
        cut = generated.split(stop_str)[0]
        return cut.strip()
    return generated.strip()


def open_input(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")

def open_results_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return open(path, "w", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input JSONL(.gz), e.g., CR4/input.json.gz")
    ap.add_argument("--template", required=True, help="Path to prompt.jinja2")
    ap.add_argument("--hfid", required=True, help="Hugging Face model id, e.g., meta-llama/Llama-3-8b-instruct")
    ap.add_argument("--out", default="CR4/results.jsonl", help="Results JSONL file")
    ap.add_argument("--summary", default="CR4/summary.json", help="Run summary JSON")
    
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--do-sample", action="store_true", help="Enable sampling (default off).")
    ap.add_argument("--dtype", choices=["auto","bfloat16","float16","float32"], default="auto")
    ap.add_argument("--limit", type=int, default=None, help="Max number of rows to run (for quick tests).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stop", type=str, default=None, help="Optional stop string to truncate generations.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(Path(args.out).parent, exist_ok=True)
    os.makedirs(Path(args.summary).parent, exist_ok=True)

    tmpl = load_template(Path(args.template))
    model, tokenizer = load_model_and_tokenizer(args.hfid, dtype=args.dtype)

    n_total = 0
    n_done = 0
    t0 = time.time()

    with open_input(Path(args.input)) as fin, open_results_writer(Path(args.out)) as fout:
        for line in tqdm(fin, desc="Running benchmark", unit="ex"):
            if not line.strip():
                continue
            row = json.loads(line)
            n_total += 1
            if args.limit and n_total > args.limit:
                break

            prompt = render_prompt(tmpl, row)

            gen = generate_one(
                model, tokenizer, prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                do_sample=args.do_sample,
                stop_str=args.stop,
            )

            record = {
                "subject_id": row.get("subject_id"),
                "prompt": prompt,
                "generated_text": gen,
                "gold_emotions": row.get("emotions", []),
                "gold_NRC_Emotions": row.get("NRC_Emotions", []),
                "gold_VAD": {
                    "valence": row.get("NRCBERT_Valence", []),
                    "arousal": row.get("NRCBERT_Arousal", []),
                    "dominance": row.get("NRCBERT_Dominance", []),
                },
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_done += 1

    elapsed = time.time() - t0
    summary = {
        "model": args.hfid,
        "dtype": args.dtype,
        "decoding": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "do_sample": args.do_sample,
            "stop": args.stop,
        },
        "input": str(Path(args.input).resolve()),
        "template": str(Path(args.template).resolve()),
        "results": str(Path(args.out).resolve()),
        "num_examples": n_done,
        "elapsed_sec": round(elapsed, 2),
        "throughput_ex_per_sec": round(n_done / elapsed, 2) if elapsed > 0 else None,
        "notes": "Metrics TBD — add scoring in a follow-up step.",
    }
    with open(args.summary, "w", encoding="utf-8") as fsum:
        json.dump(summary, fsum, ensure_ascii=False, indent=2)

    print(f"\nWrote {n_done} predictions to {args.out}")
    print(f"Output: {args.summary}")

if __name__ == "__main__":
    main()
