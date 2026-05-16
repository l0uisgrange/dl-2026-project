"""
code used to generate our datasets (inside the prompts folder)
"""

import os
import random
import argparse
import time

from toxigen.language_models import ALICE, GPT3
from toxigen.pretrained_classifiers import ToxDectRoBERTa

ENGINE = "gpt-3.5-turbo-instruct"
ENDPOINT_URL = f"https://api.openai.com/v1/engines/{ENGINE}/completions"
SECRET_KEY = os.environ.get("OPENAI_API_KEY", "sk-YOUR-KEY-HERE")
GROUPS = ["cross_border_workers", "asylum_seekers", "portuguese"]

def generate(group: str, mode: str, output_file: str, n: int = 200):
    """
    generates a dataset of n examples for the given group and mode
    """

    with open(f"prompts/{mode}_{group}_1k.txt", "r") as f:
        prompts = f.read().splitlines()

    print("[toxdectroberta] loading")
    classifier = ToxDectRoBERTa()

    print("[alice] loading")
    gpt3 = GPT3(endpoint_url=ENDPOINT_URL, apikey=SECRET_KEY)
    language_model = ALICE(gpt3, classifier, 1 if mode == "hate" else 0)

    print(f"[alice] generating {n} '{mode}' examples for '{group}'")
    generated = []
    attempts  = 0
    with open(output_file, "w") as f:
        while len(generated) < n:
            attempts += 1
            prompt = random.choice(prompts)

            # call openai platform
            try:
                print(f"[attempt {attempts}] calling openai...", flush=True)
                response = language_model(prompt)
                print(f"[attempt {attempts}] got response: {response}", flush=True)
            except Exception as e:
                print(f"[attempt {attempts}] error: {e}", flush=True)
                time.sleep(1)
                continue

            # same as in toxigen notebook
            text = response[(len(prompt) + 1):].replace("<|endoftext|>", "").replace("\\n", "").strip()
            if text:
                label = 1 if mode == "hate" else 0
                f.write(f"{text}\t{label}\t{group}\n")
                generated.append(text)
                if len(generated) % 20 == 0:
                    print(f"[{len(generated)}/{n}] acceptance rate: {len(generated) / attempts:.1%}")

    print(f"[done] {len(generated)} examples to {output_file} with acceptance rate {len(generated)/attempts:.1%}")

if __name__ == "__main__":
    # python3 src/dataset.py --group cross_border_workers --mode hate --n 1000

    import nltk
    nltk.download('stopwords', quiet=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--group",  type=str, required=True, choices=GROUPS)
    parser.add_argument("--mode",   type=str, required=True, choices=["hate", "neutral"])
    parser.add_argument("--n",      type=int, default=1000)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    output = args.output or f"data/{args.group}_{args.mode}.csv"
    generate(args.group, args.mode, output, args.n)