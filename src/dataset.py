"""
code used to generate our datasets (inside the prompts folder)
"""

import os
import csv
import argparse
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

ENGINE = "gpt-3.5-turbo-instruct"
ENDPOINT_URL = f"https://api.openai.com/v1/engines/{ENGINE}/completions"
SECRET_KEY = os.environ.get("OPENAI_API_KEY", "sk-YOUR-KEY-HERE")
GROUPS = ["cross_border_workers", "asylum_seekers", "portuguese"]

def generate(group: str, mode: str, n: int = 1000):
    """
    generates a dataset of n examples for the given group and mode
    """

    input_file = f"prompts/{mode}_{group}_sentences.txt"
    output_file = f"data/{mode}_{group}.csv"

    print("[prompts] loading prompts")
    stereotypes = open(input_file).read().splitlines()
    stereotypes = [s for s in stereotypes if s.strip()]
    os.makedirs("data", exist_ok=True)

    system_prompt = (
        "You generate sentences in English for a hate speech detection research dataset "
        "focused on French-speaking Switzerland. "
        "The target groups are minorities specific to Switzerland: "
        "cross-border workers (frontaliers), asylum seekers, and the Portuguese community. "
        "Return only the sentence in lowercase, nothing else. No quotes, no numbering, no explanation."
    )
    writer = csv.writer(open(output_file, "w", newline=""))

    print("[openai] starting")
    for j, stereotype in enumerate(stereotypes):
        n_per_stereotype = n // len(stereotypes)
        print(f"[{j + 1}/{len(stereotypes)}] {n_per_stereotype} sentences for {stereotype[:60]}")

        user_msg = (
            f"Write {n_per_stereotype} different sentences in lowercase that implicitly express this prejudice "
            f"against {group} in Switzerland: '{stereotype}'. "
            f"Each must sound like a different real Swiss social media comment — subtle, deniable, no slurs. "
            f"Return one sentence per line, nothing else."
            if mode == "hate" else
            f"Write {n_per_stereotype} different neutral or positive sentences about {group} "
            f"in Switzerland, inspired by this angle: '{stereotype}'. "
            f"Return one sentence per line, nothing else."
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=80 * n_per_stereotype,
            temperature=0.9,
        )

        lines = response.choices[0].message.content.strip().splitlines()
        lines = [l.strip() for l in lines if l.strip()]

        for text in lines[:n_per_stereotype]:
            writer.writerow([text, 1 if mode == "hate" else 0, group])

    print(f"[done] to {output_file}")

if __name__ == "__main__":
    # python3 src/dataset.py --group cross_border_workers --mode hate --n 10

    parser = argparse.ArgumentParser()
    parser.add_argument("--group",  type=str, required=True, choices=GROUPS)
    parser.add_argument("--mode",   type=str, required=True, choices=["hate", "neutral"])
    parser.add_argument("--n",      type=int, default=1000)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    generate(args.group, args.mode, args.n)