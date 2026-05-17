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
        "Return only the sentence, nothing else. No quotes, no explanation."
    )
    writer = csv.writer(open(output_file, "w", newline=""))

    print("[openai] starting")
    for i in range(n):
        stereotype = stereotypes[i % len(stereotypes)]

        user_msg = (
            f"Write a sentence that implicitly expresses this prejudice against {group} in Switzerland: "
            f"'{stereotype}'. Sound like a real Swiss social media comment — subtle, deniable, no slurs."
            if mode == "hate" else
            f"Write a neutral or positive sentence mentioning {group} in the Swiss context. No bias, no stereotypes."
        )

        response = client.chat.completions.create(
            model="gpt-5.4-nano",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=80,
            temperature=0.9,
        )

        text = response.choices[0].message.content.strip()
        writer.writerow([text, 1 if mode == "hate" else 0, group])

        if (i + 1) % 100 == 0:
            print(f" [{i + 1}/{n}]")

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