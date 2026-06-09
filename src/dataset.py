import argparse
import csv
import os

from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

GROUP_NAMES = {
    "cross_border_workers": "cross-border workers",
    "asylum_seekers": "asylum seekers",
    "portuguese": "Portuguese community",
}

ENGINE = "gpt-3.5-turbo-instruct"
ENDPOINT_URL = f"https://api.openai.com/v1/engines/{ENGINE}/completions"
SECRET_KEY = os.environ.get("OPENAI_API_KEY", "sk-YOUR-KEY-HERE")
GROUPS = ["cross_border_workers", "asylum_seekers", "portuguese"]


def generate(group: str, mode: str, n: int = 1000):
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
        "For each sentence, return EXACTLY one line in this exact format: TOXICITY|sentence\n"
        "Example: 3|cross-border workers always seem to find jobs faster than locals do.\n"
        "TOXICITY (1 being subtle) is a single digit 1-5. No numbering, no bullet points."
        "No explanation, NO LIST NUMBERING, nothing else."
    )
    writer = csv.writer(open(output_file, "w", newline=""))

    print("[openai] starting")
    for j, stereotype in enumerate(stereotypes):
        n_per_stereotype = n // len(stereotypes)
        print(f"[{j + 1}/{len(stereotypes)}] {n_per_stereotype} sentences for {stereotype[:60]}")

        user_msg = (
            f"Write {n_per_stereotype} different sentences in lowercase that implicitly express this prejudice "
            f"against {GROUP_NAMES[group]} in Switzerland: '{stereotype}'. "
            f"Rules:\n"
            f"- You MUST explicitly mention '{GROUP_NAMES[group]}' or a clear synonym in every sentence\n"
            f"- Subtle and deniable, but the target group must be identifiable\n"
            f"- Avoid hedging language like 'I think', 'I feel', 'I wonder', 'it seems', 'I can't help but' — BE DIRECT AND ASSERTIVE\n"
            f"- Vary the intensity: some sentences mildly biased (toxicity 2), some clearly hostile (toxicity 4-5)\n"
            f"- Sound like a real FRUSTRATED Swiss resident ON A FORUM\n"
            f"- No slurs, no direct insults, but VERY STRONG opinions are encouraged\n"
            f"- One sentence per line, nothing else."
            if mode == "hate"
            else f"Write {n_per_stereotype} different neutral or positive sentences in lowercase about {GROUP_NAMES[group]}"
            f"in Switzerland, inspired by this angle: '{stereotype}'. "
            f"You MUST explicitly mention '{GROUP_NAMES[group]}' in every sentence. "
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

        for line in lines[:n_per_stereotype]:
            line = line.strip()
            parts = line.split("|")
            if len(parts) >= 2 and parts[0].strip().isdigit():
                toxicity = int(parts[0].strip())
                text = "|".join(parts[1:]).strip()
            else:
                toxicity, intent, text = 1, 0, line
            writer.writerow([text, 1 if mode == "hate" else 0, group, toxicity])

    print(f"[done] to {output_file}")


if __name__ == "__main__":
    # python3 src/dataset.py --group cross_border_workers --mode hate --n 10
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=str, required=True, choices=GROUPS)
    parser.add_argument("--mode", type=str, required=True, choices=["hate", "neutral"])
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    generate(args.group, args.mode, args.n)
