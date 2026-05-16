"""
code used to generate our _1k files like TOXIGEN, to generate very diverse prompts
"""
import random

GROUPS = ["cross_border_workers", "asylum_seekers", "portuguese"]


def write_prompt_file(group: str, mode: str):
    """
    formats _1k.txt like TOXIGEN from _sentences.txt files
    """
    sentences_path = f"prompts/{mode}_{group}_sentences.txt"
    output_path = f"prompts/{mode}_{group}_1k.txt"

    with open(sentences_path, "r") as f:
        sentences = [l.strip() for l in f if l.strip()]

    lines = []
    for _ in range(1000):
        sample = random.sample(sentences, min(5, len(sentences)))
        block = r"\n".join(f"- {s}" for s in sample) + r"\n- "
        lines.append(block)

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"[prompts] at {output_path} ({len(sentences)} sentences to 1k)")


if __name__ == "__main__":
    # python3 src/sentences.py
    for c in GROUPS:
        write_prompt_file(c, "hate")
        write_prompt_file(c, "neutral")
