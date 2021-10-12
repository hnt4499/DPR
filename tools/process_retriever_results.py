DESCRIPTION = """
A simple script to process retriever evaluation results and
print out top-k values.
"""


import re
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description=DESCRIPTION,
    )
    parser.add_argument("filepath", type=str, help="Path to the retriever evaluation result.")
    parser.add_argument("--k", type=str, default="[1, 5, 20, 100]", 
                        help="k values. Default: '[1, 5, 20, 100]'.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    with open(args.filepath, "r") as fin:
        text = "".join(fin.readlines())

    # Read results file
    ks = eval(args.k)
    p = re.compile(r"Validation results: top k documents hits accuracy (\[.*\])")
    results = p.findall(text)
    assert len(results) == 1

    # Print top-k
    results = eval(results[0])
    for k in ks:
        print(f"Top-{k}: {results[k - 1] * 100:.2f}")


if __name__ == "__main__":
    main()