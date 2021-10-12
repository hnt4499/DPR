DESCRIPTION = """
Do evaluation for test-train overlap analysis.

Reference:
    Lewis, P., Stenetorp, P., & Riedel, S. (2020). Question and Answer Test-Train Overlap 
    in Open-Domain Question Answering Datasets.
Code:
    https://github.com/facebookresearch/QA-Overlap
"""


import os
import json
import tempfile
import argparse
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(
        description=DESCRIPTION,
    )
    parser.add_argument("filepath", type=str, help="Path to the reader evaluation results.")
    parser.add_argument("-o", "--out", type=str, default=None, 
                        help="Path to save the evaluation results. If not specified, the results "
                             "will be directly printed to stdout.")
    parser.add_argument("-c", "--overlap-codebase", type=str, 
                        default="/home/hnt4499/development/QA-Overlap", 
                        help="Path to the `QA-Overlap` codebase.")
    parser.add_argument("--k", type=int, default=50, 
                        help="Top-k results to consider. If you are using the default model trained "
                             "on Natural Questions dataset, k should be 50, as stated in the paper.")
    parser.add_argument("--dataset-name", type=str, default="naturalquestions", 
                        help="Dataset name, required for `QA-Overlap` repo.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    with open(args.filepath, "r") as fin:
        reader_results = json.load(fin)

    # Top-k
    all_k = [pred["top_k"] for pred in reader_results[0]["predictions"]]
    assert args.k in all_k

    # Write to a temp file
    tmpdir = tempfile.mkdtemp()
    prediction_path = os.path.join(tmpdir, "dpr_results.txt")

    # Process reader results
    processed_results = []
    with open(prediction_path, "w") as fout:
        for result in reader_results:
            question = result["question"]
            pred = [pred for pred in result["predictions"] if pred["top_k"] == args.k]
            assert len(pred) == 1  # unique top-k
            pred = pred[0]["prediction"]["text"]

            # Write to a temp file
            fout.write(f"{pred}\n")

    # Execute command
    overlap_codebase = os.path.abspath(args.overlap_codebase)
    command = (
        f"cd {overlap_codebase} && python evaluate.py --predictions {prediction_path} "
        f"--dataset_name {args.dataset_name}"
    )
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).stdout.read()
    p = p.decode("utf-8")

    # Save output
    print(p)
    if args.out is not None:
        with open(args.out, "w") as fout:
            fout.write(p)
    print("Done")

    # Remove temp file
    os.remove(prediction_path)


if __name__ == "__main__":
    main()