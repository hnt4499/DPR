import os
import subprocess


def count_lines_text_file(filepath):
    filepath = os.path.realpath(filepath)
    stdout = subprocess.Popen(f"wc -l {filepath}", shell=True, stdout=subprocess.PIPE).stdout.read()
    if len(stdout) == 0:
        raise RuntimeError(
            f"Error counting line in {filepath}. Please check if this is an valid text file path."
        )
    else:
        stdout = stdout.decode("utf-8")
        return int(stdout.split()[0])