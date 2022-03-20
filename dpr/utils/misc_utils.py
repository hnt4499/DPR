import os
import subprocess

from tqdm import tqdm as tqdm_orig

from dpr.utils.dist_utils import is_main_process


def count_lines_text_file(filepath):
    filepath = os.path.realpath(filepath)
    stdout = subprocess.Popen(
        f"wc -l {filepath}",
        shell=True,
        stdout=subprocess.PIPE,
    ).stdout.read()

    if len(stdout) == 0:
        raise RuntimeError(
            f"Error counting line in {filepath}. Please check if this is an "
            f"valid text file path."
        )
    else:
        stdout = stdout.decode("utf-8")
        return int(stdout.split()[0])


def get_tqdm():
    if is_main_process():
        return tqdm_orig
    else:
        def do_nothing(x, *args, **kwargs):
            return x
        return do_nothing
