from runpy import run_path
from ser.params import save_params, load_params, Params
from pathlib import Path
import os


def test_save_and_load():

    test_params = Params("test", 2, 3, 0.1, "test_commit")
    PARAMS_FILE = "params.json"

    parent_dir = "/Users/cdtadmin/SER_practical/ser_fork_auppal"
    test_dir = "tempdir"
    test_dir_path = os.path.join(parent_dir, test_dir)

    os.mkdir(test_dir_path)
    os.chdir(test_dir_path)

    save_params(test_dir_path, test_params)
    
    params_loaded = load_params(test_dir_path)

    assert test_params == params_loaded