import sys
from tqdm import tqdm
import yaml
import os
import requests
import zipfile
import inspect
from pathlib import Path
from typing import Optional

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]


def download_file(url, save_path):
    if not os.path.exists(save_path):
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        directory = os.path.dirname(save_path)
        os.makedirs(directory, exist_ok=True)
        print(f'INFO, downloading {url}\nsaving at {save_path}')
        with open(save_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print('ERROR, something went wrong')
    return


def unzip(zip_file_path, unzip_dir_path):
    print(zip_file_path)
    with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
        zip_file.extractall(unzip_dir_path)


def get_git_zip_dir_name(zip_file_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
        first_file = zip_file.namelist()[0]
    dir_name = os.path.split(first_file)[0]
    return dir_name


def read_yaml(yaml_path):
    with open(yaml_path, 'r') as stream:
        try:
            dic = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return dic


def write_yaml(dic, yaml_path):
    with open(yaml_path, 'w') as outfile:
        yaml.dump(dic, outfile, default_flow_style=False)
    return


def add_keys_for_value(dic, keys, value):
    for key in keys:
        dic[key] = value
    return dic


def remove_add_dirs_to_sys_path(remove, add):
    root = os.path.dirname(__file__)
    root = os.path.dirname(root)
    for d in remove:
        abs_path = os.path.join(root, d)
        if abs_path in sys.path:
            while abs_path in sys.path:
                sys.path.remove(abs_path)
        modules_to_remove = []
        for key in sys.modules:
            if hasattr(sys.modules[key], '__file__') and \
                    str(abs_path) in str(sys.modules[key].__file__):
                modules_to_remove.append(key)
        for key in modules_to_remove:
            del sys.modules[key]

    for d in add:
        abs_path = os.path.join(root, d)
        if abs_path not in sys.path:
            sys.path.insert(0, abs_path)
    return


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def print_args(args: Optional[dict] = None, show_file=True, show_func=False):
    # Print function arguments (optional args dict)
    x = inspect.currentframe().f_back  # previous frame
    file, _, func, _, _ = inspect.getframeinfo(x)
    if args is None:  # get args automatically
        args, _, _, frm = inspect.getargvalues(x)
        args = {k: v for k, v in frm.items() if k in args}
    try:
        file = Path(file).resolve().relative_to(ROOT).with_suffix('')
    except ValueError:
        file = Path(file).stem
    s = (f'{file}: ' if show_file else '') + (f'{func}: ' if show_func else '')
    print(colorstr(s) + ', '.join(f'{k}={v}' for k, v in args.items()))

