import sys
from tqdm import tqdm
import yaml
import os
import requests
import zipfile


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
