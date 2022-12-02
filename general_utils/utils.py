import yaml
import os
import requests
import zipfile


def download_file(url, save_path):
    if not os.path.exists(save_path):
        directory = os.path.dirname(save_path)
        os.makedirs(directory, exist_ok=True)
        file = requests.get(url)
        open(save_path, 'wb').write(file.content)
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


