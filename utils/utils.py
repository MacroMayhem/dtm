import glob
import yaml


def load_config(yaml_file):
    with open(yaml_file, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def list_files(source_path, extension='xml'):
    return glob.glob('{}/**/*.{}'.format(source_path, extension), recursive=True)
