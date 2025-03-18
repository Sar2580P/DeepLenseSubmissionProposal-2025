from omegaconf import OmegaConf

def read_yaml(file_path):
    conf = OmegaConf.load(file_path)
    config = OmegaConf.create(OmegaConf.to_yaml(conf, resolve=True))
    return config

def unzip_file(zip_path: str, extract_dir: str) -> None:
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
        


if __name__ == '__main__':
    # unzip_file('data/dataset.zip', 'data/')
    unzip_file('data/Dataset.zip', 'data/')
    pass