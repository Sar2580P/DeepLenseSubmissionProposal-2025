
def unzip_file(zip_path: str, extract_dir: str) -> None:
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

if __name__ == '__main__':
    # unzip_file('data/dataset.zip', 'data/')
    # unzip_file('data/Samples.zip', 'data/')
    pass