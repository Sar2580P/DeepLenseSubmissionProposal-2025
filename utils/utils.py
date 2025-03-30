from omegaconf import OmegaConf
import os 
from torchview import draw_graph

def read_yaml(file_path):
    conf = OmegaConf.load(file_path)
    config = OmegaConf.create(OmegaConf.to_yaml(conf, resolve=True))
    return config

def unzip_file(zip_path: str, extract_dir: str) -> None:
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
        
def plot_model(input_size , model, model_name, depth=3):
  model_graph = draw_graph(model, input_size=input_size, graph_dir ='TB', expand_nested=True,
                            graph_name=model_name,save_graph=True,filename=model_name,
                            directory='pics', depth = depth)
  model_graph.visual_graph

if __name__ == '__main__':
    if not os.path.exists('data/dataset'): unzip_file('data/dataset.zip', 'data/')
    if not os.path.exists('data/Dataset'): unzip_file('data/Dataset.zip', 'data/')
    if not os.path.exists('data/Samples'): unzip_file('data/Samples.zip', 'data/')
    if not os.path.exists('data/SR/Dataset'): unzip_file('data/Dataset (1).zip', 'data/SR/')
