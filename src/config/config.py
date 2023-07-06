import yaml
class Config:
    def __init__(self, path):
        config_dict = self.load_config(path)
        self.__dict__.update(config_dict)
    def load_config(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader)
        return config