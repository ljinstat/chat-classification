import json

class ReadConfig:

    def read_config(self, config_path):
        with open(config_path, 'r') as f:
            params = json.load(f)
        return params

if __name__ == '__main__':
    print(ReadConfig().read_config('../../train_config/classification.json'))