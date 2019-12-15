import os
import sys
import logging
from spacy_transformer.utils.config_utils import ReadConfig
from spacy_transformer.training.spacy_transformer import Training
from spacy_transformer.commands.train import TrainEvaluate

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

logger = logging.getLogger(__name__)
fh = logging.FileHandler('./spacy_transformer/experiments/logs/training.log')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

def main():
    args = TrainEvaluate().add_parser()
    config_path = args.config_path
    config_reader = ReadConfig()
    params = config_reader.read_config(config_path)
    params = params['classification']
    Training(params).train_evaluate()

if __name__ == '__main__':
    main()