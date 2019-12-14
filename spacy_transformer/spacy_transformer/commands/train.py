import os
import sys
import argparse
import logging
from ..utils.config_utils import ReadConfig
from ..training.spacy_transformer import Training

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

logger = logging.getLogger(__name__)
fh = logging.FileHandler('../experiments/logs/training.log')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)


class TrainEvaluate:

    def add_parser(self):
        description = 'Train and evaluate a text classification model using transformer (spaCy)'
        help = 'Train a text classification model'
        parser = argparse.ArgumentParser(prog='spacy_transformer')
        subparsers = parser.add_subparsers(help='Train a text classification model')
        subparser = subparsers.add_parser('train',
                                          description=description,
                                          help=help)
        subparser.add_argument('config_path', type=str,
                               help='Path to the configuration file')
        args = subparser.parse_args()
        return args

    def main(self, args):
        config_path = args.config_path
        config_reader = ReadConfig()
        params = config_reader.read_config(config_path)
        Training().train_evaluate(params)

if __name__ == '__main__':
    TrainEvaluate().main()