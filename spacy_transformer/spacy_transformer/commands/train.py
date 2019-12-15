import argparse


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

# if __name__ == '__main__':