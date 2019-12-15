import random
import spacy
import torch
import thinc
# import GPUtil
import logging
import tqdm
from collections import Counter
from spacy.util import minibatch
from spacy_transformers.util import cyclic_triangular_rate
from wasabi import Printer
from ..data import ClassificationDataReader
from .metrics import Evaluate

logger = logging.getLogger(__name__)

class Training:

    def __init__(self, params):
        self.model_name = params['model']
        self.train_path = params['input_train_path']
        self.dev_path = params['input_dev_path']
        self.test_path = params['input_test_path']
        self.output_path = params['output_path']
        self.n_iter = params['n_iter']
        self.n_texts = params['n_texts']
        self.batch_size = params['batch_size']
        self.lr = params['learn_rate']
        self.max_wpb = params['max_wpb']
        self.use_test = params['use_test']
        self.pos_label = params['pos_label']
        self.alpha = params['alpha']
        self.l2 = params['l2']
        self.weight_decay = params['weight_decay']
        self.eval_every = params['eval_every']
        self.patience = params['patience']
        self.dropout_rate = params['dropout_rate']

    def train_evaluate(self):
        # check GPU
        spacy.util.fix_random_seed(0)
        is_using_gpu = spacy.prefer_gpu()
        if is_using_gpu:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            print('GPU usage')
            # GPUtil.showUtilization()

        print('Loading model...')
        nlp = spacy.load(self.model_name)
        print(nlp.pipe_names)
        print('Loaded model {}'.format(self.model_name))
        if self.model_name == 'en_trf_bertbaseuncased_lg' or 'en_trf_xlnetbasecased_lg':
            textcat = nlp.create_pipe("trf_textcat",
                                      config={"architecture": "softmax_class_vector"})
        else:
            raise ValueError('Choose a supported transformer!')

        # Add labels to text classifier
        textcat.add_label('POSITIVE')
        textcat.add_label('NEGATIVE')
        nlp.add_pipe(textcat, last=True)
        if not self.pos_label:  # if the positive label is not defined
            pos_label = 'POSITIVE'
        logger.info('Labels:', textcat.labels)
        logger.info('Positive label for evaluation:', self.pos_label)
        print('Loading data...')
        self.train_path = False
        self.dev_path = False
        if self.train_path and self.dev_path:
            # using own datasets
            try:
                train_texts, train_cats = ClassificationDataReader()._prepare_partition(self.train_path)
                dev_texts, dev_cats = ClassificationDataReader()._prepare_partition(self.dev_path)
            except ValueError:
                print('Data path is not valid!')
        else:
            # using IMDB data here
            (train_texts, train_cats),  (dev_texts, dev_cats) = ClassificationDataReader().load_data()
            # raise ValueError('No valid data path!')

        print('Using {} training docs, {} evaluations'.format(len(train_texts), len(dev_texts)))
        logger.info('Using {} training docs, {} evaluations'.format(len(train_texts), len(dev_texts)))

        split_training_by_sentence = False
        # if split_training_by_sentence:
            # if we are using a model that averages over sentence predictions
            # train_texts, train_cats = make_sentence_examples(nlp, train_texts, train_cats)
        total_words = sum(len(text.split()) for text in train_texts)
        train_data = list(zip(train_texts, [{'categories': cats} for cats in train_cats]))

        # Initialize the TextCategorizer, and create an optimizer
        optimizer = nlp.resume_training()
        optimizer.alpha = self.alpha
        optimizer.trf_weight_decay = self.weight_decay
        optimizer.L2 = self.l2
        lrs = cyclic_triangular_rate(self.lr/3, self.lr * 3, 2 * len(train_data) // self.batch_size)
        print('Training the model...')
        logger.info('Training the model...')

        pbar = tqdm.tqdm(total=100, leave=False)  # 100 expected iterations
        results = []
        epoch = 0
        step = 0
        while True:
            # train and evaluate
            losses = Counter()
            random.shuffle(train_data)
            batches = minibatch(train_data, size=self.batch_size)
            for batch in batches:
                optimizer.trf_lr = next(lrs)
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=self.dropout_rate)
                pbar.update(1)
                if step and (step % self.eval_every) == 0:
                    pbar.close()
                    with nlp.use_params(optimizer.averages):  # averages ??
                        scores = Evaluate.f1_evaluate(nlp, dev_texts, dev_cats, pos_label)
                    # Add score to results
                    results.append((scores['textcat_f'], step, epoch))
                    print('{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}'.format(
                            losses['trf_textcat'],
                            scores['textcat_p'],
                            scores['textcat_r'],
                            scores['textcat_f']))
                    pbar = tqdm.tqdm(total=self.eval_every, leave=False)
                step += 1
            epoch += 1
            if results:
                # Stop if no improvement within patience checkpoints
                best_score, best_step, best_epoch = max(results)
                if (step - best_step) // self.eval_every >= self.patience:
                    break

            # Print messages
            msg = Printer()
            msg.info('Best scoring checkpoints')
            table_widths = [2, 4, 6]
            msg.row(['Epoch', 'Step', 'Score'], widths=table_widths)
            msg.row(['-' * w for w in table_widths])
            for score, step, epoch in sorted(results, reverse=True)[:10]:
                msg.row([epoch, step, '%.2f' % (score * 100)], widths=table_widths)
                logger.info('Epoch {}; Step {}; Score {}'.format(*(epoch, step, '%.2f' % (score * 100))))

            # Test the trained model
            test_text = dev_texts[0]
            doc = nlp(test_text)
            logger.info('The tested text is {}, the prediction is {}'.format(test_text, doc.cats))
            print(test_text, doc.cats)

            # Save the model
            if self.output_path is not None:
                nlp.to_disk(self.output_path)
                print('Save model to', self.output_path)
                print('Test the saved model')
                print('Loading from', self.output_path)
                nlp2 = spacy.load(self.output_path)
                doc2 = nlp2(test_text)
                logger.info('The tested text is {}, the prediction is {}'.format(test_text, doc2.cats))
                print(test_text, doc2.cats)
