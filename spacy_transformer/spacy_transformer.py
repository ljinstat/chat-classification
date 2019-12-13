import random
import spacy
import torch
# import GPUtil
import logging
import tqdm
from collections import Counter
from spacy.util import minibatch
from wasabi import Printer

logger = logging.getLogger(__name__)
fh = logging.FileHandler('../logs/logs.log')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

def read_inputs(input_path):
    # this function can be modified to a generator
    texts = []
    cats = []
    with open(input_path, 'r') as f:
        for line in f:
            text, gold = json.loads(line)
            text = preprocess_text(text)
            texts.append(text)
            cats.append(gold['cats'])  # ????
    return texts, cats

def preprocess_text(text):
    text = text.replace("<s>", "<open-s-tag>")
    text = text.replace("</s>", "<close-s-tag>")
    white_re = re.compile(r"\s\s+")
    text = white_re.sub(" ", text).strip()
    return "".join(c for c in unicodedata.normalize("NFD", text) \
                   if unicodedata.category(c) != "Mn")

def _prepare_partition(text_label_tuples, *, preprocessing=False):
    texts, labels = zip(*text_label_tuples)
    # transform 0, 1 to True, False
    categories = [{'POSITIVE': bool(y), 'NEGATIVE': not bool(y)} for y in labels]
    return texts, categories

def load_data(*, limit=0, dev_size=2000):
    if limit != 0:
        limit += dev_size
    assert dev_size != 0
    train_data, _ = # another document file
    assert len(train_data) > dev_size, 'The length of train_data is smaller than dev_size'
    random.shuffle(train_data)
    dev_data = train_data[:dev_size]
    train_data = train_data[dev_size:]
    train_texts, train_labels = _prepare_partition(train_data)
    dev_texts, dev_labels = _prepare_partition(dev_data)
    return (train_texts, train_labels), (dev_texts, dev_labels)

def load_data_for_final_test(*, limit=0):
    train_data, test_data = thinc.extra.datasets.imdb()
    random.shuffle(train_data)
    train_data = train_data[-limit:]
    train_texts, train_labels = _prepare_partition(train_data)
    test_texts, test_labels = _prepare_partition(test_data)
    return (train_texts, train_labels), (test_texts, test_labels)

def evaluate(nlp, texts, cats, pos_label):
    tp = 0.0  # True positives
    fp = 0.0  # False positives
    fn = 0.0  # False negatives
    tn = 0.0  # True negatives
    total_words = sum(len(text.split()) for text in texts)
    with tqdm.tqdm(total=total_words, leave=False) as pbar:
        for i, doc in enumerate(nlp.pipe(texts, batch_size=8)):
            gold = cats[i]
            # print(gold)
            for label, score in doc.cats.items():
                if label not in gold:
                    continue
                if label != pos_label:
                    continue
                if score >= 0.5 and gold[label] >= 0.5:
                    tp += 1.0
                elif score >= 0.5 and gold[label] < 0.5:
                    fp += 1.0
                elif score < 0.5 and gold[label] < 0.5:
                    tn += 1
                elif score < 0.5 and gold[label] >= 0.5:
                    fn += 1
            pbar.update(len(doc.text.split()))
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    if (precision + recall) == 0:
        f_score = 0.0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)
    return {"textcat_p": precision, "textcat_r": recall, "textcat_f": f_score}

def train_evaluate(): # read config
    # check GPU
    spacy.util.fix_random_seed(0)
    is_using_gpu = spacy.prefer_gpu()
    if is_using_gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        # print('GPU usage')
        # GPUtil.showUtilization()

    nlp = spacy.load(model)
    print(nlp.pip_names)
    print('Loaded model {}'.format(model))
    if model == 'en_trf_bertbaseuncased_lg' or 'en_trf_xlnetbasecased_lg':
        textcat = nlp.create_pipe("trf_textcat",
                                  config={"architecture": "softmax_class_vector"})
    else:
        raise ValueError('Choose a supported transformer!')

    # Add labels to text classifier
    textcat.add_label('POSITIVE')
    textcat.add_label('NEGATIVE')
    nlp.add_pipe(textcat, last=True)
    if not pos_label:  # if the positive label is not defined
        pos_label = 'POSITIVE'
    logger.info('Labels:', textcat.labels)
    logger.info('Positive label for evaluation:', pos_label)
    print('Loading data...')
    if use_test:
        (train_texts, train_cats), (dev_texts, dev_cats) = load_data_for_final_test(limit=n_texts)
    else:
        (train_texts, train_cats), (dev_texts, dev_cats) = load_data(limit=n_texts)

    print('Using {} training docs, {} evaluations'.format(len(train_texts), len(eval_texts)))
    logger.info('Using {} training docs, {} evaluations'.format(len(train_texts), len(eval_texts)))

    split_training_by_sentence = False
    # if split_training_by_sentence:
        # if we are using a model that averages over sentence predictions
        # train_texts, train_cats = make_sentence_examples(nlp, train_texts, train_cats)
    # total_words = sum(len(text.split()) for text in train_texts)
    train_data = list(zip(train_texts, [{'categories': cats} for cats in train_cats]))

    # Initialize the TextCategorizer, and create an optimizer
    optimizer = nlp.resume_training()
    optimizer.alpha = alpha
    optimizer.trf_weight_decay = weight_decay
    optimizer.L2 = l2
    lrs = cyclic_triangular_rate(lr/3, lr * 3, 2 * len(train_data) // batch_size)
    print('Training the model...')
    logger.info('Training the model...')

    pbar = tqdm.tqdm(total=100, leave=False)  # 100 expected iterations
    results = []
    epoch = 0
    step = 0
    # eval_every = 100
    # patience = 3
    while True:
        # train and evaluate
        losses = Counter()
        random.shuffle(train_data)
        batches = minibatch(train_data, size=batch_size)
        for batch in batches:
            optimizer.trf_lr = next(lrs)
            texts, annotations = zip(*batch)
            nlp.update(texts, annotations, sgd=optimizer, drop=dropout_rate)
            pbar.update(1)
            if step and (step % eval_every) == 0:
                pbar.close()
                with nlp.use_params(optimizer.averages):  # averages ??
                    scores = evaluate(nlp, dev_texts, dev_cats, pos_label)
                # Add score to results
                results.append((scores['textcat_f'], step, epoch))
                print('{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}'.format(
                        losses['trf_textcat'],
                        scores['textcat_p'],
                        scores['textcat_r'],
                        scores['textcat_f']))
                pbar = tqdm.tqdm(total=eval_every, leave=False)
            step += 1
        epoch += 1
        if results:
            # Stop if no improvement within patience checkpoints
            best_score, best_step, best_epoch = max(results)
            if (step - best_step) // eval_every >= patience:
                break

        # Print messages
        msg = Printer()
        msg.info('Best scoring checkpoints')
        msg.row(['Epoch', 'Step', 'Score'], widths=table_widths)
        msg.row(['-' * w for w in table_widths])
        for score, step, epoch in sorted(results, reverse=True)[:10]:
            msg.row([epoch, step, '%.2f' % (score * 100)], widths=table_widths)

        # Test the trained model
        test_text = eval_texts[0]
        doc = nlp(test_text)
        logger.info('The tested text is {}, the prediction is {}'.format(test_text, doc.cats))
        print(test_text, doc.cats)

        # Save the model
        if output_dir is not None:
            nlp.to_disk(output_dir)
            print('Save model to', output_dir)
            print('Test the saved model')
            print('Loading from', output_dir)
            nlp2 = spacy.load(output_dir)
            doc2 = nlp2(test_text)
            logger.info('The tested text is {}, the prediction is {}'.format(test_text, doc2.cats))
            print(test_text, doc2.cats)
