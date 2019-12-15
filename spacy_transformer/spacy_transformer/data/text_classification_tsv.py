import csv
import re
import unicodedata
import thinc
import random


class ClassificationDataReader:

    def preprocess_text(self, text):
        text = text.replace("<s>", "<open-s-tag>")
        text = text.replace("</s>", "<close-s-tag>")
        white_re = re.compile(r"\s\s+")
        text = white_re.sub(" ", text).strip()
        return "".join(c for c in unicodedata.normalize("NFD", text)
                       if unicodedata.category(c) != "Mn")

    def load_data(self, *, limit=1000, dev_size=200):
        """Load data from the IMDB dataset for test, splitting off a held-out set."""
        if limit != 0:
            limit += dev_size
        assert dev_size != 0
        train_data, _ = thinc.extra.datasets.imdb(limit=limit)
        assert len(train_data) > dev_size
        random.shuffle(train_data)
        dev_data = train_data[:dev_size]
        train_data = train_data[dev_size:]
        train_texts, train_labels = self._prepare_partition_imdb(train_data, preprocess=False)
        dev_texts, dev_labels = self._prepare_partition_imdb(dev_data, preprocess=False)
        return (train_texts, train_labels), (dev_texts, dev_labels)

    def read_inputs(self, input_path):
        # this function can be modified to a generator
        texts = []
        cats = []
        with open(input_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for line in reader:
                if not line:
                    continue
                text, gold = line
                text = self.preprocess_text(text)
                texts.append(text)
                if len(gold) == 1:
                    cats.append(int(gold))  # only 0 or 1
                else:
                    cats.append(gold)
        return texts, cats

    def _prepare_partition_imdb(self, text_label_tuples, *, preprocess=False):
        texts, labels = zip(*text_label_tuples)
        cats = [{"POSITIVE": bool(y), "NEGATIVE": not bool(y)} for y in labels]
        return texts, cats

    def _prepare_partition(self, input_path, preprocessing=False):
        texts, labels = self.read_inputs(input_path)
        # transform 0, 1 to True, False
        categories = [{'POSITIVE': bool(y), 'NEGATIVE': not bool(y)} for y in labels]
        return texts, categories


if __name__ == '__main__':
    texts, cats = ClassificationDataReader()._prepare_partition('../../../allennlp_cls/example_train.tsv')
    print(cats)