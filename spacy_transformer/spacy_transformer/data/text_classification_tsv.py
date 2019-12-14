import csv
import re
import unicodedata


class ClassificationDataReader:

    def preprocess_text(self, text):
        text = text.replace("<s>", "<open-s-tag>")
        text = text.replace("</s>", "<close-s-tag>")
        white_re = re.compile(r"\s\s+")
        text = white_re.sub(" ", text).strip()
        return "".join(c for c in unicodedata.normalize("NFD", text)
                       if unicodedata.category(c) != "Mn")

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

    def _prepare_partition(self, input_path, preprocessing=False):
        texts, labels = self.read_inputs(input_path)
        # transform 0, 1 to True, False
        categories = [{'POSITIVE': bool(y), 'NEGATIVE': not bool(y)} for y in labels]
        return texts, categories


if __name__ == '__main__':
    texts, cats = ClassificationDataReader()._prepare_partition('../../../allennlp_cls/example_train.tsv')
    print(cats)