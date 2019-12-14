import tqdm


class Evaluate:

    def f1_evaluate(self, nlp, texts, cats, pos_label):
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