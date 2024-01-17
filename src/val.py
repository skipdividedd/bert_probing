import torch
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import f1_score
from NeuroX.neurox.interpretation import metrics
from NeuroX.neurox.interpretation import utils

# code from https://github.com/fdalvi/NeuroX/blob/master/neurox/interpretation/linear_probe.py with small corrections to f1 computation

def _numpyfy(x):

    if isinstance(x, np.ndarray):
        return x
    return np.array(x)

def f1(preds, labels):
    preds = _numpyfy(preds)
    labels = _numpyfy(labels)
    return f1_score(y_true=labels, y_pred=preds, average='weighted')

def compute_score(preds, labels, metric):
    if metric == "f1":
        return f1(preds, labels)
    elif metric == "accuracy":
        return metrics.accuracy(preds, labels)

def evaluate_probe(
    probe,
    X,
    y,
    idx_to_class=None,
    return_predictions=False,
    source_tokens=None,
    batch_size=32,
    metric="f1",
    ):

    progressbar = utils.get_progress_bar()

    # Check if we can use GPU's for evaluation
    use_gpu = torch.cuda.is_available()

    if use_gpu:
        probe = probe.cuda()

    # always evaluate in full precision
    probe = probe.float()

    # Test the Model
    y_pred = []

    def source_generator():
        for s in source_tokens:
            for t in s:
                yield t

    src_words = source_generator()

    if return_predictions:
        predictions = []
        src_word = -1

    for inputs, labels in progressbar(
        utils.batch_generator(
            torch.from_numpy(X), torch.from_numpy(y), batch_size=batch_size
        ),
        desc="Evaluating",
    ):
        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()

        # always evaluate in full precision
        inputs = inputs.float()

        inputs = Variable(inputs)
        labels = Variable(labels)

        outputs = probe(inputs)

        if outputs.data.shape[1] == 1:
            # Regression
            predicted = outputs.data
        else:
            # Classification
            _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.cpu().numpy()

        for i in range(0, len(predicted)):
            idx = predicted[i]
            if idx_to_class:
                key = idx_to_class[idx]
            else:
                key = idx

            y_pred.append(predicted[i])

            if return_predictions:
                if source_tokens:
                    src_word = next(src_words)
                else:
                    src_word = src_word + 1
                predictions.append((src_word, key, idx_to_class[labels[i].item()]))

    y_pred = np.array(y_pred)

    result = compute_score(y_pred, y, metric)

    print("Score (%s) of the probe: %0.2f" % (metric, result))

    class_scores = {}
    class_scores["__OVERALL__"] = result

    if idx_to_class:
        for i in idx_to_class:
            class_name = idx_to_class[i]
            class_instances_idx = np.where(y == i)[0]
            y_pred_filtered = y_pred[class_instances_idx]
            y_filtered = y[class_instances_idx]
            total = y_filtered.shape
            if total == 0:
                class_scores[class_name] = 0
            else:
                class_scores[class_name] = compute_score(
                    y_pred_filtered, y_filtered, metric
                )

    if return_predictions:
        return class_scores, predictions
    return class_scores