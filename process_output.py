import os
import logging
import argparse

from collections import defaultdict, Counter

def load_heldout(filename):
    """
    Load the heldout dataset and extract the golden standard outputs.

    Args:
        filename (str): heldout tsv filename

    Returns:
        list of gold standard outputs, lines split by tabs
    """
    return [line.split("\t")[1].split() for line in open(filename).readlines()]


def load_outputs(filename):
    """
    Load the evaluation output trace and create counter objects per smaple
    per iteration with the probabilities assigned to it by the Softmax function.

    Args:
        filename (str): heldout tsv filename

    Returns:
        list of probabilities per sample per round
    """
    lines = [line.split("\t") for line in open(filename).readlines()]
    for i, sample in enumerate(lines):
        new_sample = []
        for iteration in sample:
            line = [pair.split("-") for pair in iteration.split()]
            new_iteration = Counter()
            for token, prob in line:
                new_iteration[token] = prob
            new_sample.append(new_iteration)
        lines[i] = new_sample
    return lines


def calculate_certainty_measure(heldout, evaluation_output, only_correct):
    """
    Calculates the average probability mass assigned to the correct answer
    by the softmax function.

    Args:
        heldout (list of list of str): golden standard answers
        evaluation_output (list of list of Counters): probabilities per
                                                      sample per iteration
        only_correct (bool): whether to exclude the cases in which the answer
                                                      was incorrect

    Returns:
        float, the average probability for the right ansswer
    """
    a = 0
    n = 0
    for gold, probs in zip(heldout, evaluation_output):
        for answer, iteration_probs in list(zip(gold, probs))[1:]:
            is_correct = iteration_probs.most_common(1)[0][0] == answer
            if not only_correct or (only_correct and is_correct):
                a += float(iteration_probs[answer])
                print(float(iteration_probs[answer]))
                n += 1
        print("\n")
    a = a / n
    return a


parser = argparse.ArgumentParser()
parser.add_argument('--heldout', help='heldout data without attacks', required=True)
parser.add_argument('--output', help='output without attacks', required=True)

opt = parser.parse_args()
log_format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=log_format, level=getattr(logging, "INFO"))
logging.info(opt)

heldout = load_heldout(opt.heldout)
output = load_outputs(opt.output)

# Including all data
metric = calculate_certainty_measure(heldout, output, False)
print("All: {}".format(metric))

# # Only including the data for which the answer was correct
# metric = calculate_certainty_measure(heldout, output, True)
# print("Only the correct ones: {}".format(metric))

