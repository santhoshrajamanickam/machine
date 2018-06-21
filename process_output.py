import os
import copy
from pprint import pprint
import logging
import argparse
import numpy as np
from collections import defaultdict, Counter


def parser_scan_file(file):
    word_accuracy = Counter()
    sequence_accuracy = Counter()
    with open(file) as f:
        for line in f:
            if "Length" in line and "Accuracy" in line:
                line = line.split()
                line = line[line.index('Length'):]
                length = int(line[1][:-1])
                word_accuracy[length] = float(line[4])
                sequence_accuracy[length] = float(line[7])
    return word_accuracy, sequence_accuracy


parser = argparse.ArgumentParser()
parser.add_argument('--folder', default=".")
parser.add_argument('--prefix', help='Prefix for the files to average numbers from.', required=True)

opt = parser.parse_args()
log_format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=log_format, level=getattr(logging, "INFO"))
logging.info(opt)

files = os.listdir(os.path.join(opt.folder))

word_accuracies = defaultdict(list)
sequence_accuracies = defaultdict(list)

for file in files:
    if opt.prefix in file:
        word_accuracy, sequence_accuracy = parser_scan_file(file)
        for l in word_accuracy:
            word_accuracies[l].append(word_accuracy[l])
            sequence_accuracies[l].append(sequence_accuracy[l])

# Word Accuracies ##############################################################
# Calculate the average word accuracy and sequence accuracy

minimum = Counter()
maximum = Counter()
average = Counter()

for key in word_accuracies:
    all_values = np.array(word_accuracies[key])
    average[key] = np.mean(all_values)
    minimum[key] = min(all_values)
    maximum[key] = max(all_values)

print("Word Accuracies, Average")
pprint(average)

print("Word Accuracies, Maximum")
pprint(maximum)

print("Word Accuracies, Minimum")
pprint(minimum)

# Sequence Accuracies ##########################################################
# Calculate the average word accuracy and sequence accuracy

minimum = Counter()
maximum = Counter()
average = Counter()

for key in sequence_accuracies:
    all_values = np.array(sequence_accuracies[key])
    average[key] = np.mean(all_values)
    minimum[key] = min(all_values)
    maximum[key] = max(all_values)

print("Sequence Accuracies, Average")
pprint(average)

print("Sequence Accuracies, Maximum")
pprint(maximum)

print("Sequence Accuracies, Minimum")
pprint(minimum)
