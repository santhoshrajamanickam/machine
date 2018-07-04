import sys
import subprocess
import os
import copy
import numpy as np
import argparse

from pprint import pprint
from collections import defaultdict, Counter


def extract_probabilities(foldername, subfolder, filename, model, sample):
    subfolder = os.path.join(foldername, subfolder)
    command = "python3 machine2/process_output.py --heldout {}/".format(subfolder)+\
              "{}.tsv --output attacks-evaluation-".format(filename)+\
              "output/model{}/sample{}/{}_output.tsv".format(model, sample+1, filename)
    output = subprocess.check_output(command, shell=True)

    return float(output.split()[1])


def extract_accuracies(foldername, subfolder, filename, model, sample):
    subfolder = os.path.join(foldername, subfolder)
    filename = filename.replace("_no_eos", "")
    with open("attacks-evaluation-accuracies/model{}/sample{}/{}.out".format(model, sample+1, filename), 'r') as f:
        log_file = f.read().split()

    word_accuracy, sequence_accuracy = None, None
    for i, word in enumerate(log_file):
        if word == "Word" and log_file[i+1] == "Accuracy":
            word_accuracy = log_file[i+2]
        if word == "Sequence" and log_file[i+1] == "Accuracy":
            sequence_accuracy = log_file[i+2]
            break

    return float(word_accuracy), float(sequence_accuracy)

parser = argparse.ArgumentParser()
parser.add_argument('--level', default=1, help='number of swappings', type=int)
parser.add_argument('--seq_length', default=1, help='sequence length', type=int)
parser.add_argument('--num_models', default=1, help='number of models', type=int)
opt = parser.parse_args()
level = opt.level
seq_length = str(opt.seq_length)
num_models = opt.num_models

model = 1
probabilities = defaultdict(lambda: defaultdict(list))
word_accuracies = defaultdict(lambda: defaultdict(list))
sequence_accuracies = defaultdict(lambda: defaultdict(list))

# Names of all the data tsv files
heldout_compositions = ["heldout_compositions"+seq_length+"_no_eos", "heldout_compositions"+seq_length+"_attacks", "heldout_compositions"+seq_length+"_attacks_outputs"]
heldout_tables = ["heldout_tables"+seq_length+"_no_eos", "heldout_tables"+seq_length+"_attacks", "heldout_tables"+seq_length+"_attacks_outputs"]
new_compositions = ["new_compositions"+seq_length+"_no_eos", "new_compositions"+seq_length+"_attacks", "new_compositions"+seq_length+"_attacks_outputs"]

# For every model
for j in range(1, num_models+1):
    model = j

    foldername = "machine-tasks/LookupTables/lookup-3bit/longer_compositions/llonger_compositions/"

    folders = os.listdir(foldername)
    folders.sort()

    # For every sample folder in the data
    for i, subfolder in enumerate(folders):
        # Skip sample1, the model was trained on this
        if subfolder == 'sample1' or subfolder == '.DS_Store' :
            # print(subfolder)
            continue

        # print(subfolder)
        # print(i)
        i-=1

        # Gather all the probability mass averages using the process_output script
        for filename in heldout_compositions:
            prob = extract_probabilities(foldername, subfolder, filename, model, i)
            probabilities["heldout_compositions"][filename].append(prob)
            word_acc, seq_acc = extract_accuracies(foldername, subfolder, filename, model, i)
            word_accuracies["heldout_compositions"][filename].append(word_acc)
            sequence_accuracies["heldout_compositions"][filename].append(seq_acc)
        for filename in heldout_tables:
            prob = extract_probabilities(foldername, subfolder, filename, model, i)
            probabilities["heldout_tables"][filename].append(prob)
            word_acc, seq_acc = extract_accuracies(foldername, subfolder, filename, model, i)
            word_accuracies["heldout_tables"][filename].append(word_acc)
            sequence_accuracies["heldout_tables"][filename].append(seq_acc)
        for filename in new_compositions:
            prob = extract_probabilities(foldername, subfolder, filename, model, i)
            probabilities["new_compositions"][filename].append(prob)
            word_acc, seq_acc = extract_accuracies(foldername, subfolder, filename, model, i)
            word_accuracies["new_compositions"][filename].append(word_acc)
            sequence_accuracies["new_compositions"][filename].append(seq_acc)


# Word Accuracies ##############################################################
# Calculate the average word accuracy and sequence accuracy

minimum = copy.deepcopy(word_accuracies)
maximum = copy.deepcopy(word_accuracies)
variance = copy.deepcopy(word_accuracies)

for key in word_accuracies:
    for subkey in word_accuracies[key]:
        all_values = np.array(word_accuracies[key][subkey])
        word_accuracies[key][subkey] = np.mean(all_values)
        minimum[key][subkey] = min(all_values)
        maximum[key][subkey] = max(all_values)
        variance[key][subkey] = np.var(all_values)

word_acc_file = './word_accuracies.txt'

with open(word_acc_file, 'a') as f:
    f.write("Sequence Length" + seq_length +"\n")
    f.write("Level " + str(level)+"\n")
    f.write("Word Accuracies, Average"+"\n")
    f.write(str(word_accuracies)+"\n")
    f.write("Word Accuracies, Maximum"+"\n")
    f.write(str(maximum)+"\n")
    f.write("Word Accuracies, Minimum"+"\n")
    f.write(str(minimum)+"\n")

# print("Word Accuracies, Average")
# pprint(word_accuracies)

# print("Word Accuracies, Maximum")
# pprint(maximum)
#
# print("Word Accuracies, Minimum")
# pprint(minimum)

# print("Word Accuracies, Variance")
# pprint(variance)

# Sequence Accuracies ##########################################################
# Calculate the average word accuracy and sequence accuracy

minimum = copy.deepcopy(sequence_accuracies)
maximum = copy.deepcopy(sequence_accuracies)
variance = copy.deepcopy(sequence_accuracies)

for key in sequence_accuracies:
    for subkey in sequence_accuracies[key]:
        all_values = np.array(sequence_accuracies[key][subkey])
        sequence_accuracies[key][subkey] = np.mean(all_values)
        minimum[key][subkey] = min(all_values)
        maximum[key][subkey] = max(all_values)
        variance[key][subkey] = np.var(all_values)

seq_acc_file = './seq_accuracies.txt'

with open(seq_acc_file, 'a') as f:
    f.write("Sequence Length" + seq_length + "\n")
    f.write("Level " + str(level)+"\n")
    f.write("Sequence Accuracies, Average"+"\n")
    f.write(str(sequence_accuracies)+"\n")
    f.write("Sequence Accuracies, Maximum"+"\n")
    f.write(str(maximum)+"\n")
    f.write("Sequence Accuracies, Minimum"+"\n")
    f.write(str(minimum)+"\n")

# print("Sequence Accuracies, Average")
# pprint(sequence_accuracies)
#
# print("Sequence Accuracies, Maximum")
# pprint(maximum)
#
# print("Sequence Accuracies, Minimum")
# pprint(minimum)
#
# print("Sequence Accuracies, Variance")
# pprint(variance)

# Probabilities ################################################################
# Calculate the average probabilities, minimum, maximum prob and the variance
minimum = copy.deepcopy(probabilities)
maximum = copy.deepcopy(probabilities)
variance = copy.deepcopy(probabilities)

for key in probabilities:
    for subkey in probabilities[key]:
        all_values = np.array(probabilities[key][subkey])
        probabilities[key][subkey] = np.mean(all_values)
        minimum[key][subkey] = min(all_values)
        maximum[key][subkey] = max(all_values)
        variance[key][subkey] = np.var(all_values)

prob_file = './probabilities.txt'

with open(prob_file, 'a') as f:
    f.write("Sequence Length" + seq_length + "\n")
    f.write("Level " + str(level)+"\n")
    f.write("Probabilities, Average"+"\n")
    f.write(str(probabilities)+"\n")
    f.write("Probabilities, Maximum"+"\n")
    f.write(str(maximum)+"\n")
    f.write("Probabilities, Minimum"+"\n")
    f.write(str(minimum)+"\n")

# print("Probabilities, Average")
# pprint(probabilities)
#
# print("Probabilities, Maximum")
# pprint(maximum)
#
# print("Probabilities, Minimum")
# pprint(minimum)
#
# print("Probabilities, Variance")
# pprint(variance)