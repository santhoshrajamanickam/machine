import sys
import subprocess
import os
import copy
import numpy as np

from pprint import pprint
from collections import defaultdict, Counter


def extract_numbers(foldername, subfolder, filename, model, sample):
    subfolder = os.path.join(foldername, subfolder)
    command = "python machine/process_output.py --heldout {}/".format(subfolder)+\
              "{}.tsv --output attacks-evaluation-".format(filename)+\
              "output/model{}/sample{}/{}_output.tsv".format(model, sample+1, filename)
    output = subprocess.check_output(command, shell=True)

    return float(output.split()[1])


model = 1
probabilities = defaultdict(lambda: defaultdict(list))

# Names of all the data tsv files
heldout_inputs = ["heldout_inputs_no_eos", "heldout_inputs_attacks", "heldout_inputs_attacks_outputs"]
heldout_compositions = ["heldout_compositions_no_eos", "heldout_compositions_attacks", "heldout_compositions_attacks_outputs"]
heldout_tables = ["heldout_tables_no_eos", "heldout_tables_attacks", "heldout_tables_attacks_outputs"]
new_compositions = ["new_compositions_no_eos", "new_compositions_attacks", "new_compositions_attacks_outputs"]

# For every model
for j in range(1, 6):
    model = j

    foldername = "machine-tasks/LookupTables/lookup-3bit/samples/"
    folders = os.listdir(foldername)

    # For every sample folder in the data    
    for i, subfolder in enumerate(folders):
        # Skip sample1, the model was trained on this
        if i == 0:
            continue

        # Gather all the probability mass averages using the process_output script
        for filename in heldout_inputs:
            a = extract_numbers(foldername, subfolder, filename, model, i)
            probabilities["heldout_inputs"][filename].append(a)
        for filename in heldout_compositions:
            a = extract_numbers(foldername, subfolder, filename, model, i)
            probabilities["heldout_compositions"][filename].append(a)
        for filename in heldout_tables:
            a = extract_numbers(foldername, subfolder, filename, model, i)
            probabilities["heldout_tables"][filename].append(a)
        for filename in new_compositions:
            a = extract_numbers(foldername, subfolder, filename, model, i)
            probabilities["new_compositions"][filename].append(a)

# Calculate the average, minimum, maximum prob and the variance
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


pprint(probabilities)
pprint(maximum)
pprint(minimum)
pprint(variance)