#!/bin/bash

# To run the experiments without adapting the paths, make sure the following holds for you:
# 1. Go to the folder with machine and machine-tasks as subfolders
# 2. Put four things inside this folder:
#     - the bash file for the experiment, experiment1.sh
#     - the calculate_metrics_experiment1.py file
#     - create two additional folder called `attacks-evaluation-accuracies' and `attacks-evaluation-output'
#     - in each of these subfolders create folders `model1', `model2', `model3', `model4', `model5'
#     - in each of the model folder create subfolders `sample2', `sample3', `sample4', `sample5'
# 3. update the path to the machine-zoo
# 4. Indicate the model number at the start of the bash script
# 5. now run experiment1.sh
# 6. update the model number, run the script for models 1, 2, 3, 4, and 5
# 7. now run calculate_metrics_experiment1.py
# Don't be afraid because of all the prints in your terminal, at the end of the python script you will have nice numbers

model=5
level=1

counter=1
for folder in machine-tasks/LookupTables/lookup-3bit/samples/*/; do
  if ! [ $counter -eq 1 ]; then
      echo 
      python machine-tasks/scripts/create_adversarial_dataset.py --level $level --train $folder/train.tsv --heldout $folder/heldout_inputs.tsv --ignore_output_eos --output_dir $folder
      python machine-tasks/scripts/create_adversarial_dataset.py --level $level --train $folder/train.tsv --heldout $folder/heldout_compositions.tsv --ignore_output_eos --output_dir $folder
      python machine-tasks/scripts/create_adversarial_dataset.py --level $level --train $folder/train.tsv --heldout $folder/heldout_tables.tsv --ignore_output_eos --output_dir $folder
      python machine-tasks/scripts/create_adversarial_dataset.py --level $level --train $folder/train.tsv --heldout $folder/new_compositions.tsv --ignore_output_eos --output_dir $folder
  fi
  counter=$((counter + 1))
done

echo "Created adversarial datasets."

counter=1
for folder in machine-tasks/LookupTables/lookup-3bit/samples/*/; do
  if ! [ $counter -eq 1 ]; then
    echo $counter
    python machine/evaluate.py --checkpoint_path ../machine-zoo-master/guided/gru/$model --test_data $folder/heldout_inputs_no_eos.tsv --ignore_output_eos --batch_size 1 --output --attention pre-rnn --attention_method hard --output_dir attacks-evaluation-output/model$model/sample$counter/ --log_file attacks-evaluation-accuracies/model$model/sample$counter/heldout_inputs.out
    python machine/evaluate.py --checkpoint_path ../machine-zoo-master/guided/gru/$model --test_data $folder/heldout_compositions_no_eos.tsv --ignore_output_eos --batch_size 1 --output --attention pre-rnn --attention_method hard --output_dir attacks-evaluation-output/model$model/sample$counter/  --log_file attacks-evaluation-accuracies/model$model/sample$counter/heldout_compositions.out
    python machine/evaluate.py --checkpoint_path ../machine-zoo-master/guided/gru/$model --test_data $folder/heldout_tables_no_eos.tsv --ignore_output_eos --batch_size 1 --output --attention pre-rnn --attention_method hard --output_dir attacks-evaluation-output/model$model/sample$counter/  --log_file attacks-evaluation-accuracies/model$model/sample$counter/heldout_tables.out
    python machine/evaluate.py --checkpoint_path ../machine-zoo-master/guided/gru/$model --test_data $folder/new_compositions_no_eos.tsv --ignore_output_eos --batch_size 1 --output --attention pre-rnn --attention_method hard --output_dir attacks-evaluation-output/model$model/sample$counter/  --log_file attacks-evaluation-accuracies/model$model/sample$counter/new_compositions.out
  fi
  counter=$((counter + 1))
done

counter=1
for folder in machine-tasks/LookupTables/lookup-3bit/samples/*/; do
  if ! [ $counter -eq 1 ]; then
    echo $counter
    python machine/evaluate.py --checkpoint_path ../machine-zoo-master/guided/gru/$model --test_data $folder/heldout_inputs_attacks.tsv --ignore_output_eos --batch_size 1 --output --attention pre-rnn --attention_method hard --output_dir attacks-evaluation-output/model$model/sample$counter/ --log_file attacks-evaluation-accuracies/model$model/sample$counter/heldout_inputs_attacks.out
    python machine/evaluate.py --checkpoint_path ../machine-zoo-master/guided/gru/$model --test_data $folder/heldout_compositions_attacks.tsv --ignore_output_eos --batch_size 1 --output --attention pre-rnn --attention_method hard --output_dir attacks-evaluation-output/model$model/sample$counter/  --log_file attacks-evaluation-accuracies/model$model/sample$counter/heldout_compositions_attacks.out
    python machine/evaluate.py --checkpoint_path ../machine-zoo-master/guided/gru/$model --test_data $folder/heldout_tables_attacks.tsv --ignore_output_eos --batch_size 1 --output --attention pre-rnn --attention_method hard --output_dir attacks-evaluation-output/model$model/sample$counter/  --log_file attacks-evaluation-accuracies/model$model/sample$counter/heldout_tables_attacks.out
    python machine/evaluate.py --checkpoint_path ../machine-zoo-master/guided/gru/$model --test_data $folder/new_compositions_attacks.tsv --ignore_output_eos --batch_size 1 --output --attention pre-rnn --attention_method hard --output_dir attacks-evaluation-output/model$model/sample$counter/  --log_file attacks-evaluation-accuracies/model$model/sample$counter/new_compositions_attacks.out
  fi
  counter=$((counter + 1))
done

counter=1
for folder in machine-tasks/LookupTables/lookup-3bit/samples/*/; do
  if ! [ $counter -eq 1 ]; then
    echo $counter
    python machine/evaluate.py --checkpoint_path ../machine-zoo-master/guided/gru/$model --test_data $folder/heldout_inputs_attacks_outputs.tsv --ignore_output_eos --batch_size 1 --output --attention pre-rnn --attention_method hard --output_dir attacks-evaluation-output/model$model/sample$counter/ --log_file attacks-evaluation-accuracies/model$model/sample$counter/heldout_inputs_attacks_outputs.out
    python machine/evaluate.py --checkpoint_path ../machine-zoo-master/guided/gru/$model --test_data $folder/heldout_compositions_attacks_outputs.tsv --ignore_output_eos --batch_size 1 --output --attention pre-rnn --attention_method hard --output_dir attacks-evaluation-output/model$model/sample$counter/  --log_file attacks-evaluation-accuracies/model$model/sample$counter/heldout_compositions_attacks_outputs.out
    python machine/evaluate.py --checkpoint_path ../machine-zoo-master/guided/gru/$model --test_data $folder/heldout_tables_attacks_outputs.tsv --ignore_output_eos --batch_size 1 --output --attention pre-rnn --attention_method hard --output_dir attacks-evaluation-output/model$model/sample$counter/  --log_file attacks-evaluation-accuracies/model$model/sample$counter/heldout_tables_attacks_outputs.out
    python machine/evaluate.py --checkpoint_path ../machine-zoo-master/guided/gru/$model --test_data $folder/new_compositions_attacks_outputs.tsv --ignore_output_eos --batch_size 1 --output --attention pre-rnn --attention_method hard --output_dir attacks-evaluation-output/model$model/sample$counter/  --log_file attacks-evaluation-accuracies/model$model/sample$counter/new_compositions_attacks_outputs.out
  fi
  counter=$((counter + 1))
done

echo "Evaluated the models with the i-machine-think evaluator."

