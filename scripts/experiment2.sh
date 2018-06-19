#!/usr/bin/env bash

# To run the experiments without adapting the paths, make sure the following holds for you:
# 1. Go to the folder with machine and machine-tasks as subfolders
# 2. Put four things inside this folder:
#     - the bash file for the experiment, experiment1.sh
#     - create two additional folder called `attacks-evaluation-accuracies' and `attacks-evaluation-output'
#     - in each of these subfolders create folders `model1', `model2', `model3', `model4', `model5'
#     - in each of the model folder create subfolders `sample2', `sample3', `sample4', `sample5'
# 3. update the path to the machine-zoo
# 4. Indicate the model number at the start of the bash script
# 5. Update the level
# 6. now run experiment2.sh
# Don't be afraid because of all the prints in your terminal, at the end of the python script you will have nice numbers

seq_length=5
level=1
rnn_type=lstm
num_models=5
train_folder=machine-tasks/LookupTables/lookup-3bit/samples/sample1

counter=1
for folder in machine-tasks/LookupTables/lookup-3bit/longer_compositions/llonger_compositions/*/; do
  if ! [ $counter -eq 1 ]; then
      echo
      python3 machine-tasks/scripts/create_adversarial_dataset.py --level $level --train $train_folder/train.tsv --heldout $folder/heldout_compositions${seq_length}.tsv --ignore_output_eos --output_dir $folder
      python3 machine-tasks/scripts/create_adversarial_dataset.py --level $level --train $train_folder/train.tsv --heldout $folder/heldout_tables${seq_length}.tsv --ignore_output_eos --output_dir $folder
      python3 machine-tasks/scripts/create_adversarial_dataset.py --level $level --train $train_folder/train.tsv --heldout $folder/new_compositions${seq_length}.tsv --ignore_output_eos --output_dir $folder
  fi
  counter=$((counter + 1))
done

echo "Created adversarial datasets."

for model in $(seq 1 $num_models); do

    counter=1
    for folder in machine-tasks/LookupTables/lookup-3bit/longer_compositions/llonger_compositions/*/; do
      if ! [ $counter -eq 1 ]; then
        echo $counter
        rm attacks-evaluation-output/model$model/sample$counter/*
        rm attacks-evaluation-accuracies/model$model/sample$counter/*
        python3 machine/evaluate.py --checkpoint_path ../machine-zoo/guided/$rnn_type/$model --test_data $folder/heldout_compositions5_no_eos.tsv --ignore_output_eos --batch_size 1 --output --attention pre-rnn --attention_method hard --output_dir attacks-evaluation-output/model$model/sample$counter/  --log_file attacks-evaluation-accuracies/model$model/sample$counter/heldout_compositions${seq_length}.out
        python3 machine/evaluate.py --checkpoint_path ../machine-zoo/guided/$rnn_type/$model --test_data $folder/heldout_tables5_no_eos.tsv --ignore_output_eos --batch_size 1 --output --attention pre-rnn --attention_method hard --output_dir attacks-evaluation-output/model$model/sample$counter/  --log_file attacks-evaluation-accuracies/model$model/sample$counter/heldout_tables${seq_length}.out
        python3 machine/evaluate.py --checkpoint_path ../machine-zoo/guided/$rnn_type/$model --test_data $folder/new_compositions5_no_eos.tsv --ignore_output_eos --batch_size 1 --output --attention pre-rnn --attention_method hard --output_dir attacks-evaluation-output/model$model/sample$counter/  --log_file attacks-evaluation-accuracies/model$model/sample$counter/new_compositions${seq_length}.out
      fi
      counter=$((counter + 1))
    done

    counter=1
    for folder in machine-tasks/LookupTables/lookup-3bit/longer_compositions/llonger_compositions/*/; do
      if ! [ $counter -eq 1 ]; then
        echo $counter
        python3 machine/evaluate.py --checkpoint_path ../machine-zoo/guided/$rnn_type/$model --test_data $folder/heldout_compositions5_attacks.tsv --ignore_output_eos --batch_size 1 --output --attention pre-rnn --attention_method hard --output_dir attacks-evaluation-output/model$model/sample$counter/  --log_file attacks-evaluation-accuracies/model$model/sample$counter/heldout_compositions${seq_length}_attacks.out
        python3 machine/evaluate.py --checkpoint_path ../machine-zoo/guided/$rnn_type/$model --test_data $folder/heldout_tables5_attacks.tsv --ignore_output_eos --batch_size 1 --output --attention pre-rnn --attention_method hard --output_dir attacks-evaluation-output/model$model/sample$counter/  --log_file attacks-evaluation-accuracies/model$model/sample$counter/heldout_tables${seq_length}_attacks.out
        python3 machine/evaluate.py --checkpoint_path ../machine-zoo/guided/$rnn_type/$model --test_data $folder/new_compositions5_attacks.tsv --ignore_output_eos --batch_size 1 --output --attention pre-rnn --attention_method hard --output_dir attacks-evaluation-output/model$model/sample$counter/  --log_file attacks-evaluation-accuracies/model$model/sample$counter/new_compositions${seq_length}_attacks.out
      fi
      counter=$((counter + 1))
    done

    counter=1
    for folder in machine-tasks/LookupTables/lookup-3bit/longer_compositions/llonger_compositions/*/; do
      if ! [ $counter -eq 1 ]; then
        echo $counter
        python3 machine/evaluate.py --checkpoint_path ../machine-zoo/guided/$rnn_type/$model --test_data $folder/heldout_compositions5_attacks_outputs.tsv --ignore_output_eos --batch_size 1 --output --attention pre-rnn --attention_method hard --output_dir attacks-evaluation-output/model$model/sample$counter/  --log_file attacks-evaluation-accuracies/model$model/sample$counter/heldout_compositions${seq_length}_attacks_outputs.out
        python3 machine/evaluate.py --checkpoint_path ../machine-zoo/guided/$rnn_type/$model --test_data $folder/heldout_tables5_attacks_outputs.tsv --ignore_output_eos --batch_size 1 --output --attention pre-rnn --attention_method hard --output_dir attacks-evaluation-output/model$model/sample$counter/  --log_file attacks-evaluation-accuracies/model$model/sample$counter/heldout_tables${seq_length}_attacks_outputs.out
        python3 machine/evaluate.py --checkpoint_path ../machine-zoo/guided/$rnn_type/$model --test_data $folder/new_compositions5_attacks_outputs.tsv --ignore_output_eos --batch_size 1 --output --attention pre-rnn --attention_method hard --output_dir attacks-evaluation-output/model$model/sample$counter/  --log_file attacks-evaluation-accuracies/model$model/sample$counter/new_compositions${seq_length}_attacks_outputs.out
      fi
      counter=$((counter + 1))
    done
    echo "Evaluated the models with the i-machine-think evaluator."
done

echo "Print Results!!"
python3 machine/scripts/calculate_metrics_experiment2.py
