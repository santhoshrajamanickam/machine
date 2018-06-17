#!/usr/bin/env bash

seq_length=5
start_level=1
end_level=4
num_models=5
train_folder=machine-tasks/LookupTables/lookup-3bit/samples/sample1

for level in $(seq $start_level $end_level); do
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
            python3 machine/evaluate.py --checkpoint_path ../machine-zoo/guided/gru/$model --test_data $folder/heldout_compositions5_no_eos.tsv --ignore_output_eos --batch_size 1 --output --attention pre-rnn --attention_method hard --output_dir attacks-evaluation-output/model$model/sample$counter/  --log_file attacks-evaluation-accuracies/model$model/sample$counter/heldout_compositions${seq_length}.out
            python3 machine/evaluate.py --checkpoint_path ../machine-zoo/guided/gru/$model --test_data $folder/heldout_tables5_no_eos.tsv --ignore_output_eos --batch_size 1 --output --attention pre-rnn --attention_method hard --output_dir attacks-evaluation-output/model$model/sample$counter/  --log_file attacks-evaluation-accuracies/model$model/sample$counter/heldout_tables${seq_length}.out
            python3 machine/evaluate.py --checkpoint_path ../machine-zoo/guided/gru/$model --test_data $folder/new_compositions5_no_eos.tsv --ignore_output_eos --batch_size 1 --output --attention pre-rnn --attention_method hard --output_dir attacks-evaluation-output/model$model/sample$counter/  --log_file attacks-evaluation-accuracies/model$model/sample$counter/new_compositions${seq_length}.out
          fi
          counter=$((counter + 1))
        done

        counter=1
        for folder in machine-tasks/LookupTables/lookup-3bit/longer_compositions/llonger_compositions/*/; do
          if ! [ $counter -eq 1 ]; then
            echo $counter
            python3 machine/evaluate.py --checkpoint_path ../machine-zoo/guided/gru/$model --test_data $folder/heldout_compositions5_attacks.tsv --ignore_output_eos --batch_size 1 --output --attention pre-rnn --attention_method hard --output_dir attacks-evaluation-output/model$model/sample$counter/  --log_file attacks-evaluation-accuracies/model$model/sample$counter/heldout_compositions${seq_length}_attacks.out
            python3 machine/evaluate.py --checkpoint_path ../machine-zoo/guided/gru/$model --test_data $folder/heldout_tables5_attacks.tsv --ignore_output_eos --batch_size 1 --output --attention pre-rnn --attention_method hard --output_dir attacks-evaluation-output/model$model/sample$counter/  --log_file attacks-evaluation-accuracies/model$model/sample$counter/heldout_tables${seq_length}_attacks.out
            python3 machine/evaluate.py --checkpoint_path ../machine-zoo/guided/gru/$model --test_data $folder/new_compositions5_attacks.tsv --ignore_output_eos --batch_size 1 --output --attention pre-rnn --attention_method hard --output_dir attacks-evaluation-output/model$model/sample$counter/  --log_file attacks-evaluation-accuracies/model$model/sample$counter/new_compositions${seq_length}_attacks.out
          fi
          counter=$((counter + 1))
        done

        counter=1
        for folder in machine-tasks/LookupTables/lookup-3bit/longer_compositions/llonger_compositions/*/; do
          if ! [ $counter -eq 1 ]; then
            echo $counter
            python3 machine/evaluate.py --checkpoint_path ../machine-zoo/guided/gru/$model --test_data $folder/heldout_compositions5_attacks_outputs.tsv --ignore_output_eos --batch_size 1 --output --attention pre-rnn --attention_method hard --output_dir attacks-evaluation-output/model$model/sample$counter/  --log_file attacks-evaluation-accuracies/model$model/sample$counter/heldout_compositions${seq_length}_attacks_outputs.out
            python3 machine/evaluate.py --checkpoint_path ../machine-zoo/guided/gru/$model --test_data $folder/heldout_tables5_attacks_outputs.tsv --ignore_output_eos --batch_size 1 --output --attention pre-rnn --attention_method hard --output_dir attacks-evaluation-output/model$model/sample$counter/  --log_file attacks-evaluation-accuracies/model$model/sample$counter/heldout_tables${seq_length}_attacks_outputs.out
            python3 machine/evaluate.py --checkpoint_path ../machine-zoo/guided/gru/$model --test_data $folder/new_compositions5_attacks_outputs.tsv --ignore_output_eos --batch_size 1 --output --attention pre-rnn --attention_method hard --output_dir attacks-evaluation-output/model$model/sample$counter/  --log_file attacks-evaluation-accuracies/model$model/sample$counter/new_compositions${seq_length}_attacks_outputs.out
          fi
          counter=$((counter + 1))
        done
        echo "Evaluated the models with the i-machine-think evaluator."
    done
done