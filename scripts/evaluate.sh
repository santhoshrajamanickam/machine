#!/usr/bin/env bash


LENGTH=10
LEVEL=level_3

OUTPUT_DIR=samples/attacks/longer_compositions/$LENGTH/$LEVEL/

TEST_PATH1_1=samples/attacks/longer_compositions/$LENGTH/$LEVEL/heldout_compositions${LENGTH}_no_eos.tsv
TEST_PATH1_2=samples/attacks/longer_compositions/$LENGTH/$LEVEL/heldout_compositions${LENGTH}_attacks.tsv
TEST_PATH1_3=samples/attacks/longer_compositions/$LENGTH/$LEVEL/heldout_compositions${LENGTH}_attacks_outputs.tsv
TEST_PATH1_4=samples/attacks/longer_compositions/$LENGTH/$LEVEL/heldout_compositions${LENGTH}_no_eos_output.tsv
TEST_PATH1_5=samples/attacks/longer_compositions/$LENGTH/$LEVEL/heldout_compositions${LENGTH}_attacks_output.tsv
TEST_PATH1_6=samples/attacks/longer_compositions/$LENGTH/$LEVEL/heldout_compositions${LENGTH}_attacks_outputs_output.tsv

TEST_PATH2_1=samples/attacks/longer_compositions/$LENGTH/$LEVEL/heldout_tables${LENGTH}_no_eos.tsv
TEST_PATH2_2=samples/attacks/longer_compositions/$LENGTH/$LEVEL/heldout_tables${LENGTH}_attacks.tsv
TEST_PATH2_3=samples/attacks/longer_compositions/$LENGTH/$LEVEL/heldout_tables${LENGTH}_attacks_outputs.tsv
TEST_PATH2_4=samples/attacks/longer_compositions/$LENGTH/$LEVEL/heldout_tables${LENGTH}_no_eos_output.tsv
TEST_PATH2_5=samples/attacks/longer_compositions/$LENGTH/$LEVEL/heldout_tables${LENGTH}_attacks_output.tsv
TEST_PATH2_6=samples/attacks/longer_compositions/$LENGTH/$LEVEL/heldout_tables${LENGTH}_attacks_outputs_output.tsv

TEST_PATH3_1=samples/attacks/longer_compositions/$LENGTH/$LEVEL/new_compositions${LENGTH}_no_eos.tsv
TEST_PATH3_2=samples/attacks/longer_compositions/$LENGTH/$LEVEL/new_compositions${LENGTH}_attacks.tsv
TEST_PATH3_3=samples/attacks/longer_compositions/$LENGTH/$LEVEL/new_compositions${LENGTH}_attacks_outputs.tsv
TEST_PATH3_4=samples/attacks/longer_compositions/$LENGTH/$LEVEL/new_compositions${LENGTH}_no_eos_output.tsv
TEST_PATH3_5=samples/attacks/longer_compositions/$LENGTH/$LEVEL/new_compositions${LENGTH}_attacks_output.tsv
TEST_PATH3_6=samples/attacks/longer_compositions/$LENGTH/$LEVEL/new_compositions${LENGTH}_attacks_outputs_output.tsv


echo "\n\nEvaluate model on test data"
python3 evaluate.py --checkpoint_path './pretrained/best_model/' --test_data $TEST_PATH1_3 --attention 'pre-rnn' --attention_method 'hard' --ignore_output_eos --output --output_dir $OUTPUT_DIR --batch_size 1
python3 evaluate.py --checkpoint_path './pretrained/best_model/' --test_data $TEST_PATH2_3 --attention 'pre-rnn' --attention_method 'hard' --ignore_output_eos --output --output_dir $OUTPUT_DIR --batch_size 1
python3 evaluate.py --checkpoint_path './pretrained/best_model/' --test_data $TEST_PATH3_3 --attention 'pre-rnn' --attention_method 'hard' --ignore_output_eos --output --output_dir $OUTPUT_DIR --batch_size 1

#python3 evaluate.py --checkpoint_path './pretrained/best_model/' --test_data $TEST_PATH1_1 --ignore_output_eos --output --output_dir $OUTPUT_DIR --batch_size 1
#python3 evaluate.py --checkpoint_path './pretrained/best_model/' --test_data $TEST_PATH2_1 --ignore_output_eos --output --output_dir $OUTPUT_DIR --batch_size 1
#python3 evaluate.py --checkpoint_path './pretrained/best_model/' --test_data $TEST_PATH3_1 --ignore_output_eos --output --output_dir $OUTPUT_DIR --batch_size 1

python3 process_output.py --heldout $TEST_PATH1_3 --output $TEST_PATH1_6
python3 process_output.py --heldout $TEST_PATH2_3 --output $TEST_PATH2_6
python3 process_output.py --heldout $TEST_PATH3_3 --output $TEST_PATH3_6
