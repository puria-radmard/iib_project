run_path="/home/alta/BLTSpeaking/exp-pr450/lent_shell_scripts/acquisition_distillation/run_cifar_logits_generation.sh"

txt_base="/home/alta/BLTSpeaking/exp-pr450/lent_logs/acquisition_distillation_10/log"
save_dir_base="/home/alta/BLTSpeaking/exp-pr450/lent_logs/acquisition_distillation_10/trained_logits"

# subset_size=$1
# subset_index_path=$2
# log_base=$3
# mirrored_config_json_path=$4

indices_folder_base='/home/alta/BLTSpeaking/exp-pr450/lent_logs/UNUSABLE_cifar10_densenet_al_baseline/round_history'

mirrored_config_json_path="${indices_folder_base}-4/config.json"

$run_path ${txt_base}_01.txt 7501  "${indices_folder_base}-2/round_3/labelled_set.txt"  $save_dir_base  $mirrored_config_json_path 
$run_path ${txt_base}_02.txt 15001 "${indices_folder_base}-2/round_6/labelled_set.txt"  $save_dir_base  $mirrored_config_json_path 
$run_path ${txt_base}_03.txt 20001 "${indices_folder_base}-2/round_8/labelled_set.txt"  $save_dir_base  $mirrored_config_json_path 
