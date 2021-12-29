run_path="/home/alta/BLTSpeaking/exp-pr450/shell_scripts/preresnet/basic_al/run_train_basic_al.sh"
txt_base="/home/alta/BLTSpeaking/exp-pr450/logs/preresnet_al_baseline_100/al_baseline_log"
save_dir_base="/home/alta/BLTSpeaking/exp-pr450/logs/preresnet_al_baseline_100/round_history"

# run_path="/home/alta/BLTSpeaking/exp-pr450/shell_scripts/preresnet/basic_al/train_basic_al.sh"
# save_dir_base="/home/alta/BLTSpeaking/exp-pr450/logs/preresnet_al_baseline_100/round_history"
# $run_path 0.05 0.05 rand 2 $save_dir_base

# initProp
# roundProp
# acquisitionFunction
# roundEpochs
# saveDir

# Repeat 1
$run_path ${txt_base}_01.txt 0.05 0.05 rand 50 $save_dir_base
$run_path ${txt_base}_02.txt 0.05 0.10 rand 50 $save_dir_base
$run_path ${txt_base}_03.txt 0.05 0.05 maxent 50 $save_dir_base
$run_path ${txt_base}_04.txt 0.05 0.10 maxent 50 $save_dir_base
$run_path ${txt_base}_05.txt 0.05 0.05 margin 50 $save_dir_base
$run_path ${txt_base}_06.txt 0.05 0.10 margin 50 $save_dir_base
$run_path ${txt_base}_07.txt 0.05 0.05 kldiff 50 $save_dir_base
$run_path ${txt_base}_08.txt 0.05 0.10 kldiff 50 $save_dir_base
$run_path ${txt_base}_09.txt 0.05 0.05 lc 50 $save_dir_base
$run_path ${txt_base}_10.txt 0.05 0.10 lc 50 $save_dir_base

# Repeat 2
$run_path ${txt_base}_11.txt 0.05 0.05 rand 50 $save_dir_base
$run_path ${txt_base}_12.txt 0.05 0.10 rand 50 $save_dir_base
$run_path ${txt_base}_13.txt 0.05 0.05 maxent 50 $save_dir_base
$run_path ${txt_base}_14.txt 0.05 0.10 maxent 50 $save_dir_base
$run_path ${txt_base}_15.txt 0.05 0.05 margin 50 $save_dir_base
$run_path ${txt_base}_16.txt 0.05 0.10 margin 50 $save_dir_base
$run_path ${txt_base}_17.txt 0.05 0.05 kldiff 50 $save_dir_base
$run_path ${txt_base}_18.txt 0.05 0.10 kldiff 50 $save_dir_base
$run_path ${txt_base}_19.txt 0.05 0.05 lc 50 $save_dir_base
$run_path ${txt_base}_20.txt 0.05 0.10 lc 50 $save_dir_base

# Repeat 3
$run_path ${txt_base}_21.txt 0.05 0.05 rand 50 $save_dir_base
$run_path ${txt_base}_22.txt 0.05 0.10 rand 50 $save_dir_base
$run_path ${txt_base}_23.txt 0.05 0.05 maxent 50 $save_dir_base
$run_path ${txt_base}_24.txt 0.05 0.10 maxent 50 $save_dir_base
$run_path ${txt_base}_25.txt 0.05 0.05 margin 50 $save_dir_base
$run_path ${txt_base}_26.txt 0.05 0.10 margin 50 $save_dir_base
$run_path ${txt_base}_27.txt 0.05 0.05 kldiff 50 $save_dir_base
$run_path ${txt_base}_28.txt 0.05 0.10 kldiff 50 $save_dir_base
$run_path ${txt_base}_29.txt 0.05 0.05 lc 50 $save_dir_base
$run_path ${txt_base}_30.txt 0.05 0.10 lc 50 $save_dir_base



# for local submission

# train_path="/home/alta/BLTSpeaking/exp-pr450/shell_scripts/preresnet/basic_al/train_basic_al.sh"
# txt_base="/home/alta/BLTSpeaking/exp-pr450/logs/preresnet_al_baseline/"
# save_dir_base="/home/alta/BLTSpeaking/exp-pr450/logs/preresnet_al_baseline/round_history"
# export CUDA_VISIBLE_DEVICES=1 # change this!

# $train_path 0.05 0.05 rand 50 $save_dir_base > ${txt_base}_31.txt
# $train_path 0.05 0.10 rand 50 $save_dir_base > ${txt_base}_32.txt
# $train_path 0.05 0.05 maxent 50 $save_dir_base > ${txt_base}_33.txt
# $train_path 0.05 0.10 maxent 50 $save_dir_base > ${txt_base}_34.txt
# $train_path 0.05 0.05 margin 50 $save_dir_base > ${txt_base}_35.txt
# $train_path 0.05 0.10 margin 50 $save_dir_base > ${txt_base}_36.txt
# $train_path 0.05 0.05 kldiff 50 $save_dir_base > ${txt_base}_37.txt
# $train_path 0.05 0.10 kldiff 50 $save_dir_base > ${txt_base}_38.txt
# $train_path 0.05 0.05 lc 50 $save_dir_base > ${txt_base}_39.txt
# $train_path 0.05 0.10 lc 50 $save_dir_base > ${txt_base}_40.txt
