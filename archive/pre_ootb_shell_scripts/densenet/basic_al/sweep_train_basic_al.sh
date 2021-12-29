run_path="/home/alta/BLTSpeaking/exp-pr450/shell_scripts/densenet/basic_al/run_train_basic_al.sh"

dataset=$1

if [ "$dataset" = "cifar10" ]; then 
    txt_base="/home/alta/BLTSpeaking/exp-pr450/logs/densenet_al_baseline/al_baseline_log"
    save_dir_base="/home/alta/BLTSpeaking/exp-pr450/logs/densenet_al_baseline/round_history"
elif [ "$dataset" = "cifar100" ]; then 
    txt_base="/home/alta/BLTSpeaking/exp-pr450/logs/densenet_al_baseline_100/al_baseline_log"
    save_dir_base="/home/alta/BLTSpeaking/exp-pr450/logs/densenet_al_baseline_100/round_history"
else
    echo "Bad dataset"
    exit 1
fi;

# initProp
# roundProp
# acquisitionFunction
# roundEpochs
# saveDir

# Repeat 1
# $run_path ${txt_base}_01.txt 0.05 0.05 rand 100 $save_dir_base $dataset
# $run_path ${txt_base}_02.txt 0.05 0.10 rand 100 $save_dir_base $dataset
# $run_path ${txt_base}_03.txt 0.05 0.05 maxent 100 $save_dir_base $dataset
# $run_path ${txt_base}_04.txt 0.05 0.10 maxent 100 $save_dir_base $dataset
# $run_path ${txt_base}_05.txt 0.05 0.05 margin 100 $save_dir_base $dataset
# $run_path ${txt_base}_06.txt 0.05 0.10 margin 100 $save_dir_base $dataset
# $run_path ${txt_base}_07.txt 0.05 0.05 kldiff 100 $save_dir_base $dataset
# $run_path ${txt_base}_08.txt 0.05 0.10 kldiff 100 $save_dir_base $dataset
# $run_path ${txt_base}_09.txt 0.05 0.05 lc 100 $save_dir_base $dataset
# $run_path ${txt_base}_10.txt 0.05 0.10 lc 100 $save_dir_base $dataset

# Repeat 2
# $run_path ${txt_base}_11.txt 0.05 0.05 rand 100 $save_dir_base $dataset
# $run_path ${txt_base}_12.txt 0.05 0.10 rand 100 $save_dir_base $dataset
# $run_path ${txt_base}_13.txt 0.05 0.05 maxent 100 $save_dir_base $dataset
# $run_path ${txt_base}_14.txt 0.05 0.10 maxent 100 $save_dir_base $dataset
# $run_path ${txt_base}_15.txt 0.05 0.05 margin 100 $save_dir_base $dataset
# $run_path ${txt_base}_16.txt 0.05 0.10 margin 100 $save_dir_base $dataset
# $run_path ${txt_base}_17.txt 0.05 0.05 kldiff 100 $save_dir_base $dataset
# $run_path ${txt_base}_18.txt 0.05 0.10 kldiff 100 $save_dir_base $dataset
# $run_path ${txt_base}_19.txt 0.05 0.05 lc 100 $save_dir_base $dataset
# $run_path ${txt_base}_20.txt 0.05 0.10 lc 100 $save_dir_base $dataset

# Repeat 3
# $run_path ${txt_base}_21.txt 0.05 0.05 rand 100 $save_dir_base $dataset
# $run_path ${txt_base}_22.txt 0.05 0.10 rand 100 $save_dir_base $dataset
# $run_path ${txt_base}_23.txt 0.05 0.05 maxent 100 $save_dir_base $dataset
# $run_path ${txt_base}_24.txt 0.05 0.10 maxent 100 $save_dir_base $dataset
# $run_path ${txt_base}_25.txt 0.05 0.05 margin 100 $save_dir_base $dataset
# $run_path ${txt_base}_26.txt 0.05 0.10 margin 100 $save_dir_base $dataset
# $run_path ${txt_base}_27.txt 0.05 0.05 kldiff 100 $save_dir_base $dataset
# $run_path ${txt_base}_28.txt 0.05 0.10 kldiff 100 $save_dir_base $dataset
# $run_path ${txt_base}_29.txt 0.05 0.05 lc 100 $save_dir_base $dataset
# $run_path ${txt_base}_30.txt 0.05 0.10 lc 100 $save_dir_base $dataset

# 0.2 added each time, all repeats
# $run_path ${txt_base}_31.txt 0.05 0.20 rand 100 $save_dir_base $dataset
# $run_path ${txt_base}_32.txt 0.05 0.20 maxent 100 $save_dir_base $dataset
# $run_path ${txt_base}_33.txt 0.05 0.20 margin 100 $save_dir_base $dataset
# $run_path ${txt_base}_34.txt 0.05 0.20 kldiff 100 $save_dir_base $dataset
# $run_path ${txt_base}_35.txt 0.05 0.20 lc 100 $save_dir_base $dataset
# $run_path ${txt_base}_36.txt 0.05 0.20 rand 100 $save_dir_base $dataset
# $run_path ${txt_base}_37.txt 0.05 0.20 maxent 100 $save_dir_base $dataset
# $run_path ${txt_base}_38.txt 0.05 0.20 margin 100 $save_dir_base $dataset
# $run_path ${txt_base}_39.txt 0.05 0.20 kldiff 100 $save_dir_base $dataset
# $run_path ${txt_base}_40.txt 0.05 0.20 lc 100 $save_dir_base $dataset
# $run_path ${txt_base}_41.txt 0.05 0.20 rand 100 $save_dir_base $dataset
# $run_path ${txt_base}_42.txt 0.05 0.20 maxent 100 $save_dir_base $dataset
# $run_path ${txt_base}_43.txt 0.05 0.20 margin 100 $save_dir_base $dataset
# $run_path ${txt_base}_44.txt 0.05 0.20 kldiff 100 $save_dir_base $dataset
# $run_path ${txt_base}_45.txt 0.05 0.20 lc 100 $save_dir_base $dataset


# BALD 

# $run_path ${txt_base}_46.txt 0.05 0.05 bald 100 $save_dir_base $dataset
# $run_path ${txt_base}_47.txt 0.05 0.10 bald 100 $save_dir_base $dataset
# $run_path ${txt_base}_48.txt 0.05 0.20 bald 100 $save_dir_base $dataset
# $run_path ${txt_base}_49.txt 0.05 0.05 bald 100 $save_dir_base $dataset
# $run_path ${txt_base}_50.txt 0.05 0.10 bald 100 $save_dir_base $dataset
# $run_path ${txt_base}_51.txt 0.05 0.20 bald 100 $save_dir_base $dataset
# $run_path ${txt_base}_52.txt 0.05 0.05 bald 100 $save_dir_base $dataset
# $run_path ${txt_base}_53.txt 0.05 0.10 bald 100 $save_dir_base $dataset
# $run_path ${txt_base}_54.txt 0.05 0.20 bald 100 $save_dir_base $dataset
