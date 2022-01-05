run_path="/home/alta/BLTSpeaking/exp-pr450/lent_shell_scripts/cifar_labelled_classification_recalibration/run_train.sh"

dataset=$1

if [ "$dataset" = "cifar10" ]; then 
    txt_base="/home/alta/BLTSpeaking/exp-pr450/lent_logs/cifar10_labelled_classification_recalibration/log"
    save_dir_base="/home/alta/BLTSpeaking/exp-pr450/lent_logs/cifar10_labelled_classification_recalibration/config"
elif [ "$dataset" = "cifar100" ]; then 
    txt_base="/home/alta/BLTSpeaking/exp-pr450/lent_logs/cifar100_labelled_classification_recalibration/log"
    save_dir_base="/home/alta/BLTSpeaking/exp-pr450/lent_logs/cifar100_labelled_classification_recalibration/config"
elif [ "$dataset" = "test" ]; then 
    echo "Implement test"
    exit 1
else
    echo "Bad dataset"
    exit 1    
fi;

# dataset=$1
# architecture_name=$2
    # default_simple_cifar_convolutional_classifier
    # default_mini_resnet_classifier
    # default_mini_densenet_classifier
# minibatch_prop=$3
# initial_lr=$4
# num_initial_epochs=$5
# save_dir=$6



$run_path ${txt_base}_01.txt default_simple_cifar_convolutional_classifier  0.01 0.001 30 $save_dir_base
$run_path ${txt_base}_02.txt default_mini_resnet_classifier                 0.01 0.001 30 $save_dir_base
$run_path ${txt_base}_03.txt default_mini_densenet_classifier               0.01 0.001 30 $save_dir_base
$run_path ${txt_base}_04.txt default_simple_cifar_convolutional_classifier  0.01 0.001 30 $save_dir_base
$run_path ${txt_base}_05.txt default_mini_resnet_classifier                 0.01 0.001 30 $save_dir_base
$run_path ${txt_base}_06.txt default_mini_densenet_classifier               0.01 0.001 30 $save_dir_base
$run_path ${txt_base}_07.txt default_simple_cifar_convolutional_classifier  0.01 0.001 30 $save_dir_base
$run_path ${txt_base}_08.txt default_mini_resnet_classifier                 0.01 0.001 30 $save_dir_base
$run_path ${txt_base}_09.txt default_mini_densenet_classifier               0.01 0.001 30 $save_dir_base

