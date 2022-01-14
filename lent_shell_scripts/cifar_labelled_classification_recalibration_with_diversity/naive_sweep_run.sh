run_path="/home/alta/BLTSpeaking/exp-pr450/lent_shell_scripts/cifar_labelled_classification_recalibration_with_diversity/run_train.sh"

dataset=$1

if [ "$dataset" = "cifar10" ]; then 
    txt_base="/home/alta/BLTSpeaking/exp-pr450/lent_logs/cifar10_labelled_classification_recalibration_with_diversity/naive_log"
    save_dir_base="/home/alta/BLTSpeaking/exp-pr450/lent_logs/cifar10_labelled_classification_recalibration_with_diversity/naive_round_history"
elif [ "$dataset" = "cifar100" ]; then 
    txt_base="/home/alta/BLTSpeaking/exp-pr450/lent_logs/cifar100_labelled_classification_recalibration_with_diversity/naive_log"
    save_dir_base="/home/alta/BLTSpeaking/exp-pr450/lent_logs/cifar100_labelled_classification_recalibration_with_diversity/naive_round_history"
elif [ "$dataset" = "test" ]; then 
    run_path="/home/alta/BLTSpeaking/exp-pr450/lent_shell_scripts/cifar_labelled_classification_recalibration_with_diversity/train.sh"
    txt_base="/home/alta/BLTSpeaking/exp-pr450/lent_logs/cifar100_labelled_classification_recalibration_with_diversity/test_naive_log"
    save_dir_base="/home/alta/BLTSpeaking/exp-pr450/lent_logs/cifar100_labelled_classification_recalibration_with_diversity/test_naive_round_history"
    $run_path cifar100 default_simple_cifar_convolutional_classifier 0.01 0.001 1 $save_dir_base
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



$run_path ${txt_base}_01.txt $dataset default_simple_cifar_convolutional_classifier  0.02 0.001 30 $save_dir_base
$run_path ${txt_base}_02.txt $dataset default_simple_cifar_convolutional_classifier  0.02 0.0001 30 $save_dir_base
$run_path ${txt_base}_03.txt $dataset default_simple_cifar_convolutional_classifier  0.02 0.00001 30 $save_dir_base

