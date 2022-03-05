from itertools import groupby
from operator import itemgetter
from torch import nn
import torch, argparse, os

from classes_losses import multitask_distillation
from classes_losses.multitask_distillation import MultiTaskSymmetricTemperatureKLDistillationLoss, SymmetricTemperatureKLDistillationLoss, no_lambda_scheduling
from classes_losses.acquisition_prediction_losses import *
from active_learning.acquisition.logit_acquisition import *

from classes_utils.base.model import MultitaskHead
from config.ootb.convolutional_classification import *
from cifar_repo.cifar import transform_train
from training_scripts.cifar_daf_scripts import train_daf_acquisition_regression

from classes_utils.cifar.data import CIFAR10, CIFAR100
from util_functions.base import config_savedir


parser = argparse.ArgumentParser()

# Losses that can be used for multitask, and by themselves
viable_secondary_losses = [
    'entropy_log_mse', 'entropy_mse', 'entropy_bce', 'lc_bce', 
    'lc_parametric_jacobian_bce', 'lc_histogram_jacobian_bce',
    'entropy_parametric_jacobian_bce', 'entropy_histogram_jacobian_bce',
    'entropy_soft_rank', 'lc_soft_rank'
]

# parser.add_argument('--architecture_name', type=str, required=True, help="Drawn from config.ootb.convolutional_classification")
parser.add_argument('--densenet_daf_depth', type=int, required=False, help='Depth = 3n+4 for DenseNet DAF')
parser.add_argument('--byol_daf', action='store_true', required=False, help='Use a BYOL DAF instead of a DenseNet one - see byol_binary_linear_classification')
parser.add_argument('--logit_bank_path', type=str, required=True, help="The file that holds the stored logits and labelled image indices")
parser.add_argument('--include_trainset', action='store_true', help="i.e. do we include the images that the densenet was trained on in the regression problem?")
parser.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'cifar100'])
parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--batch_size', type=int, required=True, help='Batch size when training regressor')
parser.add_argument('--num_epochs', type=int, required=True, help='Number of epochs to train model')
parser.add_argument(
    '--training_mode', type=str, 
    choices = viable_secondary_losses + ['posterior_distillation', 'posterior_distillation_multitask'],
    help='Follows from Lent Week 3 report', required=True
)
parser.add_argument(
    '--objective_uniformalisation', type=str, choices=['no_un', 'online_un']
)
parser.add_argument('--test_prop', type=float, required=True, help='Proportion of CIFAR used for test set')
parser.add_argument("--save_dir", required=True, type=str, default=None)

parser.add_argument(
    '--coarse_cifar100', action='store_true', 
    help='If active with dataset=cifar100, will group distillation target to the coarse labels; if active with dataset=cifar10, will throw error'
)

posterior_distillation_arguments = parser.add_argument_group("posterior distillation")

posterior_distillation_arguments.add_argument(
    '--multitask_training_mode', type=str, required=False, default=None,
    choices = viable_secondary_losses + ['None'],
    help="When using multitask mode, the second head has the option to emulate any of these"   
)
posterior_distillation_arguments.add_argument(
    '--distillation_temperature', type=float, required=False, default=None,
    help="Symmetric temperature applied to both the prediction and target logits for posterior distillation"
)
posterior_distillation_arguments.add_argument(
    '--multitask_acquisition_target_weight', type=float, required=False, default=None,
    help="e.g.: Loss = (1 - multitask_acquisition_target_weight) * posterior distillation loss + multitask_acquisition_target_weight * acquisition prediction loss, but this depends on our lambda_scheduler_option"
)
posterior_distillation_arguments.add_argument(
    '--lambda_scheduler_option', type=str, required=False, default='no_lambda_scheduling', choices=multitask_distillation.scheduler_functions,
    help="Scheduling we apply to multitask_acquisition_target_weight, which may not even be a weight, remember!"
)


def prepare_for_jacobian(train_loader, dataset, training_mode):

    # Get the indices for the training data, which we can use for fitting beta
    base1_training_indices = train_loader.dataset.indices

    # train_loader is a subset of a subset, so we have to convert indices
    base2_training_indices = list(map(train_loader.dataset.dataset.indices.__getitem__, base1_training_indices))

    # Using the converted indices, we can get all the training pairs (images + targets)
    training_pairs = list(map(train_loader.dataset.dataset.dataset.__getitem__, base2_training_indices))

    # Finally, we can get the actual targets from here
    # If we're not in multitask mode, each tp in training_pairs will be (image, scalar target)
    if isinstance(training_pairs[0][1], np.ndarray):
        training_targets = np.array([tp[1] for tp in training_pairs])
    # Otherwise, each tp will be (image, [logits, scalar target])
    elif isinstance(training_pairs[0][1], list):
        training_targets = np.array([tp[1][1] for tp in training_pairs])

    # Build the base transform to get targets into [0,1] range, ready for Jacobian method
    if 'entropy' in training_mode:
        C = 100. if dataset == 'cifar100' else 10.
        base_transform = lambda x: x/torch.log(torch.tensor(C))
    elif 'lc' in training_mode:
        base_transform = lambda x: x + 1.

    return training_targets, base_transform
        

def configure_muzzle_for_cifar(model, training_mode, num_classes, second_training_mode=None):
    ## Add the correct head(s) to the model, based on the training mode
    
    # Currently this only works for InterfaceFriendlyModel types
    if isinstance(model, InterfaceFriendlyModel):

        # Add just a sigmoid
        if 'bce' in training_mode:
            model.muzzle = nn.Sigmoid()

        # Add a multitraining mode muzzle, which splits posterior from acquisition prediction
        elif training_mode in ['posterior_distillation_multitask']:
            # Treat the second training mode (i.e. the secondary target for multitask) the same way
            model.muzzle = MultitaskHead(sigmoid_target = ('bce' in second_training_mode), num_classes = num_classes)
    else:
        raise NotImplementedError('Final non-linearity for non-PyTorch ready networks')

    return model.to(device)


def generate_acquisition_regression_dataset(logit_bank_path, acquisition_classes, dataset_name, batch_size, test_prop, include_trainset, byol_option):
    
    # Recover logits and labelled indices
    data = torch.load(logit_bank_path)
    logits = data['ordered_logits']

    # Assert we have the right dataset name
    num_classes = logits.size(1)
    assert num_classes == int(dataset_name[5:])

    # Get the right dataset class
    dataset_class = CIFAR10 if dataset_name == 'cifar10' else CIFAR100
    original_master_dataset = dataset_class(root='./data', train=True, download=True, transform=transform_train, target_transform=None)
    
    # Make sure again we have the right data
    num_data = len(original_master_dataset.targets)
    assert len(logits) == num_data

    # Might be needed - data indices that the logit generating model had seen before
    seen_subset = set(data['subset_indices'])

    # Data we can use in the regression training
    usable_set = []

    # For each data point

    for i in range(len(original_master_dataset.targets)):

        # Decide whether we're even using this one - i.e. either including everything, or including unseen only
        if include_trainset or (i not in seen_subset):
            usable_set.append(i)
        else:
            continue

        # Generate the targets from the stored logits, one using each
        ith_acq = [acquisition_class._score_from_logits(logits[i]).cpu().numpy() for acquisition_class in acquisition_classes]

        # Replace it in the dataset - length >1 only occurs for multitask learning, otherwise put the only value back in
        original_master_dataset.targets[i] = ith_acq if len(ith_acq) > 1 else ith_acq[0]

    # We just have a list of indices now, which access the BYOL embeddings
    if byol_option:
        original_master_dataset.data = list(range(len(original_master_dataset.data)))
        
        # Update the dataset index method to bypass ``img = Image.fromarray(img)''
        original_master_dataset.disable_img()

    # Subset master_dataset by the data we're using

    master_dataset = torch.utils.data.Subset(original_master_dataset, usable_set)

    # And split the dataset
    test_length = int(np.floor(len(usable_set) * test_prop))
    train_length = len(usable_set) - test_length
    train_dataset, test_dataset = torch.utils.data.random_split(master_dataset, [train_length, test_length])

    # Turn the regression datasets into a dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # If using the BYOL embeddings, swe won't have an image to transform
    if byol_option:
        train_loader.dataset.dataset.dataset.disable_img()
        test_loader.dataset.dataset.dataset.disable_img()

    return train_loader, test_loader, test_dataset


def get_unseen_model_output(test_set, batch_size, trained_model):

    # Get the unseen data for this run
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    ## Initialise output and acquisition data
    preds_cache = []
    acquisition_cache = []

    # Evaluation only!
    with torch.no_grad():

        print('\n')
        for inputs, targets in tqdm(test_loader):

            inputs = inputs.to(device)
                
            _, decodings = trained_model(inputs)
            try:
                preds = decodings[0].to('cpu')
            except:
                preds = [d.to('cpu') for d in decodings[0]]

            # Append to cache
            acquisition_cache.append(targets)
            preds_cache.append(preds)

    return acquisition_cache, preds_cache


def get_acquisition_classes(training_mode, coarsify, second_training_mode=None):

    # What are we using as the reference posterior ctegorical?
    identity_acquisiton_type = CoarseCIFAR100Acquisition if coarsify else IdentityAcquisition

    # Direct posterior `prediction'
    if training_mode in ['posterior_distillation']:
        acquisition_classes = [identity_acquisiton_type]
    
    # Multitask prediction => one objective with identity, one based on second objective
    elif training_mode in ['posterior_distillation_multitask']:
        acquisition_classes = [
            identity_acquisiton_type,
            # Use same rules for the second objective as the first one
            MaximumEntropyAcquisition if 'entropy' in second_training_mode 
                else LowestConfidenceAcquisition if 'lc' in second_training_mode else None
        ]

    # Simple acquisition class prediction
    else:
        acquisition_classes = [
            MaximumEntropyAcquisition if 'entropy' in training_mode 
                else LowestConfidenceAcquisition if 'lc' in training_mode else None
        ]

    return acquisition_classes


def get_acquisition_prediction_criterion_class(training_mode, dataset, objective_uniformalisation, train_loader, test_loader, second_training_mode=None, args=None, save_dir=None, lambda_scheduling='no_lambda_scheduling'):

    # Get the loss function we are looking for

    # Start with the posterior distillation case - for 
    if training_mode in ['bag_of_phonemes_distillation']:
        pass
    
    # Start with the posterior distillation case - for 
    elif training_mode in ['posterior_distillation']:
        assert 'cifar' in dataset
        criterion = SymmetricTemperatureKLDistillationLoss(temperature=args.distillation_temperature)

    elif training_mode in ['entropy_soft_rank', 'lc_soft_rank']:
        criterion = SpearmanRankLoss()

    elif 'multitask' in training_mode:
        # USED TO BE: training_mode in ['posterior_distillation_multitask']: BUT more got added for speech!
        # Getting the acquisition loss for the multitask setting does not need second_training_mode or args
        # Nice chance to use recursion
        secondary_criterion = get_acquisition_prediction_criterion_class(second_training_mode, dataset, objective_uniformalisation, train_loader, test_loader)

        # Get the scheduling function we are using from the same file as MultiTaskSymmetricTemperatureKLDistillationLoss
        lambda_scheduler = getattr(multitask_distillation, lambda_scheduling)

        # Get the number of steps in each full epoch
        steps_in_epoch = len(train_loader) + len(test_loader)

        # The actial criterion
        criterion = MultiTaskSymmetricTemperatureKLDistillationLoss(
            acquisition_criterion=secondary_criterion, 
            acquisition_target_weight_parameter=args.multitask_acquisition_target_weight, 
            lambda_scheduler=lambda_scheduler,
            steps_in_epoch=steps_in_epoch,
            logging_file=os.path.join(save_dir, 'multitask_loss_hist.txt'),
            temperature=args.distillation_temperature,
        )

    elif training_mode in ['lc_bce']:
        # Make targets negative
        criterion = RescaleBCELoss(- torch.tensor(1.))

    elif training_mode in ['entropy_bce']:
        # Normalise targets in range
        C = 100. if dataset == 'cifar100' else 10.
        criterion = RescaleBCELoss(torch.log(torch.tensor(C)))

    elif training_mode in ['entropy_log_mse', 'entropy_mse']:
        criterion = LogNormalMSELoss() if training_mode == 'entropy_log_mse' else nn.MSELoss()
        
        if objective_uniformalisation != 'no_un':
            raise Exception('No ON if using MSE, not BCE')

    elif training_mode in ['lc_parametric_jacobian_bce', 'lc_histogram_jacobian_bce', 'entropy_parametric_jacobian_bce', 'entropy_histogram_jacobian_bce']:

        # Get the targets used to initialise the jacobian transform, and the transform needed after the model output
        training_targets, base_transform = prepare_for_jacobian(train_loader, dataset, training_mode)

        # Now build the actual loss function
        if 'parametric' in training_mode:
            criterion = SqrtExponentialJacobianTransformedBCELoss(torch.tensor(training_targets), lr = 0.0001, beta_0 = torch.tensor(6), num_steps=500, base_transform = base_transform, reduction = 'mean')
        elif 'histogram' in training_mode:
            criterion = NonParametricJacobianTransformedBCELoss(torch.tensor(training_targets), num_bins = 500, base_transform = base_transform, reduction = 'mean')

    if objective_uniformalisation == 'online_un':
        criterion = AdditionImportanceWeightedBCELoss(criterion, 500, fix_after=len(train_loader.dataset))

    return criterion


class IdentityAcquisition(UnitwiseAcquisition):
    """Just a way of getting logits back in an interface friendly way"""
    @staticmethod
    def _score_from_logits(logits):
        return logits


class CoarseCIFAR100Acquisition(UnitwiseAcquisition):
    """
        Group CIFAR100 classes together at the logit level
        reference: https://github.com/ryanchankh/cifar100coarse/blob/master/cifar100coarse.py
    """
    coarse_labels = torch.tensor([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                                3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                                6, 11,  5, 10,  7,  6, 13, 15,  3, 15, 
                                0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                                5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                                16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                                10, 3,  2, 12, 12, 16, 12,  1,  9, 19, 
                                2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                                16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                                18,  1,  2, 15,  6,  0, 17,  8, 14, 13])

    @staticmethod
    def _score_from_logits(logits):
        """Logits will come in with shape"""
        
        # Get the probabilities so we can add them in linear space
        distribution = torch.softmax(logits, 0)

        # Group the probabilities together - slow right now but better than a for loop
        grouped_probabilities = torch.tensor(
            [
                sum(map(itemgetter(1), group)) for key, group 
                in groupby(sorted(zip(CoarseCIFAR100Acquisition.coarse_labels, distribution)), key=itemgetter(0))
            ]
        )

        # Convert back to logits
        grouped_logits = torch.distributions.Categorical(grouped_probabilities).logits

        return grouped_logits


if __name__ == '__main__':

    args = parser.parse_args()

    # Initialise the regression mode we are using

    # Configure dataset dependent number of classes, for later things
    base_num_classes = num_classes = 10 if args.dataset == 'cifar10' else 100
    # This is number of classes for our distillation head => we need to change things here
    if args.coarse_cifar100:
        if base_num_classes == 100:
            base_num_classes = 20
        elif base_num_classes == 10:
            raise Exception('Cannot have --coarse_cifar100 active if dataset=cifar10')
    
    ## Get the model + any modifications
    # These classes have >1D output
    num_classes = (
        base_num_classes if args.training_mode == 'posterior_distillation' 
        else base_num_classes + 1 if args.training_mode == 'posterior_distillation_multitask' 
        else None
    )

    # Premade model classes
    if args.byol_daf:
        path = (
            '/home/alta/BLTSpeaking/exp-pr450/data/byol_embeddings_10.pkl' if args.dataset == 'cifar10' 
            else '/home/alta/BLTSpeaking/exp-pr450/data/byol_embeddings_100.pkl'
        )
        model = byol_binary_linear_classification(embedding_cache_path=path, regression_mode=True, num_classes=num_classes).to(device)
    else:
        model = default_mini_densenet_classifier(depth=args.densenet_daf_depth, regression_mode=True, num_classes=num_classes).to(device)   

    # Get the acquisition class we require
    acquisition_classes = get_acquisition_classes(args.training_mode, args.coarse_cifar100, args.multitask_training_mode)

    # Add the final sigmoid if predicting LC, or the dual head for multitask posterior distillation
    model = configure_muzzle_for_cifar(model, args.training_mode, base_num_classes, args.multitask_training_mode)

    # Get the regression CIFAR dataset required
    train_loader, test_loader, test_dataset = generate_acquisition_regression_dataset(
        args.logit_bank_path, acquisition_classes, args.dataset, args.batch_size, 
        args.test_prop, args.include_trainset, args.byol_daf
    )

    # Get where we're saving the model
    save_dir = config_savedir(args.save_dir, args)

    # Get loss + OU we need
    criterion = get_acquisition_prediction_criterion_class(
        args.training_mode, args.dataset, args.objective_uniformalisation, train_loader, 
        test_loader, args.multitask_training_mode, args, save_dir, args.lambda_scheduler_option
    )

    # Standard DenseNet optimiser
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    regressor, results = train_daf_acquisition_regression(
        regressor=model,
        optimizer=optimizer,
        scheduler=None,
        scheduler_epochs=[],
        decodings_criterion=criterion,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        num_epochs=args.num_epochs,
        show_print=True
    )

    with open(os.path.join(save_dir, 'results.json'), 'w') as jf:
        json.dump(results, jf)

    torch.save(regressor.state_dict(), os.path.join(save_dir, 'model.mdl'))

    # The results we need
    # TODO: make this for train_dataset too
    all_acquisition, all_preds = get_unseen_model_output(test_dataset, args.batch_size, regressor)

    torch.save(
        {'all_acquisition': all_acquisition, 'all_preds': all_preds}, 
        os.path.join(save_dir, 'acquisition_results.pkl')
    )

    torch.save(
        {'criterion': criterion},
        os.path.join(save_dir, 'criterion.pkl')
    )
