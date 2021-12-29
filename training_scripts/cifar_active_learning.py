import os, torch, random, json
from torchvision.datasets.cifar import CIFAR10, CIFAR100
import torch.utils.data as data
from torch import nn
from cifar_repo.utils.logger import Logger
from classes_utils.ensemble import AudioEncoderBasicEnsemble

import active_learning as al
from classes_utils.cifar.data import CIFAR10Subset, CIFAR100Subset
from cifar_repo.cifar import (
    parser, transform_train, transform_test, make_model, train, test,
    adjust_learning_rate
)

device = (
    torch.device('cuda') if torch.cuda.is_available()
    else torch.device('cpu')
)
use_cuda = torch.cuda.is_available()

parser.add_argument('--initProp', type=float, help='proportion of dataset to start with')
parser.add_argument('--roundProp', type=float, help='proportion of dataset to add each round')
parser.add_argument('--totalBudgetProp', type=float, help='proportion of dataset to stop at')
parser.add_argument('--acquisitionFunction', type=str, help='acquisition function to use')
parser.add_argument('--roundEpochs', type=int, help='Number of epochs to train after each acquisition')
parser.add_argument('--saveDir', type=str, help='Where to make directories to save performances & indices')

def make_ensemble(args, ensemble_size, num_classes):
    "Make a resnet model"
    ensemble = AudioEncoderBasicEnsemble(
        encoder_type=make_model,
        ensemble_size=ensemble_size,
        encoder_ensemble_kwargs={'args': args, 'num_classes': num_classes, 'variational': False},
    )
    return ensemble

class ModelWrapper(nn.Module):
    def __init__(self):
        super(ModelWrapper, self).__init__()
        self.model = None
        self.midloop = False
    def reinit_model(self, new_model):
        self.model = new_model
    def forward(self, x, *args, **kwargs):
        x = x.to(device)
        output = self.model(x)
        if self.midloop:
            # Training, so return logits
            return output
        else:
            # i.e. if not an ensemble, update regular ALAttribute using this
            if len(output) == 1:
                output = output[0]
            return {'last_logits': output}

if __name__ == '__main__':

    al.disable_tqdm()

    args = parser.parse_args()
    state = {k: v for k, v in args._get_kwargs()}

    # Random seed
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    print('\nSeed:', args.manualSeed, '\n')
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.manualSeed)




    if args.dataset == 'cifar10':
        train_dataset, test_dataset, num_classes = CIFAR10Subset, CIFAR10, 10
        print('Using cifar10')
    else:
        train_dataset, test_dataset, num_classes = CIFAR100Subset, CIFAR100, 100
        print('Using cifar100')

    test_image_dataset = test_dataset(root='./data', train=False, download=True, transform=transform_test)
    testloader = data.DataLoader(test_image_dataset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    train_image_dataset = train_dataset(root='./data', train=True, download=True, transform=transform_train, original_transform=transform_test, init_indices=[], target_transform=None)


    ensemble_size = 5 if args.acquisitionFunction == 'bald' else 1
    if args.acquisitionFunction == 'bald':
        ensemble_size = 5
        # Normal initialisation doesn't work because labels size != predictions size, so
        # we need to manually input prediction shape
        initialisation = torch.zeros(len(train_image_dataset.data), ensemble_size, num_classes)
        last_logits = al.dataset_classes.StochasticAttribute(
            'last_logits', initialisation, ensemble_size, cache=True
        )
    else:
        ensemble_size = 1
        last_logits = al.dataset_classes.ALAttribute(
            'last_logits', torch.zeros(len(train_image_dataset.data), num_classes), cache=True
        )


    train_dataset = al.dataset_classes.DimensionlessDataset(
        data=torch.tensor(range(len(train_image_dataset.data))),
        labels=torch.tensor(range(len(train_image_dataset.data))),
        costs=torch.tensor([1. for _ in range(len(train_image_dataset.data))]),
        index_class=al.annotation_classes.DimensionlessIndex,
        semi_supervision_agent=None,
        data_reading_method=lambda x: train_image_dataset.get_original_data(x),
        label_reading_method=lambda x: train_image_dataset.get_original_label(x),
        al_attributes=[last_logits],
        is_stochastic=(ensemble_size != 1)
    )

    if args.acquisitionFunction == 'maxent':
        acquisition = al.acquisition.MaximumEntropyAcquisition(train_dataset)
    elif args.acquisitionFunction == 'lc':
        acquisition = al.acquisition.LowestConfidenceAcquisition(train_dataset)
    elif args.acquisitionFunction == 'margin':
        acquisition = al.acquisition.MarginAcquisition(train_dataset)
    elif args.acquisitionFunction == 'rand':
        acquisition = al.acquisition.RandomBaselineAcquisition(
            train_dataset, lambda *x: ()
        )
    elif args.acquisitionFunction == 'kldiff':
        acquisition = al.acquisition.PredsKLAcquisition(train_dataset)
    elif args.acquisitionFunction == 'bald':
        acquisition = al.acquisition.BALDAcquisition(train_dataset)
    else:
        raise ValueError(args.acquisitionFunction)

    round_cost = train_dataset.total_cost * args.roundProp
    print('Adding each round: ', round_cost, ' images')

    total_budget = train_dataset.total_cost * args.totalBudgetProp
    print('Will stop at: ', total_budget, ' images')

    div_pol = al.batch_querying.NoPolicyBatchQuerying()
    window_class = al.annotation_classes.DimensionlessAnnotationUnit
    selector = al.selector.DimensionlessSelector(
        round_cost=round_cost, 
        acquisition=acquisition,
        window_class=window_class,
        diversity_policy=div_pol,
        selection_mode='argmax'
    )

    classification_model = ModelWrapper()

    agent = al.agent.ActiveLearningAgent(
        train_set=train_dataset,
        batch_size=args.train_batch,
        selector_class=selector,
        model=classification_model,
        device=device,
        budget=total_budget
    )

    i=0
    while True:
        try:
            os.mkdir(f"{args.saveDir}-{i}")
            break
        except:
            i+=1
        if i>60:
            raise Exception("Too many folders!")
    args.saveDir = f"{args.saveDir}-{i}"
    print(f'\nSave directory: {args.saveDir}\n')

    saveable_args = vars(args)
    with open(os.path.join(args.saveDir, 'config.json'), 'w') as f:
        json.dump(saveable_args, f)

    agent.init(args.initProp)

    round_num = 0
    for _ in agent:

        # We are using the pytorch dataloader, so need to move the batches to the CIFARSubset classes
        new_indices = []
        for ls in agent.labelled_set:
            new_indices.extend(ls)
        train_image_dataset.indices = new_indices
        
        trainloader = data.DataLoader(train_image_dataset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

        # Make logging path based on args.saveDir
        round_num += 1
        round_dir = os.path.join(args.saveDir, f'round_{round_num}')
        os.mkdir(round_dir)

        title = 'cifar-100-' + args.arch + str(round_num)
        logger = Logger(os.path.join(round_dir, f'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

        with open(os.path.join(round_dir, f'labelled_set.txt'), 'w') as f:
            for i in new_indices:
                f.write(str(i))
                f.write('\n')
        
        # Reinitialise the classification_model and the optimizer
        classification_model.reinit_model(make_ensemble(args, ensemble_size, num_classes).to(device))
        optimizer = torch.optim.SGD(classification_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()

        print(f'\n\nRound: {round_num} | {len(train_image_dataset)} labelled')

        # Train and val
        for epoch in range(args.roundEpochs):

            adjust_learning_rate(optimizer, epoch, args, state)

            print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.roundEpochs, state['lr']))

            # Need to return logits drectly while in training/val script
            classification_model.midloop = True
            train_loss, train_acc = train(trainloader, classification_model, criterion, optimizer, epoch, use_cuda)
            test_loss, test_acc = test(testloader, classification_model, criterion, epoch, use_cuda)

            # Need to return attribute dictionary while for active learning agent
            classification_model.midloop = False
            print(f'train_loss: {train_loss} | train_acc: {train_acc} | test_loss: {test_loss} | test_acc: {test_acc}')

            # append logger file
            logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])
