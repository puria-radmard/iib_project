import torch, json, os
from config import device
from tqdm import tqdm
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader
from cifar_repo.cifar import transform_test

from interfaces.cifar_train_on_subset import parser, train_cifar_subset_main, Config
from util_functions.base import config_savedir, torch_manual_script

parser.add_argument("--logits_save_path", type=str, help='Where we are saving the acquisition values')


def generate_logits(evaluation_model, args, subset_indices):

    # Set evaluation_model to testing mode
    evaluation_model.eval()
    
    # Get the correct dataset and verify in logs
    if args.dataset == 'cifar10':
        cifar_dataset, num_classes = CIFAR10, 10
        print(f'Generating cifar10 target logits')
    elif args.dataset == 'cifar100':
        cifar_dataset, num_classes = CIFAR100, 100
        print(f'Generating cifar100 target logits')

    # Test time transformation applied only
    transform = transform_test

    # We use the training set as this will give us useful comparisons downstream
    train_image_dataset = cifar_dataset(
        root='./data', train=True, download=True, transform=transform, target_transform=None
    )
    
    # Strictly unshuffled dataloader
    evalutation_dataloader = DataLoader(train_image_dataset, batch_size=args.train_batch, shuffle=False, num_workers=4)

    # Initialise the logits tensor
    all_logits = torch.empty(0, num_classes).to('cpu')

    # Again ensure evaluation only
    with torch.no_grad():

        # Iterate batches in unshuffled loader
        for i, (images, targets) in tqdm(enumerate(evalutation_dataloader)):

            # Get (trained) model output - in a list wrapper
            batch_logits = evaluation_model(images.to(device))[0].detach().cpu()

            # Append new batch's logits onto the saving array
            all_logits = torch.cat([all_logits, batch_logits])

    save_dict = {
        'ordered_logits': all_logits,
        # Save this as well for ease of access later
        'subset_indices': subset_indices
    }

    return save_dict
            



if __name__ == '__main__':

    # Args passed to this
    main_args = parser.parse_args()
    cmd_args = Config(vars(main_args))

    # Collect cmd_args from mirrord config, for model construction
    with open(cmd_args.mirrored_config_json_path, 'r') as jf:
        cmd_args.update(json.load(jf))

    # Configure save_base
    cmd_args.log_base = config_savedir(cmd_args.log_base, cmd_args)

    state = cmd_args.__dict__

    # Validate dataset
    assert cmd_args.dataset == 'cifar10' or cmd_args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

    # Use CUDA
    use_cuda = torch.cuda.is_available()

    # Random seed
    torch_manual_script(cmd_args)

    # Get the subset we are training on
    with open(cmd_args.subset_index_path, 'r') as f:
        indices = f.read().split('\n')[:-1]
        assert len(indices) == cmd_args.subset_size
    subset_indices = list(map(lambda x: int(x), indices))

    # Train the model on the desired subset
    model = train_cifar_subset_main(subset_indices, cmd_args, state)

    # Get the logits (and replciate the desired subset) using the trained model
    save_dict = generate_logits(model, cmd_args, subset_indices)

    # Save save_dict
    save_path = os.path.join(cmd_args.log_base, 'evaluations.pkl')
    torch.save(save_dict, save_path)
