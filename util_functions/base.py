import os, torch, json, random

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
use_cuda = torch.cuda.is_available()


def load_state_dict(model, weights_path):
    if weights_path != None:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)


def get_config_from_file(file_path):
    with open(file_path, "r") as jfile:
        kw = json.load(jfile)
    return kw


def length_aware_loss(crit_output, lengths):
    loss_mask = torch.ones_like(crit_output)
    for j, lm in enumerate(loss_mask):
        lm[lengths[j] :] = 0
    crit_output *= loss_mask
    return crit_output, loss_mask


def batch_trace(x):
    torch.diagonal(x, dim1=-2, dim2=-1).sum(-1)


def batch_outer(z_i, z):
    return torch.einsum("bi,bj->bij", (z_i - z, z_i, z))


def config_savedir(base_save_dir, args):

    i=0
    while True:
        try:
            save_dir=f"{base_save_dir}-{i}"
            os.mkdir(save_dir)
            break
        except:
            i+=1
        if i>200:
            raise Exception("Too many folders!")

    saveable_args = vars(args)
    config_json_path = os.path.join(save_dir, "config.json")

    with open(config_json_path, "w") as jfile:
        json.dump(saveable_args, jfile)

    print(f"Config dir : {save_dir}\n", flush=True)

    return save_dir


def torch_manual_script(args):
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    print('\nSeed:', args.manualSeed, '\n')
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.manualSeed)
