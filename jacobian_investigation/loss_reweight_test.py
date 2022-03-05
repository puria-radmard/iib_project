import json
import sys, torch
from torch import nn
from math import ceil
from sklearn import datasets
from matplotlib import pyplot as plt
from torch.nn.functional import sigmoid
from torch.utils.data import Dataset, DataLoader
from torch.distributions.exponential import Exponential
from scipy.stats import spearmanr
from util_functions.jacobian import truncated_sqrt_exponential_pdf
from classes_losses.acquisition_prediction_losses import ImportanceWeightedBCELoss, NonParametricJacobianTransformedBCELoss, SqrtExponentialJacobianTransformedBCELoss, AdditionImportanceWeightedBCELoss, RescaleBCELoss



class LogRegDataset(Dataset):

    def __init__(self, x, t):
        self.x = x
        self.t = t
        super(LogRegDataset, self).__init__()

    def __getitem__(self, index):
        return self.x[index], self.t[index]

    def __len__(self):
        return len(self.x)


class LogRegModel(nn.Module):

    def __init__(self):
        super(LogRegModel, self).__init__()
        self.layer = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1, 1)
        x = self.layer(x)
        x = self.sigmoid(x)
        return x


class Simple2DNN(nn.Module):

    def __init__(self):
        super(Simple2DNN, self).__init__()

        hidden_size = 5

        self.layers = nn.Sequential(
            nn.Linear(2, hidden_size), nn.Tanh(), 
            # nn.Linear(hidden_size, hidden_size), nn.Tanh(), 
            nn.Linear(hidden_size, 1, bias = False), nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


def train_model(model, trainset, testset, opt, crit, num_epochs):

    # Initialise loss/rank curves
    train_loss, test_loss, rank_history = [], [], []

    for epoch in range(num_epochs):

        # Initialise average losses
        epoch_train_loss, epoch_test_loss = 0, 0
        num_train_points, num_test_points = len(trainset.dataset), len(testset.dataset)

        model.train()
    
        # Train loop
        for batch in trainset:

            opt.zero_grad()
            
            # Unpack batch
            x, t = batch
            
            # Predictions
            y = model(x)

            # Get the loss value
            loss = crit(y, t)

            # Step
            loss.backward()
            opt.step()

            epoch_train_loss += loss.item() / num_train_points

            model.eval()

            test_batches = []
            test_preds = []

        with torch.no_grad():

            # Test loop
            for batch in trainset:

                opt.zero_grad()
                
                # Unpack batch
                x, t = batch
                
                # Predictions
                y = model(x)

                # Save these for the rank metrics
                test_batches.append(t)
                test_preds.append(y)

                # Get the loss value
                loss = crit(y, t)

                epoch_test_loss += loss.item() / num_test_points

        # Calculate rank
        all_ts = torch.concat(test_batches).numpy()
        all_ys = torch.concat(test_preds).numpy()
        s_rank_coeff = spearmanr(all_ys, all_ts)[0]

        rank_history.append(s_rank_coeff)
        train_loss.append(epoch_train_loss)
        test_loss.append(epoch_test_loss)

        print(f'Epoch {epoch} || Train loss: {round(epoch_train_loss, 5)} | Test loss: {round(epoch_test_loss, 5)} | Test rank {round(s_rank_coeff, 5)}')

    return train_loss, test_loss, rank_history


def generate_poly_dataset():
    def bias_regression_function(x, a = 1.1, m = 2.3, S = 3.3):
        """
            Desmose form:
                f\left(x\right)\ =\ 0.65+\frac{\left(\frac{x}{m}-\left(ax\right)^{4}-\left(x\right)^{14}+0.75\right)}{S}\left\{-1\le x\le1\right\}
        """
        numerator = (x/m) - (a*x)**4 - x**14 + 0.75
        return 0.35 - numerator / S

    data_x = 2 * torch.rand(dataset_size) - 1
    data_f = bias_regression_function(data_x)
    data_t = torch.clip(data_f + 0.1 * torch.randn_like(data_f), 0., 1.)

    return data_x, data_t


def generate_sigmoid_dataset():

    x = torch.linspace(-1, 1, 50)
    t = sigmoid(x)

    # Generate the data
    dist = Exponential(0.2)
    data_x = dist.sample_n(dataset_size) - 15
    data_t = sigmoid(data_x/6.)

    # Apply transforms
    data_x /= 12.
    data_x += torch.randn_like(data_x)
    data_t -= data_t.min()
    data_t /= data_t.max()
    # data_t = data_t + (torch.randn_like(data_t) * 0.05)

    return data_x, data_t


def generate_2D_dataset():
    
    # How many times more class 0 than class 1 (class 0 being sampled from a lower mean distribution)
    bias_ratio = 1.2

    # How many datapoints to generate then crop
    _N = int(2 * bias_ratio * dataset_size/(bias_ratio + 1))

    # Generate a balanced number of class 0s and 1s
    X, y = datasets.make_moons(_N, noise=0.2, shuffle=False, random_state=42)
    y = torch.tensor(y).float(); y += torch.rand_like(y)*0.1
    y, X = zip(*sorted(zip(y, X), key=lambda x: x[0]))
    X, y = torch.tensor(X), torch.tensor(y)

    # Crop dataset to induce bias
    X, y = torch.stack([X[:dataset_size]])[0], torch.stack([y[:dataset_size]])[0]

    # More 0s than 1s, and they have the lower mean
    class_0_mean = 4.5
    class_1_mean = 20

    # Sample the whole thing
    class_0_samples = Exponential(1/class_0_mean).sample(y.shape)
    class_1_samples = Exponential(1/class_1_mean).sample(y.shape)

    # Filter which sample gets what
    y = class_0_samples * (y<0.5).to(int) + class_1_samples * (y>=0.5).to(int)

    # Bound outputs
    y = 2*(torch.sigmoid(y/50) - 0.5)

    return X.float(), y.float()



if  __name__ == '__main__':

    dataset_size = 5000

    if sys.argv[1] == 'show_dataset':
        data_x, data_t = generate_2D_dataset()
        
        fig = plt.figure(figsize = (8, 4))
        ax1 = fig.add_subplot(122, projection = '3d')
        ax2 = fig.add_subplot(121)

        ax1.scatter(data_x[:,0].numpy(), data_x[:,1].numpy(), data_t.numpy(), c=data_t.numpy())
        ax1.view_init(azim=50)
        ax2.hist(data_t.numpy(), 50)

        ax2.set_xlabel()

        fig.savefig('jacobian_investigation/2D_dataset.png')

        plt.show()

        exit()    

    # Trial parameters, dataset, and model we are using
    if True:
        dataset_name = sys.argv[1]

        jacobian = sys.argv[2]
        assert jacobian in ['j', 'nj', 'jnp']

        reweigher = sys.argv[3]
        assert reweigher in ['rw', 'rwoff', 'nrw']

        if dataset_name == 'poly':
            data_x, data_t = generate_poly_dataset()
            model = LogRegModel()

        elif dataset_name == 'sigmoid':
            data_x, data_t = generate_sigmoid_dataset()
            model = LogRegModel()

        elif dataset_name == '2D':
            data_x, data_t = generate_2D_dataset()
            model = Simple2DNN()

    # Training parameters
    lr = 0.005
    num_epochs = 500
    test_prop = float(sys.argv[4])
    batch_size = 256

    # Sort out data
    testset_size = ceil(test_prop * dataset_size)
    trainset_size = dataset_size - ceil(test_prop * dataset_size)
    master_dataset = LogRegDataset(data_x, data_t)
    train_dataset, test_dataset = torch.utils.data.random_split(master_dataset, [trainset_size, testset_size])
    train_dataloader, test_dataloader = DataLoader(train_dataset, batch_size=batch_size), DataLoader(test_dataset, batch_size=batch_size)

    # Get this for the Jacobian case
    train_targets = data_t[train_dataset.indices]
    test_targets = data_t[test_dataset.indices]

    # Optimiser
    optimiser = torch.optim.SGD(lr=lr, params=model.parameters())

    # Star of the show: loss function
    # do we use Jacobian or not?
    if jacobian == 'j':
        criterion_type = SqrtExponentialJacobianTransformedBCELoss
        criterion_args = [train_targets, 0.00001, torch.tensor(6), 200, lambda x: x]
    if jacobian == 'jnp':
        criterion_type = NonParametricJacobianTransformedBCELoss
        criterion_args = [train_targets, 500, lambda x: x]
    elif jacobian == 'nj':
        criterion_type = RescaleBCELoss
        criterion_args = [torch.tensor(1)]

    # do we do loss reweighting or now?
    if reweigher == 'rw':
        # For now, fix the num bins to 50, and only update once
        criterion = AdditionImportanceWeightedBCELoss(criterion_type(*criterion_args), 50, dataset_size)
    elif reweigher == 'rwoff':
        criterion = AdditionImportanceWeightedBCELoss(criterion_type(*criterion_args), 50, train_targets)
    elif reweigher == 'nrw':
        criterion = criterion_type(*criterion_args)

    print('Starting training')

    train_loss, test_loss, rank_history = train_model(model, train_dataloader, test_dataloader, optimiser, criterion, num_epochs)

    # Plot dataset and results
    fig, axs = plt.subplots(2, 3, figsize = (15, 10))

    # First dataset/function plotting method only works for 1D input
    if dataset_name != '2D':
        axs[0, 0].scatter(data_x.numpy(), data_t.numpy())
        axs[0, 0].set_title('dataset')

        # Get the model curve
        model.eval()
        x = torch.linspace(-1, 4, 50)
        fitted_function = model(x).detach().reshape(-1)
        axs[0, 0].plot(x.numpy(), fitted_function.numpy())

    if jacobian in ['j', 'jnp']:
        # If using the Jacobian distribution, histogram the transformed training data
        transformed_targets = criterion._transform_target(test_targets)
        axs[0, 2].set_title(f'J transformed data')
        axs[0, 2].hist(transformed_targets.numpy(), 50, density = True)

    if jacobian in ['j']:
        # If using the Jacobian distribution, histogram the transformed training data
        transformed_targets = criterion._transform_target(test_targets)
        axs[0, 2].set_title(f'J transformed data, $\\beta = {criterion.beta.item()}$')
        axs[0, 2].hist(transformed_targets.numpy(), 50, density = True)

        # Also plot the fitted truncated sqrt exponential pdf
        x = torch.linspace(0, 1, 50)
        y = truncated_sqrt_exponential_pdf(x, criterion.beta)
        axs[0, 1].plot(x, y)

    axs[0, 1].set_title('target distribution')
    axs[0, 1].hist(data_t.numpy(), 50, density = True)

    axs[1, 0].plot(train_loss); axs[1, 0].set_title('train_loss')
    axs[1, 1].plot(test_loss);  axs[1, 1].set_title('test_loss')

    # Scatted the model predictions
    # preds = model(data_x[test_dataset.indices]).detach().reshape(-1)
    # axs[1, 2].scatter(test_targets.numpy(), preds.numpy())

    axs[1, 2].plot(rank_history)

    # axs[1, 2].plot([0, 1], [0, 1], color = 'black')

    if len(sys.argv) > 5:
        results = {
            'jacobian_setting': jacobian,
            'reweigher_setting': reweigher,
            'test_prop': test_prop,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'rank_history': rank_history
        }
        json_path = sys.argv[5]
        with open(json_path, 'w') as f:
            json.dump(results, f)
    else:
        fig.savefig(f'jacobian_investigation/{dataset_name}-{sys.argv[2]}-{sys.argv[3]}-{test_prop}.png')


