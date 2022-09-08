import copy
from argparse import ArgumentParser

import torch
from torch import nn
from torch import optim

from learn2rank.utils import PointwiseVariableRankRegressionDataset
from learn2rank.utils import eval_rank_metrics
from learn2rank.utils import get_dataloaders
from learn2rank.utils import get_flattened_split
from learn2rank.utils import get_unnormalized_variable_rank
from learn2rank.utils import numpy_dataset_paths
from learn2rank.utils import unflatten_data


class NeuralNetwork(nn.Module):
    def __init__(self, args):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(37, args.dim_l1),
            nn.ReLU(),
            nn.Linear(args.dim_l1, args.dim_l2),
            nn.ReLU(),
            nn.Linear(args.dim_l2, 1))

    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x


def weighted_mse(y1, y2, wt):
    return torch.sum(wt * ((y1 - y2) ** 2))


def train_epoch(tr_loader, model, optimizer, criteria):
    model.train()
    running_loss = 0
    counter = 0
    for x, y, wt in tr_loader:
        y_pred = model(x)
        y_pred = torch.squeeze(y_pred)
        loss = criteria(y, y_pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        counter += 1
    running_loss /= counter

    return running_loss


def val_epoch(val_loader, model, criteria):
    model.eval()
    with torch.no_grad():
        for x, y, wt in val_loader:
            y_pred = model(x)
            y_pred = torch.squeeze(y_pred)
            val_loss = criteria(y, y_pred).item()

            return val_loss


def train_nn(args):
    print('Neural Network: Point-wise Rank regression')

    num_items = int(args.dataset_name.split("_")[1])
    dataset = numpy_dataset_paths[args.dataset_name]
    x_train, y_train, weights_train = get_flattened_split(args, dataset['train'])
    x_val, y_val, weights_val = get_flattened_split(args, dataset['val'])
    x_test, y_test, weights_test = get_flattened_split(args, dataset['test'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    assert 'train' in dataset

    # Load data
    tr_dataset = PointwiseVariableRankRegressionDataset(x_train, y_train, weights_train, device)
    val_dataset = PointwiseVariableRankRegressionDataset(x_val, y_val, weights_val, device)
    test_dataset = PointwiseVariableRankRegressionDataset(x_test, y_test, weights_test, device)
    tr_loader, val_loader, test_loader = get_dataloaders(args, tr_dataset, val_dataset,
                                                         test_dataset)

    # Optimizer, loss and model
    criteria = nn.MSELoss()  # Fixed for now
    model = NeuralNetwork(args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    metrics = {'train': [], 'val': [], 'test': []}

    best_loss = 1000
    best_model = NeuralNetwork(args).to(device)

    for epoch in range(args.n_epochs):
        # tr_loss = train_epoch(tr_loader, model, optimizer, criteria)
        model.train()
        running_loss = 0
        counter = 0
        for x, y, wt in tr_loader:
            y_pred = model(x)
            y_pred = torch.squeeze(y_pred)
            loss = criteria(y, y_pred)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            counter += 1
        running_loss /= counter

        # val_loss = val_epoch(val_loader, model, criteria)
        model.eval()
        with torch.no_grad():
            for x, y, wt in val_loader:
                y_pred = model(x)
                y_pred = torch.squeeze(y_pred)
                val_loss = criteria(y, y_pred).item()

        if best_loss > val_loss:
            print()
            print('Update best model...', best_loss, val_loss)
            best_loss = copy.copy(val_loss)
            best_model.load_state_dict(model.state_dict())

        print(f"Epoch {epoch + 1}: Train Loss {running_loss}, Val Loss {val_loss}")

    best_model.eval()
    with torch.no_grad():
        y_train_pred = best_model(torch.from_numpy(x_train).float().to(device)).squeeze().numpy()
        y_val_pred = best_model(torch.from_numpy(x_val).float().to(device)).squeeze().numpy()

        y_pred_unflattened = {
            'train': unflatten_data(y_train_pred, num_items),
            'val': unflatten_data(y_val_pred, num_items)
        }
        y_pred_unnorm_rank = {
            'train': get_unnormalized_variable_rank(y_pred_unflattened['train']),
            'val': get_unnormalized_variable_rank(y_pred_unflattened['val'])
        }

        y_unflattened = {
            'train': unflatten_data(y_train, num_items),
            'val': unflatten_data(y_val, num_items)
        }
        y_unnorm_rank = {
            'train': get_unnormalized_variable_rank(y_unflattened['train']),
            'val': get_unnormalized_variable_rank(y_unflattened['val']),
        }

        print()
        print(f'Train rank metrics...')
        metrics.update(eval_rank_metrics(y_unnorm_rank['train'], y_pred_unnorm_rank['train']))

        print()
        print(f'Validation rank metrics...')
        metrics.update(eval_rank_metrics(y_unnorm_rank['val'], y_pred_unnorm_rank['val']))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--weighted_loss', type=int, default=0)
    parser.add_argument('--dataset_name', type=str, default='3_60_mat')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--n_epochs', type=int, default=250)
    parser.add_argument('--dim_l1', type=int, default=128)
    parser.add_argument('--dim_l2', type=int, default=128)
    args = parser.parse_args()

    train_nn(args)
