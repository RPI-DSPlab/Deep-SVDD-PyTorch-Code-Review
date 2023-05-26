import click
import torch
import pandas as pd

from deepSVDD import DeepSVDD
from datasets.main import load_dataset


def calculate_label_score(data, net):
    """
    Calculate labels and scores for given data.

    Parameters:
    data (Tuple[torch.Tensor]): Tuple of inputs, labels and indices from the DataLoader.
        inputs (torch.Tensor): Input data.
        labels (torch.Tensor): Ground truth labels.
        idx (torch.Tensor): Indices of the data.
    net (DeepSVDD): The neural network model to use for prediction.

    Returns:
    List[Tuple[int, int, float]]: List of tuples with indices, labels and calculated scores.
    """
    inputs, labels, idx = data
    inputs = inputs.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    outputs = net(inputs)
    dist = torch.sum((outputs - net.c) ** 2, dim=1)

    if net.objective == 'soft-boundary':
        scores = dist - net.R ** 2
    else:
        scores = dist

    # Save triples of (idx, label, score) in a list
    idx_label_score = list(zip(idx.cpu().data.numpy().tolist(),
                               labels.cpu().data.numpy().tolist(),
                               scores.cpu().data.numpy().tolist()))

    return idx_label_score


@click.command()
@click.argument('dataset_name', type=click.Choice(['mnist', 'cifar10']))
@click.argument('net_name', type=click.Choice(['mnist_LeNet', 'cifar10_LeNet', 'cifar10_LeNet_ELU']))
@click.argument('load_model', type=click.Path(exists=True))
@click.argument('data_path', type=click.Path(exists=True))
@click.argument('save_path', type=click.Path())
@click.option('--device', type=str, default='cuda', help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
@click.option('--normal_class', type=int, default=0,
              help='Specify the normal class of the dataset (all other classes are considered anomalous).')
def main(dataset_name, net_name, load_model, data_path, save_path, device, normal_class):
    # Load data
    dataset = load_dataset(dataset_name, data_path, normal_class)

    # Initialize DeepSVDD model and set neural network \phi
    deep_SVDD = DeepSVDD('one-class', 0.1)
    deep_SVDD.set_network(net_name)

    # Load Deep SVDD model (radius R, center c, network weights, and possibly autoencoder weights)
    deep_SVDD.load_model(model_path=load_model, load_ae=False)

    # Calculate scores for the entire CIFAR10 dataset
    if dataset_name == 'cifar10':
        all_scores = []
        for data in dataset.test_loader:
            inputs, labels = data
            idx = torch.arange(len(inputs))
            data = (inputs, labels, idx)
            scores = calculate_label_score(data, deep_SVDD)
            all_scores.extend(scores)

        # Save scores to a CSV file
        df_scores = pd.DataFrame(all_scores, columns=['Index', 'Label', 'Score'])
        df_scores.to_csv(save_path + '/scores.csv', index=False)
        print('Anomaly scores saved to %s.' % (save_path + '/scores.csv'))


if __name__ == '__main__':
    main()
