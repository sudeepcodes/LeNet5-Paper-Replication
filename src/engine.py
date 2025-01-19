from collections import defaultdict
from typing import Tuple

import torch
from tqdm.auto import tqdm

from src.utils import accuracy_fn


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """
    Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
        model: A PyTorch model to be trained.
        dataloader: A DataLoader instance for the model to be trained on.
        loss_fn: A PyTorch loss function to minimize.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of training loss and training accuracy metrics.
        In the form (train_loss, train_accuracy). For example:
        (0.1112, 0.8743)
    """

    cumulative_train_loss, cumulative_train_acc = 0, 0
    model.to(device)

    model.train()
    for batch_no, (X_train, y_train) in enumerate(dataloader):
        X_train, y_train = X_train.to(device), y_train.to(device)

        y_logits = model(X_train)

        batch_loss = loss_fn(y_logits, y_train)
        cumulative_train_loss += batch_loss
        cumulative_train_acc += accuracy_fn(y_true=y_train,
                                            y_pred=torch.argmax(torch.softmax(y_logits, dim=1), dim=1))

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

    loss_per_epoch = cumulative_train_loss / len(dataloader)
    acc_per_epoch = cumulative_train_acc / len(dataloader)
    return loss_per_epoch, acc_per_epoch


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """
    Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
        model: A PyTorch model to be trained.
        dataloader: A DataLoader instance for the model to be trained on.
        loss_fn: A PyTorch loss function to minimize.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of testing loss and testing accuracy metrics.
        In the form (test_loss, test_accuracy). For example:
        (0.1112, 0.8743)
    """

    cumulative_test_loss, cumulative_test_acc = 0, 0
    model.to(device)

    model.eval()
    with torch.inference_mode():
        for X_test, y_test in dataloader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            y_logits = model(X_test)

            batch_loss = loss_fn(y_logits, y_test)
            cumulative_test_loss += batch_loss
            cumulative_test_acc += accuracy_fn(y_true=y_test,
                                               y_pred=torch.argmax(torch.softmax(y_logits, dim=1), dim=1))

        loss_per_epoch = cumulative_test_loss / len(dataloader)
        acc_per_epoch = cumulative_test_acc / len(dataloader)
        return loss_per_epoch, acc_per_epoch


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: torch.device):
    """
    Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
        model: A PyTorch model to be trained and tested.
        train_dataloader: A DataLoader instance for the model to be trained on.
        test_dataloader: A DataLoader instance for the model to be tested on.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        loss_fn: A PyTorch loss function to calculate loss on both datasets.
        epochs: An integer indicating how many epochs to train for.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A dictionary of training and testing loss as well as training and
        testing accuracy metrics. Each metric has a value in a list for
        each epoch.
        In the form: {train_loss: [...],
                      train_acc: [...],
                      test_loss: [...],
                      test_acc: [...]}
        For example if training for epochs=2:
                     {train_loss: [2.0616, 1.0537],
                      train_acc: [0.3945, 0.3945],
                      test_loss: [1.2641, 1.5706],
                      test_acc: [0.3400, 0.2973]}
    """

    results = defaultdict(list)
    for epoch in tqdm(range(epochs)):
        print(f'Epoch: {epoch}\n--------------')
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)

        print(f'Train Loss: {train_loss} | Train Acc: {train_acc}')
        print(f'Test Loss: {test_loss} | Test Acc: {test_acc}')

        results['train_loss'].append(train_loss.cpu().item())
        results['train_acc'].append(train_acc)
        results['test_loss'].append(test_loss.cpu().item())
        results['test_acc'].append(test_acc)
    return results
