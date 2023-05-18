import argparse
import os
import time

from torch import optim
from tqdm import tqdm
import torch
import torch.nn as nn

from transformerLite import TransformerLite
from get_data import PCMDataSet
from torch.utils.data import Dataset, DataLoader
SEQUANCE_LEN =1000


def main(config):
    device = torch.device('cuda:0')


def train(train_iter, model, optimizer, lr_scheduler, criterion, MAX_LENGTH=SEQUANCE_LEN, GRADIENT_CLIPPING=1.0):
    avg_loss = 0
    correct = 0
    total = 0

    # Switch to train mode
    model.train()

    # Iterate through batches
    for batch in tqdm(train_iter):
        inp,label = batch
        inp = inp.to('cuda:0')
        label = label.to('cuda:0')
        # # TODO#############
        # inp = batch.data  # pcm file bytes(mini batch size(10),t=4096,k=1)
        # label = batch.label  # (batchsize,4) four coded corresponding to 4 places
        # ############################
        # inp = torch.ones(2,20000,2).to('cuda:0')
        if inp.size(1) > MAX_LENGTH:
            inp = inp[:, 10000:10000+MAX_LENGTH,:]
        output = model(inp)  # output(batchsize,k=1,4)
        output = output.squeeze()  # output(batchsize,4)
        loss = criterion(output, label)

        loss.backward()

        # Clip gradients if the total gradient vector has a length > 1, we
        # clip it back down to 1.
        if GRADIENT_CLIPPING > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIPPING)

        optimizer.step()
        lr_scheduler.step()

        # Keep track of loss and accuracy
        avg_loss += loss.item()
        predicted = torch.argmax(output, dim=1)
        total += inp.size(0)
        correct += (predicted == torch.argmax(label, dim=1)).sum().item()
        # print(100 * correct / total)
    return avg_loss / len(train_iter), 100 * correct / total


def run(epochs=150, k=2, heads=8, t=SEQUANCE_LEN, BATCH_SIZE=10):
    model = TransformerLite(t=t, k=k, heads=heads)
    model = model.to('cuda:0')
    # Create loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=1e-4)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda i: min(i / (10_000 / BATCH_SIZE), 1.0))
    dataset = PCMDataSet("./0517_data")
    train_iter = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    # train_data, test_data = dataloader.split(split_ratio=0.8)

    # Training loop
    for epoch in range(epochs):
        print(f'\n Epoch {epoch}')

        # Train on data
        train_loss, train_acc = train(train_iter,
                                      model,
                                      optimizer,
                                      lr_scheduler,
                                      criterion)
        print(f'\nloss:{train_loss},acc:{train_acc}')

    print(f'\nFinished.loss:{train_loss},acc:{train_acc}')

torch.cuda.empty_cache()
run()