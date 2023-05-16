import argparse
import os
import time

from torch import optim
from tqdm import tqdm
import torch
import torch.nn as nn

from transformerLite import TransformerLite

SEQUANCE_LEN = 10000


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

        optimizer.zero_grad()
        # # TODO#############
        # inp = batch.data  # pcm file bytes(mini batch size(10),t=4096,k=1)
        # label = batch.label  # (batchsize,4) four coded corresponding to 4 places
        # ############################
        inp = torch.ones(2,SEQUANCE_LEN,2).to('cuda:0')
        #label = torch.rand(2,4).to("cuda:0")
        label = torch.Tensor([[0,1,0,0],[0,1,0,0]]).to('cuda:0')
        if inp.size(1) > MAX_LENGTH:
            inp = inp[:, :MAX_LENGTH, :]
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
        print(100 * correct / total)
    return avg_loss / len(train_iter), 100 * correct / total


def run(epochs=10, k=2, heads=2, t=SEQUANCE_LEN, BATCH_SIZE=2):
    model = TransformerLite(t=t, k=k, heads=heads)
    model = model.to('cuda:0')
    # Create loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=1e-4)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda i: min(i / (10_000 / BATCH_SIZE), 1.0))
    # TODO#####################
    # folder_path = "Desktop/SMS_data_sample"
    # files_iterator = (os.path.join(folder_path, file_name) for file_name in os.listdir(folder_path) if
    #                   os.path.isfile(os.path.join(folder_path, file_name)))
    #######################
    train_iter = range(0,10)

    # Training loop
    for epoch in range(epochs):
        print(f'\n Epoch {epoch}')

        # Train on data
        train_loss, train_acc = train(train_iter,
                                      model,
                                      optimizer,
                                      lr_scheduler,
                                      criterion)

    print(f'\nFinished.loss:{train_loss},acc:{train_acc}')

torch.cuda.empty_cache()
run()