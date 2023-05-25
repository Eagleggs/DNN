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
from torch.utils.data import random_split
SEQUANCE_LEN =2000


def train(train_iter, model, optimizer, lr_scheduler, criterion, MAX_LENGTH=SEQUANCE_LEN, GRADIENT_CLIPPING=1.0):
    avg_loss = 0
    correct = 0
    total = 0

    # Switch to train mode
    # model.train()

    # Iterate through batches
    for batch in tqdm(train_iter):
        inp,label = batch
        inp = inp.to('cuda:0')
        label = label.to('cuda:0')
        # if inp.size(1) > MAX_LENGTH:
        #     inp = inp[:, 9000:9000+MAX_LENGTH,:]
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

def test(test_iter, model,MAX_LENGTH=SEQUANCE_LEN):
    correct = 0
    total = 0

    for batch in tqdm(test_iter):
        inp,label = batch
        inp = inp.to('cuda:0')
        label = label.to('cuda:0')
        # if inp.size(1) > MAX_LENGTH:
        #     inp = inp[:, 9000:9000+MAX_LENGTH,:]
        output = model(inp)  # output(batchsize,k=1,4)
        output = output.squeeze()  # output(batchsize,4)


        predicted = torch.argmax(output, dim=1)
        total += inp.size(0)
        correct += (predicted == torch.argmax(label, dim=1)).sum().item()
    return 100 * correct / total

def run(epochs=120, k=2, heads=8, t=SEQUANCE_LEN, BATCH_SIZE=10):
    model = TransformerLite(t=t, k=k, heads=heads)
    model = model.to('cuda:0')
    # Create loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=1e-4,weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda i: min(i / (10_000 / BATCH_SIZE), 1.0))
    dataset = PCMDataSet("./0524_25_data")
    train_size = int(0.8 * len(dataset))  # 90% for training
    test_size = len(dataset) - train_size  # Remaining 10% for testing
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    test_dataset_list = list(test_dataset)
    train_dataset_list = list(train_dataset)
    num_test_data = len(test_dataset_list)
    num_train_data = len(train_dataset_list)
    # Print the number of data points in the test set
    print("Number of data points in the test set:", num_test_data)
    print("Number of data points in the train set:", num_train_data)
    train_iter = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_iter = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=True)
    best_test_acc = 0
    # Training loop
    for epoch in range(epochs):
        print(f'\n Epoch {epoch}')

        # Train on data
        train_loss, train_acc = train(train_iter,
                                      model,
                                      optimizer,
                                      lr_scheduler,
                                      criterion)
        test_acc =test(test_iter,model)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model, 'model_best.pt')
        print(f'\n training loss:{train_loss},training acc:{train_acc}')
        print(f'\nFinished.test acc:{test_acc}')
    torch.save(model, 'model_final.pt')

torch.cuda.empty_cache()
run()