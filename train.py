import os, sys
import time
import h5py
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
from provider import ModelNet40
from models.dgcnn import dgcnn
import metrics 

np.random.seed(42)
torch.manual_seed(42)

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

conf = metrics.config()
conf.epochs = 64
conf.batch_size = 32
conf.learning_rate = 0.001
conf.lr_shrink_rate = 0.8
conf.lr_min = 0.00001
conf.regularization = 5e-4
conf.N = 1024 # max is 2048
conf.nCls = 40
conf.k = 20
conf.cuda = True
conf.workers = 1
conf.print_freq = 50

def create_dataset():
    train_dataset = ModelNet40("train", channel="first", points=conf.N)
    test_dataset = ModelNet40("test", channel="first", points=conf.N)
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
            batch_size = conf.batch_size,
            shuffle = True,
            num_workers = conf.workers,
            pin_memory = True,
            sampler = None,
            worker_init_fn = lambda work_id:np.random.seed(work_id))

    test_loader = torch.utils.data.DataLoader(test_dataset,
            batch_size = conf.batch_size,
            shuffle = True,
            num_workers = conf.workers,
            pin_memory = True,
            worker_init_fn = lambda work_id:np.random.seed(work_id))

    return train_loader, test_loader

def train(train_loader, model, criterion, optimizer, epoch, device):
    print("==> Train Epoch {}".format(epoch))
    model.train() # switch to train mode
    end = time.time()
    nCorrect = 0
    nTotal = 0
    totalLoss = 0
    totalTime = 0
    nBatch = len(train_loader)
    for i, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)
        torch.cuda.synchronize()
        data_time = time.time() - end

        # compute pred
        end = time.time()
        pred = model(data)
        loss = criterion(pred, labels)

        totalLoss += loss.item()
        optimizer.zero_grad()
        loss.backward() # compute gradient and do SGD step
        optimizer.step()
        torch.cuda.synchronize()
        gpu_time = time.time() - end

        # measure accuracy and record loss
        end = time.time()
        acc, nc, nt = metrics.accuracy(pred, labels)
        nCorrect += nc
        nTotal += nt
        totalTime += data_time + gpu_time

        if i % conf.print_freq == 0:
            print("    training batch {}/{}: acc {:.2f}, time {:.2f}s".format(i+1, nBatch, acc, data_time+gpu_time))
    
    acc = nCorrect/nTotal
    print("+=> Epoch {:3}: Accuracy {:.4f}, Loss {:.2f}, Time {:.2f}s".format(epoch, acc, totalLoss, totalTime))

def test(test_loader, model, epoch, device):
    print("==> Test Epoch {}".format(epoch))
    model.eval() # switch to evaluate mode
    end = time.time()
    nCorrect = 0
    nTotal = 0
    totalTime = 0
    nBatch = len(test_loader)
    for i, (data, labels) in enumerate(test_loader):
        data, labels = data.cuda(), labels.cuda()
        torch.cuda.synchronize()
        data_time = time.time() - end

        # compute output
        end = time.time()
        with torch.no_grad():
            pred = model(data)
        torch.cuda.synchronize()
        gpu_time = time.time() - end

        # measure accuracy and record loss
        end = time.time()
        acc, nc, nt = metrics.accuracy(pred, labels)
        nCorrect += nc
        nTotal += nt
        totalTime += data_time + gpu_time

        if conf.print_freq and i % conf.print_freq == 0:
            print("    testing batch {}/{}: acc {:.2f}, time {:.2f}s".format(i+1, nBatch, acc, data_time+gpu_time))
    
    acc = nCorrect/nTotal
    print("++> Test Epoch {:3}: Accuracy {:.4f} ({}/{}), Time {:.2f}s".format(epoch, acc, nCorrect, nTotal, totalTime))

def adjust_learning_rate(optimizer, epoch):
    lr = conf.learning_rate * (conf.lr_shrink_rate ** (epoch // 5))
    lr = max(conf.lr_min, lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    print(conf)
    train_loader, test_loader = create_dataset()
    device = torch.device("cuda" if conf.cuda else "cpu")
    model = dgcnn(conf).to(device)

    optimizer = optim.Adam(model.parameters(), lr=conf.learning_rate, weight_decay=conf.regularization)
    #optimizer = optim.SGD(model.parameters(), lr=conf.learning_rate, momentum=0.9)

    #criterion = metrics.get_loss_cls
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(0, conf.epochs):
        adjust_learning_rate(optimizer, epoch)

        train(train_loader, model, criterion, optimizer, epoch, device) # train for one epoch
        test(test_loader, model, epoch, device) # evaluate on validation set

if __name__ == "__main__":
    main()
