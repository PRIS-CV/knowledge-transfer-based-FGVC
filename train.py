from __future__ import print_function
import os

import logging
import torch
import shutil
import torch.optim as optim
import random
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.autograd import Variable

from utils import *
from test import test
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"



util_param = 0

address = "/data/changdongliang/"
save_dir = '/data/zhangsiqing/restart'


batch_size = 32
num_workers = 4

def train_softlabel(nb_epoch, resume=False, start_epoch=0, model_path=None, teacher_path=None, global_count=1, dataset="Birds2", modelname="resnet50", usage="test", params=0):
    test_dir = address + dataset + '/test'
    train_dir = address + dataset + "/train"

    print(dataset)

    store_name = '{}_{}_{}'.format(dataset, modelname, usage)

    # from utils import proFgress_bar
    logging.basicConfig(level=logging.INFO)

    # setup output
    exp_dir = store_name

    try:
        os.stat(exp_dir)
    except:
        os.makedirs(exp_dir)
    logging.info("OPENING " + exp_dir + '/{}_results_train{}.csv'.format(usage, str(global_count)))
    logging.info("OPENING " + exp_dir + '/{}_results_test{}.csv'.format(usage, str(global_count)))

    results_train_file = open(exp_dir + '/{}_results_train{}.csv'.format(usage, str(global_count)), 'w')
    results_train_file.write('epoch, train_acc,train_loss\n')
    results_train_file.flush()

    results_test_file = open(exp_dir + '/{}_results_test{}.csv'.format(usage, str(global_count)), 'w')
    results_test_file.write('epoch, test_acc,test_loss\n')
    results_test_file.flush()

    use_cuda = torch.cuda.is_available()
    print(use_cuda)

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.Resize((600, 600)),
        transforms.RandomCrop(448, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    # Model
    if resume:
        net = torch.load(model_path)
    else:
        net = load_model(model_name='{}'.format(modelname), dataset=dataset, pretrain=True, require_grad=True)


    teacher_nets = []
    # try:
    for path in teacher_path:
        teacher_net = torch.load(path)
        for param in teacher_net.parameters():
            param.requires_grad = False
        teacher_nets.append(teacher_net)
    # except:
    #     pass

    if use_cuda:
        net.classifier.cuda()
        net.features.cuda()

        net.classifier = torch.nn.DataParallel(net.classifier, device_ids=[0, 1, 2, 3])
        net.features = torch.nn.DataParallel(net.features, device_ids=[0, 1, 2, 3])

        cudnn.benchmark = True

    CELoss = nn.CrossEntropyLoss()
    optimizer = optim.SGD([
        {'params': net.classifier.parameters(), 'lr': 0.01},
        {'params': net.features.parameters(), 'lr': 0.001}
    ],
        momentum=0.9, weight_decay=5e-4)

    max_val_acc = 0
    for epoch in range(start_epoch, nb_epoch):
        print('\nEpoch: %d' % epoch)
        optimizer.param_groups[0]['lr'] = cosine_anneal_schedule(epoch, nb_epoch, 0.01)
        optimizer.param_groups[1]['lr'] = cosine_anneal_schedule(epoch, nb_epoch, 0.001)

        for param_group in optimizer.param_groups:
            print(param_group['lr'])

        net.train()
        train_loss = 0
        correct = 0
        total = 0
        idx = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            idx = batch_idx
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            features, outputs = net(inputs)

            loss2 = 0
            for t_net in teacher_nets:
                featuret, _ = t_net(inputs)
                loss2 += FeatureLoss(features, featuret, util_param) / len(teacher_nets)
            
            loss1 = CELoss(outputs, targets)
            if len(teacher_nets) > 0:
                w = loss1.item() / loss2.item() * params
                loss = loss1 + loss2 * w
            else:
                loss = loss1

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            if batch_idx % 50 == 0:
                print('Step: %d | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                batch_idx, train_loss / (batch_idx + 1), 100. * float(correct) / float(total), correct, total))

        print('Step: %d | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                batch_idx, train_loss / (batch_idx + 1), 100. * float(correct) / float(total), correct, total))

        train_acc = 100. * float(correct) / float(total)
        train_loss = train_loss / (idx + 1)
        logging.info('Iteration %d, train_acc = %.5f,train_loss = %.6f' % (epoch, train_acc, train_loss))
        results_train_file.write('%d, %.4f,%.4f\n' % (epoch, train_acc, train_loss))
        results_train_file.flush()

        val_acc, val_loss = test(net, CELoss, batch_size, num_workers, test_dir)
        if val_acc > max_val_acc:
            max_val_acc = val_acc
            temp = os.path.join(save_dir, store_name)
            if not os.path.exists(temp):
                os.makedirs(temp)
            torch.save(net, temp + '/model{}.pth'.format(str(global_count)))
        logging.info('Iteration %d, test_acc = %.4f,test_loss = %.4f' % (epoch, val_acc, val_loss))
        results_test_file.write('%d, %.4f,%.4f\n' % (epoch, val_acc, val_loss))
        results_test_file.flush()

def process(dataset, modelname, usage, global_count):

    folder = "/data/zhangsiqing/restart/{}_{}_{}/".format(dataset, modelname, usage)

    teacher_path = []
    
    if global_count <= 0:
        pass
    elif global_count == 1:
    	return
    else:
        teacher_path= [folder + "model1.pth"]
    params = [0, 0, 0.5, 0.1, 0.05]

    print(teacher_path)
    # print(global_count + 1)
    print(params[global_count])
    print('model{}.pth'.format(str(global_count + 1)))

    train_softlabel(
            nb_epoch=100, 
            resume=False, 
            start_epoch=0, 
            model_path=None, 
            teacher_path=teacher_path, 
            global_count=global_count + 1, 
            dataset=dataset, 
            modelname=modelname, 
            usage=usage,
            params=params[global_count]
            )

def process2(dataset, modelname, usage, global_count):

    folder = "/data/zhangsiqing/restart/{}_{}_{}/".format(dataset, modelname, usage)

    teacher_path = [folder + "model1.pth"]
    
    for index in range(2, global_count):
        teacher_path.append(folder + "model{}.pth".format(index + 1))

    # params = [0, 0, 0.5, 0.1, 0.05]

    print(teacher_path)
    # print(global_count + 1)
    # print(params[global_count])
    print('model{}.pth'.format(str(global_count + 1)))

    train_softlabel(
            nb_epoch=100, 
            resume=False, 
            start_epoch=0, 
            model_path=None, 
            teacher_path=teacher_path, 
            global_count=global_count + 1, 
            dataset=dataset, 
            modelname=modelname, 
            usage=usage,
            params=0.1
            )

if __name__=="__main__":
    count = 7
    dataset="StandCars"
    modelname="resnet50"
    usage="car"
    
    # folder = "/data/zhangsiqing/restart/{}_{}_{}/".format(dataset, modelname, usage)
    # if not os.path.exists(folder):
    #     os.makedirs(folder)
    # param = "/data/zhangsiqing/restart/{}_{}_{}/".format(dataset, modelname, 'bn_param')
    # shutil.copy(os.path.join(param, 'model1.pth'), os.path.join(folder, 'model1.pth'))
    
    for index in range(3, count):
        process2(dataset, modelname, usage, index)

    # count = 7
    # dataset="StandCars"
    # modelname="resnet50"
    # usage="param512"
    
    # folder = "/data/zhangsiqing/restart/{}_{}_{}/".format(dataset, modelname, usage)
    # if not os.path.exists(folder):
    #     os.makedirs(folder)
    
    # for index in range(count):
    #     process(dataset, modelname, usage, index)