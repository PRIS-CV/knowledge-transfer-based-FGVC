import torch
import torchvision
from torch.autograd import Variable
from torchvision import transforms
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def test_ensemble(nets_path, dataset, batch_size):
    use_cuda = torch.cuda.is_available()
    correct = 0
    total = 0

    nets = []
    for path in nets_path:
        net = torch.load(path)
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        # net= torch.nn.DataParallel(net)
        net.eval()
        nets.append(net)

    transform_test = transforms.Compose([
        transforms.Scale((600, 600)),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    # testset = torchvision.datasets.ImageFolder(root='./test', transform=transform_test)
    testset = torchvision.datasets.ImageFolder(root='/data/changdongliang/{}/test'.format(dataset),
                                               transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=8)

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = 0
        for net in nets:
            outputs += net(inputs)[1]
        # outputs = net(inputs[0].unsqueeze(0))

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        if batch_idx % 50 == 0:
            print('Step: %d | Acc: %.3f%% (%d/%d)' % (
            batch_idx, 100. * float(correct) / total, correct, total))
            
    print('Step: %d | Acc: %.3f%% (%d/%d)' % (
            batch_idx, 100. * float(correct) / total, correct, total))
    test_acc = 100. * float(correct) / total

    return test_acc


# for name in os.listdir("."):
if __name__ == "__main__":
    # datasets = ['Aircraft', 'StandCars', 'Birds2']
    datasets = ['Birds2', 'Aircraft']
    # modelnames = ['resnet50', 'vgg16']
    modelnames = ['resnet50']
    # usages = ['param', 'param2', 'param3']
    usages = ['car']
    name1 = datasets[0]
    name2 = modelnames[0]
    name3 = usages[0]
    if True:
        nets_path = ['/data/zhangsiqing/restart/{}_{}_{}/model1.pth'.format(name1, name2, "param"),
                     '/data/zhangsiqing/restart/{}_{}_{}/model2.pth'.format(name1, name2, "param" ),
                     # '/data/zhangsiqing/restart/{}_{}_{}/model1.pth'.format(name1, name2, "param2"),
                     '/data/zhangsiqing/restart/{}_{}_{}/model2.pth'.format(name1, name2, "param2"),
                     # '/data/zhangsiqing/restart/{}_{}_{}/model1.pth'.format(name1, name2, "param3"),
                     # '/data/zhangsiqing/restart/{}_{}_{}/model2.pth'.format(name1, name2, "param3"),
                     # '/data/zhangsiqing/restart/{}_{}_{}/model8.pth'.format(name1, name2, name3),
                     ]
        test_ensemble(
            nets_path=nets_path,
            dataset=name1, 
            batch_size=16)
        print(1)
    if True:
        nets_path = [
        # '/data/zhangsiqing/restart/{}_{}_{}/model1.pth'.format(name1, name2, "param"),
                     # '/data/zhangsiqing/restart/{}_{}_{}/model2.pth'.format(name1, name2, "param" ),
                     # '/data/zhangsiqing/restart/{}_{}_{}/model1.pth'.format(name1, name2, "param2"),
                     # '/data/zhangsiqing/restart/{}_{}_{}/model2.pth'.format(name1, name2, "param2"),
                     # '/data/zhangsiqing/restart/{}_{}_{}/model1.pth'.format(name1, name2, "param3"),
                     # '/data/zhangsiqing/restart/{}_{}_{}/model2.pth'.format(name1, name2, "param3"),
                     # '/data/zhangsiqing/restart/{}_{}_{}/model8.pth'.format(name1, name2, name3),
                     ]
        test_ensemble(
            nets_path=nets_path,
            dataset=name1, 
            batch_size=16)
        print(2)
    if True:
        nets_path = [
        # '/data/zhangsiqing/restart/{}_{}_{}/model1.pth'.format(name1, name2, "param"),
        #              '/data/zhangsiqing/restart/{}_{}_{}/model2.pth'.format(name1, name2, "param" ),
                     '/data/zhangsiqing/restart/{}_{}_{}/model1.pth'.format(name1, name2, "param2"),
                     # '/data/zhangsiqing/restart/{}_{}_{}/model2.pth'.format(name1, name2, "param2"),
                     # '/data/zhangsiqing/restart/{}_{}_{}/model1.pth'.format(name1, name2, "param3"),
                     # '/data/zhangsiqing/restart/{}_{}_{}/model2.pth'.format(name1, name2, "param3"),
                     # '/data/zhangsiqing/restart/{}_{}_{}/model8.pth'.format(name1, name2, name3),
                     ]
        test_ensemble(
            nets_path=nets_path,
            dataset=name1, 
            batch_size=16)
        print(3)
    if True:
        nets_path = [
        # '/data/zhangsiqing/restart/{}_{}_{}/model1.pth'.format(name1, name2, "param"),
        #              '/data/zhangsiqing/restart/{}_{}_{}/model2.pth'.format(name1, name2, "param" ),
        #              '/data/zhangsiqing/restart/{}_{}_{}/model1.pth'.format(name1, name2, "param2"),
                     '/data/zhangsiqing/restart/{}_{}_{}/model2.pth'.format(name1, name2, "param2"),
                     # '/data/zhangsiqing/restart/{}_{}_{}/model1.pth'.format(name1, name2, "param3"),
                     # '/data/zhangsiqing/restart/{}_{}_{}/model2.pth'.format(name1, name2, "param3"),
                     # '/data/zhangsiqing/restart/{}_{}_{}/model8.pth'.format(name1, name2, name3),
                     ]
        test_ensemble(
            nets_path=nets_path,
            dataset=name1, 
            batch_size=16)
        print(4)
    if True:
        nets_path = [
        # '/data/zhangsiqing/restart/{}_{}_{}/model1.pth'.format(name1, name2, "param"),
        #              '/data/zhangsiqing/restart/{}_{}_{}/model2.pth'.format(name1, name2, "param" ),
        #              '/data/zhangsiqing/restart/{}_{}_{}/model1.pth'.format(name1, name2, "param2"),
        #              '/data/zhangsiqing/restart/{}_{}_{}/model2.pth'.format(name1, name2, "param2"),
                     '/data/zhangsiqing/restart/{}_{}_{}/model1.pth'.format(name1, name2, "param3"),
                     # '/data/zhangsiqing/restart/{}_{}_{}/model2.pth'.format(name1, name2, "param3"),
                     # '/data/zhangsiqing/restart/{}_{}_{}/model8.pth'.format(name1, name2, name3),
                     ]
        test_ensemble(
            nets_path=nets_path,
            dataset=name1, 
            batch_size=16)
        print(5)
    if True:
        nets_path = [
        # '/data/zhangsiqing/restart/{}_{}_{}/model1.pth'.format(name1, name2, "param"),
        #              '/data/zhangsiqing/restart/{}_{}_{}/model2.pth'.format(name1, name2, "param" ),
        #              '/data/zhangsiqing/restart/{}_{}_{}/model1.pth'.format(name1, name2, "param2"),
        #              '/data/zhangsiqing/restart/{}_{}_{}/model2.pth'.format(name1, name2, "param2"),
        #              '/data/zhangsiqing/restart/{}_{}_{}/model1.pth'.format(name1, name2, "param3"),
                     '/data/zhangsiqing/restart/{}_{}_{}/model2.pth'.format(name1, name2, "param3"),
                     # '/data/zhangsiqing/restart/{}_{}_{}/model8.pth'.format(name1, name2, name3),
                     ]
        test_ensemble(
            nets_path=nets_path,
            dataset=name1, 
            batch_size=16)
        print(6)