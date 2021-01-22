import torch.nn as nn


class model_bn_resnet50(nn.Module):
    def __init__(self, model, feature_size, classes_num):
        super(model_bn_resnet50, self).__init__()
        self.features = nn.Sequential(*list(model.children())[:-2])

        self.max = nn.MaxPool2d(kernel_size=14, stride=14)
        self.num_ftrs = 2048

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            # nn.Dropout(0.5),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(feature_size, classes_num),
        )

    def forward(self, x):
        features = self.features(x)
        x = self.max(features)
        # print("x.shape", x.shape)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return features, x


class model_bn_vgg16(nn.Module):
    def __init__(self, model, feature_size, classes_num):
        super(model_bn_vgg16, self).__init__()
        self.features = nn.Sequential(*list(model.children())[:-2])
        self.max = nn.MaxPool2d(kernel_size=14, stride=14)
        self.num_ftrs = 512

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            # nn.Dropout(0.5),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(feature_size, classes_num),
        )

    def forward(self, x):
        features = self.features(x)
        # print(features.shape, "features")
        x = self.max(features)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return features, x

class model_bn_vgg16_dry(nn.Module):
    def __init__(self, model, feature_size, classes_num):
        super(model_bn_vgg16_dry, self).__init__()
        self.features = nn.Sequential(*list(model.children())[:-2])
        self.max = nn.MaxPool2d(kernel_size=2, stride=2)
        self.num_ftrs = 512 * 7 * 7

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            # nn.Dropout(0.5),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(feature_size, classes_num),
        )

    def forward(self, x):
        features = self.features(x)
        # print(features.shape, "features")
        x = self.max(features)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return features, x
