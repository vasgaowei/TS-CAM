import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math

__all__ = ['vgg16_cam']


model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
}


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


configs_dict = {
    'cam': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512,
                  512, 'M', 512, 512, 512]
}


class VGG_CAM(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG_CAM, self).__init__()
        self.features = features
        self.num_classes = num_classes  # L

        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)

        if init_weights:
            self._initialize_weights()

        features_layers = list(map(id, self.features.parameters()))
        self.features_params = filter(lambda p: id(p) in features_layers, self.parameters())
        self.classifier_params = filter(lambda p: id(p) not in features_layers, self.parameters())

    def forward(self, x, return_cam=False):
        x = self.features(x)
        x = self.conv6(x)
        x = self.relu(x)
        pre_logit = self.avgpool(x)
        pre_logit = pre_logit.view(pre_logit.size(0), -1)
        logits = self.fc(pre_logit)

        if return_cam:
            feature_map = x.detach().clone()
            fc_weights = self.fc.weight.view(
                1, self.num_classes, feature_map.shape[1], 1, 1)  # 1 * L * C * 1 * 1
            feature = feature_map.unsqueeze(1)  # N * 1 * C * H * W
            cams = (feature * fc_weights).sum(2)  # N * L * H * W

            return logits, cams
        return logits

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def vgg16_cam(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = True
    model = VGG_CAM(make_layers(configs_dict['cam']), **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['vgg16'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


if __name__ == "__main__":
    import torch
    import torch.nn.functional as F

    input = torch.randn((2, 3, 196, 196))
    model = vgg16_cam(pretrained=True, num_classes=200)
    logits, cams = model(input, return_cam=True)
    print(logits.shape)
    print(cams.shape)

    cams_t = F.upsample_bilinear(cams, [14, 14])

    print(cams_t.shape)

