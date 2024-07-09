import torch
import torchvision
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
from transformers import AutoImageProcessor, ViTMAEForPreTraining, ViTModel



def build_classifier(network_type, contrastive=False, input_chans=3):
    
    net = Classifier(network_type, contrastive=contrastive, input_chans=input_chans)

    return net


class Classifier(torch.nn.Module):
    def __init__(self, network_type, contrastive=False, state_dict=None, config=None, input_chans=3):
        super().__init__()
        self.network_type = network_type
        self.contrastive = contrastive



        # Build the network with pretrained weights
        if network_type == 'resnet18':

            weights = ResNet18_Weights.DEFAULT
            net = resnet18(weights=weights)

            num_filters = net.fc.in_features

            # Get the feature extractor
            layers = list(net.children())[:-1]
            self.feature_extractor = torch.nn.Sequential(*layers)


        if network_type == 'resnet50':
            weights = torchvision.models.ResNet50_Weights.DEFAULT
            net = torchvision.models.resnet50(weights=weights)

            num_filters = net.fc.in_features

            # Get the feature extractor
            layers = list(net.children())[:-1]
            if input_chans != 3:
                layers[0] = torch.nn.Conv2d(input_chans, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.feature_extractor = torch.nn.Sequential(*layers)



        if network_type == 'swin':
            weights = torchvision.models.Swin_T_Weights.DEFAULT
            net = torchvision.models.swin_t(weights=weights)

            num_filters = net.head.in_features

            # Get the feature extractor
            # layers = list(net.children())[:-1]
            self.feature_extractor = net.features  # torch.nn.Sequential(*layers)

            # Get the classifier
            self.norm = net.norm
            # self.permute = net.permute
            self.avgpool = net.avgpool
            self.flatten = torch.nn.Flatten(1)

        if network_type == 'vit.py':
            self.image_processor = AutoImageProcessor.from_pretrained("facebook/vit.py-mae-base")
            self.feature_extractor = ViTModel.from_pretrained(None, state_dict=state_dict, config=config)
            num_filters = 768

            # Define the transforms
            weights = ResNet18_Weights.DEFAULT #Transforms are same as for pretrained resnet18
            # self.transform_mean = self.image_processor.image_mean
            # self.transform_mean = self.image_processor.image_std
            # self.t = ResNet18_Weights.DEFAULT.transforms()
            # self.t.mean = torch.tensor(self.transform_mean)
            # self.t.std = torch.tensor(self.transform_std)

            self.resize = torchvision.transforms.Resize((224, 224), antialias=True)



        self.dropout = torch.nn.Dropout2d(p=0.2)


        # Define the classifier portion
        # self.classifier = torch.nn.Linear(num_filters, 1)
        if self.contrastive:
            self.contrastive_proj = torch.nn.Sequential(torch.nn.Linear(num_filters, 4*128),
                                                    torch.nn.ReLU(inplace=True),
                                                  torch.nn.Linear(4*128, 128))


        self.classifier = torch.nn.Sequential(torch.nn.Linear(num_filters, 256),
                                              torch.nn.ReLU(),
                                              torch.nn.Linear(256, 128),
                                              torch.nn.ReLU(),
                                              torch.nn.Linear(128, 1))



        self.transform_mean = weights.transforms().mean
        self.transform_std = weights.transforms().std

        self.t = weights.transforms()
        self.t.mean = torch.tensor(self.t.mean)
        self.t.std = torch.tensor(self.t.std)


    def get_features(self, x):

        batch_size = x.shape[0]

        if self.network_type == 'swin':
            x = self.feature_extractor(x)
            x = self.norm(x)
            x = x.permute(0, 3, 1, 2)
            x = self.avgpool(x)
            feats = self.dropout(x)
            #feats = self.flatten(x)

        elif self.network_type == 'vit.py':
            # Make into 224x224
            x = self.resize(x)

            x = self.feature_extractor(x)

            feats = x[0]

            # Take the first token
            feats = feats[:, 0, :]


        else:
            feats = self.feature_extractor(x)
            feats = self.dropout(feats)
            #feats.flatten(1)

        feats = feats.view(batch_size, -1)

        return feats

    def classify(self, feats):
        y = self.classifier(feats)
        return y

    def get_contrastive_proj(self, feats):
        return self.contrastive_proj(feats)

    def forward(self, x):
        # Get the features
        feats = self.get_features(x)

        # Classify
        y = self.classify(feats)

        return y