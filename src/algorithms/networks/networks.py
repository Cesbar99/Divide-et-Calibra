import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ConvNeXt_Tiny_Weights
import timm
from tab_transformer_pytorch import FTTransformer, TabTransformer

class ScaledLogits(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, logits):
        return logits / self.temperature
    
##### DEEP MLP FOR SYNTHETIC TABULAR DATA #####
class DeepMLP(nn.Module):
    def __init__(self, input_dim, output_dim, temperature=1.0):
        super().__init__()
        self.temperature = temperature

        self.main = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            #nn.Linear(128, 128),
            #nn.ReLU(),
            #nn.Linear(128, 128),
            #nn.ReLU(),
            #nn.Linear(128, 256),
            #nn.ReLU(),
            #nn.Linear(256, 128),
            #nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.output_layer = nn.Linear(64, output_dim)
        self.scaler = ScaledLogits(temperature)

    def forward(self, x):
        latent = self.main(x)
        logits = self.output_layer(latent)        
        return self.scaler(logits)
    

class MnistArch(nn.Module):
    def __init__(self, temperature=1.0):        
        super(MnistArch, self).__init__()
        self.num_channels = 1
        self.image_size = 28
        self.num_labels = 10

        layers = [
            nn.Conv2d(1, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 200),  # After convs and pooling, image size is reduced
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 10)
        ]        
        
        self.model = nn.Sequential(*layers)
        self.scaler = ScaledLogits(temperature)

    def forward(self, x):
        logits = self.model(x) 
        return self.scaler(logits)   

class TissueMNISTConvNeXtTiny(nn.Module):
    """Model for classification based on torchvision ConvNeXt Tiny."""

    def __init__(self, temperature=1.0, num_labels=8):
        super(TissueMNISTConvNeXtTiny, self).__init__()
        
        self.n_classes = num_labels
        self.scaler = ScaledLogits(temperature)

        # Pretrained ConvNeXt Tiny
        self.convnext_tiny = models.convnext_tiny(
            weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        )

        # In torchvision ConvNeXt Tiny, classifier is:
        # Sequential(LayerNorm2d/LayerNorm, Flatten(1), Linear(768, 1000))
        self.num_features = self.convnext_tiny.classifier[2].in_features

        # Replace classifier with your custom head
        self.convnext_tiny.classifier = nn.Sequential(
            self.convnext_tiny.classifier[0],   # keep pretrained norm
            self.convnext_tiny.classifier[1],   # keep flatten
            nn.Dropout(p=0.2),
            nn.Linear(self.num_features, num_labels)
        )        

        # Freeze everything
        for param in self.convnext_tiny.parameters():
            param.requires_grad = False

        # Unfreeze last feature stage + classifier
        # ConvNeXt Tiny features layout in torchvision:
        # features[0] = stem
        # features[1] = stage 1
        # features[2] = downsample
        # features[3] = stage 2
        # features[4] = downsample
        # features[5] = stage 3
        # features[6] = downsample
        # features[7] = stage 4 (last stage)
        for name, param in self.convnext_tiny.named_parameters():
            if "features.7" in name or "classifier" in name:
                param.requires_grad = True

    def repr(self, x):
        # Backbone features
        x = self.convnext_tiny.features(x)
        x = self.convnext_tiny.avgpool(x)

        # Stop before final Linear layer
        x = self.convnext_tiny.classifier[0](x)  # norm
        x = self.convnext_tiny.classifier[1](x)  # flatten

        return x

    def forward(self, x):
        logits = self.convnext_tiny(x)
        return self.scaler(logits)
    
class TissueMNISTResNet50(nn.Module):
    """Model for just classification.
    The architecture of our model is the same as standard DenseNet121
    """

    def __init__(self, temperature=1.0, num_labels=8):
        super(TissueMNISTResNet50, self).__init__()      
        self.scaler = ScaledLogits(temperature)
        self.resnet50 = models.resnet50(weights='IMAGENET1K_V2')
        self.num_features = self.resnet50.fc.in_features
        self.resnet50.layer4 = nn.Sequential(
            self.resnet50.layer4,
            nn.Dropout(p=0.2)
        )
        self.resnet50.fc = nn.Sequential(
                #nn.Dropout(p=0.1),                
                nn.Linear(self.num_features, num_labels))
        self.n_classes = num_labels
        
        # for param in self.resnet50.parameters():
        #     param.requires_grad = False        
        # for name, param in self.resnet50.named_parameters():
        #     if 'layer4' in name or 'fc' in name: #
        #         param.requires_grad = True
        for param in self.resnet50.parameters():
            param.requires_grad = False        
        for name, param in self.resnet50.named_parameters():
            if 'layer4' in name or 'fc' in name: #
                param.requires_grad = True

    def repr(self, x):
        # See note [TorchScript super()]
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)

        x = self.resnet50.layer1(x)
        x = self.resnet50.layer2(x)
        x = self.resnet50.layer3(x)
        x = self.resnet50.layer4(x)

        x = self.resnet50.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):                
        logits = self.resnet50(x)
        return self.scaler(logits)
    
    # def extract_features(self, batch):
    #     x, y = batch        
    #     feats = self.repr(x)  # feature representation
    #     preds = torch.argmax(self.resnet50.fc(feats), dim=-1).view(-1,1)  # predicted class
    #     # Create dict in the same format as predict outputs
    #     out = {
    #         "logits": feats,                  # replace logits with features
    #         "preds": preds,     # dummy preds
    #         "true": y
    #     }
    #     return out
         
                        
class TissueMnistVit(nn.Module):
    """Model for just classification.
    The architecture of our model is the same as standard DenseNet121
    """

    def __init__(self, temperature=1.0, num_labels=8):
        super(TissueMnistVit, self).__init__()      
        self.scaler = ScaledLogits(temperature)
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_labels, in_chans=3)
        print(self.vit)
        
        # Freeze all parameters
        for param in self.vit.parameters():
            param.requires_grad = False

        # Unfreeze last transformer block
        for name, param in self.vit.named_parameters():
            if 'blocks.11' in name:
                param.requires_grad = True

        # Unfreeze classifier head
        for param in self.vit.get_classifier().parameters():
            param.requires_grad = True

    def repr(self, x):
        """
        Extract feature representations (before classification head).
        Equivalent to the ResNet repr but for ViT.
        """
        # timm ViTs provide forward_features to get embeddings before head        
        tokens = self.vit.forward_features(x)
        feats = self.vit.forward_head(tokens, pre_logits=True)
        return feats   
    
    def forward(self, x):
        logits = self.vit(x)                
        return self.scaler(logits)          
    
class PathMnistVit(nn.Module):
    """Model for just classification.
    The architecture of our model is the same as standard DenseNet121
    """

    def __init__(self, temperature=1.0, num_labels=9):
        super(PathMnistVit, self).__init__()      
        self.scaler = ScaledLogits(temperature)
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_labels, in_chans=3)
        print(self.vit)
        
        # Freeze all parameters
        for param in self.vit.parameters():
            param.requires_grad = False

        # Unfreeze last transformer block
        for name, param in self.vit.named_parameters():
            if 'blocks.11' in name:
                param.requires_grad = True

        # Unfreeze classifier head
        for param in self.vit.get_classifier().parameters():
            param.requires_grad = True

    def forward(self, x):
        logits = self.vit(x)                
        return self.scaler(logits)       
    
class PathMNISTResNet50(nn.Module):
    """Model for just classification.
    The architecture of our model is the same as standard DenseNet121
    """

    def __init__(self, temperature=1.0, num_labels=9):
        super(PathMNISTResNet50, self).__init__()      
        self.scaler = ScaledLogits(temperature)
        self.resnet50 = models.resnet50(weights='IMAGENET1K_V2')
        self.num_features = self.resnet50.fc.in_features
        self.resnet50.layer4 = nn.Sequential(
            self.resnet50.layer4,
            nn.Dropout(p=0.2)
        )
        self.resnet50.fc = nn.Sequential(
                nn.Dropout(p=0.1),                
                nn.Linear(self.num_features, num_labels))
        self.n_classes = num_labels
        
        # for param in self.resnet50.parameters():
        #     param.requires_grad = False        
        # for name, param in self.resnet50.named_parameters():
        #     if 'layer4' in name or 'fc' in name: #
        #         param.requires_grad = True
        for param in self.resnet50.parameters():
            param.requires_grad = False        
        for name, param in self.resnet50.named_parameters():
            if 'layer4' in name or 'fc' in name: #
                param.requires_grad = True

    def repr(self, x):
        # See note [TorchScript super()]
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)

        x = self.resnet50.layer1(x)
        x = self.resnet50.layer2(x)
        x = self.resnet50.layer3(x)
        x = self.resnet50.layer4(x)

        x = self.resnet50.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        logits = self.resnet50(x)
        return self.scaler(logits)  
    
class PathMnistDenseNet121(nn.Module):
    """Model for just classification.
    The architecture of our model is the same as standard DenseNet121
    """

    def __init__(self, temperature=1.0, num_labels=9):
        super(PathMnistDenseNet121, self).__init__()      
        self.scaler = ScaledLogits(temperature)
        self.model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_labels)
        
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True
        for name, child in self.model.features.named_children():
            if name in ['denseblock4', 'norm5']:
                for param in child.parameters():
                    param.requires_grad = True
        
    def forward(self, x):
        logits = self.model(x)
        return self.scaler(logits)  
    
class Cifar10ResNet50(nn.Module):
    """Model for just classification.
    The architecture of our model is the same as standard DenseNet121
    """

    def __init__(self, temperature=1.0, num_labels=10):
        super(Cifar10ResNet50, self).__init__()      
        self.scaler = ScaledLogits(temperature)
        self.resnet50 = models.resnet50(weights='IMAGENET1K_V2')
        self.num_features = self.resnet50.fc.in_features
        self.resnet50.layer4 = nn.Sequential(
            self.resnet50.layer4,
            nn.Dropout(p=0.2)
        )
        self.resnet50.fc = nn.Sequential(
                #nn.Dropout(p=0.1),                
                nn.Linear(self.num_features, num_labels))
        self.n_classes = num_labels
        
        for param in self.resnet50.parameters():
            param.requires_grad = False        
        for name, param in self.resnet50.named_parameters():
            if 'layer4' in name or 'fc' in name: #
                param.requires_grad = True

    def repr(self, x):
        # See note [TorchScript super()]
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)

        x = self.resnet50.layer1(x)
        x = self.resnet50.layer2(x)
        x = self.resnet50.layer3(x)
        x = self.resnet50.layer4(x)

        x = self.resnet50.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        logits = self.resnet50(x)
        return self.scaler(logits)      

class Cifar10ConvNeXtTiny(nn.Module):
    """Model for classification based on torchvision ConvNeXt Tiny."""

    def __init__(self, temperature=1.0, num_labels=10):
        super(Cifar10ConvNeXtTiny, self).__init__()
        
        self.n_classes = num_labels
        self.scaler = ScaledLogits(temperature)

        # Pretrained ConvNeXt Tiny
        self.convnext_tiny = models.convnext_tiny(
            weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        )

        # In torchvision ConvNeXt Tiny, classifier is:
        # Sequential(LayerNorm2d/LayerNorm, Flatten(1), Linear(768, 1000))
        self.num_features = self.convnext_tiny.classifier[2].in_features

        # Replace classifier with your custom head
        self.convnext_tiny.classifier = nn.Sequential(
            self.convnext_tiny.classifier[0],   # keep pretrained norm
            self.convnext_tiny.classifier[1],   # keep flatten
            nn.Dropout(p=0.2),
            nn.Linear(self.num_features, num_labels)
        )        

        # Freeze everything
        for param in self.convnext_tiny.parameters():
            param.requires_grad = False

        # Unfreeze last feature stage + classifier
        # ConvNeXt Tiny features layout in torchvision:
        # features[0] = stem
        # features[1] = stage 1
        # features[2] = downsample
        # features[3] = stage 2
        # features[4] = downsample
        # features[5] = stage 3
        # features[6] = downsample
        # features[7] = stage 4 (last stage)
        for name, param in self.convnext_tiny.named_parameters():
            if "features.7" in name or "classifier" in name:
                param.requires_grad = True

    def repr(self, x):
        # Backbone features
        x = self.convnext_tiny.features(x)
        x = self.convnext_tiny.avgpool(x)

        # Stop before final Linear layer
        x = self.convnext_tiny.classifier[0](x)  # norm
        x = self.convnext_tiny.classifier[1](x)  # flatten

        return x

    def forward(self, x):
        logits = self.convnext_tiny(x)
        return self.scaler(logits)
    
class Cifar10Vit(nn.Module):
    """Model for just classification.
    The architecture of our model is the same as standard DenseNet121
    """

    def __init__(self, temperature=1.0, num_labels=10):
        super(Cifar10Vit, self).__init__()      
        self.scaler = ScaledLogits(temperature)
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_labels, in_chans=3)
        print(self.vit)
        
        # Freeze all parameters
        for param in self.vit.parameters():
            param.requires_grad = False

        # Unfreeze last transformer block
        for name, param in self.vit.named_parameters():
            if 'blocks.11' in name:
                param.requires_grad = True

        # Unfreeze classifier head
        for param in self.vit.get_classifier().parameters():
            param.requires_grad = True
            
    def repr(self, x):
        tokens = self.vit.forward_features(x)
        feats = self.vit.forward_head(tokens, pre_logits=True)
        return feats

    def forward(self, x):
        logits = self.vit(x)                
        return self.scaler(logits)   
    
class Cifar10LTVit(nn.Module):
    """Model for just classification.
    The architecture of our model is the same as standard DenseNet121
    """

    def __init__(self, temperature=1.0, num_labels=10):
        super(Cifar10LTVit, self).__init__()      
        self.scaler = ScaledLogits(temperature)
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_labels, in_chans=3)
        print(self.vit)
        
        # Freeze all parameters
        for param in self.vit.parameters():
            param.requires_grad = False

        # Unfreeze last transformer block
        for name, param in self.vit.named_parameters():
            if 'blocks.11' in name:
                param.requires_grad = True

        # Unfreeze classifier head
        for param in self.vit.get_classifier().parameters():
            param.requires_grad = True

    def forward(self, x):
        logits = self.vit(x)                
        return self.scaler(logits)        


class Cifar100ResNet50(nn.Module):
    """Model for just classification.
    The architecture of our model is the same as standard DenseNet121
    """

    def __init__(self, temperature=1.0, num_labels=100):
        super(Cifar100ResNet50, self).__init__()      
        self.scaler = ScaledLogits(temperature)
        self.resnet50 = models.resnet50(weights='IMAGENET1K_V2')
        self.num_features = self.resnet50.fc.in_features
        self.resnet50.layer4 = nn.Sequential(
            self.resnet50.layer4            
        )
        self.resnet50.fc = nn.Sequential(
                nn.Dropout(p=0.1),                                             
                nn.Linear(self.num_features, num_labels))
        self.n_classes = num_labels
        
        for param in self.resnet50.parameters():
            param.requires_grad = False        
        for name, param in self.resnet50.named_parameters():
            if 'layer4' in name or 'fc' in name: #if 'fc' in name: 
                param.requires_grad = True

    def repr(self, x):
        # See note [TorchScript super()]
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)

        x = self.resnet50.layer1(x)
        x = self.resnet50.layer2(x)
        x = self.resnet50.layer3(x)
        x = self.resnet50.layer4(x)

        x = self.resnet50.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        logits = self.resnet50(x)
        return self.scaler(logits)   
    
class Cifar100ConvNeXtTiny(nn.Module):
    """Model for classification based on torchvision ConvNeXt Tiny."""

    def __init__(self, temperature=1.0, num_labels=100):
        super(Cifar100ConvNeXtTiny, self).__init__()
        
        self.n_classes = num_labels
        self.scaler = ScaledLogits(temperature)

        # Pretrained ConvNeXt Tiny
        self.convnext_tiny = models.convnext_tiny(
            weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        )

        # In torchvision ConvNeXt Tiny, classifier is:
        # Sequential(LayerNorm2d/LayerNorm, Flatten(1), Linear(768, 1000))
        self.num_features = self.convnext_tiny.classifier[2].in_features

        # Replace classifier with your custom head
        self.convnext_tiny.classifier = nn.Sequential(
            self.convnext_tiny.classifier[0],   # keep pretrained norm
            self.convnext_tiny.classifier[1],   # keep flatten
            nn.Dropout(p=0.2),
            nn.Linear(self.num_features, num_labels)
        )        

        # Freeze everything
        for param in self.convnext_tiny.parameters():
            param.requires_grad = False

        # Unfreeze last feature stage + classifier
        # ConvNeXt Tiny features layout in torchvision:
        # features[0] = stem
        # features[1] = stage 1
        # features[2] = downsample
        # features[3] = stage 2
        # features[4] = downsample
        # features[5] = stage 3
        # features[6] = downsample
        # features[7] = stage 4 (last stage)
        for name, param in self.convnext_tiny.named_parameters():
            if "features.7" in name or "classifier" in name:
                param.requires_grad = True

    def repr(self, x):
        # Backbone features
        x = self.convnext_tiny.features(x)
        x = self.convnext_tiny.avgpool(x)

        # Stop before final Linear layer
        x = self.convnext_tiny.classifier[0](x)  # norm
        x = self.convnext_tiny.classifier[1](x)  # flatten

        return x

    def forward(self, x):
        logits = self.convnext_tiny(x)
        return self.scaler(logits)   
    
class Food101ResNet50(nn.Module):
    """Model for just classification.
    The architecture of our model is the same as standard DenseNet121
    """

    def __init__(self, temperature=1.0, num_labels=101):
        super(Food101ResNet50, self).__init__()      
        self.scaler = ScaledLogits(temperature)
        self.resnet50 = models.resnet50(weights='IMAGENET1K_V2')
        self.num_features = self.resnet50.fc.in_features
        self.resnet50.layer4 = nn.Sequential(
            self.resnet50.layer4            
        )
        self.resnet50.fc = nn.Sequential(
                nn.Dropout(p=0.1),                                             
                nn.Linear(self.num_features, num_labels))
        self.n_classes = num_labels
        
        for param in self.resnet50.parameters():
            param.requires_grad = False        
        for name, param in self.resnet50.named_parameters():
            if 'layer4' in name or 'fc' in name: #if 'fc' in name: 
                param.requires_grad = True

    def repr(self, x):
        # See note [TorchScript super()]
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)

        x = self.resnet50.layer1(x)
        x = self.resnet50.layer2(x)
        x = self.resnet50.layer3(x)
        x = self.resnet50.layer4(x)

        x = self.resnet50.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        logits = self.resnet50(x)
        return self.scaler(logits)     
    
    
class Cifar100ResNet152(nn.Module):
    """Model for just classification.
    The architecture of our model is the same as standard DenseNet121
    """

    def __init__(self, temperature=1.0, num_labels=100):
        super(Cifar100ResNet152, self).__init__()      
        self.scaler = ScaledLogits(temperature)
        self.resnet152 = models.resnet152(weights='IMAGENET1K_V2')
        self.num_features = self.resnet152.fc.in_features
        
        self.resnet152.fc = nn.Sequential(
                nn.Dropout(p=0.5),                
                nn.Linear(self.num_features, num_labels))
        self.n_classes = num_labels
        
        for param in self.resnet152.parameters():
            param.requires_grad = False        
        for name, param in self.resnet152.named_parameters():
            if 'layer4' in name or 'fc' in name: #
                param.requires_grad = True

    def repr(self, x):
        # See note [TorchScript super()]
        x = self.resnet152.conv1(x)
        x = self.resnet152.bn1(x)
        x = self.resnet152.relu(x)
        x = self.resnet152.maxpool(x)

        x = self.resnet152.layer1(x)
        x = self.resnet152.layer2(x)
        x = self.resnet152.layer3(x)
        x = self.resnet152.layer4(x)

        x = self.resnet152.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        logits = self.resnet152(x)
        return self.scaler(logits)
    
    
class Food101ResNet152(nn.Module):
    """Model for just classification.
    The architecture of our model is the same as standard DenseNet121
    """

    def __init__(self, temperature=1.0, num_labels=101):
        super(Food101ResNet152, self).__init__()      
        self.scaler = ScaledLogits(temperature)
        self.resnet152 = models.resnet152(weights='IMAGENET1K_V2')
        self.num_features = self.resnet152.fc.in_features
        
        self.resnet152.fc = nn.Sequential(
                nn.Dropout(p=0.5),                
                nn.Linear(self.num_features, num_labels))
        self.n_classes = num_labels
        
        for param in self.resnet152.parameters():
            param.requires_grad = False        
        for name, param in self.resnet152.named_parameters():
            if 'layer4' in name or 'fc' in name: #
                param.requires_grad = True

    def repr(self, x):
        # See note [TorchScript super()]
        x = self.resnet152.conv1(x)
        x = self.resnet152.bn1(x)
        x = self.resnet152.relu(x)
        x = self.resnet152.maxpool(x)

        x = self.resnet152.layer1(x)
        x = self.resnet152.layer2(x)
        x = self.resnet152.layer3(x)
        x = self.resnet152.layer4(x)

        x = self.resnet152.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        logits = self.resnet152(x)
        return self.scaler(logits)               
    
class Cifar100ResNet18(nn.Module):
    """Model for just classification.
    The architecture of our model is the same as standard DenseNet121
    """

    def __init__(self, temperature=1.0, num_labels=100):
        super(Cifar100ResNet18, self).__init__()      
        self.scaler = ScaledLogits(temperature)
        self.resnet18 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.num_features = self.resnet18.fc.in_features
        
        self.resnet18.fc = nn.Sequential(
                nn.Dropout(p=0.5),                
                nn.Linear(self.num_features, num_labels))
        self.n_classes = num_labels
        
        for param in self.resnet18.parameters():
            param.requires_grad = False        
        for name, param in self.resnet18.named_parameters():
            if 'layer4' in name or 'fc' in name: #
                param.requires_grad = True

    def repr(self, x):
        # See note [TorchScript super()]
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)

        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)

        x = self.resnet18.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        logits = self.resnet18(x)
        return self.scaler(logits)   
   
class Food101ResNet18(nn.Module):
    """Model for just classification.
    The architecture of our model is the same as standard DenseNet121
    """

    def __init__(self, temperature=1.0, num_labels=101):
        super(Food101ResNet18, self).__init__()      
        self.scaler = ScaledLogits(temperature)
        self.resnet18 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.num_features = self.resnet18.fc.in_features
        
        self.resnet18.fc = nn.Sequential(
                nn.Dropout(p=0.5),                
                nn.Linear(self.num_features, num_labels))
        self.n_classes = num_labels
        
        for param in self.resnet18.parameters():
            param.requires_grad = False        
        for name, param in self.resnet18.named_parameters():
            if 'layer4' in name or 'fc' in name: #
                param.requires_grad = True

    def repr(self, x):
        # See note [TorchScript super()]
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)

        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)

        x = self.resnet18.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        logits = self.resnet18(x)
        return self.scaler(logits)      
    
class Cifar100DenseNet121(nn.Module):
    """Model for just classification.
    The architecture of our model is the same as standard DenseNet121
    """

    def __init__(self, temperature=1.0, num_labels=100):
        super(Cifar100DenseNet121, self).__init__()      
        self.scaler = ScaledLogits(temperature)
        self.model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_labels)
        
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True
        for name, child in self.model.features.named_children():
            if name in ['denseblock4', 'norm5']:
                for param in child.parameters():
                    param.requires_grad = True

    def repr(self, x):
        # Extract features by passing through the 'features' module
        x = self.model.features(x)
        # Apply global average pooling
        x = nn.functional.relu(x, inplace=True)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        # Flatten the output to get the representation vector
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        logits = self.model(x)
        return self.scaler(logits)  
        
    
class Cifar100Vit(nn.Module):
    """Model for just classification.
    The architecture of our model is the same as standard DenseNet121
    """

    def __init__(self, temperature=1.0, num_labels=100):
        super(Cifar100Vit, self).__init__()              
        self.scaler = ScaledLogits(temperature)
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_labels, in_chans=3)
        print(self.vit)
        
        # Freeze all parameters
        for param in self.vit.parameters():
            param.requires_grad = False

        # Unfreeze last transformer block
        for name, param in self.vit.named_parameters():
            if 'blocks.11' in name:
                param.requires_grad = True

        # Unfreeze classifier head
        for param in self.vit.get_classifier().parameters():
            param.requires_grad = True
    
    def repr(self, x):
        """
        Extract feature representations (before classification head).
        Equivalent to the ResNet repr but for ViT.
        """
        # timm ViTs provide forward_features to get embeddings before head        
        tokens = self.vit.forward_features(x)
        feats = self.vit.forward_head(tokens, pre_logits=True)
        return feats        
    
    def forward(self, x):
        logits = self.vit(x)                
        return self.scaler(logits)   

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        if self.dropout:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = F.relu(self.fc1(x))
        if self.dropout:
            x = self.dropout(x)
        x = self.norm2(x)
        x = self.fc2(x)
        return x + residual  # Residual connection

class ResMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, depth=4, dropout=0.1):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(depth)])
        self.output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x_cat, x_num = x
        x = torch.cat([x_cat, x_num], dim=1) 
        x = F.relu(self.input_layer(x))
        x = self.blocks(x)
        return self.output_layer(x)    
    
class Wheather_FTT(nn.Module):
    """Model for just classification.
    The architecture of our model is the same as standard DenseNet121
    """
    def __init__(self, category_counts, numerical_features, temperature=1.0, num_labels=5):
        super(Wheather_FTT, self).__init__()      
        self.scaler = ScaledLogits(temperature)   
        # cont_mean_std = torch.randn(115, 2)     
        dim = 64
        self.ftt = FTTransformer( # 
            categories = category_counts,                               # tuple containing the number of unique values within each category (10, 5, 6, 5, 8)
            num_continuous = len(numerical_features),                   # number of continuous values
            dim = dim, #32                                               # dimension, paper set at 32
            dim_out = num_labels,                                       # binary prediction, but could be anything
            depth = 6, #6                                               # depth, paper recommended 6
            heads = 8, #8                                               # heads, paper recommends 8
            attn_dropout = 0.1,                                         # post-attention dropout
            ff_dropout = 0.1                                           # feed forward dropout
            #mlp_hidden_mults = (4, 2),                                 # relative multiples of each hidden dimension of the last mlp to logits
            #mlp_act = nn.ReLU()                                        # activation for final mlp, defaults to relu, but could be anything else (selu etc)
            #continuous_mean_std = cont_mean_std # (optional) - normalize the continuous values before layer norm
        )        
        # Replace the original classification head
        self.ftt.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, num_labels)
        )
        print(self.ftt)    

    def forward(self, x):
        x_cat, x_num = x
        logits = self.ftt(x_cat, x_num)                
        return self.scaler(logits)   

    def repr(self, x):
        """
        Return the penultimate representation, i.e. the feature vector
        right before the final classification Linear layer.
        """
        x_cat, x_num = x

        assert x_cat.shape[-1] == self.ftt.num_categories, (
            f'you must pass in {self.ftt.num_categories} values for your categories input'
        )
        assert x_num.shape[-1] == self.ftt.num_continuous, (
            f'you must pass in {self.ftt.num_continuous} values for your numerical input'
        )

        # categorical embeddings
        xs = []
        if self.ftt.num_unique_categories > 0:
            x_cat = x_cat + self.ftt.categories_offset
            x_cat = self.ftt.categorical_embeds(x_cat)
            xs.append(x_cat)

        # numerical embeddings
        if self.ftt.num_continuous > 0:
            x_num = self.ftt.numerical_embedder(x_num)
            xs.append(x_num)

        # concatenate tokens
        x = torch.cat(xs, dim=1)

        # prepend CLS token
        b = x.shape[0]
        cls_tokens = self.ftt.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # transformer
        x = self.ftt.transformer(x)

        # CLS representation
        x = x[:, 0]

        # penultimate representation = everything in to_logits except final Linear
        features = self.ftt.to_logits[:-1](x) # x 
        return features
    
class CovType_FTT(nn.Module):
    """Model for just classification.
    The architecture of our model is the same as standard DenseNet121
    """
    def __init__(self, category_counts, numerical_features, temperature=1.0, num_labels=10):
        super(CovType_FTT, self).__init__()      
        self.scaler = ScaledLogits(temperature)
        self.ftt = FTTransformer(
            categories = category_counts,                               # tuple containing the number of unique values within each category (10, 5, 6, 5, 8)
            num_continuous = len(numerical_features),                   # number of continuous values
            dim = 32,                                                   # dimension, paper set at 32
            dim_out = num_labels,                                       # binary prediction, but could be anything
            depth = 6,                                                  # depth, paper recommended 6
            heads = 8,                                                  # heads, paper recommends 8
            attn_dropout = 0.1,                                         # post-attention dropout
            ff_dropout = 0.1                                            # feed forward dropout
            #mlp_hidden_mults = (4, 2),                                 # relative multiples of each hidden dimension of the last mlp to logits
            #mlp_act = nn.ReLU()                                        # activation for final mlp, defaults to relu, but could be anything else (selu etc)
            #continuous_mean_std = cont_mean_std # (optional) - normalize the continuous values before layer norm
        )        
        print(self.ftt)

    def forward(self, x):
        x_cat, x_num = x
        logits = self.ftt(x_cat, x_num)                
        return self.scaler(logits)        
    
class Otto_FTT(nn.Module):
    """Model for just classification.
    The architecture of our model is the same as standard DenseNet121
    """
    def __init__(self, numerical_features, temperature=1.0, num_labels=9):
        super(Otto_FTT, self).__init__()      
        self.scaler = ScaledLogits(temperature)
        self.ftt = FTTransformer(
            categories = [],                                            # tuple containing the number of unique values within each category (10, 5, 6, 5, 8)
            num_continuous = numerical_features,                        # number of continuous values
            dim = 32,                                                   # dimension, paper set at 32
            dim_out = num_labels,                                       # binary prediction, but could be anything
            depth = 6,                                                  # depth, paper recommended 6
            heads = 8,                                                  # heads, paper recommends 8
            attn_dropout = 0.1,                                         # post-attention dropout
            ff_dropout = 0.1                                            # feed forward dropout
            #mlp_hidden_mults = (4, 2),                                 # relative multiples of each hidden dimension of the last mlp to logits
            #mlp_act = nn.ReLU()                                        # activation for final mlp, defaults to relu, but could be anything else (selu etc)
            #continuous_mean_std = cont_mean_std # (optional) - normalize the continuous values before layer norm
        )        
        print(self.ftt)

    def forward(self, x):
        x_categ, x_num = x        
        logits = self.ftt(x_categ, x_num)                
        return self.scaler(logits)        
    
class Cifar10OODArch(nn.Module):
    def __init__():
        super().__init__()
        pass
    
class Cifar100Arch(nn.Module):
    def __init__():
        super().__init__()
        pass
    
class Cifar100LongTailArch(nn.Module):
    def __init__():
        super().__init__()
        pass
    
class ImagenetArch(nn.Module):
    def __init__():
        super().__init__()
        pass
    
class ImagenetOODArch(nn.Module):
    def __init__():
        super().__init__()
        pass
    
class ImagenetLongTailArch(nn.Module):
    def __init__():
        super().__init__()
        pass
