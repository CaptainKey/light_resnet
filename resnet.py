import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from imagenet import classes
from torchsummary import summary
import logging as log
# Bloc ResNet

log.basicConfig(filename='debug.log',format='%(levelname)s : %(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',level=log.INFO,filemode='w')

class ResNetBlock(nn.Module):
    # Initialisation
    def __init__(self, in_depth, out_depth, stride=1):
        # Le block Resnet prend en entree un tenseur de taille [BATCH, in_depth, Y0, X0] et retourne un tenseur de taille [BATCH, out_depth, Y1, X1]
        super(ResNetBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_depth, out_depth, kernel_size=3,stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_depth)

        self.conv2 = nn.Conv2d(out_depth, out_depth, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_depth)

        self.in_depth = in_depth
        self.out_depth = out_depth

        self.downsampling = False

        # Si les dimensions d entree et de sortie de sont pas egales
        # ex : in_depth = 64 | out_depth = 128
        if self.in_depth != self.out_depth:
            self.downsampling = nn.Sequential(
                                    nn.Conv2d(in_depth, out_depth, kernel_size=1, stride=2, bias=False),
                                    nn.BatchNorm2d(out_depth)
                            )

    def forward(self, x):
        log.info('BEGIN RESNET BLOCK')
        residu = x
        log.info('residu : {} '.format(residu.shape))

        x = F.relu(self.bn1(self.conv1(x)))
        log.info('CONV1/BN1/RELU : {}'.format(x.shape))

        x = self.bn2(self.conv2(x))
        log.info('CONV2/BN2 :  {}'.format(x.shape))


        if self.downsampling:
            # Si downsampling != False
            # Donc si la dimension la dimension en sortie != dimension en entree 
            log.info('DIMENSION RESIDU != X : {} != {}'.format(residu.shape,x.shape))
            residu = self.downsampling(residu)
            log.info('NOUVELLE DIMENSION RESIDU :  {}'.format(residu.shape))

        x += residu
        log.info('AJOUT DU RESIDU {}'.format(x.shape))
        x = F.relu(x)

        log.info('END RESNET BLOCK\n\n')

        return x


class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()

        self.current_depth = 64

      
        self.conv1 = nn.Conv2d(3, self.current_depth, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 =  nn.BatchNorm2d(self.current_depth)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.resnetblock1 = self.make_layer(64)
        self.resnetblock2 = self.make_layer(128, stride=2)
        self.resnetblock3 = self.make_layer(256, stride=2)
        self.resnetblock4 = self.make_layer(512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(512, 1000)



    def make_layer(self, depth, stride=1):
        layers = []

        layers.append(ResNetBlock(self.current_depth, depth, stride))

        # On met à jour la dimension 
        self.current_depth = depth 
        
        layers.append(ResNetBlock(self.current_depth, depth)) 
        layers.append(ResNetBlock(self.current_depth, depth)) 

        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        log.info('CONV1/BN1/RELU : {}'.format(x.shape))
        x = self.maxpool(x)
        log.info('MAXPOOL :  {}\n\n'.format(x.shape))

        x = self.resnetblock1(x)
        log.info('resnetblock1 :  {}'.format(x.shape))

        x = self.resnetblock2(x)
        log.info('resnetblock1 :  {}'.format(x.shape))

        x = self.resnetblock3(x)
        log.info('resnetblock3 :  {}'.format(x.shape))

        x = self.resnetblock4(x)
        log.info('resnetblock4 : {}'.format(x.shape))

        x = self.avgpool(x)
        log.info('avgpool :  {}'.format(x.shape))

        x = torch.flatten(x, 1)
        log.info('flatten :  {}'.format(x.shape))

        x = self.fc(x)
        log.info('fc : {}'.format(x.shape))

        return x



# Instance de Resnet
net = ResNet()

# Utilisation du package torchsummary pour avoir un resume de l inference du reseau
summary(net, input_size=(3, 224, 224))

# Creation d un tenseur random de dimension [1, 3, 224, 224]
test_tensor = torch.rand(1,3,224,224)

# Inference du model
out = net(test_tensor)

# Recupération des 5 classes avec le score le plus haut
values,indices = torch.topk(out.data,5)

# Conversion des tensors en numpy array
top = indices.numpy()[0]
scores = values.numpy()[0]

log.info("##############################")
log.info("#########PREDICTION###########")
log.info("##############################")
for i,data in enumerate(zip(top,scores)):
    classe, score = data
    # Affichage de la classe
    log.info("Top {} => {} ".format(i+1,classes[classe]))

