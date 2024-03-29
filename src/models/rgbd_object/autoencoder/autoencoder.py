import torch
import torch.nn as nn
import torch.nn.functional as F


class TestAutoencoder(nn.Module) :
    """
    Autoencoder neural network
    Inspired by: https://www.researchgate.net/publication/369855548/figure/fig2/AS:11431281139291982@1680837293399/Architecture-of-Convolutional-Autoencoder.ppm
    """

    def __init__(self):
        super().__init__()

        # Encoder
        self.pool = nn.MaxPool2d(2, 2) # [(input_width - 2) / 2 + 1, (input_height - 2) / 2 + 1, input_depth]

        self.conv1_1 = nn.Conv2d(3, 32, 7, 1, "same")    # Conv -> [256, 256, 32]
        self.conv1_2 = nn.Conv2d(32, 32, 7, 1, "same")   # Conv -> [256, 256, 32]
                                                         # Pool -> [128, 128, 32]
        self.conv2_1 = nn.Conv2d(32, 64, 5, 1, "same")   # Conv -> [128, 128, 64]
        self.conv2_2 = nn.Conv2d(64, 128, 5, 1, "same")  # Conv -> [128, 128, 128]
                                                         # Pool -> [64, 64, 128]
        self.conv3_1 = nn.Conv2d(128, 128, 3, 1, "same") # Conv -> [64, 64, 128]
        self.conv3_2 = nn.Conv2d(128, 128, 3, 1, "same") # Conv -> [64, 64, 128]
                                                         # Pool -> [32, 32, 128]
        self.conv4_1 = nn.Conv2d(128, 128, 3, 1, "same") # Conv -> [32, 32, 128]
        self.conv4_2 = nn.Conv2d(128, 128, 3, 1, "same") # Conv -> [32, 32, 128]
                                                         # Pool -> [16, 16, 128]
        self.fc_1 = nn.Linear(16 * 16 * 128, 512)        # FC   -> [512]
        self.fc_2 = nn.Linear(512, 256)                  # FC   -> [256]
        self.fc_3 = nn.Linear(256, 16 * 16 * 128)        # FC   -> [16 * 16 * 128]
        
        # Decoder
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest") # https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html
                                                          # Upsamp -> [32, 32, 128]
        self.tconv1_1 = nn.Conv2d(128, 128, 3, 1, "same") # T Conv -> [32, 32, 128]
        self.tconv1_2 = nn.Conv2d(128, 128, 3, 1, "same") # T Conv -> [32, 32, 128]
                                                          # Upsamp -> [64, 64, 128]
        self.tconv2_1 = nn.Conv2d(128, 128, 3, 1, "same") # T Conv -> [64, 64, 128]
        self.tconv2_2 = nn.Conv2d(128, 128, 3, 1, "same") # T Conv -> [64, 64, 128]
                                                          # Upsamp -> [128, 128, 128]
        self.tconv3_1 = nn.Conv2d(128, 64, 5, 1, "same")  # T Conv -> [128, 128, 64]
        self.tconv3_2 = nn.Conv2d(64, 32, 5, 1, "same")   # T Conv -> [128, 128, 32]
                                                          # Upsamp -> [256, 256, 32]
        self.tconv4_1 = nn.Conv2d(32, 32, 7, 1, "same")   # T Conv -> [256, 256, 32]
        self.tconv4_2 = nn.Conv2d(32, 3, 7, 1, "same")    # T Conv -> [256, 256, 16]

        # Batch normalisation
        self.batchnorm32 = nn.BatchNorm2d(32)
        self.batchnorm128 = nn.BatchNorm2d(128)
        self.batchnorm256 = nn.BatchNorm2d(256)

    def forward(self, x):
        # Encoder
        # print("Conv 1")
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))

        # print("Pool 1")
        x = self.pool(x)

        # print("Batch Norm 32")
        x = self.batchnorm32(x)

        # print("Conv 2")
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))

        # print("Pool 2")
        x = self.pool(x)

        # print("Batch Norm 128")
        x = self.batchnorm128(x)

        # print("Conv 3")
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))    

        # print("Pool 3")
        x = self.pool(x)

        # print("Batch Norm 128")
        x = self.batchnorm128(x)

        # print("Conv 4")
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))    

        # print("Pool 4")
        x = self.pool(x)

        # print("Batch Norm 128")
        x = self.batchnorm128(x)

        # print("Flatten")
        x = torch.flatten(x, 1)

        # print("Fully Connected")
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        encoded = x.detach().clone()
        x = F.relu(self.fc_3(x))

        # Decoder
        x = torch.reshape(x, (-1, 128, 16, 16))

        # print("Upsample 1")
        x = self.upsample(x)

        # print("Trans Conv 1")
        x = F.relu(self.tconv1_1(x))
        x = F.relu(self.tconv1_2(x))

        # print("Upsample 2")
        x = self.upsample(x)

        # print("Batch Norm 256")
        x = self.batchnorm128(x)

        # print("Trans Conv 2")
        x = F.relu(self.tconv2_1(x))
        x = F.relu(self.tconv2_2(x))

        # print("Upsample 3")
        x = self.upsample(x)

        # print("Batch Norm 64")
        x = self.batchnorm128(x)

        # print("Trans Conv 3")
        x = F.relu(self.tconv3_1(x))
        x = F.relu(self.tconv3_2(x))

        # print("Upsample 4")
        x = self.upsample(x)

        # print("Batch Norm 32")
        x = self.batchnorm32(x)

        # print("Trans Conv 4")
        x = F.relu(self.tconv4_1(x))
        decoded = F.relu(self.tconv4_2(x))

        return encoded, decoded


class TestAutoencoder_skip(nn.Module) :
    """
    Autoencoder neural network
    Inspired by: https://www.researchgate.net/publication/369855548/figure/fig2/AS:11431281139291982@1680837293399/Architecture-of-Convolutional-Autoencoder.ppm
    """

    def __init__(self):
        super().__init__()

        # Encoder
        self.pool = nn.MaxPool2d(2, 2) # [(input_width - 2) / 2 + 1, (input_height - 2) / 2 + 1, input_depth]

        self.conv1_1 = nn.Conv2d(3, 32, 7, 1, 3)    
        self.conv1_2 = nn.Conv2d(32, 32, 7, 1, 3)   
                                                         
        self.conv2_1 = nn.Conv2d(32, 64, 5, 1, 2)
        self.conv2_2 = nn.Conv2d(64, 128, 5, 1, 2) 
                                                
        self.conv3_1 = nn.Conv2d(128, 128, 3, 1, 1) 
        self.conv3_2 = nn.Conv2d(128, 128, 3, 1, 1) 
                                                        
        self.conv4_1 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv4_2 = nn.Conv2d(128, 128, 3, 1, 1) 
                                                        
        #self.fc_1 = nn.Linear(8 * 8 * 128, 512)
        self.fc_1 = nn.Linear(32768, 512)
        self.fc_2 = nn.Linear(512, 256)         
        self.fc_3 = nn.Linear(256, 32768)          
        
        # Decoder
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest") # https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html

        self.tconv1_1 = nn.Conv2d(128, 128, 3, 1, 1) 
        self.tconv1_2 = nn.Conv2d(128, 128, 3, 1, 1) 
                                                    
        self.tconv2_1 = nn.Conv2d(128, 128, 3, 1, 1)  
        self.tconv2_2 = nn.Conv2d(128, 128, 3, 1, 1)   

        self.tconv3_1 = nn.Conv2d(128, 64, 5, 1, 2)   
        self.tconv3_2 = nn.Conv2d(64, 32, 5, 1, 2)  
                                                    
        self.tconv4_1 = nn.Conv2d(32, 32, 7, 1, 3)   
        self.tconv4_2 = nn.Conv2d(32, 3, 7, 1, 3)   
                                                          
        #self.tconv5_1 = nn.Conv2d(32, 3, 3, 1, "same")    # T Conv -> [128, 128, 3]

        # Batch normalisation
        self.batchnorm32 = nn.BatchNorm2d(32)
        self.batchnorm64 = nn.BatchNorm2d(64)
        self.batchnorm128 = nn.BatchNorm2d(128)
        self.batchnorm256 = nn.BatchNorm2d(256)

    def forward(self, x):
        # Encoder
        # print("Conv 1")
        x = F.relu(self.conv1_1(x))
        x1 = F.relu(self.conv1_2(x)) # skip-1

        # print("Pool 1")
        x = self.pool(x1)

        # print("Batch Norm 32")
        x = self.batchnorm32(x)

        # print("Conv 2")
        x = F.relu(self.conv2_1(x))
        x2 = F.relu(self.conv2_2(x)) # skip-2

        # print("Pool 2")
        x = self.pool(x2)

        # print("Batch Norm 128")
        x = self.batchnorm128(x)

        # print("Conv 3")
        x = F.relu(self.conv3_1(x))
        x3 = F.relu(self.conv3_2(x)) # skip-3

        # print("Pool 3")
        x = self.pool(x3)

        # print("Batch Norm 128")
        x = self.batchnorm128(x)

        # print("Conv 4")
        x = F.relu(self.conv4_1(x))
        x4 = F.relu(self.conv4_2(x))    

        # print("Pool 4")
        x = self.pool(x4)

        # print("Batch Norm 128")
        x = self.batchnorm128(x)

        # print("Flatten")
        x = torch.flatten(x, 1)

        # print("Fully Connected")
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        encoded = x
        x = F.relu(self.fc_3(x))

        # Decoder
        # print("Reshape")
        #x = torch.reshape(encoded, (-1, 256, 4, 4))
        x = torch.reshape(x, (-1, 128, 16, 16))

        # print("Upsample 1")
        x = self.upsample(x)

        # print("Trans Conv 1")
        x = F.relu(self.tconv1_1(x))
        x = F.relu(self.tconv1_2(x))

        # print("Upsample 2")
        x = self.upsample(x)

        # print("Batch Norm 128")
        x = self.batchnorm128(x)

        # print("Trans Conv 2")
        x = F.relu(self.tconv2_1(x)) + x3 # skip-3 added here
        x = F.relu(self.tconv2_2(x))

        # print("Upsample 3")
        x = self.upsample(x)

        # print("Batch Norm 128")
        x = self.batchnorm128(x)

        # print("Trans Conv 3")
        x = F.relu(self.tconv3_1(x)) + x2 # skip-2 added here
        x = F.relu(self.tconv3_2(x))

        # print("Upsample 4")
        #x = self.upsample(x)

        # print("Batch Norm 32")
        x = self.batchnorm32(x)

        # print("Trans Conv 4")
        x = F.relu(self.tconv4_1(x)) + x1 # skip-1 added here 
        x = F.relu(self.tconv4_2(x))

        # print("Upsample 5")
        x = self.upsample(x)

        # print("Trans Conv 5")
        #decoded = F.relu(self.tconv5_1(x))
        decoded = F.relu(x)

        return encoded, decoded
