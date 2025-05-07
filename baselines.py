import torch
import torch.nn as nn
import math
from model import AgeEncoding

class AlexNet3D(nn.Module):
    """
    3D adaptation of AlexNet for Alzheimer's disease detection.
    Modified to work with 3D brain MRI scans and include age information.
    
    The original AlexNet architecture has been adapted to work with 3D inputs
    and the specific requirements of the Alzheimer's disease classification task.
    """
    def __init__(self, use_age=True):
        """
        Initialize the AlexNet3D model.
        
        Parameters:
        -----------
        use_age : bool
            Whether to incorporate age information
        """
        super(AlexNet3D, self).__init__()
        
        self.use_age = use_age
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Conv1
            nn.Conv3d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            
            # Conv2
            nn.Conv3d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            
            # Conv3
            nn.Conv3d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv4
            nn.Conv3d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv5
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
        )
        
        # Age encoding if enabled
        if use_age:
            self.age_encoder = AgeEncoding(d_model=512)
            
        # Calculate the feature size dynamically
        dummy_input = torch.zeros(1, 1, 96, 96, 96)
        self.feature_size = self._calculate_output_shape(dummy_input)
        
        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.feature_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
        )
        
        # Final classification layer
        self.final_layer = nn.Linear(1024, 3)  # 3 classes: CN, MCI, AD
        
        print(f"Feature size after convolutions: {self.feature_size}")
        
        # Initialize weights using He initialization
        self._initialize_weights()
    
    def _calculate_output_shape(self, x):
        """
        Calculate the output shape of the convolutional layers.
        
        Parameters:
        -----------
        x : torch.Tensor
            Dummy input of shape (batch_size, channels, height, width, depth)
        
        Returns:
        --------
        int
            Total number of features after flattening
        """
        x = self.features(x)
        return x.numel()
    
    def _initialize_weights(self):
        """
        Initialize the weights of the network using He initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, age=None):
        """
        Forward pass through the network.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input 3D brain scan tensor of shape (batch_size, 1, 96, 96, 96)
        age : torch.Tensor or None
            Age values tensor of shape (batch_size,), required if use_age=True
        
        Returns:
        --------
        torch.Tensor
            Output logits of shape (batch_size, 3)
        """
        # Ensure input has channel dimension
        if x.dim() == 4:
            x = x.unsqueeze(1)  # Add channel dimension if not present
        
        # Forward through feature extraction layers
        x = self.features(x)
        
        # Flatten features
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # Forward through classifier
        x = self.classifier(x)
        
        # Add age information if enabled
        if self.use_age and age is not None:
            age_encoding = self.age_encoder(age)
            x = x + age_encoding
        
        # Final classification
        x = self.final_layer(x)
        
        return x


class ResidualBlock3D(nn.Module):
    """
    3D Residual Block for ResNet3D
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.InstanceNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm3d(out_channels)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNet3D(nn.Module):
    """
    3D adaptation of ResNet for Alzheimer's disease detection.
    Designed to work with 3D brain MRI scans and include age information.
    
    This implementation is based on the ResNet architecture but adapted 
    for 3D inputs and the specific requirements of Alzheimer's disease classification.
    """
    def __init__(self, block=ResidualBlock3D, layers=[2, 2, 2, 2], use_age=True):
        """
        Initialize the ResNet3D model.
        
        Parameters:
        -----------
        block : nn.Module
            Residual block module
        layers : list
            Number of blocks in each layer
        use_age : bool
            Whether to incorporate age information
        """
        super(ResNet3D, self).__init__()
        
        self.use_age = use_age
        self.in_channels = 64
        
        # Initial layers
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.InstanceNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Average pooling
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Age encoding if enabled
        if use_age:
            self.age_encoder = AgeEncoding(d_model=512)
        
        # Calculate the feature size dynamically
        dummy_input = torch.zeros(1, 1, 96, 96, 96)
        self.feature_size = self._calculate_output_shape(dummy_input)
        
        # Fully connected layer
        self.fc = nn.Linear(self.feature_size, 1024)
        self.final_layer = nn.Linear(1024, 3)  # 3 classes: CN, MCI, AD
        
        print(f"Feature size after convolutions: {self.feature_size}")
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        """
        Create a residual layer with the specified number of blocks.
        
        Parameters:
        -----------
        block : nn.Module
            Residual block module
        out_channels : int
            Number of output channels
        blocks : int
            Number of residual blocks in the layer
        stride : int
            Stride for the first block
        
        Returns:
        --------
        nn.Sequential
            Residual layer
        """
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.InstanceNorm3d(out_channels)
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _calculate_output_shape(self, x):
        """
        Calculate the output shape after the convolutional layers.
        
        Parameters:
        -----------
        x : torch.Tensor
            Dummy input of shape (batch_size, channels, height, width, depth)
        
        Returns:
        --------
        int
            Total number of features after flattening
        """
        # Forward through the feature extraction layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        return x.size(1)
    
    def _initialize_weights(self):
        """
        Initialize the weights of the network using He initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm3d):
                # InstanceNorm3d may not have learnable parameters (weight and bias could be None)
                # Only initialize if they exist
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, age=None):
        """
        Forward pass through the network.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input 3D brain scan tensor of shape (batch_size, 1, 96, 96, 96)
        age : torch.Tensor or None
            Age values tensor of shape (batch_size,), required if use_age=True
        
        Returns:
        --------
        torch.Tensor
            Output logits of shape (batch_size, 3)
        """
        # Ensure input has channel dimension
        if x.dim() == 4:
            x = x.unsqueeze(1)  # Add channel dimension if not present
        
        # Forward through initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Forward through residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling
        x = self.avgpool(x)
        
        # Flatten features
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # Forward through fully connected layer
        x = self.fc(x)
        
        # Add age information if enabled
        if self.use_age and age is not None:
            age_encoding = self.age_encoder(age)
            x = x + age_encoding
        
        # Final classification
        x = self.final_layer(x)
        
        return x


# Create a deeper ResNet with more layers
def ResNet3D50(use_age=True):
    """
    Creates a ResNet-50 like model with 3D convolutions for Alzheimer's disease detection.
    
    Parameters:
    -----------
    use_age : bool
        Whether to incorporate age information
    
    Returns:
    --------
    ResNet3D
        Initialized ResNet3D model
    """
    return ResNet3D(ResidualBlock3D, [3, 4, 6, 3], use_age)