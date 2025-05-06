import torch
import torch.nn as nn
import math

class AgeEncoding(nn.Module):
    """
    Age encoding class inspired by positional encoding in the transformer model.
    Encodes age values into vectors that can be integrated with CNN features.
    
    The encoding is defined as:
    AE(age, 2i) = sin(age/10000^(2i/d_model))
    AE(age, 2i+1) = cos(age/10000^(2i/d_model))
    
    Where:
    - age is the patient's age value (rounded to 0.5 decimal places)
    - i is the dimension index
    - d_model is the size of the encoding
    """
    def __init__(self, d_model=512, age_min=0, age_max=120, age_step=0.5):
        """
        Initialize the age encoding module.
        
        Parameters:
        -----------
        d_model : int
            Size of the age encoding vector
        age_min : float
            Minimum possible age value
        age_max : float
            Maximum possible age value
        age_step : float
            Step size for discretizing age values
        """
        super(AgeEncoding, self).__init__()
        
        self.d_model = d_model
        self.age_min = age_min
        self.age_max = age_max
        self.age_step = age_step
        
        # Create encoder layers to transform age encoding to match visual representation
        self.encoder = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.LayerNorm(512),
            nn.Linear(512, 1024)
        )
        
        # Pre-compute all possible age encodings
        self.num_ages = int((age_max - age_min) / age_step) + 1
        self.register_buffer('age_encodings', self._create_age_encodings())
    
    def _create_age_encodings(self):
        """
        Pre-compute all possible age encodings.
        
        Returns:
        --------
        torch.Tensor
            Tensor of shape (num_ages, d_model) containing all possible age encodings
        """
        # Create all possible age values
        ages = torch.arange(self.age_min, self.age_max + self.age_step, self.age_step)
        
        # Create position encodings for all ages
        position_encodings = torch.zeros(len(ages), self.d_model)
        
        # Apply sinusoidal encoding
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        
        for idx, age in enumerate(ages):
            position_encodings[idx, 0::2] = torch.sin(age * div_term)
            position_encodings[idx, 1::2] = torch.cos(age * div_term)
        
        return position_encodings
    
    def forward(self, age):
        """
        Encode age values into vectors.
        
        Parameters:
        -----------
        age : torch.Tensor
            Tensor of age values of shape (batch_size,)
        
        Returns:
        --------
        torch.Tensor
            Encoded age vectors of shape (batch_size, 1024)
        """
        # Discretize age values to the nearest step
        age_indices = ((age - self.age_min) / self.age_step).round().long()
        
        # Clamp indices to valid range
        age_indices = torch.clamp(age_indices, 0, self.num_ages - 1)
        
        # Get corresponding encodings
        encodings = self.age_encodings[age_indices]
        
        # Transform the encodings to match visual representation
        return self.encoder(encodings)

class AlzheimerCNN(nn.Module):
    """
    3D Convolutional Neural Network for Alzheimer's disease detection.
    Distinguishes between Cognitively Normal (CN), Mild Cognitive Impairment (MCI), 
    and Alzheimer's Disease (AD) using structural brain MRI scans.
    
    Key features of this architecture:
    - Instance normalization instead of batch normalization
    - Small-sized kernels in first layer to avoid early spatial downsampling
    - Wide architecture with large numbers of filters
    - Age information integration
    """
    def __init__(self, widening_factor=8, use_age=True):
        """
        Initialize the CNN model.
        
        Parameters:
        -----------
        widening_factor : int
            Factor to increase the number of filters in each layer
        use_age : bool
            Whether to incorporate age information
        """
        super(AlzheimerCNN, self).__init__()
        
        self.widening_factor = widening_factor
        self.use_age = use_age
        
        # Block 1: Small kernel size (1x1x1) to prevent early spatial downsampling
        self.block1 = nn.Sequential(
            nn.Conv3d(1, 4 * widening_factor, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.InstanceNorm3d(4 * widening_factor),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=3, stride=2)
        )
        
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv3d(4 * widening_factor, 32 * widening_factor, kernel_size=3, stride=1, padding=0, dilation=2),
            nn.InstanceNorm3d(32 * widening_factor),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=3, stride=2)
        )
        
        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv3d(32 * widening_factor, 64 * widening_factor, kernel_size=5, stride=1, padding=2, dilation=2),
            nn.InstanceNorm3d(64 * widening_factor),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=3, stride=2)
        )
        
        # Block 4
        self.block4 = nn.Sequential(
            nn.Conv3d(64 * widening_factor, 64 * widening_factor, kernel_size=3, stride=1, padding=1, dilation=2),
            nn.InstanceNorm3d(64 * widening_factor),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=5, stride=2)
        )
        
        # Age encoding if enabled
        if use_age:
            self.age_encoder = AgeEncoding(d_model=512)
        
        # Calculate the feature size dynamically
        dummy_input = torch.zeros(1, 1, 96, 96, 96)
        self.feature_size = self._calculate_output_shape(dummy_input)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.feature_size, 1024)
        self.fc2 = nn.Linear(1024, 3)  # 3 output classes: CN, MCI, AD
        
        print(f"Feature size after convolutions: {self.feature_size}")
    
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
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x.numel()
    
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
        
        # Forward through convolutional blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        # Flatten features
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # FC1
        x = self.fc1(x)
        
        # Add age information if enabled
        if self.use_age and age is not None:
            age_encoding = self.age_encoder(age)
            x = x + age_encoding
        
        # FC2 (Output layer)
        x = self.fc2(x)
        
        return x