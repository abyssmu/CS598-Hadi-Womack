import os
import zipfile
import io
import numpy as np
import torch
from torch.utils import data

class AlzheimerDataset(data.Dataset):
    """
    Dataset for loading Alzheimer's disease MRI scans from preprocessed zip files.
    The dataset handles loading scans from multiple zip files, where each file contains
    numpy arrays for brain scans along with metadata including patient group, ID, and age.
    """
    def __init__(self, data_dir, transform=None):
        """
        Initialize the dataset.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing the processed zip files
        transform : callable, optional
            Optional transform to be applied on a sample
        """
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        
        # Find all zip files in the directory
        zip_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
                    if f.endswith('.zip')]
        
        # Load sample information from all zip files
        for zip_path in zip_files:
            self._load_zip_index(zip_path)
            
        print(f"Loaded {len(self.samples)} samples from {len(zip_files)} zip files")
    
    def _load_zip_index(self, zip_path):
        """
        Load sample information from a zip file.
        
        Parameters:
        -----------
        zip_path : str
            Path to the zip file
        """
        try:
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                for filename in zipf.namelist():
                    if filename.endswith('.npz'):
                        # Parse filename to extract metadata
                        # Format: group-patient_id-image_id-age-sex.npz
                        parts = filename.split('.')[0].split('-')
                        
                        # Handle filenames with and without age/sex information
                        if len(parts) >= 3:
                            group = parts[0]
                            patient_id = parts[1]
                            image_id = parts[2]
                            
                            # Extract age if available (default to None)
                            age = None
                            if len(parts) >= 4 and parts[3].replace('.', '', 1).isdigit():
                                age = float(parts[3])
                            
                            # Extract sex if available (default to None)
                            sex = None
                            if len(parts) >= 5:
                                sex = parts[4]
                            
                            # Convert group to class index
                            label = {'CN': 0, 'MCI': 1, 'AD': 2}.get(group, -1)
                            
                            if label != -1:  # Valid group
                                self.samples.append({
                                    'zip_path': zip_path,
                                    'filename': filename,
                                    'group': group,
                                    'label': label,
                                    'patient_id': patient_id,
                                    'image_id': image_id,
                                    'age': age,
                                    'sex': sex
                                })
        except Exception as e:
            print(f"Error loading zip file {zip_path}: {e}")
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Parameters:
        -----------
        idx : int
            Index of the sample
        
        Returns:
        --------
        dict
            Dictionary containing the sample data and metadata
        """
        sample_info = self.samples[idx]
        
        # Load the actual data from the zip file
        with zipfile.ZipFile(sample_info['zip_path'], 'r') as zipf:
            with zipf.open(sample_info['filename']) as f:
                buffer = io.BytesIO(f.read())
                data_npz = np.load(buffer)
                scan_data = data_npz['data']
        
        # Convert to tensor
        scan_tensor = torch.from_numpy(scan_data).float()
        
        # Add channel dimension if missing
        if scan_tensor.dim() == 3:
            scan_tensor = scan_tensor.unsqueeze(0)
        
        # Apply transforms if any
        if self.transform:
            scan_tensor = self.transform(scan_tensor)
        
        # Create result dictionary
        result = {
            'scan': scan_tensor,
            'label': sample_info['label'],
            'group': sample_info['group'],
            'patient_id': sample_info['patient_id'],
            'image_id': sample_info['image_id']
        }
        
        # Add age if available
        if sample_info['age'] is not None:
            result['age'] = torch.tensor(sample_info['age']).float()
        
        # Add sex if available
        if sample_info['sex'] is not None:
            # Convert sex to numeric: 0 for male (M), 1 for female (F)
            sex_value = 1 if sample_info['sex'].upper() == 'F' else 0
            result['sex'] = torch.tensor(sex_value).long()
        
        return result

def create_data_loaders(data_dir, batch_size=4, num_workers=4, 
                       train_val_test_split=(0.7, 0.15, 0.15), transform=None, 
                       shuffle=True, random_seed=42):
    """
    Create data loaders for training, validation, and testing.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the processed zip files
    batch_size : int
        Batch size for the data loaders
    num_workers : int
        Number of workers for the data loaders
    train_val_test_split : tuple
        Proportions of training, validation, and test sets
    transform : callable, optional
        Optional transform to be applied on the samples
    shuffle : bool
        Whether to shuffle the datasets
    random_seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    tuple
        (train_loader, val_loader, test_loader)
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Load the full dataset
    full_dataset = AlzheimerDataset(data_dir=data_dir, transform=transform)
    
    # Calculate split sizes
    dataset_size = len(full_dataset)
    train_size = int(dataset_size * train_val_test_split[0])
    val_size = int(dataset_size * train_val_test_split[1])
    test_size = dataset_size - train_size - val_size
    
    # Split the dataset
    indices = list(range(dataset_size))
    if shuffle:
        np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create samplers
    train_sampler = data.SubsetRandomSampler(train_indices)
    val_sampler = data.SubsetRandomSampler(val_indices)
    test_sampler = data.SubsetRandomSampler(test_indices)
    
    # Create data loaders
    train_loader = data.DataLoader(
        full_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers
    )
    
    val_loader = data.DataLoader(
        full_dataset, batch_size=batch_size, sampler=val_sampler,
        num_workers=num_workers
    )
    
    test_loader = data.DataLoader(
        full_dataset, batch_size=batch_size, sampler=test_sampler,
        num_workers=num_workers
    )
    
    print(f"Created data loaders with {train_size} training, {val_size} validation, "
          f"and {test_size} test samples")
    
    return train_loader, val_loader, test_loader