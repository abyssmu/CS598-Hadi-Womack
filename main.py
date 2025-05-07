from pipeline import AlzheimerPipeline
from dataset import create_data_loaders
from model import AlzheimerCNN
from baselines import AlexNet3D, ResNet3D, ResNet3D50

import os
import torch

def create_pipeline(model_name='alzheimer_cnn', model_path=None, data_dir=None, 
                    output_dir='results', batch_size=4, num_workers=4, 
                    device=None, widening_factor=8, use_age=True):
    """
    Create a pipeline for testing and visualizing different CNN models for Alzheimer's detection.
    
    Parameters:
    -----------
    model_name : str
        Name of the model to use ('alzheimer_cnn', 'alexnet3d', 'resnet3d', 'resnet3d50')
    model_path : str or None
        Path to the model checkpoint file
    data_dir : str or None
        Directory containing the processed data files
    output_dir : str
        Directory to save results
    batch_size : int
        Batch size for the data loaders
    num_workers : int
        Number of workers for the data loaders
    device : torch.device or None
        Device to run the model on (CPU or GPU)
    widening_factor : int
        Widening factor for the AlzheimerCNN model
    use_age : bool
        Whether to use age information
    
    Returns:
    --------
    AlzheimerPipeline
        Pipeline object
    """
    # Set device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Create model based on the specified name
    if model_name.lower() == 'alzheimer_cnn':
        model = AlzheimerCNN(widening_factor=widening_factor, use_age=use_age)
        model_type = 'Custom Architecture'
    elif model_name.lower() == 'alexnet3d':
        model = AlexNet3D(use_age=use_age)
        model_type = 'AlexNet (3D)'
    elif model_name.lower() == 'resnet3d':
        model = ResNet3D(use_age=use_age)
        model_type = 'ResNet-18 (3D)'
    elif model_name.lower() == 'resnet3d50':
        model = ResNet3D50(use_age=use_age)
        model_type = 'ResNet-50 (3D)'
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    print(f"Created {model_type} model")
    
    # Load model if path provided
    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {model_path} (epoch {checkpoint.get('epoch', 'unknown')})")
    
    # Update output directory to include model name
    output_dir = os.path.join(output_dir, model_name)
    
    # Create pipeline
    pipeline = AlzheimerPipeline(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        output_dir=output_dir
    )
    
    return pipeline

def main():
    # Model settings
    DATA_DIR = 'processed_data'
    batch_size = 4
    num_workers = 2
    widening_factor = 8
    use_age = True
    num_epochs = 10
    learning_rate = 0.001
    group_samples = 300
    
    # Models to compare
    models = ['alzheimer_cnn', 'alexnet3d', 'resnet3d']
    
    # Base output directory
    base_output_dir = f'comparison_results_epochs-{num_epochs}_num_samples-{group_samples}'
    data_dir = f'{DATA_DIR}_{group_samples}/'
    
    # Run comparison for each model
    for model_name in models:
        print(f"\n{'='*50}")
        print(f"Running pipeline for model: {model_name}")
        print(f"{'='*50}\n")
        
        # Create output directory for this model
        output_dir = os.path.join(base_output_dir, model_name)
        
        # Create pipeline
        pipeline = create_pipeline(
            model_name=model_name,
            data_dir=data_dir,
            output_dir=output_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            widening_factor=widening_factor,
            use_age=use_age
        )
        
        # Train the model
        pipeline.train(num_epochs=num_epochs, lr=learning_rate, weight_decay=1e-4, patience=10)
        
        # Test the model
        pipeline.test()
        
        # Run inference pipeline
        pipeline.run_inference_pipeline(pipeline.test_loader, num_samples=10)
        
        # Analyze age effects
        pipeline.analyze_age_effects(pipeline.test_loader, num_age_bins=5)
        
        # Export metrics
        pipeline.export_metrics()
        
        print(f"\nPipeline for {model_name} completed successfully!")
    
    print("\nAll model comparisons completed!")

if __name__ == "__main__":
    main()