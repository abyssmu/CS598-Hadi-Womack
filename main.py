from pipeline import AlzheimerPipeline
from dataset import create_data_loaders, DATA_DIR
from model import AlzheimerCNN

import os
import torch

def create_pipeline(model_path=None, data_dir=DATA_DIR, output_dir='results',
				   batch_size=4, num_workers=4, device=None, widening_factor=8, use_age=True):
	"""
	Create a pipeline for testing and visualizing the Alzheimer's CNN model.
	
	Parameters:
	-----------
	model_path : str or None
		Path to the model checkpoint file
	data_dir : str
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
		Widening factor for the model
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
	
	# Create model
	model = AlzheimerCNN(widening_factor=widening_factor, use_age=use_age)
	
	# Load model if path provided
	if model_path and os.path.exists(model_path):
		checkpoint = torch.load(model_path, map_location=device)
		model.load_state_dict(checkpoint['model_state_dict'])
		print(f"Loaded model from {model_path} (epoch {checkpoint.get('epoch', 'unknown')})")
	
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

if __name__ == "__main__":
	output_dir = 'results'
	batch_size = 4
	num_workers = 2
	widening_factor = 8
	use_age = True
	num_epochs = 10
	learning_rate = 0.001

	# Create pipeline
	pipeline = create_pipeline(
		data_dir=DATA_DIR,
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

	# Visualize saliency maps
	#pipeline.visualize_saliency_maps(num_samples=3)

	# Run inference pipeline
	pipeline.run_inference_pipeline(pipeline.test_loader, num_samples=10)

	# Analyze age effects
	pipeline.analyze_age_effects(pipeline.test_loader, num_age_bins=5)

	# Export metrics
	pipeline.export_metrics()

	print("Pipeline completed successfully!")