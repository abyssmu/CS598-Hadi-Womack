import zipfile
import os
import random
import nibabel as nib
import numpy as np
import io
import pandas as pd
from nilearn import image, datasets
from scipy.ndimage import zoom, gaussian_filter
import multiprocessing as mp

# Extract all the files with '.nii' extension from a zip file using the zipfile module
# Add on the original zip file name to each file name to keep track of where they came from
def extract_file_names(zip_filename):
	nii_files = []
	try:
		with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
			for file_info in zip_ref.infolist():
				if file_info.filename.endswith('.nii'):
					# Append the original zip file name to the nii file name
					nii_files.append(f"{zip_filename}:{file_info.filename}")
	except zipfile.BadZipFile:
		print(f"Error: {zip_filename} is not a valid zip file.")
	return nii_files

# Get all the zip files in the current directory
def get_zip_files():
	zip_files = []
	for file in os.listdir():
		if file.endswith('.zip'):
			zip_files.append(file)
	return zip_files

# Extract all the filenames with the given group name in the file name from a list of zip files
def extract_group_files(zip_files, group_name):
	group_files = []
	for f in zip_files:
		if group_name in f:
			group_files.append(f)
	return group_files

# Create a 2D list of all the '.nii' files in the zip files that match the group name
def create_grouped_file_list():
	group_names = ['AD', 'CN', 'MCI']
	zip_files = get_zip_files()
	grouped_files = []

	for group_name in group_names:
		group_files = extract_group_files(zip_files, group_name)
		grouped_nii_files = []
		for zip_file in group_files:
			nii_files = extract_file_names(zip_file)
			grouped_nii_files.extend(nii_files)
		grouped_files.append(grouped_nii_files)

	return grouped_files

# Create a random subset of files of size n
def create_random_subset(file_list, n):
	if len(file_list) <= n:
		return file_list
	else:
		return random.sample(file_list, n)

# Collect all the files and put them into their respective groups, then create random subsets
def collect_and_subset_files(n):
	AD, CN, MCI = create_grouped_file_list()

	AD_subset = create_random_subset(AD, n)
	CN_subset = create_random_subset(CN, n)
	MCI_subset = create_random_subset(MCI, n)

	return AD_subset, CN_subset, MCI_subset

class Patient:
	def __init__(self):
		self.patient_id = None
		self.image_id = None
		self.group = None
		self.data = None
		self.patient_age = None
		self.patient_sex = None
		self.mni_image = None  # Add field for registered image

	# Print the Patient object in a readable format
	def __repr__(self):
		return (f"Patient(patient_id={self.patient_id}, image_id={self.image_id}, "
				f"group={self.group}, age={self.patient_age}, sex={self.patient_sex}), "
				f"data_shape={self.data.shape if self.data is not None else None}, "
				f"mni_image={'Available' if self.mni_image is not None else 'Not available'})")

def random_crop_3d(image, crop_size):
	"""
	Randomly crop a 3D image to the specified size
	
	Parameters:
	-----------
	image : 3D numpy array
		The image to crop
	crop_size : tuple
		The size of the crop (x, y, z)
		
	Returns:
	--------
	3D numpy array
		The cropped image
	"""
	x, y, z = image.shape
	cx, cy, cz = crop_size
	
	# Check if cropping is possible
	if x < cx or y < cy or z < cz:
		# If image is smaller than crop size, resize it
		zoom_factors = (cx/x, cy/y, cz/z)
		return zoom(image, zoom_factors, order=1)
	
	# Calculate random starting positions
	x_start = random.randint(0, x - cx)
	y_start = random.randint(0, y - cy)
	z_start = random.randint(0, z - cz)
	
	# Crop the image
	cropped = image[x_start:x_start+cx, y_start:y_start+cy, z_start:z_start+cz]
	return cropped

def gaussian_blur(image, sigma=0.5):
	"""
	Apply Gaussian blur to an image
	
	Parameters:
	-----------
	image : 3D numpy array
		The image to blur
	sigma : float
		Standard deviation for Gaussian kernel
		
	Returns:
	--------
	3D numpy array
		The blurred image
	"""
	return gaussian_filter(image, sigma=sigma)

def process_batch(batch_files, group, is_training, output_dir, process_id=0):
	"""
	Process a batch of files and save them to a temporary file
	
	Parameters:
	-----------
	batch_files : list
		List of files to process
	group : str
		Group name (AD, CN, or MCI)
	is_training : bool
		Whether this is training data
	output_dir : str
		Output directory for temporary file
	process_id : int
		Process ID for file naming
		
	Returns:
	--------
	str
		Path to the temporary file containing processed data
	"""
	# Create a temporary file to store batch results
	os.makedirs(output_dir, exist_ok=True)
	temp_file = os.path.join(output_dir, f"{group}_batch_{process_id}.tmp")
	
	# Load MNI template once for the whole batch
	mni_template = datasets.load_mni152_template()
	
	# Process each file in the batch
	results = []
	for file_idx, file in enumerate(batch_files):
		try:
			# Split the file name
			zip_name, nii_name = file.split(':', 1)
			
			# Extract patient_id and image_id
			path_parts = nii_name.split('/')
			patient_id = path_parts[1] if len(path_parts) > 1 else "unknown"
			image_id = path_parts[4] if len(path_parts) > 4 else f"img_{random.randint(0, 10000)}"
			
			# Try to extract patient age and sex from the path or filename
			# This is an approximation - you may need to adjust this based on your actual data structure
			try:
				# Look for CSV files with metadata
				csv_files = [f for f in os.listdir('patient_info/') if f.endswith('.csv') and group in f]
				if csv_files:
					df = pd.read_csv(csv_files[0])
					# Match patient ID in the dataframe
					matched_row = df[df['Subject'] == patient_id]
					if not matched_row.empty:
						# If we found a matching row, extract age and sex
						patient_age = str(matched_row['Age'].values[0])
						patient_sex = matched_row['Sex'].values[0]
					else:
						# Default values if not found
						patient_age = "unknown_age"
						patient_sex = "unknown_sex"
				else:
					# Default values if no CSV found
					patient_age = "unknown_age"
					patient_sex = "unknown_sex"
			except Exception as e:
				print(f"Warning: Could not extract demographics for {patient_id}: {e}")
				patient_age = "unknown_age"
				patient_sex = "unknown_sex"
			
			# Load the NIfTI file
			with zipfile.ZipFile(zip_name, 'r') as zip_ref:
				with zip_ref.open(nii_name) as f:
					file_data = f.read()
					fileobj = io.BytesIO(file_data)
					img = nib.FileHolder(fileobj=fileobj)
					img = nib.Nifti1Image.from_file_map({'header': img, 'image': img})
					data = img.get_fdata()
			
			# 1. Register to MNI space
			img_for_norm = nib.Nifti1Image(data, img.affine)
			
			registered_img = image.resample_img(
				img_for_norm,
				target_affine=mni_template.affine,
				target_shape=(121, 145, 121),
				interpolation='linear'
			)
			
			# 2. Get registered data
			registered_data = registered_img.get_fdata()
			
			# 3. Apply Gaussian blur for augmentation (if training)
			if is_training and random.random() < 0.5:  # 50% chance as in paper
				sigma = random.uniform(0, 1.5)  # As specified in the paper
				registered_data = gaussian_blur(registered_data, sigma)
			
			# 4. Random crop to 96x96x96
			cropped_data = random_crop_3d(registered_data, (96, 96, 96))
			
			# 5. Normalize intensity
			cropped_data = (cropped_data - np.mean(cropped_data)) / np.std(cropped_data)
			
			# Generate filename with age and sex included
			processed_file_name = f"{group}-{patient_id}-{image_id}-{patient_age}-{patient_sex}.npz"
			
			# Save as numpy compressed array
			buffer = io.BytesIO()
			np.savez_compressed(
				buffer,
				data=cropped_data,
				group=group,
				patient_id=patient_id,
				image_id=image_id,
				patient_age=patient_age,
				patient_sex=patient_sex
			)
			buffer.seek(0)
			
			results.append((processed_file_name, buffer.getvalue()))
			
			# Free memory
			del data, img_for_norm, registered_img, registered_data, cropped_data, buffer
			
			# Print progress
			if (file_idx + 1) % 5 == 0 or file_idx == len(batch_files) - 1:
				print(f"Process {process_id}: Processed {file_idx + 1}/{len(batch_files)} files in batch")
			
		except Exception as e:
			print(f"Process {process_id}: Error processing {file}: {e}")
			# Continue with next file
	
	# Save results to temporary file
	with open(temp_file, 'wb') as f:
		import pickle
		pickle.dump(results, f)
	
	return temp_file

def process_files_parallel(file_list, group, output_dir, is_training, n_processes=None):
	"""
	Process a list of files in parallel using batch processing
	
	Parameters:
	-----------
	file_list : list
		List of files to process
	group : str
		Group name (AD, CN, or MCI)
	output_dir : str
		Output directory
	is_training : bool
		Whether this is training data
	n_processes : int or None
		Number of processes to use (None for CPU count)
		
	Returns:
	--------
	str
		Path to the zip file
	"""
	if n_processes is None:
		n_processes = mp.cpu_count()
	
	# Limit to reasonable number based on system resources
	n_processes = min(n_processes, 12)  # Arbitrary limit to prevent system overload
	
	# Create output directory if it doesn't exist
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	
	# Create temp directory for batch processing
	temp_dir = os.path.join(output_dir, 'temp')
	os.makedirs(temp_dir, exist_ok=True)
	
	# Define output zip filename
	zip_filename = f"{group}_processed_data.zip"
	zip_path = os.path.join(output_dir, zip_filename)
	
	# Divide files into batches for each process
	batch_size = (len(file_list) + n_processes - 1) // n_processes
	batches = [file_list[i:i+batch_size] for i in range(0, len(file_list), batch_size)]
	
	print(f"Processing {len(file_list)} files using {n_processes} processes in {len(batches)} batches")
	
	# Process batches in parallel
	pool = mp.Pool(processes=n_processes)
	batch_tasks = []
	for i, batch in enumerate(batches):
		batch_tasks.append(pool.apply_async(
			process_batch, 
			args=(batch, group, is_training, temp_dir, i)
		))
	
	# Get results from all processes
	temp_files = []
	for task in batch_tasks:
		temp_files.append(task.get())
	
	# Close the pool
	pool.close()
	pool.join()
	
	# Combine all results into one zip file
	with zipfile.ZipFile(zip_path, 'w') as zipf:
		file_count = 0
		for temp_file in temp_files:
			try:
				with open(temp_file, 'rb') as f:
					import pickle
					results = pickle.load(f)
				
				for filename, data in results:
					zipf.writestr(filename, data)
					file_count += 1
				
				# Remove temporary file
				os.remove(temp_file)
			except Exception as e:
				print(f"Error processing temporary file {temp_file}: {e}")
	
	# Clean up temp directory
	try:
		os.rmdir(temp_dir)
	except:
		pass
	
	print(f"Successfully processed {file_count} files out of {len(file_list)}")
	return zip_path

def process_all_groups(n_samples=None, output_dir='processed_data', n_processes=None):
	"""
	Process all groups and create datasets
	
	Parameters:
	-----------
	n_samples : int or None
		Number of samples per group (None for all available)
	output_dir : str
		Directory to save the processed data
	n_processes : int or None
		Number of processes to use for multiprocessing (None for CPU count)
	
	Returns:
	--------
	list
		Paths to the processed data zip files
	"""
	# Collect files
	if n_samples is not None:
		AD, CN, MCI = collect_and_subset_files(n_samples)
	else:
		AD, CN, MCI = create_grouped_file_list()
	
	# Process each group in parallel
	ad_zip = process_files_parallel(AD, group='AD', output_dir=output_dir, is_training=True, n_processes=n_processes)
	cn_zip = process_files_parallel(CN, group='CN', output_dir=output_dir, is_training=True, n_processes=n_processes)
	mci_zip = process_files_parallel(MCI, group='MCI', output_dir=output_dir, is_training=True, n_processes=n_processes)
	
	return [ad_zip, cn_zip, mci_zip]

if __name__ == "__main__":
	import argparse
	
	# Create output directory
	output_dir = 'processed_data'
	os.makedirs(output_dir, exist_ok=True)
	
	# Clear previous output files
	for filename in os.listdir(output_dir):
		file_path = os.path.join(output_dir, filename)
		if os.path.isfile(file_path):
			os.remove(file_path)
	
	# Set the number of samples and processes
	n_samples = 300  # Adjust as needed
	n_processes = None  # Use all available CPU cores
	
	print(f"Starting processing with {n_samples} samples per group and {n_processes if n_processes else 'all available'} processes")
	
	# Process all groups
	processed_zips = process_all_groups(
		n_samples=n_samples,
		output_dir=output_dir,
		n_processes=n_processes
	)
	
	print(f"Processed data saved to: {processed_zips}")
	print("Processing complete!")
	
	# Print statistics about the processed data
	total_files = 0
	for zip_path in processed_zips:
		if os.path.exists(zip_path):
			with zipfile.ZipFile(zip_path, 'r') as zipf:
				file_count = len(zipf.namelist())
				total_files += file_count
				print(f" - {os.path.basename(zip_path)}: {file_count} files")
	
	print(f"Total processed files: {total_files}")