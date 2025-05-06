### Steps To Run

1. The following package imports are required to run:

	* from nilearn import image, datasets
	* from scipy.ndimage import gaussian_filter, zoom
	* from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
	* from torch.utils import data
	* from torch.utils.tensorboard import SummaryWriter
	* from tqdm import tqdm
	* import matplotlib.pyplot as plt
	* import nibabel as nib
	* import numpy as np
	* import pandas as pd
	* import seaborn as sns
	* import torch
	* import torch.nn as nn
	* import torch.nn.functional as F
	* import torch.optim as optim

Use this command to `pip install` all of the base packages at once:
`pip install nilearn scipy scikit-learn torch tensorboard tqdm matplotlib nibabel numpy pandas seaborn`

2. Download the repository and keep the Python files in the same folder.

3. Follow the instructions listed below for downloading the data.

4. Process the data wherever you please, but copy/paste the CSV files and the `processed_data` folder into the folder with the Python files.

5. Use `main.py` to run the ML pipeline.
	* In `if __name__ == 'main':` at the bottom, you a free to change the variables at the top of the section. Specifically, and of the following may be altered:

		* batch_size = 4
		* num_workers = 2
		* widening_factor = 8
		* use_age = True
		* num_epochs = 10
		* learning_rate = 0.001
		* n_samples = 5

Note: `n_samples` needs to match the number of samples that are generated for each group. For instance, we used 5, 20, and 300 which generated 15, 60, and 900 files respectively.

### Downloading The Data

Registration through the Image and Data Archive (IDA) are needed to download any of the relevant data. This can be done at the [IDA website](https://ida.loni.usc.edu/collaboration/access/appLicense.jsp;jsessionid=B61EA74690BFB07100C07269CCC5B8F8). After this has been done, you can use the following instructions from the [IDA homepage](https://ida.loni.usc.edu/login.jsp).

1. Click on "Search \& Download" at the top of the home page and then click on "Advanced Image Search".
2. In the "PROJECT/PHASE" section, ensure that only ADNI is selected.
3. In the "SUBJECT" section, check "MCI".
4. In the "IMAGE" section, ensure only "MRI" is checked and copy/paste "MPR; GradWarp; B1 Correction; N3" into the "Image Description" box. (Note: the search function appears to be finicky. Make sure to copy/paste the string as-is.)
5. In the "IMAGING PROTOCOL" section, scroll to the bottom to the "Weighting" section and check "T1".
6. Hit "SEARCH" in the top right.
7. In the new tab, click on "Select all" in the top right.
8. Click "Add to Collection" right next to it.
9. In the second box, name it something descriptive (with "MCI" in the name) and click "Ok".
10. Navigate back to the "Advanced Search" tab and repeat steps 2-9 with "AD" and "CN" in place of "MCI".

At this point, there should be three collections for "MCI", "AD", and "CN". In the "Data Collections" tab, on the left side of the window, there will be a section titled "My Collections". Expand that and expand one of the three collections that you downloaded. Click on "Not Downloaded". Now, on the right side of the window check the "All" box to highlight all samples. At the top of the window, click the "CSV" button on the left to download all of the sample information as a CSV, and click either the "1-CLICK-DOWNLOAD" or "ADVANCED DOWNLOAD" buttons. The IDA website will compile the data into zip files and present you with a popup or window (depending on which download option you selected) listing each of the zip files to download.
