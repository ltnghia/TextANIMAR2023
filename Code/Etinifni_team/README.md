# Etinifni_team

## Blender Installation

1. Download and Install Blender:
   - Visit the official Blender website at [https://www.blender.org/download/](https://www.blender.org/download/).
   - Download the appropriate version for your operating system (Windows, macOS, Linux).
   - Run the installer and follow the on-screen instructions to complete the installation.

## Manual Data Pre-processing

### Download dataset

1. Download the original dataset folder from [drive](https://drive.google.com/drive/folders/1PotQ4wmSRDcWwoW6pfURv4dmfKRDoUDn?usp=drive_link).
2. Unzip the .zip file.
3. Unzip the _References.zip_ file in _3D_Model_References_ directory to a new directory with the same name.
4. Download [multiviews_v3_rotated folder](https://drive.google.com/file/d/11qo9UwyTXJc6zSj6WlITTnIJw_f4Cp2-/view?usp=drive_link) and move downloaded_zip_file/etinifni-SHREC-2023-TEXT-ANIMAR-main/content/original/multiviews_v3_rotated to the corresponding folder.

### Manual adjustment

1. Open Blender:
   - Launch Blender by double-clicking on the application icon.
   - Select all default objects `(Camera, Cube, Light)` and delete them.
     
2. Rotate the objects to the right position:
   - Import a 3D .obj file in _References_ directory by click on `File` in the top menu and select `Import`. We choose `Wavefront (.obj)`.
   - Adjust the imported object to have the standing position along the z-axis.
   - Click on `File` in the top menu and select `Export`. We choose `Wavefront (.obj)`.
   - Repeat the process until all objects are processed.
  
3. Rename the files:
   - For each file, rename the file with the current number of processed files.
   - Store the mapping between the original name to the new name into **mapping.txt** file. Each row is a pair of 2 values: **(original, new_name)**
   - Repeat the process until all files are processed.
  
4. Create new directory:
   - Create new directory named _ProcessedObjFiles_.
   - Move all the processed 3D .obj files in to the directory.
  
5. Create mapped_TextQuery_Train.csv:
   - Open _TextQuery_Train.csv_ in Train directory.
   - Copy and paste the context of ID column outside for mapping in part 6.
   - Change the context of ID column to `text_1, text_2, ... , text_100`.
   - Save the new version with name **mapped_TextQuery_Train.csv**.
  
6. Create mapped_TextQuery_GT_Train.csv:
   - Open _TextQuery_GT_Train.csv_ in Train directory.
   - Map the context of "Text Query ID" column in this file to the context of ID column in _mapped_TextQuery_Train.csv_ by the copied content in part 5.
   - Save the new version with name **mapped_TextQuery_GT_Train.csv**.

### Generating views from processed dataset

1. Open Blender:
   - Launch Blender by double-clicking on the application icon.
   - The Blender interface will open, displaying a 3D viewport and various panels.

2. Open the File:
   - Click on `File` in the top menu and select `Open` or use the shortcut `Ctrl + O`.
   - Navigate to the location where style file is saved and select it. Click `Open` to load the file.

3. Switch to the Scripting Tab:
   - Click on the `Scripting` tab located at the top of the Blender interface.

4. Edit the Code:
   - Locate the code section where the **input** (.obj file) and **output** (results) directories are specified.
   - Update the paths in the code to match your desired **input** and **output** directories.

5. Run the Code:
   - Click on the `Run Script` button in the top-right corner of the text editor panel. Alternatively, use the shortcut `Alt + P`.
   - Blender will execute the code and result will be saved in output directories.

## Training and Inference 
- For simplicity, we store the folder containing all views extracted from 3D Objects (12 views for each object) into _content/original_ directory. 
- All files .csv which contain sentences after preprocessing (change wrong word) are also stored in the _content/original_ directory.
- For implementation, you only need to run the below scripts.
- File etinifni_TextANIMAR2023.csv will be saved in root directory.
### Install CLIP library
``` bash
pip install git+https://github.com/openai/CLIP.git
```
### Prepare training and validation set for training phase
``` bash
python preprocess.py
```
### Train model
``` bash
python train.py
```
### Inference model
``` bash
python test.py
```

### Note
If you want to train the model (text-image) on full 12 view images of each 3D object instead of training on only two view of each 3D object. You can change the code in lines 6-8 in script preprocess.py as below. This help to get 0.48 score in public test and 0.42 score in private test (Nearest Neighbor).
- **Old code**
``` bash
df = pd.read_csv('./content/original/mapped_TextQuery_multiview_rotated_GT_train.csv')
train = df.loc[0:607, :]
test = df.loc[608:, :]
```
- **New code**
``` bash
df = pd.read_csv('./content/original/mapped_TextQuery_multiview_GT_train.csv')
train = df.loc[0:3647, :]
test = df.loc[3648:, :]
```
