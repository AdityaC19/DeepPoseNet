## Installation steps:

# SMPLify-x
### Create new environment
```
conda create --name smplx0 python=3.8
conda activate smplx0
```
### Clone the repository and run setup.py
```
git clone https://github.com/AdityaC19/smplify-x.git
python setup.py install
```
### Install packages
```
pip install smplx
pip install pyrender
conda install -c conda-forge trimesh
pip install -r requirements.txt
```
### Model Loading
```
Link for SMPL: https://smpl.is.tue.mpg.de/
```
Download the model from the link and save in a directory with the following structure
```
models
|__ smpl
      |__ SMPL_FEMALE.pkl
      |__ SMPL_MALE.pkl
      |__ SMPL_NEUTRAL.pkl
```

### Fitting
After installing the smpl package and downloading the model parameters you should be able to run the main.py script to visualize the results. For this step you have to install the pyrender and trimesh packages. \
\
Run the following command to execute the code:
```
python smplifyx/main.py \
      --config cfg_files/fit_smpl.yaml \
      --data_folder DATA_FOLDER
      --output_folder SMPL_OUTPUT \
      --visualize=True/False \
      --model_folder models \
      --vposer_ckpt vposer_v1_0 --part_segm_fn smplx_parts_segm.pkl

```
where the DATA_FOLDER should contain two subfolders, images, where the images are located, and keypoints, where the OpenPose output should be stored.
