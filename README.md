

## Installation steps:

_Create new env_
```
conda create --name smplx0 python=3.8
conda activate smplx0
```
_Clone the repository and run setup.py_
```
git clone https://github.com/vchoutas/smplx
python setup.py install
```
_Install_
```
pip install smplx
pip install pyrender
conda install -c conda-forge trimesh
```
_Download SMPL-X model_
```
Link: https://smpl-x.is.tue.mpg.de/
```
_Removing Chumpy objects from the model data_
```
pip install chumpy
pip install tqdm
cd smplx
python tools/clean_ch.py --input-models path-to-model --output-folder output-folder
```
_Run the demo file_
```
python examples/demo.py --model-folder path-to-model --plot-joints=True --gender="neutral"
```

