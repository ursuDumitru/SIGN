# For tensorflow to be able to use GPUs

## Pip packages
```bash
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit && \
python3 -m pip install --upgrade setuptools pip wheel && \
python3 -m pip install nvidia-pyindex && \
python3 -m pip install nvidia-cuda-runtime-cu11
# pip install tf-nightly[and-cuda] # or
pip install tensorflow[and-cuda]==2.13.0
pip install tensorrt==8.5.1.7

# python3 -m pip install --upgrade tensorrt && \
```

## Setup paths

```bash
CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)")) && \
TENSORRT_PATH=$(dirname $(python -c "import tensorrt;print(tensorrt.__file__)")) && \
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH && \
export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH && \
export LD_LIBRARY_PATH=$TENSORRT_PATH/lib:$LD_LIBRARY_PATH && \
export PATH=/home/aicore/ursu/protoc/bin:$PATH


# export LD_LIBRARY_PATH=/usr/local/cuda-18.2/lib64:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=$CONDA_PREFIX/tensorrt_libs/:$LD_LIBRARY_PATH && \
```

<!-- ??? -->
```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d

echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'TENSORRT_PATH=$(dirname $(python -c "import tensorrt;print(tensorrt.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

<!-- to check if tf sees the gpus -->
```bash
python3 -c "import tensorrt;print(tensorrt.__file__)"
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```