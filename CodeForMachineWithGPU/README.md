# For tensorflow to be able to use GPUs

## Pip packages
```bash
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit && \
python3 -m pip install --upgrade setuptools pip wheel && \
python3 -m pip install nvidia-pyindex && \
python3 -m pip install nvidia-cuda-runtime-cu11 && \
pip install tf-nightly[and-cuda]
```

## Setup paths

```bash
CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)")) && \
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH && \
export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/tensorrt_libs/:$LD_LIBRARY_PATH: && \
export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH && \
export PATH=/home/aicore/ursu/protoc/bin:$PATH
```