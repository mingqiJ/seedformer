#!/bin/bash

cd ./codes/pointnet2_ops_lib
python setup.py install

cd ../Chamfer3D
TORCH_CUDA_ARCH_LIST="7.0+PTX" python setup.py install
