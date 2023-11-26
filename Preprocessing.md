# Data preprocessing

According to the explanation from the authors in [Preprocessing.md](https://github.com/sangyun884/HR-VITON/blob/main/Preprocessing.md). At least 6 steps are needed for getting all the required inputs of the model. Thanks to this [detailed instructions](https://github.com/sangyun884/HR-VITON/issues/45), we have reproduced all of the required preprocessing steps.

We've added one more preprocessing step and modified other steps in order to get better try-on results.

Please check the following notebook if you want to try on your own data 
<a target="_blank" href="https://colab.research.google.com/drive/1nmDHjGH3HKEmawXWdyooWNcGBl9qtFv8?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

> [!NOTE]
> We reproduced all of the preprocessing steps on the above Colab. The instructions below will show you how to preprocess the data step by step.

## 1. Remove background
## 2. OpenPose
(1) Install OpenPose
```
import os
from os.path import exists, join, basename, splitext

git_repo_url = 'https://github.com/CMU-Perceptual-Computing-Lab/openpose.git'
project_name = splitext(basename(git_repo_url))[0]
if not exists(project_name):
  # see: https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/949
  # install new CMake because of CUDA10
  !wget -q https://cmake.org/files/v3.13/cmake-3.13.0-Linux-x86_64.tar.gz
  !tar xfz cmake-3.13.0-Linux-x86_64.tar.gz --strip-components=1 -C /usr/local
  # clone openpose
  !git clone -q --depth 1 $git_repo_url
  !sed -i 's/execute_process(COMMAND git checkout master WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}\/3rdparty\/caffe)/execute_process(COMMAND git checkout f019d0dfe86f49d1140961f8c7dec22130c83154 WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}\/3rdparty\/caffe)/g' openpose/CMakeLists.txt
  # install system dependencies
  !apt-get -qq install -y libatlas-base-dev libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler libgflags-dev libgoogle-glog-dev liblmdb-dev opencl-headers ocl-icd-opencl-dev libviennacl-dev
  # install python dependencies
  !pip install -q youtube-dl
  # build openpose
  with open("/content/openpose/CMakeLists.txt","r") as f:
    lines = list()
    for line in f:
      if '78287B57CF85FA89C03F1393D368E5B7' in line:
        line = line.replace('78287B57CF85FA89C03F1393D368E5B7','d41d8cd98f00b204e9800998ecf8427e')
      if '78287b57cf85fa89c03f1393d368e5b7' in line:
        line = line.replace('78287b57cf85fa89c03f1393d368e5b7','d41d8cd98f00b204e9800998ecf8427e')
      if 'e747180d728fa4e4418c465828384333' in line:
        line = line.replace('e747180d728fa4e4418c465828384333','d41d8cd98f00b204e9800998ecf8427e')
      if 'a82cfc3fea7c62f159e11bd3674c1531' in line:
        line = line.replace('a82cfc3fea7c62f159e11bd3674c1531','d41d8cd98f00b204e9800998ecf8427e')
      lines.append(line)
    f.close()
  # replacing the hash of the CMakeLists.txt
  with open("/content/openpose/CMakeLists.txt","w") as f:
    for line in lines:
        f.writelines(line)
    f.close()
  !cd openpose && rm -rf build || true && mkdir build && cd build && cmake .. -DUSE_CUDNN=OFF && make -j`nproc`
```
(2) Get all needed models then put them in `./openpose/models`. You can get all of it from this [link](https://drive.google.com/file/d/15Ir-yP6dYupibzO7sJCBEyWy9Y0AdZFR/view?usp=sharing).

(3) Run
```
!./build/examples/openpose/openpose.bin --image_dir {image_path} --hand --disable_blending --display 0 --write_json {json_path} --write_images {img_path} --num_gpu 1 --num_gpu_start 0
```
Then JSON files will be saved under `../json_path` and images will be saved under `../img_path`.

The image result looks like

![](/figures/14673_00_rendered.png)

More details about the results can be found at [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose).

## 3. DensePose
(1) Get the repository of detectron2

We've modified some code from the original repo of detectron2, you can get the modified version from this [link](https://drive.google.com/file/d/1wThrFf3NOzhgl2_L7AraKubZOKrs1lHd/view?usp=sharing), put it in your drive then unzip it into Colab.

(2) Install dependencies
```
!python -m pip install -e detectron2
```

(3) Install packages for DensePose
```
%cd /content/detectron2/projects/DensePose
!pip install av>=8.0.3 opencv-python-headless>=4.5.3.56 scipy>=1.5.4 torch torchvision
%cd /content
```

(4) Run
```
!python /content/detectron2/projects/DensePose/apply_net_test.py \
show /content/detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml \
https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl \
{image_path} dp_segm -v
```
If you want to use CPU, add `--opts MODEL.DEVICE cpu` to the end of the above command.

Then you can get results that look like



## 4. Cloth Mask
## 5. Human Parse
## 6. Parse Agnostic
## 7. Human Agnostic
