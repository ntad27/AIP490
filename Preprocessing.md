# Data preprocessing

According to the explanation from the authors in [Preprocessing.md](https://github.com/sangyun884/HR-VITON/blob/main/Preprocessing.md). At least 6 steps are needed for getting all the required inputs of the model. Thanks to this [detailed instructions](https://github.com/sangyun884/HR-VITON/issues/45), we have reproduced all of the required preprocessing steps.

We've added one more preprocessing step and modified other steps in order to get better try-on results.

Please check the following notebook for more details
<a target="_blank" href="https://colab.research.google.com/drive/1nmDHjGH3HKEmawXWdyooWNcGBl9qtFv8?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

You should first create these folders and put them under `./data/test` when you want to test the model, or `./data/train` when you want to retrain it.

![](/figures/folders.png)

> [!NOTE]
> We reproduced all of the preprocessing steps on Colab. The instructions below will show you how to preprocess the data step by step.

## 1. Remove background
This is our additional step to improve the model's accuracy. We used the transparent-background package to remove the background of both `./data/test/image` and `./data/test/cloth`. It's kind of easy to process, just follow the below instructions:

(1) Install transparent-background package
```
!pip install transparent-background
```

(2) Run
> [!IMPORTANT]
> Remember to download [the checkpoint](https://drive.google.com/file/d/12QZJJ26JyOELd5ERsbMOxaCIDl-6rJzW/view?usp=sharing), put it in your drive then paste its path in `ckpt={checkpoint_path}` before running the below code.
```
import cv2
import os
import numpy as np
from PIL import Image
from transparent_background import Remover

# Load model
remover = Remover(fast=False, jit=False, device='cuda:0', ckpt={checkpoint_path})

input_folder = {image_path}
output_folder = {image_path}

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        img = Image.open(input_path).convert('RGB')  # read image
        out = remover.process(img, type='white')  # change backround with white color

        # Convert the processed image to a NumPy array
        out_np = np.array(out)

        # Save the image using PIL
        Image.fromarray(out_np).save(output_path)  # save result

print("Finished.")
```
Just change the path of `input_folder` and `output_folder` when removing cloth's background.

We've tried other methods but we found out this one gave the best results and was the easiest one to process. The package has some other usages, more details can be found at [transparent-background](https://github.com/plemeri/transparent-background).

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

![](/figures/densepose.png)

More details about the DensePose results can be found at [detectron2](https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose).

## 4. Cloth Mask
We also used the transparent-background package for this step because it created better results than the original one.

> [!IMPORTANT]
> Remember to download [the checkpoint](https://drive.google.com/file/d/12QZJJ26JyOELd5ERsbMOxaCIDl-6rJzW/view?usp=sharing), put it in your drive then paste its path in `ckpt={checkpoint_path}` before running the below code.
```
import cv2
import os
import numpy as np
from PIL import Image
from transparent_background import Remover

# Load model
remover = Remover(fast=False, jit=False, device='cuda:0', ckpt={checkpoint_path})

input_folder = {cloth_path}
output_folder = {cloth_mask_path}

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        img = Image.open(input_path)  # read image
        out = remover.process(img, type='map') # object map only

        # Convert the processed image to a NumPy array
        out_np = np.array(out)

        im = Image.fromarray(out_np).convert('L')

        # Save the image using PIL
        im.save(output_path)  # save result

print("Finished.")
```
Then you can get results that look like

![](/figures/cloth_mask.png)

## 5. Human Parse
This may be the hardest step. The authors used TensorFlow 1.x for this step and they had to create a virtual Python environment to run it, so it wouldn't be synchronized to other steps. Luckily we've been able to upgrade to TensorFlow 2.0, therefore it's easier for us to process.

(1) Get the zip file from this [link](https://drive.google.com/file/d/1eX_O-KflZe31eVOubsrwotpFFbwiK9mL/view?usp=sharing), put it in your drive then unzip it into Colab.

(2) Install required packages
```
!pip install matplotlib opencv-python Pillow scipy
!pip install ipykernel
!pip install pandas
!pip install -r /content/CIHP_PGN_v2/requirement.txt
```

> [!NOTE]
> The below instructions are our proposed method, you can try other methods if that generate better results.

(3) Resize images from `768 x 1024` to `192 x 256`

We used [CIHP_PGN](https://github.com/Engineering-Course/CIHP_PGN) method for this step and it gives better performance on 192x256 images so we have to resize the images to 192x256 then upsize them back to 768x1024 later. You can use any resize method, this is our proposed one:
```
import os
from PIL import Image

def resize_images(input_dir, output_dir, new_size):
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_file in image_files:
        input_image_path = os.path.join(input_dir, image_file)
        output_image_path = os.path.join(output_dir, image_file)

        image = Image.open(input_image_path)

        resized_image = image.resize(new_size, Image.ANTIALIAS)

        resized_image.save(output_image_path)

input_dir = {image_path}
output_dir = "/content/CIHP_PGN_v2/datasets/images"

new_size = (192, 256)

resize_images(input_dir, output_dir, new_size)
```
(4) Run
```
%cd /content/CIHP_PGN_v2
!python inference_pgn.py
%cd /content
```
(5) Store the visualize files (file ends with `_vis.png`) in `./image-parse-v3-visualize`. **The black images are what we really need** because their values are from 0-20 and are consistent with the model.

(6) 

## 6. Parse Agnostic
Here is the parse label and corresponding body parts. We'll leave it here in case you need it.
```
0 - Background
1 - Hat
2 - Hair
3 - Glove
4 - Sunglasses
5 - Upper-clothes
6 - Dress
7 - Coat
8 - Socks
9 - Pants
10 - Tosor-skin
11 - Scarf
12 - Skirt
13 - Face
14 - Left-arm
15 - Right-arm
16 - Left-leg
17 - Right-leg
18 - Left-shoe
19 - Right-shoe
```
(1) Install packages
```
!pip install Pillow tqdm
```
(2) Run
```
import json
from os import path as osp
import os
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

def get_im_parse_agnostic(im_parse, pose_data, w=768, h=1024):
    label_array = np.array(im_parse)
    parse_upper = ((label_array == 5).astype(np.float32) +
                    (label_array == 6).astype(np.float32) +
                    (label_array == 7).astype(np.float32))
    parse_neck = (label_array == 10).astype(np.float32)

    r = 10
    agnostic = im_parse.copy()

    # mask arms
    for parse_id, pose_ids in [(14, [2, 5, 6, 7]), (15, [5, 2, 3, 4])]:
        mask_arm = Image.new('L', (w, h), 'black')
        mask_arm_draw = ImageDraw.Draw(mask_arm)
        i_prev = pose_ids[0]
        for i in pose_ids[1:]:
            if (pose_data[i_prev, 0] == 0.0 and pose_data[i_prev, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                continue
            mask_arm_draw.line([tuple(pose_data[j]) for j in [i_prev, i]], 'white', width=r*10)
            pointx, pointy = pose_data[i]
            radius = r*4 if i == pose_ids[-1] else r*15
            mask_arm_draw.ellipse((pointx-radius, pointy-radius, pointx+radius, pointy+radius), 'white', 'white')
            i_prev = i
        parse_arm = (np.array(mask_arm) / 255) * (label_array == parse_id).astype(np.float32)
        agnostic.paste(0, None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))

    # mask torso & neck
    agnostic.paste(0, None, Image.fromarray(np.uint8(parse_upper * 255), 'L'))
    agnostic.paste(0, None, Image.fromarray(np.uint8(parse_neck * 255), 'L'))

    return agnostic


if __name__ =="__main__":
    data_path = {dataroot_path}
    output_path = {parse_agnostic_path}

    os.makedirs(output_path, exist_ok=True)

    for im_name in tqdm(os.listdir(osp.join(data_path, 'image'))):

        # load pose image
        pose_name = im_name.replace('.jpg', '_keypoints.json')

        try:
            with open(osp.join(data_path, 'openpose_json', pose_name), 'r') as f:
                pose_label = json.load(f)
                pose_data = pose_label['people'][0]['pose_keypoints_2d']
                pose_data = np.array(pose_data)
                pose_data = pose_data.reshape((-1, 3))[:, :2]
        except IndexError:
            print(pose_name)
            continue

        # load parsing image
        parse_name = im_name
        parse_name = im_name.replace('.jpg', '.png')
        im_parse = Image.open(osp.join(data_path, 'image-parse-v3', parse_name))
        agnostic = get_im_parse_agnostic(im_parse, pose_data)
        #parse_name = parse_name.replace("jpg","png")
        agnostic.save(osp.join(output_path, parse_name))
```
You can check the results under `./test/image-parse-agnostic-v3.2`. Of course it's all black as well the `./test./image-parse-v3`. To ensure you're getting the right agnostic parse images, do below:
```
import numpy as np
from PIL import Image

im_ori = Image.open('./test/image-parse-v3/06868_00.png')
im = Image.open('./test/image-parse-agnostic-v3.2/06868_00.png')
print(np.unique(np.array(im_ori)))
print(np.unique(np.array(im)))
```
The output may look like:
```
[ 0  2  5  9 10 13 14 15]
[ 0  2  9 13 14 15]
```
You can also visualize it by:
```
np_im = np.array(im)
np_im[np_im==2] = 151
np_im[np_im==9] = 178
np_im[np_im==13] = 191
np_im[np_im==14] = 221
np_im[np_im==15] = 246
Image.fromarray(np_im)
```

## 7. Human Agnostic
Steps are almost the same as the **Parse Agnostic**.

(1) Install packages (if you have installed them above you don't need to install them again)
```
!pip install Pillow tqdm
```
(2) Run
```
import json
from os import path as osp
import os
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

def get_img_agnostic(img, parse, pose_data):
    parse_array = np.array(parse)
    parse_head = ((parse_array == 4).astype(np.float32) +
                    (parse_array == 13).astype(np.float32) +
                    (parse_array == 2).astype(np.float32))
    parse_lower = ((parse_array == 9).astype(np.float32) +
                    (parse_array == 12).astype(np.float32) +
                    (parse_array == 16).astype(np.float32) +
                    (parse_array == 17).astype(np.float32) +
                    (parse_array == 18).astype(np.float32) +
                    (parse_array == 19).astype(np.float32))


    agnostic = img.copy()
    agnostic_draw = ImageDraw.Draw(agnostic)

    length_a = np.linalg.norm(pose_data[5] - pose_data[2])
    length_b = np.linalg.norm(pose_data[12] - pose_data[9])
    point = (pose_data[9] + pose_data[12]) / 2
    pose_data[9] = point + (pose_data[9] - point) / length_b * length_a
    pose_data[12] = point + (pose_data[12] - point) / length_b * length_a
    r = int(length_a / 16) + 1

    # mask arms
    agnostic_draw.line([tuple(pose_data[i]) for i in [2, 5]], 'gray', width=r*10)
    for i in [2, 5]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')
    for i in [3, 4, 6, 7]:
        if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
            continue
        agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'gray', width=r*10)
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')

    # mask torso
    for i in [9, 12]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*3, pointy-r*6, pointx+r*3, pointy+r*6), 'gray', 'gray')
    agnostic_draw.line([tuple(pose_data[i]) for i in [2, 9]], 'gray', width=r*6)
    agnostic_draw.line([tuple(pose_data[i]) for i in [5, 12]], 'gray', width=r*6)
    agnostic_draw.line([tuple(pose_data[i]) for i in [9, 12]], 'gray', width=r*12)
    agnostic_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 12, 9]], 'gray', 'gray')

    # mask neck
    pointx, pointy = pose_data[1]
    agnostic_draw.rectangle((pointx-r*7, pointy-r*7, pointx+r*7, pointy+r*7), 'gray', 'gray')
    agnostic.paste(img, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
    agnostic.paste(img, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))

    return agnostic

if __name__ =="__main__":
    data_path = {dataroot_path}
    output_path = {human_agnostic_path}

    os.makedirs(output_path, exist_ok=True)

    for im_name in tqdm(os.listdir(osp.join(data_path, 'image'))):

        # load pose image
        pose_name = im_name.replace('.jpg', '_keypoints.json')

        try:
            with open(osp.join(data_path, 'openpose_json', pose_name), 'r') as f:
                pose_label = json.load(f)
                pose_data = pose_label['people'][0]['pose_keypoints_2d']
                pose_data = np.array(pose_data)
                pose_data = pose_data.reshape((-1, 3))[:, :2]
        except IndexError:
            print(pose_name)
            continue

        # load parsing image
        im = Image.open(osp.join(data_path, 'image', im_name))
        label_name = im_name.replace('.jpg', '.png')
        im_label = Image.open(osp.join(data_path, 'image-parse-v3', label_name))

        agnostic = get_img_agnostic(im, im_label, pose_data)

        agnostic.save(osp.join(output_path, im_name))
```
The results will look like:

![](/figures/human_agnostic.png)

## Conclusion
Thank you for reading. It's not easy to get all this done. Before you run the HR-VITON model with your preprocessed data, note that each person's image needs a corresponding cloth image even though it's not used while inference. If you don't want this behavior, feel free to change the source code manually or just add some random images with the same name as person images. After it's all done, suppose you're testing 5 people images and 3 cloth images, which are all unpaired, you should end up with 3 images under `./cloth` and 3 images under `./cloth-mask`; 5 images under each other dirs: `agnostic-v3.2`, `image`, `image-densepose`, `image-parse-agnostic-v3.2`, `image-parse-v3`, `openpose_img` and `openpose_json`.

The complete result will looks like this:

![](/figures/01_01.png)
