# AIP490 - Virtual Shirt Fitting Model Based on Deep Learning and Computer Vision

## Model's architecture
The figures below show the architecture of the model from the data preprocessing step to the main [HR-VITON](https://github.com/sangyun884/HR-VITON) model we used in this project.

![Main flow](/figures/base.png)

## Dataset
We train and evaluate our model using the dataset from [High-Resolution Virtual Try-On with Misalignment and Occlusion-Handled Conditions](https://github.com/sangyun884/HR-VITON), the original dataset was from [VITON-HD: High-Resolution Virtual Try-On via Misalignment-Aware Normalization](https://github.com/shadow2496/VITON-HD).

To download the dataset, please check the following links:
- Our preprocessed dataset: [link](https://drive.google.com/file/d/1iHoiyTnRF2lMFN95f37s8-4G2-Plp5Zb/view?usp=sharing)
- Authors' preprocessed dataset: [link](https://drive.google.com/file/d/190xa7nb92KNWc4EF9pxP0YJ8pWu1NkU8/view?usp=sharing)

After you download the dataset, create a `./data` folder and put it under.

If you want to know more about how we preprocess the data, please check the [Preprocessing.md](https://github.com/ntad27/AIP490/blob/main/Preprocessing.md)

## Inference
Here are the download links for each model checkpoint:
- Author's try-on condition generator: [link](https://drive.google.com/file/d/1l81F7eShSg5mOorpwY5xEWla06KaQ76Y/view?usp=drive_link)
- Our retrain 30000 steps try-on condition generator: [link](https://drive.google.com/file/d/1WcH0E-W6DKEFQhD5FpKNVpWxmfaJtSL5/view?usp=sharing)
- Author's try-on image generator: [link](https://drive.google.com/file/d/1LBkpO5HO3KYUGSXU_SNeQfWOarUh5lTO/view?usp=drive_link)
- We assume that you have obtained all the checkpoints and stored them in `./eval_models/weights/v0.1`.

We've built a web app demo, please check the following notebook for detailed instructions 
<a target="_blank" href="https://colab.research.google.com/drive/1qyTB0-70KAorx3VmVFkNo3QVZ35gJGYm?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Train try-on condition generator
```
python train_condition.py --name test --cuda {True} --gpu_ids {gpu_ids} --dataroot {dataroot_path} --datamode train --data_list {path_to_trainpairs.txt} --keep_step 300000 --Ddownx2 --Ddropout --lasttvonly --interflowloss --occlusion
```

## Train try-on image generator
```
python train_generator.py --cuda {True} --name test -b 4 -j 8 --gpu_ids {gpu_ids} --fp16 --tocg_checkpoint {condition generator ckpt path} --occlusion --dataroot {data_path}
```

To use "--fp16" option, you should install the [apex](https://github.com/NVIDIA/apex.git) library.

## Acknowledgements
- Tran Ngoc Xuan Tin
- Duong Vien Thach
- https://github.com/sangyun884/HR-VITON
- https://github.com/shadow2496/VITON-HD
- https://github.com/plemeri/transparent-background
- https://github.com/CMU-Perceptual-Computing-Lab/openpose
- https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose
- https://github.com/Engineering-Course/CIHP_PGN
