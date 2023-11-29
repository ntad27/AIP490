# AIP490 - Virtual Shirt Fitting Model Based on Deep Learning and Computer Vision

## Dataset
We train and evaluate our model using the dataset from [HR-VITON — Official PyTorch Implementation](https://drive.google.com/file/d/11d1IKZ-jsK9mx0BSQmxrEqLxAA00C3IO/view?usp=drive_link) and [Dress Code](https://github.com/aimagelab/dress-code).

To download the datasets, please check the following links:
- Our preprocessed HR-VITON dataset: [link](https://drive.google.com/file/d/1iHoiyTnRF2lMFN95f37s8-4G2-Plp5Zb/view?usp=sharing)
- Authors' preprocessed HR-VITON dataset: [link](https://drive.google.com/file/d/190xa7nb92KNWc4EF9pxP0YJ8pWu1NkU8/view?usp=sharing)
- We cannot provide you with the Dress Code Dataset due to copyright issues. But you can get it by filling out the [request form](https://docs.google.com/forms/d/e/1FAIpQLSeWVzxWcj3JSALtthuw-2QDAbf2ymiK37sA4pRQD4tZz2vqsw/viewform) then the author will email you for more detailed instructions.

We assume that you have downloaded it into `./data`.

If you want to know more about how we preprocess the data, please check the [Preprocessing.md](https://github.com/ntad27/AIP490/blob/main/Preprocessing.md)

## Inference
Here are the download links for each model checkpoint:
- Try-on condition generator: [link](https://drive.google.com/file/d/1l81F7eShSg5mOorpwY5xEWla06KaQ76Y/view?usp=sharing)
- Try-on image generator: [link](https://drive.google.com/file/d/1LBkpO5HO3KYUGSXU_SNeQfWOarUh5lTO/view?usp=sharing)
- We assume that you have downloaded all of the checkpoints into `./eval_models/weights/v0.1`.

We used [Anvil](https://anvil.works/) to build the web app demo, please check the following notebook for detailed instructions 
<a target="_blank" href="https://colab.research.google.com/drive/1qyTB0-70KAorx3VmVFkNo3QVZ35gJGYm?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Train try-on condition generator
```
python train_condition.py --cuda {True} --gpu_ids {gpu_ids} --Ddownx2 --Ddropout --lasttvonly --interflowloss --occlusion
```

## Train try-on image generator
```
python train_generator.py --cuda {True} --name test -b 4 -j 8 --gpu_ids {gpu_ids} --fp16 --tocg_checkpoint {condition generator ckpt path} --occlusion --dataroot {data_path}
```

To use "--fp16" option, you should install the [apex](https://github.com/NVIDIA/apex.git) library.

## Acknowledgements
- https://github.com/sangyun884/HR-VITON
- https://github.com/shadow2496/VITON-HD
- https://github.com/aimagelab/dress-code
