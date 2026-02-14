## TIP
<img src="流程.png" width="1001" alt="">

## Setup
We provide the environment configuration file exported by Anaconda, which can help you build up conveniently.
```bash
conda env create -f environment.yml
conda activate cam 
```  

## Datasets
Download the required datasets CIFAR-10, CIFAR-100, and Tiny-ImageNet-200 into the data directory.

For the generative model used by GIFD, download it into the biggan-deep-256 directory, including pytorch_model.bin and config.json.

## GIFD

The code related to GIFD in this project is stored in the `GIFD_Gradient_Inversion_Attack` directory and is adopted from the following repository:

https://github.com/ffhibnese/GIFD_Gradient_Inversion_Attack
## Run
For convergence analysis, run the following command.

```bash
cd logs/高频噪声不影响模型收敛
python run.py --method tip --gpu 0
```  

For evaluating the defense performance, run the following command.

```bash
cd logs/防御效果和热力图
python run_attack_and_define.py --image_folder sampled_images/cifar10_32 --basepath logs/防御效果和热力图/cifar10 --model_path data/resnet18_cifar10.pth --class_name cifar10 --img_size 32 --epsilon 5
```  




## Convergence  
<img src="result_acc.png" width="1045" alt="The preprint will not be shown on the platform.">

