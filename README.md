# NetVLAD-Based Visual Geo-localization: Enhancements and Experimentations

Welcome to the Geo-localization Enhancement project repository! This repository contains code and resources related to our efforts to improve large-scale visual place recognition using the "NetVLAD: CNN architecture for weakly supervised place recognition" model.

This project is based on the CVPR 2022 (Oral) paper titled "[Deep Visual Geo-localization Benchmark](https://arxiv.org/abs/2204.03444)." We have extended the benchmark and conducted experiments to enhance the model's robustness, especially in the night domain, and made general improvements to achieve more accurate place recognition.

For more information please check out the [website](https://deep-vg-bench.herokuapp.com/)!

<img src="https://github.com/gmberton/gmberton.github.io/blob/main/images/vg_system.png" width="90%">

## Dataset
We use four diverse datasets to evaluate our model's performance: Pitts30k (training), Tokyo-night, Tokyo-xs, and sf-xs. These datasets represent various real-world scenarios and test the model's robustness in different conditions.


## Methodology
Our project focuses on several key areas of improvement:

 1. **Night Domain Robustness**: We have implemented data augmentation techniques, including functional transformers, to improve the model's performance in recognizing places in low-		light conditions.
 2. **General Improvements**: We experimented with image resizing, various optimizers (ASGD and AdamW), and different hyperparameters (learning rate, weight decay) to optimize the base 	model's performance.
 3. **Post-processing**: We've introduced post-processing steps to refine the model's predictions, including clustering and ranking approaches to improve the top-k candidates retrieved 	with the kNN algorithm.
 4. **Multi-Scale Testing**: We evaluated the impact of multi-scaling on model accuracy by considering image resolutions at different scales, both upscaling and downscaling.
 5. **Smart Data Augmentation**: We explored various data augmentation techniques, particularly to enhance the model's performance in the challenging night-time domain.

## Evaluation Metric
We evaluate model performance using recall@1 (R@1) and recall@5 (R@5) metrics. An image is considered correctly localized if at least one of the top N database images is within 25 meters of the query image's ground truth location.
 
## Setup
Before you begin experimenting with this toolbox, your dataset should be organized in a directory tree as such:

```
.
├── benchmarking_vg
└── datasets_vg
    └── datasets
        └── pitts30k
            └── images
                ├── train
                │   ├── database
                │   └── queries
                ├── val
                │   ├── database
                │   └── queries
                └── test
                    ├── database
                    └── queries
```
The [datasets_vg](https://github.com/gmberton/datasets_vg) repo can be used to download a number of datasets. Detailed instructions on how to download datasets are in the repo. Note that many datasets are available, and _pitts30k_ is just an example.



## Running experiments
### Basic experiment
For a basic experiment run

`$ python3 train.py --dataset_name=pitts30k`

this will train a ResNet-18 + NetVLAD on Pitts30k.
The experiment creates a folder named `./logs/default/YYYY-MM-DD_HH-mm-ss`, where checkpoints are saved, as well as an `info.log` file with training logs and other information, such as model size, FLOPs and descriptors dimensionality.

You can find more information on setup and runnig on the [Deep Visual Geo-localization Benchmark](https://github.com/gmberton/deep-visual-geo-localization-benchmark.git) repository.


## Acknowledgements
Parts of this repo are inspired by the following great repositories:
- [Deep Visual Geo-localization Benchmark](https://github.com/gmberton/deep-visual-geo-localization-benchmark.git)
- [NetVLAD's original code](https://github.com/Relja/netvlad) (in MATLAB)
- [NetVLAD layer in PyTorch](https://github.com/lyakaap/NetVLAD-pytorch)
- [NetVLAD training in PyTorch](https://github.com/Nanne/pytorch-NetVlad/)
- [GeM](https://github.com/filipradenovic/cnnimageretrieval-pytorch)
- [Deep Image Retrieval](https://github.com/naver/deep-image-retrieval)
- [Mapillary](https://github.com/mapillary/mapillary_sls)
- [Compact Convolutional Transformers](https://github.com/SHI-Labs/Compact-Transformers)


**Keywords**:Image retrival, Deep learning, Geo-localization, NetVLAD

