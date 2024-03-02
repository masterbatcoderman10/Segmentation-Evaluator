# Image Segmentation Evaluation Pipeline

This repository contains the code for the image segmentation evaluation pipeline. This pipeline is built to provide comprehensive qualitative and quantitative analysis of image segmentation algorithms. While the pipeline is intended to be used with PyTorch models, it can in principle be used with Keras models as they possess a `predict` method.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Test](#test)
- [Improvements](#improvements)

## Installation

The pipeline requires the following packages in order to function properly:
- `numpy`
- `matplotlib`
- `opencv-python`
- `miseval`

This last library - `miseval`, is a library built for reliable evaluation of image segmentation algorithms. 

```
@Article{misevalMUELLER2022,
  title={MISeval: a Metric Library for Medical Image Segmentation Evaluation},
  author={Dominik Müller, Dennis Hartmann, Philip Meyer, Florian Auer, Iñaki Soto-Rey, Frank Kramer},
  year={2022},
  journal={Studies in health technology and informatics},
  volume={294},
  number={},
  pages={33-37},
  doi={10.3233/shti220391},
  eprint={2201.09395},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

## Usage

The pipeline is designed in a modular way, consisting of multiple stages each of which conduct a separate type of evaluation. The pipeline is available as a class, `EvalPipeline`, which can be instantiated with the following parameters:

- `dataloader`: The DataLoader object that contains the images and masks to be evaluated.
- `n`: The number of classes in the dataset.
- `model_dict` : A dictionary containing different models, this pipeline was designed to evaluate multiple models at once.
- `class_list` : A list of class names, used for visualization purposes.
- `color_dict` : A dictionary mapping numbers to tuples of RGB colors, used for visualization purposes.

### Stages

Other metrics specified on the `miseval` library can be used by passing in an array of metric names to the `metrics` parameter for each stage.

The pipeline consists of the following stages:

1. `stage_one`: This stage conducts quantitative analysis where the model's performance is evaluated using the following metrics: sensitivity, specificity, intersection over union, and dice coefficient. The metrics are averaged over all classes and the results are saved to a csv file.

```python
eval_pipeline.stage_one(model_keys=['model1', 'model2'...], save_path='path/to/save.csv')
```

2. `stage_two`: This stage conducts an analysis similar to stage one, however, the metrics for individual classes without averaging are saved to a csv and JSON file.

```python
eval_pipeline.stage_two(model_keys=['model1', 'model2'...], save_path='path/to/save.json', csv_path='path/to/save.csv')
```

## Example

A comprehensive usage example is coming soon.

However, in the `tests/outputs` directory, there are example outputs from the pipeline. These outputs were generated for a multi-class image segmentation on Diabetes Retinopathy dataset. 

More details on this project are available [in this repository](!https://github.com/masterbatcoderman10/DR_Segmentation_Analysis).