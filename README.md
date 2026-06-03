# Hierachical Coding

Official implementation of the article: "SeSeCo: Selective semantic compression of images
" [[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=11406894)

## Abstract

In image compression, in applications targeting extremely low bitrates (0.01 bpp), where the reconstruction distortion can be severe, it makes sense to prioritize parts of the image that are more relevant than others. In this paper, we propose a semantic compression framework that integrates user or application preferences to compress image parts based on their semantic representation. We design a guide for trained diffusion models that takes into account the preferences for describing objects with varying accuracies. We show that we are able to preserve the selected objects while also preserving the semantic and global aspect of the image without any retraining or fine-tuning.

## Install

This code was developed in python 3.12.
To install download the repository and install the packages in requirements.txt in your environment:
```commandline
pip install -r req.txt
```

Since the gradients of the model are computed, a large amount of GPU memory is required.

## Run the project

Edit the config.yml file to specify the path to your images, cache and output folders:

```yaml
misc:
  quality_parameter: 5
  name: "image"
  data_dir: "./images" #Path to the image folder
  cache_dir: "./cache/" #Path to cache models
  save_dir: "./output_folder/" #Path to the output folder
  verbose: True
```

And run simply using:
```commandline
python main.py -c config.py
```

## License

This work is licensed under the terms of the MIT license.

## Citation

If you use the work released here for your research, please cite this paper:
```
@article{bordin2026seseco,
  title={SeSeCo: Selective semantic compression of images},
  author={Bordin, Tom and Maugey, Thomas and Barbarossa, Sergio},
  journal={IEEE Open Journal of Signal Processing},
  year={2026},
  publisher={IEEE}
}
```