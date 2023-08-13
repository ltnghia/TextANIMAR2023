# THP_team

## Environment

- **Linux:** 18.04.5
- **GPU:** NVIDIA GeForce GTX 1080 Ti 11GB
- **CUDA Version:** 10.2
- **Torch Version:** 1.10.0

## Instructions

## 1. Generating 2D Images from 3D Models

We generate 2D images following the instructions from [BlenderPhong](https://github.com/WeiTang114/BlenderPhong) repository with modifications in the `phong.py` file.

1. Download [blender-2.79-linux-glibc219-i686](https://www.blender.org/).
2. Follow the instructions in the repository to generate 2D images.
3. Our generated data can be found in [this Google Drive folder](https://drive.google.com/drive/folders/1njdHzfOZxfTJoWndiAM1aQqSdxUz2Io4?usp=sharing).

## 2. Preprocessing

We perform preprocessing on the generated images:

1. Crop and resize:
```bash
python preprocess/center_crop_224.py
--root path/to/2d_image_folder
--output-dir path/to/output
--is-sketch
```


2. Get Canny edges and apply Dilation Morphology:
```bash
python preprocess/get_canny_dilate.py
--root path/to/2d_image_folder
--output-canny path/to/canny_output
--output-canny-dilate path/to/canny_dilate_output
--is-sketch
```

## 3. Feature Extractor

We extract and save CLIP features of 3D models and text queries:

1. Extract and save model features:
```bash
python feature_extractor.py
--model-root path/to/model_2D_images
--model-output-dir path/to/output_model_feature
--query-file path/to/TextQuery_Test.csv
--text-output-dir path/to/output_text_feature
```

## 4. Retrieval

We perform retrieval using CLIP features:

1. Retrieval by CLIP feature:
```bash
python text_retrieval.py
--model-feature-dir path/to/model_feature
--text-feature-dir path/to/text_feature
--model-original-dir path/to/original_2D_images
--output-vis-dir path/to/visualize_folder
```