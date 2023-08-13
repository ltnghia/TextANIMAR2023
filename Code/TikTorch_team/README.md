# TikTorch_team

## 1. Installation
<span id='2-install'></span>

Before installing the repo, we need to install the CUDA driver version >=11.6

We will create a new conda environment:

```bash
conda create -n animar python=3.8 -y
conda activate animar
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

Consider download the original submission from [here](https://drive.google.com/file/d/1eKcgI2og05u0jiqXNvCevw1EF7tw3TSZ/view?usp=drive_link) for full data and weights.

## 2. Training

Training command:

```bash
python train_prompt_ringview.py \
    --view-cnn-backbone efficientnet_v2_s \
    --rings-path data/TextANIMAR2023/3D_Model_References/generated_models \
    --used-rings 3 \
    --train-csv-path data/kfold/train_tex_1.csv \
    --test-csv-path data/kfold/test_tex_1.csv \
    --batch-size 2 \
    --epochs 60 \
    --latent-dim 128 \
    --output-path exps_kfold \
    --view-seq-embedder mha \
    --num-rings-mhas 2 \
    --num-heads 4 \
    --lr-obj 3e-5\
    --lr-txt 3e-5 \
    --reduce-lr \
    --dropout 0.2 \
    --text-model openai/clip-vit-base-patch32 
```
## 3. Inference

Retrieval command:

```bash
python retrieve_prompt_ringview.py \
    --rings-path data/TextANIMAR2023/3D_Model_References/generated_models \
    --info-json exps_kfold/ringview_exp_1/args.json \
    --obj-csv-path data/TextANIMAR2023/3D_Model_References/References.csv \
    --txt-csv-path data/TextANIMAR2023/Test/TextQuery_Test.csv \
    --obj-weight exps_kfold/ringview_exp_1/weights/best_obj_embedder.pth \
    --txt-weight exps_kfold/ringview_exp_1/weights/best_query_embedder.pth \
    --output-path ./text_predicts
```
The retrieval results will be on the directory `text_predicts/ringview_predict_{num}`. We use the file `submission.csv` in this directory to submit the results.

## 4. (Extra) Data manipulation
Please read download_zip_file/TikTorch-TextANIMAR/data/README.md for more details.

## 5. References

This repository is based on the [official baseline](https://github.com/nhtlongcs/SHREC23-ANIMAR-BASELINE) of the organizers.
