import argparse
import spacy
import json
import shutil
import glob
import clip
import os
from PIL import Image
import numpy as np
import torch


def get_text_metadata(text_query_test_file):
    all_data = {}

    with open(text_query_test_file, 'r') as tq:
        tq_lines = tq.readlines()

    for line in tq_lines[1:]:
        line = line.strip()
        text_id, text = line.split(";")
        if text_id not in all_data:
            all_data[text_id] = text
    return all_data


@torch.no_grad()
def get_text_all_case(text_output_dir, query_file):
    nlp = spacy.load('en_core_web_sm')
    # pre_fix = "the "

    all_text_metadata = get_text_metadata(query_file)
    model, preprocess = clip.load("ViT-B/32")
    model.cuda().eval()

    for text_id in all_text_metadata:
        text = all_text_metadata[text_id]
        # full text
        doc = nlp(text)
        det, amod, nsubj, aux, root = "", "", "", "", ""
        for tok in doc:
            # print(tok.dep_)
            if tok.dep_ == "det":
                det = str(tok)
                break

        for tok in doc:
            # print(tok.dep_)
            if tok.dep_ == "amod":
                amod = str(tok)
                break

        for tok in doc:
            if tok.dep_ == "nsubj":
                nsubj = str(tok)
                break

        for tok in doc:
            if tok.dep_ == "aux":
                aux = str(tok)
                break

        for tok in doc:
            if tok.dep_ == "ROOT":
                root = str(tok)
                break

        full_text = text
        animal_text = " ".join([det, nsubj])
        animal_amod_text = " ".join([det, amod, nsubj])
        animal_amod_verb_text = " ".join([det, amod, nsubj, aux, root])

        text_list = [full_text, animal_text, animal_amod_text, animal_amod_verb_text]
        text_tokens = clip.tokenize(text_list).cuda()

        text_features = model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # save vision feature
        output_path = os.path.join(text_output_dir, text_id + ".pt")
        torch.save(text_features, output_path)
        print("Text feature saved at: ", output_path)


def get_model_feature(model_folder_path, model_output_dir):
    all_views = ["view_1", "view_2", "view_3", "view_4"]
    all_model_id = os.listdir(os.path.join(model_folder_path, all_views[0]))

    model, preprocess = clip.load("ViT-B/32")
    model.cuda().eval()

    for model_id in all_model_id:
        all_view_path = [os.path.join(model_folder_path, view, model_id) for view in all_views]
        images = []  # batch size = 48
        for view_path in all_view_path:
            img_paths = glob.glob(view_path + "/*")
            for img_path in img_paths:
                image = Image.open(img_path).convert("RGB")
                images.append(preprocess(image))
        images_input = torch.tensor(np.stack(images)).cuda()
        with torch.no_grad():
            features = model.encode_image(images_input).float()
            features /= features.norm(dim=-1, keepdim=True)

            # save vision feature
            output_path = os.path.join(model_output_dir, model_id + ".pt")
            torch.save(features, output_path)
            print("Model feature (48, 512) saved at: ", output_path)


def main(args):
    get_model_feature(args["model-root"], args["model-output-dir"])
    get_text_all_case(args["test-output-dir"], args["query-file"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-root', type=str,
                        default='./multi_view_image',
                        help='directory store 2D images in 4 views')
    parser.add_argument('--model-output-dir', type=str,
                        default='./output_model_feature',
                        help='directory save 3D model feature')

    parser.add_argument('--query-file', type=str,
                        default='./TextQuery_Test.csv',
                        help='test query file path')
    parser.add_argument('--text-output-dir', type=str,
                        default='./output_text_feature',
                        help='directory save text feature')

    args = parser.parse_args()
    main(args)
