import argparse
import shutil
import glob
import os
import numpy as np
import torch


def sort_func(e):
    return e[1]


@torch.no_grad()
def text_image_retrieval(model_path, text_path, output_vis_dir, model_original_dir):
    num_side = 48
    num_view = 4
    num_max = 4
    num_text_case = 4
    top_k = 711  # number of candidates
    all_model_path = glob.glob(model_path + "/*")
    all_text_path = glob.glob(text_path + "/*")

    # load text feature
    num_text_feature = len(all_text_path)
    all_text_feature = torch.zeros((num_text_feature * num_text_case, 512))
    for i, text_path in enumerate(all_text_path):
        text_feature = torch.load(text_path)
        all_text_feature[i * num_text_case: (i + 1) * num_text_case] = text_feature

    # load model feature
    num_model_feature = len(all_model_path)
    all_model_feature = torch.zeros((num_model_feature * num_side, 512))
    for i, model_path in enumerate(all_model_path):
        model_feature = torch.load(model_path)
        all_model_feature[i * num_side: (i + 1) * num_side] = model_feature

    # get similarity for each sketch
    write_str = ""
    all_model_feature = all_model_feature.cpu().numpy()
    all_text_feature = all_text_feature.cpu().numpy()
    for i in range(len(all_text_feature)):
        final_score = np.zeros(shape=(len(all_model_path),), dtype=np.float64)
        text_feature = all_text_feature[i * num_text_case: (i + 1) * num_text_case]
        similarity = text_feature @ all_model_feature.T  # (4, 711 * 48)
        similarity = similarity.reshape(num_text_case, num_model_feature, num_view, -1)  # shape = (4, 711, 4, 12)

        for s, similarity_cate in enumerate(similarity):
            cate_score = np.zeros(shape=(num_text_case,), dtype=np.float64)
            # get top 4
            for j, multi_views in enumerate(similarity_cate):
                multi_view_score = np.zeros(shape=(num_view,), dtype=np.float64)
                for k, view in enumerate(multi_views):
                    top_4_index = np.argpartition(view, -num_max)[-num_max:]
                    top_4_max = view[top_4_index]
                    view_score = 1. * np.sum(top_4_max) / num_max
                    multi_view_score[k] = view_score
                model_score = np.max(multi_view_score)
                cate_score[j] = model_score
            final_score[s] = cate_score  # shape = (711,)

        # get top k candidates
        top_k_score_index = np.argpartition(final_score, -top_k)[-top_k:]
        top_k_score = final_score[top_k_score_index]

        index_value = list(np.stack((top_k_score_index, top_k_score), axis=1))
        index_value.sort(key=sort_func, reverse=True)
        index_value = np.array(index_value)  # shape=(100,)

        # save candidate with sketch in the same folder for visualize
        top_k_score_index = index_value[:, 0].astype(np.int32)
        top_k_score = index_value[:, 1]

        text_feature_path = all_text_path[i]
        text_name = os.path.basename(text_feature_path)
        text_id = text_name.split(".")[0]

        output_folder = os.path.join(output_vis_dir, text_id)
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

        # write result
        write_str = write_str + text_id
        for j, ik in enumerate(top_k_score_index):
            model_id = os.path.basename(all_model_path[ik]).split(".")[0]
            write_str = write_str + "," + model_id
            if j < 50:
                model_folder = os.path.join(model_original_dir, "view_1", model_id)
                img_path = glob.glob(model_folder + "/*")[0]
                new_img_path = os.path.join(output_folder, str(round(top_k_score[j], 8)) + ".png")
                # new_img_path = os.path.join(output_folder, model_id + ".png")
                shutil.copyfile(img_path, new_img_path)
        write_str = write_str + "\n"

        if i % 20 == 0:
            print("Done {} / {}".format(i, num_text_feature))

    output_folder = r"/media/phongquang/SSD2T_3/workspace/Phong/project/VisionLanguage/TextANIMAR/source/TextANIMAR/output_analyse"
    csv_output_path = os.path.join(output_folder, "THP_TextANIMAR2023.csv")
    with open(csv_output_path, "w") as wf:
        wf.write(write_str)
    print("Result saved at: ", csv_output_path)


def main():
    """ Step 1: get text feature """
    # save_dir = "/media/phongquang/SSD2T_3/workspace/Phong/project/VisionLanguage/TextANIMAR/data/SketchANIMAR2023/Final/feature"
    # text_query_test_file = "/media/phongquang/SSD2T_3/workspace/Phong/project/VisionLanguage/TextANIMAR/data/SketchANIMAR2023/Final/TextQuery_Test.csv"
    # get_text_all_case(save_dir, text_query_test_file)

    """ Step 2: get model feature """
    model_original_dir = "all_views_224_crop"

    """ Step 3: phase 1 retrieval """
    model_dir = "features/model"
    text_dir = "features/text_animal_amod_verb"
    output_vis_dir = "result_visualize/text_max_6/text_animal_amod_verb"
    text_image_retrieval(model_dir,
                         text_dir,
                         output_vis_dir,
                         model_original_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-feature-dir', type=str,
                        default='./output_model_feature',
                        help='directory store model feature')
    parser.add_argument('--text-feature-dir', type=str,
                        default='./output_text_feature',
                        help='directory store text feature')
    parser.add_argument('--model-original-dir', type=str,
                        default='./model_original_dir',
                        help='directory store 2D image of 3D object model')
    parser.add_argument('--output-vis-dir', type=str,
                        default='./output_vis_dir',
                        help='directory save visualize image')

    args = parser.parse_args()
    main(args)
