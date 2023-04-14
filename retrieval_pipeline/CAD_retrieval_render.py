# coding: utf-8
import argparse
import os
import sys

parser = argparse.ArgumentParser(description="CAD Model Retrieval")
parser.add_argument("--config", type=str, default='ScanNet.ini', help="Configuration file")
parser.add_argument("--data_split", type=str, default="", help="data split")
parser.add_argument("--device", type=str, default="", help="device")

os.environ["CUDA_VISIBLE_DEVICES"] = parser.parse_args().device
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import torch
import time
import pickle
from config import load_config
from retrieval_pipeline.CAD_Search_Algos.CAD_search_algos import CAD_Search_Algos
from retrieval_pipeline.CAD_Search_Algos.prepare_scene import Prepare_Scene
from retrieval_pipeline.utils_CAD_retrieval import compose_cad_transforms, load_textured_cad_model
import open3d as o3d
import copy
from misc.utils import shapenet_category_dict


def main(args):
    # Setup
    if args.device == '':
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
    else:
        device = 'cuda:' + str(torch.cuda.current_device())

    config_path = os.path.join(parent, 'config', args.config)
    config = load_config(config_path)['CAD_retrieval']
    config_general = load_config(config_path)['general']
    data_split = args.data_split

    prepare_scene_obj = Prepare_Scene(config, config_general, data_split, parent, device)
    scene_list = prepare_scene_obj.load_scene_list()

    for scene_cnt, scene_name in enumerate(scene_list):
        print(scene_name)

        out_path = os.path.join(parent, config_general['results_path'], scene_name, config['out_folder'])
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        pkl_out_path = os.path.join(parent, config_general['results_path'], scene_name, config['out_folder'],
                                    scene_name + '.pkl')

        data_path = os.path.join(parent, config_general['data_path'], config['data_folder'], scene_name,
                                 scene_name + '.pkl')
        if not os.path.exists(data_path):
            continue

        pkl_file = open(data_path, 'rb')
        scene_obj = pickle.load(pkl_file)

        with torch.no_grad():
            obj_mesh_all = None
            mesh_scene = prepare_scene_obj.prepare_scene(scene_obj)

            for count, box_item in enumerate(scene_obj.bbox3d_list):

                cad_transformations, transform_dict = compose_cad_transforms(box_item,
                                                                             config.getstruct('rotations'),
                                                                             config.getint('num_scales')
                                                                             )

                num_scales = config.getint('num_scales')
                rotations = config.getstruct('rotations')

                n_views_selected, scene_mesh, mesh_bg, renderer, depth_gt, depth_bg, mask_gt, \
                    mask_depth_valid_render_gt, max_depth_gt, mesh_obj, depth_sensor, \
                    mask_depth_valid_sensor = prepare_scene_obj.prepare_box_item_for_rendering(box_item,
                                                                                               scene_obj.inst_seg_3d,
                                                                                               mesh_scene,
                                                                                               scene_name,
                                                                                               num_scales, rotations)

                ret_obj = CAD_Search_Algos(parent, config, config_general, renderer, box_item, n_views_selected, device,
                                           mesh_obj,
                                           max_depth_gt, depth_bg, mask_gt, depth_gt, depth_sensor,
                                           mask_depth_valid_sensor,
                                           mask_depth_valid_render_gt, cad_transformations, mesh_bg)

                start = time.time()

                loss_list, idx_list, orientation_list, obj_id_list = ret_obj.run_exhaustive_search()

                if loss_list is None:
                    assert False

                print('Object Category = ' + str(box_item.cls_name))
                print('Search Time = ' + str(time.time() - start))
                print('----------------------------------')

                out_folder = os.path.join(out_path, str(count) + '_' + str(box_item.cls_name))
                if not os.path.exists(out_folder):
                    os.makedirs(out_folder)

                cad_transformations_list = []
                transform_dict_list = []

                for cad_index in range(5):

                    if cad_index >= len(idx_list):
                        break

                    cad_transform_base = cad_transformations[orientation_list[cad_index]].to('cpu')
                    synset_id = shapenet_category_dict.get(box_item.cls_name)
                    obj_id = obj_id_list[cad_index]
                    model_path = os.path.join(parent, config_general['shapenet_core_path'], synset_id, obj_id, 'models',
                                              'model_normalized.obj')

                    cad_transformations_list.append(cad_transform_base)
                    transform_dict_list.append(transform_dict[orientation_list[cad_index]])
                    cad_model_o3d = load_textured_cad_model(model_path, cad_transform_base, box_item.cls_name)

                    tmp = o3d.io.write_triangle_mesh(os.path.join(out_folder, str(cad_index) + ".ply"), cad_model_o3d)

                    if cad_index == 0:
                        if obj_mesh_all is None:
                            obj_mesh_all = cad_model_o3d
                        else:
                            obj_mesh_all += cad_model_o3d

                box_item.add_obj_id_retrieval_list(list(obj_id_list))
                box_item.add_transform3D(cad_transformations_list, transform_dict_list)

                del ret_obj, cad_transformations, transform_dict, cad_transformations_list, transform_dict_list
                torch.cuda.empty_cache()

            mesh_full_bg = copy.deepcopy(mesh_scene)
            mesh_full_bg.remove_vertices_by_index(prepare_scene_obj.all_obj_idx_list)

            out_path_bg_mesh = os.path.join(parent, config_general['results_path'], scene_name)
            tmp = o3d.io.write_triangle_mesh(os.path.join(out_path_bg_mesh, scene_name + '_mesh_bg.ply'), mesh_full_bg)
            tmp = o3d.io.write_triangle_mesh(os.path.join(out_path, scene_name + "_cad_retrieval.ply"), obj_mesh_all)

            pkl_out_file = open(pkl_out_path, 'wb')
            pickle.dump(scene_obj, pkl_out_file)
            pkl_out_file.close()


if __name__ == "__main__":
    main(parser.parse_args())
