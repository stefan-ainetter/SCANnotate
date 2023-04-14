# coding: utf-8
import argparse
import os
import sys

parser = argparse.ArgumentParser(description="Differentiable CAD Model Pose Refinement")
parser.add_argument("--config", type=str, default='ScanNet.ini', help="Configuration file")
parser.add_argument("--data_split", type=str, default="", help="data split")
parser.add_argument("--device", type=str, default="", help="device")

os.environ["CUDA_VISIBLE_DEVICES"] = parser.parse_args().device
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import torch
import copy
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
import pickle
from PoseRefinementModel import PoseRefineModel
from config import load_config
from retrieval_pipeline.CAD_Search_Algos.prepare_scene import Prepare_Scene
from pytorch3d.ops import sample_points_from_meshes
import numpy as np
from pytorch3d.structures import Pointclouds
from retrieval_pipeline.losses import chamfer_distance_one_way
from misc.utils import shapenet_category_dict
from retrieval_pipeline.utils_CAD_retrieval import load_textured_cad_model
import open3d as o3d


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
    config = load_config(config_path)['Pose_Refinement']
    config_general = load_config(config_path)['general']

    n_iter = config.getint('niter')

    w_sil = config.getfloat('weight_sil')
    w_depth = config.getfloat('weight_depth')
    w_sensor = config.getfloat('weight_sensor')
    w_chamfer = config.getfloat('weight_chamfer')

    num_sampled_points = config.getint('num_sampled_points')

    num_k = config.getint('num_k')

    adam_lr = config.getfloat('adam_lr')
    adam_wd = config.getfloat('adam_wd')

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

        data_path = os.path.join(parent, config_general['results_path'], scene_name, config['data_folder'],
                                 scene_name + '.pkl')

        if not os.path.exists(data_path):
            continue

        pkl_file = open(os.path.join(data_path), 'rb')
        scene_obj = pickle.load(pkl_file)

        mesh_scene = prepare_scene_obj.prepare_scene(scene_obj)

        global_cnt = 0
        obj_mesh_all = None

        for count, box_item in enumerate(scene_obj.bbox3d_list):

            num_scales = config.getint('num_scales')
            rotations = config.getstruct('rotations')

            n_views_selected, scene_mesh, mesh_bg, renderer, depth_gt, depth_bg, mask_gt, mask_depth_valid_render_gt, \
                max_depth_gt, mesh_obj, depth_sensor, \
                mask_depth_valid_sensor = prepare_scene_obj.prepare_box_item_for_rendering(box_item,
                                                                                           scene_obj.inst_seg_3d,
                                                                                           mesh_scene,
                                                                                           scene_name,
                                                                                           num_scales, rotations)

            points_target_sampled = sample_points_from_meshes(mesh_obj,
                                                              num_samples=num_sampled_points,
                                                              return_normals=False, return_textures=False)

            pcl_obj_target = Pointclouds(points=points_target_sampled).to(device=device)

            if box_item.obj_id_similarity_list is None:
                cad_obj_for_refine_list = box_item.obj_id_retrieval_list[0:num_k]
            else:
                if isinstance(box_item.obj_id_similarity_list, list):
                    cad_obj_for_refine_list = box_item.obj_id_similarity_list
                else:
                    assert False

            transform_dict = box_item.transform_dict_list[0]
            scale_func_init = transform_dict['scale_transform'].to(device)
            rotate_func_init = transform_dict['rotate_transform'].to(device)
            translate_func_init = transform_dict['translate_transform'].to(device)
            transform_refine = scale_func_init.compose(rotate_func_init.compose(translate_func_init))

            cad_loss_list = []
            obj_id_list = []
            trans_mat_final_list = []

            for cad_model_cnt, cad_obj_id_for_refine in enumerate(cad_obj_for_refine_list):
                cad_obj_class = box_item.cls_name
                synset_id = shapenet_category_dict.get(cad_obj_class)
                obj_id = cad_obj_id_for_refine

                cad_obj_path = os.path.join(parent, config_general['shapenet_path'], synset_id, obj_id, 'models',
                                            'model_normalized.obj')

                verts, faces_, _ = load_obj(cad_obj_path, load_textures=False, device=device)
                faces = faces_.verts_idx

                cad_mesh = Meshes(verts=[verts.squeeze(dim=0)], faces=[faces])

                # Initialize PoseRefine model
                model = PoseRefineModel(bg_mesh=mesh_bg, obj_mesh=cad_mesh,
                                        scale_func_init=scale_func_init,
                                        rotate_func_init=rotate_func_init,
                                        translate_func_init=translate_func_init,
                                        renderer=renderer, device=device,
                                        num_views=n_views_selected, depth_gt=depth_gt, mask_gt=mask_gt,
                                        depth_bg=depth_bg,
                                        max_depth_gt=max_depth_gt, depth_sensor=depth_sensor,
                                        mask_depth_valid_sensor=mask_depth_valid_sensor,
                                        mask_depth_valid_render_gt=mask_depth_valid_render_gt,
                                        depth_out_path=None)

                best_trans_mat = copy.deepcopy(transform_refine)
                min_loss = np.inf

                # Create an optimizer
                optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr, weight_decay=adam_wd)

                for i in range(n_iter):

                    optimizer.zero_grad()
                    loss_sil, loss_depth, loss_sensor, tmesh, depth_final = model()

                    if loss_sil is None:
                        break

                    points_cad_sampled_ = sample_points_from_meshes(tmesh, num_samples=num_sampled_points,
                                                                    return_normals=False, return_textures=False)

                    pcl_cad = Pointclouds(points=points_cad_sampled_).to(device=device)

                    chamfer_dist_x = chamfer_distance_one_way(x=pcl_obj_target.points_padded(),
                                                              y=pcl_cad.points_padded(),
                                                              point_reduction='mean',
                                                              batch_reduction=None,
                                                              norm=1)

                    loss_sil *= w_sil
                    loss_depth *= w_depth
                    loss_sensor *= w_sensor
                    chamfer_dist_x *= w_chamfer

                    loss_render = loss_sil + loss_depth + loss_sensor

                    loss_render_mean = torch.mean(loss_render, dim=0)
                    loss_final = loss_render_mean + chamfer_dist_x

                    loss_final.backward()
                    optimizer.step()

                    if i % 50 == 0:
                        print('Optimization step: ' + str(i))
                        print('Optimizing (loss %.4f)' % loss_final.data)

                    if (loss_final.item() < min_loss):
                        min_loss = loss_final.item()
                        best_trans_mat = model.transform_refine

                    # threshold for early stopping
                    elif loss_final.item() > (min_loss * 1.2):
                        break

                print('-------------------')
                cad_loss_list.append(min_loss)
                trans_mat_final_list.append(best_trans_mat)
                obj_id_list.append(obj_id)

            global_cnt += 1

            obj_id = obj_id_list[np.argmin(np.asarray(cad_loss_list))]
            synset_id = shapenet_category_dict.get(box_item.cls_name)
            model_path = os.path.join(parent, config_general['shapenet_core_path'], synset_id, obj_id, 'models',
                                      'model_normalized.obj')

            transform_best = trans_mat_final_list[np.argmin(np.asarray(cad_loss_list))]
            cad_model_o3d = load_textured_cad_model(model_path, transform_best, box_item.cls_name, device)

            if obj_mesh_all is None:
                obj_mesh_all = cad_model_o3d
            else:
                obj_mesh_all += cad_model_o3d

            box_item.add_transform3D_refined([transform_best])
            box_item.add_obj_id_refined(obj_id)

        pkl_out_file = open(pkl_out_path, 'wb')
        pickle.dump(scene_obj, pkl_out_file)
        pkl_out_file.close()

        tmp = o3d.io.write_triangle_mesh(os.path.join(out_path, scene_name + "_cad_retrieval_refined.ply"),
                                         obj_mesh_all)

if __name__ == "__main__":
    main(parser.parse_args())
