# coding: utf-8
import argparse
import os
import sys

parser = argparse.ArgumentParser(description="CAD Model Clustering and Cloning")
parser.add_argument("--config", type=str, default='ScanNet.ini', help="Configuration file")
parser.add_argument("--data_split", type=str, default="", help="data split")
parser.add_argument("--device", type=str, default="", help="device")

os.environ["CUDA_VISIBLE_DEVICES"] = parser.parse_args().device
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import torch
from pytorch3d.loss import chamfer_distance as chamfer_dist_py3d
from pytorch3d.transforms import Transform3d
import pickle
from config import load_config
from pytorch3d.io import load_obj
from misc.utils import shapenet_category_dict
from pytorch3d.structures.meshes import join_meshes_as_batch
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Pointclouds
from render_pipeline.CAD_Search_Algos.prepare_scene import Prepare_Scene
from pytorch3d.structures import Meshes
from losses import loss_IOU_render_sensor, chamfer_distance_one_way
import numpy as np
from render_pipeline.utils_CAD_retrieval import load_textured_cad_model
import open3d as o3d


def prepare_meshes_for_clustering(shapenet_path, scene_obj, target_cls, num_sampled_points, device):
    mesh_list = []
    obj_idx_map = []

    for count_main, box_item_ in enumerate(scene_obj.bbox3d_list):

        if box_item_.cls_name != target_cls:
            continue
        obj_idx_map.append(count_main)

        synset_id = shapenet_category_dict.get(box_item_.cls_name)
        obj_id = box_item_.obj_id_retrieval_list[0]
        cad_obj_path = os.path.join(parent, shapenet_path, synset_id, obj_id, 'models', 'model_normalized.obj')
        verts, faces, _ = load_obj(cad_obj_path, load_textures=False, device=device)

        faces = faces.verts_idx

        scale_func_init = Transform3d(device=device).scale(x=torch.as_tensor(box_item_.scale[0]).float(),
                                                           y=torch.as_tensor(box_item_.scale[1]).float(),
                                                           z=torch.as_tensor(box_item_.scale[2]).float()
                                                           )

        tverts = scale_func_init.transform_points(verts.squeeze(dim=0))

        tmp_mesh = Meshes(
            verts=[tverts],
            faces=[faces]
        )

        mesh_list.append(tmp_mesh)

    mesh_batch = join_meshes_as_batch(mesh_list, include_textures=False)

    points_gt_sampled = sample_points_from_meshes(mesh_batch, num_samples=num_sampled_points,
                                                  return_normals=False, return_textures=False)

    return points_gt_sampled, obj_idx_map


def calc_sim_loss_mat(pcl_gt, chamfer_threshold):
    loss_list = []

    for obj_idx, pcl in enumerate(pcl_gt):
        pcl_sample = pcl.extend(len(pcl_gt))
        loss_chamfer, loss_normals = chamfer_dist_py3d(x=pcl_sample.points_padded(), y=pcl_gt.points_padded(),
                                                       point_reduction='mean',
                                                       batch_reduction=None)

        loss_chamfer_np = loss_chamfer.cpu().detach().numpy()
        loss_list.append(loss_chamfer_np)

    loss_mat_full = np.asarray(loss_list)
    row, col = np.diag_indices(loss_mat_full.shape[0])
    loss_mat = np.copy(loss_mat_full)
    loss_mat[row, col] = 1
    loss_mat = np.triu(loss_mat)
    loss_mat = np.triu(loss_mat)
    loss_mat[loss_mat > chamfer_threshold] = 1.

    cluster_list = []

    triu_shape = loss_mat[np.triu_indices(loss_mat.shape[0])].shape[0]

    for i in range(triu_shape):
        A = loss_mat[np.triu_indices(loss_mat.shape[0])]

        if np.min(A) == 1.:
            break

        row_ids, col_ids = np.where(loss_mat == np.min(A))
        for (row, col) in zip(row_ids, col_ids):
            if not cluster_list:
                cluster_list.append([row, col])
                loss_mat[row, col] = 1
                break

            for cluster in cluster_list:
                appended = False
                if row in cluster:
                    if not col in cluster:
                        cluster.append(col)
                    appended = True
                    break

                if col in cluster:
                    if not row in cluster:
                        cluster.append(row)
                    appended = True
                    break

            if not appended:
                cluster_list.append([row, col])
            loss_mat[row, col] = 1

    return cluster_list


def merge_clusters(cluster_list, obj_idx_map):
    for cluster_cnt_i, cluster_i in enumerate(cluster_list):
        del_list = []
        for cluster_cnt_j, cluster_j in enumerate(cluster_list):
            if cluster_cnt_i == cluster_cnt_j:
                continue
            if bool(set(cluster_i) & set(cluster_j)):
                new_list = list(set(cluster_i).union(set(cluster_j)))
                cluster_list[cluster_cnt_i] = new_list
                cluster_i = new_list
                del_list.append(cluster_j)
        for del_cluster in del_list:
            cluster_list.remove(del_cluster)

    cluster_ids_real_list = []
    for cluster in cluster_list:
        new_cluster = []
        for id in cluster:
            new_cluster.append(obj_idx_map[id])
        cluster_ids_real_list.append(new_cluster)

    return cluster_ids_real_list


def object_clustering(shapenet_path, scene_obj, num_sampled_points, device, target_cls='table',
                      chamfer_threshold=0.01):
    pcl_list, obj_idx_map = prepare_meshes_for_clustering(shapenet_path, scene_obj, target_cls, num_sampled_points,
                                                          device)

    if len(pcl_list) < 2:
        return None

    pcl_gt = Pointclouds(points=pcl_list).to(device=device)

    cluster_list = calc_sim_loss_mat(pcl_gt, chamfer_threshold)

    cluster_ids_list = merge_clusters(cluster_list, obj_idx_map)

    return cluster_ids_list


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
    config = load_config(config_path)['CAD_retrieval_similarity']
    config_general = load_config(config_path)['general']

    shapenet_path = config_general['shapenet_path']

    rotations = config.getstruct('rotations')
    num_orientations_per_mesh = len(rotations)

    w_sil = config.getfloat('weight_sil')
    w_depth = config.getfloat('weight_depth')
    w_sensor = config.getfloat('weight_sensor')
    w_chamfer = config.getfloat('weight_chamfer')

    num_sampled_points_clustering = config.getint('num_sampled_points_clustering')
    num_sampled_points = config.getint('num_sampled_points')
    target_cls_list = config.getstruct('target_cls_list')
    chamfer_threshold = config.getfloat('chamfer_threshold')

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

        pkl_file = open(data_path, 'rb')
        scene_obj = pickle.load(pkl_file)

        with torch.no_grad():
            obj_mesh_all = None
            mesh_scene = prepare_scene_obj.prepare_scene(scene_obj)
            all_clustered_obj_id_list = []
            for target_cls in target_cls_list:

                cluster_idx_lists = object_clustering(shapenet_path, scene_obj, num_sampled_points_clustering, device,
                                                      target_cls,
                                                      chamfer_threshold)
                if cluster_idx_lists is None:
                    continue

                for cluster_idx_list in cluster_idx_lists:
                    loss_mat_list = []

                    for count_main in cluster_idx_list:
                        box_item_main = scene_obj.bbox3d_list[count_main]

                        num_scales = config.getint('num_scales')
                        rotations = config.getstruct('rotations')

                        n_views_selected, scene_mesh, mesh_bg, renderer, depth_gt, depth_bg, mask_gt, \
                            mask_depth_valid_render_gt, max_depth_gt, mesh_obj, depth_sensor, \
                            mask_depth_valid_sensor = prepare_scene_obj.prepare_box_item_for_rendering(box_item_main,
                                                                                                       scene_obj.inst_seg_3d,
                                                                                                       mesh_scene,
                                                                                                       scene_name,
                                                                                                       num_scales,
                                                                                                       rotations)

                        points_gt_sampled = sample_points_from_meshes(mesh_obj, num_samples=num_sampled_points,
                                                                      return_normals=False, return_textures=False)
                        pcl_gt = Pointclouds(points=points_gt_sampled).to(device=device)

                        transform_refine_main = box_item_main.transform3d_list[0].to(device)
                        loss_per_box_list = []

                        for count in cluster_idx_list:
                            box_item = scene_obj.bbox3d_list[count]

                            synset_id = shapenet_category_dict.get(box_item.cls_name)
                            obj_id = box_item.obj_id_retrieval_list[0]
                            cad_obj_path = os.path.join(parent, shapenet_path, synset_id, obj_id, 'models',
                                                        'model_normalized.obj')

                            verts, faces, _ = load_obj(cad_obj_path, load_textures=False, device=device)
                            faces = faces.verts_idx
                            tverts = transform_refine_main.transform_points(verts.squeeze(dim=0))

                            tmp_mesh = Meshes(
                                verts=[tverts],
                                faces=[faces]
                            )
                            points_cad_sampled = sample_points_from_meshes(tmp_mesh, num_samples=num_sampled_points,
                                                                           return_normals=False, return_textures=False)
                            pcl_cad = Pointclouds(points=points_cad_sampled).to(device=device)

                            tmp_mesh = tmp_mesh.extend(n_views_selected * num_orientations_per_mesh * num_scales)
                            fragments = renderer(meshes_world=tmp_mesh)
                            depth_pred = fragments.zbuf
                            depth_pred[fragments.pix_to_face == -1] = max_depth_gt
                            mask_depth_valid_render_pred = torch.zeros_like(depth_pred).to(device)
                            mask_depth_valid_render_pred[fragments.pix_to_face != -1] = 1

                            mask_pred = torch.zeros_like(depth_pred)
                            mask_depth_bg = torch.zeros_like(depth_pred)

                            mask_pred[depth_pred < depth_bg] = 1.
                            mask_depth_bg[depth_pred >= depth_bg] = 1.

                            mask_combined = torch.zeros_like(mask_pred)
                            mask_combined[mask_pred == 1] = 1.
                            mask_combined[mask_gt == 1] = 1.

                            depth_final = depth_pred * mask_pred + depth_bg * mask_depth_bg

                            loss_sil, loss_depth, loss_sensor = loss_IOU_render_sensor(mask_pred, mask_gt, depth_gt,
                                                                                       depth_final,
                                                                                       mask_combined, depth_sensor,
                                                                                       mask_depth_valid_sensor,
                                                                                       mask_depth_valid_render_gt,
                                                                                       mask_depth_valid_render_pred)

                            chamfer_dist_x = chamfer_distance_one_way(x=pcl_gt.points_padded(),
                                                                      y=pcl_cad.points_padded(),
                                                                      point_reduction='mean',
                                                                      batch_reduction=None)

                            loss_sil *= w_sil
                            loss_depth *= w_depth
                            loss_sensor *= w_sensor
                            chamfer_dist_x *= w_chamfer

                            loss = loss_sil + loss_depth + loss_sensor

                            loss_reshape = torch.reshape(loss,
                                                         (n_views_selected, num_orientations_per_mesh * num_scales))

                            # losses have to be summed up for n_views
                            loss_render_mean = torch.mean(loss_reshape, dim=0)
                            loss_final = loss_render_mean + chamfer_dist_x
                            loss_per_box_list.append(loss_final.item())

                        loss_mat_list.append(np.asarray(loss_per_box_list))

                    loss_mat = np.asarray(loss_mat_list)
                    loss_per_obj = np.sum(loss_mat, axis=0)  # sum over all views for each object
                    best_obj_id = cluster_idx_list[np.argmin(loss_per_obj)]

                    obj_id_best = scene_obj.bbox3d_list[best_obj_id].obj_id_retrieval_list[0]
                    cad_obj_class = scene_obj.bbox3d_list[best_obj_id].cls_name
                    synset_id = shapenet_category_dict.get(cad_obj_class)
                    cad_obj_path = os.path.join(parent, config_general['shapenet_core_path'], synset_id, obj_id_best,
                                                'models',
                                                'model_normalized.obj')

                    cluster_idx_list_new = cluster_idx_list

                    if not all_clustered_obj_id_list:
                        all_clustered_obj_id_list = cluster_idx_list_new
                    else:
                        all_clustered_obj_id_list += cluster_idx_list_new

                    # save best results
                    for count in cluster_idx_list_new:
                        box_item = scene_obj.bbox3d_list[count]
                        box_item.add_obj_id_similarity([obj_id_best])

                        # Visualize best model
                        transform_refine = box_item.transform3d_list[0].to(device)
                        cad_model_o3d = load_textured_cad_model(cad_obj_path, transform_refine, box_item.cls_name,
                                                                device)

                        if obj_mesh_all is None:
                            obj_mesh_all = cad_model_o3d
                        else:
                            obj_mesh_all += cad_model_o3d

            # Visualization of remaining objects
            for count_main, box_item_ in enumerate(scene_obj.bbox3d_list):
                if count_main in all_clustered_obj_id_list:
                    continue

                box_item_.add_obj_id_similarity(None)

                cad_obj_class = box_item_.cls_name
                obj_id = box_item_.obj_id_retrieval_list[0]

                synset_id = shapenet_category_dict.get(cad_obj_class)
                cad_obj_path_new = os.path.join(parent, config_general['shapenet_core_path'], synset_id, obj_id,
                                                'models',
                                                'model_normalized.obj')

                if os.path.exists(cad_obj_path_new):
                    cad_obj_path = cad_obj_path_new

                transform_refine = box_item_.transform3d_list[0].to(device)

                cad_model_o3d = load_textured_cad_model(cad_obj_path, transform_refine, box_item_.cls_name, device)

                if obj_mesh_all is None:
                    obj_mesh_all = cad_model_o3d
                else:
                    obj_mesh_all += cad_model_o3d

            pkl_out_file = open(pkl_out_path, 'wb')
            pickle.dump(scene_obj, pkl_out_file)
            pkl_out_file.close()
            tmp = o3d.io.write_triangle_mesh(os.path.join(out_path, scene_name + "_cad_retrieval_sim.ply"),
                                             obj_mesh_all)


if __name__ == "__main__":
    main(parser.parse_args())
