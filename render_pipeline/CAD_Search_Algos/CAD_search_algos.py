import os
import time

import numpy as np
import torch
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Pointclouds
from pytorch3d.structures.meshes import join_meshes_as_batch
from pytorch3d.structures.pointclouds import join_pointclouds_as_batch
from torch.utils.data import DataLoader

from render_pipeline.losses import loss_IOU_render_sensor, chamfer_distance_one_way
from render_pipeline.shapenet_core import ShapeNetCore_SCANnotate
from misc.utils import shapenet_category_dict


def collate_batched_pcls(batch):
    """
    Take a list of objects in the form of dictionaries and merge them
    into a single dictionary.

    Args:
        batch: List of dictionaries containing information about objects
            in the dataset.

    Returns:
        collated_dict: Dictionary of collated lists.
    """
    if batch is None or len(batch) == 0:
        return None
    collated_dict = {}
    for k in batch[0].keys():
        collated_dict[k] = [d[k] for d in batch]
    mesh_all_list = []
    points_all = None

    for mesh, points in zip(collated_dict["mesh_list"], collated_dict["pcl_list"]):
        mesh_all_list.append(mesh)
        if points_all is None:
            points_all = points
        else:
            points_all = torch.cat((points_all, points), dim=0)

    collated_dict['mesh'] = join_meshes_as_batch(mesh_all_list, include_textures=False)
    collated_dict['pcl'] = join_pointclouds_as_batch(points_all)
    return collated_dict


class CAD_Search_Algos(object):
    def __init__(self, parent, config, config_general, renderer, box_item, n_views, device, mesh_obj, max_depth_gt,
                 depth_bg, mask_gt, depth_gt, depth_sensor, mask_depth_valid_sensor,
                 mask_depth_valid_render_gt, cad_transformations, mesh_bg, transform_dict=None):

        self.config = config
        self.config_general = config_general
        self.renderer = renderer
        self.box_item = box_item
        self.shapenet_path = os.path.join(parent, config_general['shapenet_path'])
        self.n_views = n_views
        self.img_scale = config.getfloat('img_scale')
        self.batch_size = config.getint('batch_size')
        self.rotations = config.getstruct('rotations')
        self.num_scales = config.getint('num_scales')
        self.num_orientations_per_mesh = len(self.rotations)

        self.w_sil = config.getfloat('weight_sil')
        self.w_depth = config.getfloat('weight_depth')
        self.w_sensor = config.getfloat('weight_sensor')
        self.w_chamfer = config.getfloat('weight_chamfer')

        self.num_sampled_points = config.getint('num_sampled_points')
        self.num_workers = config.getint('num_workers')

        self.mesh_obj = mesh_obj
        self.mesh_bg = mesh_bg
        self.max_depth_GT = max_depth_gt
        self.depth_bg = depth_bg
        self.mask_GT = mask_gt
        self.depth_GT = depth_gt
        self.depth_sensor = depth_sensor
        self.mask_depth_valid_sensor = mask_depth_valid_sensor
        self.mask_depth_valid_render_GT = mask_depth_valid_render_gt

        self.cad_transformations = cad_transformations
        self.transform_dict = transform_dict
        self.shapenet_dataset = None

        self.device = device

    def run_exhaustive_search(self):

        nan_loss_bool = False
        synset_id = shapenet_category_dict.get(self.box_item.cls_name)

        if synset_id is None:
            assert False

        self.shapenet_dataset = ShapeNetCore_SCANnotate(self.shapenet_path, load_textures=False, synsets=synset_id,
                                                        transform=None, scale_func=None, rotations=self.rotations,
                                                        num_views=self.n_views,
                                                        cad_transformations=self.cad_transformations,
                                                        num_sampled_points=self.num_sampled_points,
                                                        cls_name=self.box_item.cls_name)

        self.shapenet_loader = DataLoader(self.shapenet_dataset, shuffle=False, batch_size=self.batch_size,
                                          collate_fn=collate_batched_pcls,
                                          pin_memory=True, num_workers=self.num_workers, drop_last=True)

        points_target_sampled = sample_points_from_meshes(self.mesh_obj, num_samples=self.num_sampled_points,
                                                          return_normals=False, return_textures=False)

        pcl_target_ = Pointclouds(points=points_target_sampled)
        pcl_target = pcl_target_.extend(N=self.num_orientations_per_mesh * self.num_scales).to(
            device=self.device)

        loss_list = []
        model_list = []
        idx_list = []
        orientation_list = []
        obj_id_list = []
        start = time.time()
        for it, batch in enumerate(self.shapenet_loader):

            if it == 51:
               break

            cad_mesh_batch = batch['mesh'].to(device=self.device)
            cad_pcl_batch = batch['pcl'].to(device=self.device)

            fragments = self.renderer(meshes_world=cad_mesh_batch)
            depth_pred = fragments.zbuf
            depth_pred[fragments.pix_to_face == -1] = self.max_depth_GT
            mask_depth_valid_render_pred = torch.zeros_like(depth_pred).to(self.device)
            mask_depth_valid_render_pred[fragments.pix_to_face != -1] = 1

            mask_pred = torch.zeros_like(depth_pred)
            mask_depth_bg = torch.zeros_like(depth_pred)

            mask_pred[depth_pred < self.depth_bg] = 1.
            mask_depth_bg[depth_pred >= self.depth_bg] = 1.

            mask_combined = torch.zeros_like(mask_pred)
            mask_combined[mask_pred == 1] = 1.
            mask_combined[self.mask_GT == 1] = 1.

            depth_final = depth_pred * mask_pred + self.depth_bg * mask_depth_bg

            loss_sil, loss_depth, loss_sensor = loss_IOU_render_sensor(mask_pred, self.mask_GT, self.depth_GT,
                                                                       depth_final,
                                                                       mask_combined, self.depth_sensor,
                                                                       self.mask_depth_valid_sensor,
                                                                       self.mask_depth_valid_render_GT,
                                                                       mask_depth_valid_render_pred)

            # L1 norm for point distance here
            chamfer_dist_x = chamfer_distance_one_way(x=pcl_target.points_padded(),
                                                      y=cad_pcl_batch.points_padded(),
                                                      point_reduction='mean',
                                                      batch_reduction=None,
                                                      norm=1)

            loss_sil *= self.w_sil
            loss_depth *= self.w_depth
            loss_sensor *= self.w_sensor

            chamfer_dist_x *= self.w_chamfer

            loss_sil_reshape = torch.reshape(loss_sil, (self.n_views, self.batch_size * self.num_orientations_per_mesh *
                                                        self.num_scales))
            loss_depth_reshape = torch.reshape(loss_depth,
                                               (self.n_views, self.batch_size * self.num_orientations_per_mesh *
                                                self.num_scales))
            loss_sensor_reshape = torch.reshape(loss_sensor,
                                                (self.n_views, self.batch_size * self.num_orientations_per_mesh *
                                                 self.num_scales))

            loss_render = loss_sil_reshape + loss_depth_reshape + loss_sensor_reshape
            loss_render_mean = torch.mean(loss_render, dim=0)

            loss_final = loss_render_mean + chamfer_dist_x

            if torch.any(torch.isnan(loss_final)):
                nan_loss_bool = True
                break

            loss_batch_min = torch.min(loss_final)
            min_idx = torch.argmin(loss_final).item()

            batch_idx = int(np.floor(min_idx / (self.num_orientations_per_mesh * self.num_scales)))
            orientation_list.append(min_idx)
            loss_list.append(float(loss_batch_min.cpu().detach().numpy()))

            model_list.append(batch['path'][batch_idx])
            idx_list.append(batch['id'][batch_idx])
            obj_id_list.append(batch['obj_id'][batch_idx])

            if it % 100 == 0:
                print(it)
                print('Time = ' + str(time.time() - start))

            del batch, fragments, depth_pred, depth_final, mask_pred, mask_depth_bg, loss_final, loss_batch_min

        del pcl_target

        if nan_loss_bool:
            return None, None, None, None
        else:
            loss_list, idx_list, orientation_list, obj_id_list = \
                zip(*sorted(zip(loss_list, idx_list, orientation_list, obj_id_list)))
            return loss_list, idx_list, orientation_list, obj_id_list
