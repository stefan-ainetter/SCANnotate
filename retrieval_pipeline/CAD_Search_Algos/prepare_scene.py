import copy
import os

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from pytorch3d.structures import Meshes

from retrieval_pipeline.PyTorch3DRenderer.Torch3DRenderer import initialize_renderer
from retrieval_pipeline.load_ScanNet_data import load_axis_alignment_mat
from retrieval_pipeline.utils_CAD_retrieval import cut_meshes
from retrieval_pipeline.utils_CAD_retrieval import load_depth_img
from misc.utils import transform_ScanNet_to_py3D, alignPclMesh, transform_ARKIT_to_py3D
from pytorch3d.structures.meshes import join_meshes_as_batch

import cv2

class Prepare_Scene():
    def __init__(self, config, config_general, data_split, parent_dir, device):
        self.config = config
        self.config_general = config_general
        self.data_split = data_split
        self.parent = parent_dir
        self.dataset_name = self.config_general['dataset']
        self.device = device
        print('Initialization: device=' + str(self.device))

        self.all_obj_idx_list = None
        self.num_scales = None
        self.rotations = None

    def load_scene_list(self):

        data_path = self.config_general['data_path']
        if self.data_split == '':
            scene_list = os.listdir(os.path.join(self.parent, data_path, 'scans'))
        else:
            if self.dataset_name == 'ScanNet':
                data_split_path = os.path.join(self.parent, 'data/ScanNet_splits', self.data_split)
            else:
                print('data_splits only available for ScanNet dataset; current dataset: ' + str(self.dataset_name))
                assert False
            if not os.path.exists(data_split_path):
                print('data_split file not found: ' + str(self.data_split))
                assert False

            text_file = open(data_split_path, "r")
            scene_list_init = text_file.readlines()
            scene_list = []
            for scene_name in scene_list_init:
                scene_name = scene_name.rstrip()
                scene_list.append(scene_name)

        return scene_list

    def prepare_scene(self, scene_obj):

        self.all_obj_idx_list = []
        data_path = self.config_general['data_path']

        if self.dataset_name == 'ScanNet':

            meta_file_path = os.path.join(self.parent, data_path, 'scans', scene_obj.scene_name,
                                          scene_obj.scene_name + '.txt')

            T_mat = transform_ScanNet_to_py3D()
            align_mat = load_axis_alignment_mat(meta_file_path=meta_file_path)
            align_mat = np.reshape(np.asarray(align_mat), (4, 4))

            scene_path = os.path.join(self.parent, data_path, 'scans', scene_obj.scene_name,
                                      scene_obj.scene_name + '_vh_clean_2.ply')

        elif self.dataset_name == 'custom_dataset':
            T_mat = transform_ScanNet_to_py3D()
            align_mat = np.eye(4)

            scene_path = os.path.join(self.parent, data_path, 'scans', scene_obj.scene_name,
                                      scene_obj.scene_name + '_detection.ply')

        elif self.dataset_name == 'ARKitScenes':
            T_mat = transform_ARKIT_to_py3D()
            align_mat = np.eye(4)

            scene_path = os.path.join(self.parent, data_path, 'scans', scene_obj.scene_name,
                                      scene_obj.scene_name + '_3dod_mesh.ply')
        else:
            assert False

        mesh_scene = o3d.io.read_triangle_mesh(scene_path)
        mesh_scene = alignPclMesh(mesh_scene, axis_align_matrix=align_mat, T=T_mat)

        return mesh_scene

    def prepare_GT_data(self,scene_mesh, mesh_bg, renderer, n_views, device):
        with torch.no_grad():
            scene_mesh = scene_mesh.extend(n_views * self.config.getint('batch_size') * 1)

            fragments_gt = renderer(meshes_world=scene_mesh.to(device))
            depth_gt = fragments_gt.zbuf

            mask_depth_valid_render_gt = torch.zeros_like(depth_gt)
            mask_depth_valid_render_gt[fragments_gt.pix_to_face != -1] = 1

            depth_gt[fragments_gt.pix_to_face == -1] = torch.max(depth_gt[fragments_gt.pix_to_face > 0])
            max_depth_gt = torch.max(depth_gt)

            mesh_bg = mesh_bg.extend(n_views * self.config.getint('batch_size') * 1)

            fragments_bg = renderer(meshes_world=mesh_bg.to(device))
            depth_bg = fragments_bg.zbuf
            depth_bg[fragments_bg.pix_to_face == -1] = max_depth_gt

            mask_gt = torch.zeros_like(depth_gt)
            mask_bg = torch.zeros_like(depth_gt)
            mask_gt[depth_gt < depth_bg] = 1.
            mask_bg[depth_gt >= depth_bg] = 1.

            del fragments_gt, fragments_bg
        return depth_gt, depth_bg, mask_gt, mask_depth_valid_render_gt, mesh_bg

    def prepare_box_item_for_rendering(self, box_item, inst_seg_3d, mesh_scene, scene_name, num_scales, rotations):

        self.num_scales = num_scales
        self.rotations = rotations

        n_views = self.config.getint('n_views')
        data_path = self.config_general['data_path']

        indices = np.where(inst_seg_3d == box_item.inst_seg_id)[0].tolist()

        self.all_obj_idx_list = self.all_obj_idx_list + indices

        mesh_tmp = copy.deepcopy(mesh_scene)
        mesh_bg, mesh_obj = cut_meshes(mesh_tmp, indices)

        scene_mesh = Meshes(
            verts=[torch.tensor(np.asarray(mesh_scene.vertices)).float()],
            faces=[torch.tensor(np.asarray(mesh_scene.triangles))],
        )

        view_parameters = box_item.view_params

        if len(view_parameters['views']) < n_views:
            n_views = len(view_parameters['views'])

        views_select = np.linspace(0, len(view_parameters['views']) - 1, n_views).astype(int)
        R = view_parameters['R'][views_select].squeeze(axis=1)
        T = view_parameters['T'][views_select].squeeze(axis=1)
        intrinsics = view_parameters['intrinsics']

        depth_img_list = np.asarray(view_parameters['depth_img_path'])[views_select].tolist()
        depth_imgs = []

        if self.dataset_name == 'ScanNet':
            for depth_cnt, depth_frame_name in enumerate(depth_img_list):
                depth_img_path_new = os.path.join(self.parent, data_path, 'extracted', scene_name, 'depths',
                                                  depth_frame_name)
                depth_img = load_depth_img(depth_img_path_new)
                depth_imgs.append(depth_img)

        elif self.dataset_name == 'custom_dataset':
            for depth_cnt, depth_frame_name in enumerate(depth_img_list):
                depth_img_path_new = os.path.join(self.parent, data_path, 'extracted', scene_name, 'depths',
                                                  depth_frame_name + '.png')
                depth_img = load_depth_img(depth_img_path_new)
                depth_imgs.append(depth_img)
                #depth_out_path = '/home/stefan/PycharmProjects/SCANnotate_test/data/custom_dataset/debug/depth_out'
                #depth_norm = ((depth_img / np.max(depth_img)) * 255).astype('uint8')
                #cv2.imwrite(os.path.join(depth_out_path, str(depth_cnt) + '_depth.png'), depth_norm)

        elif self.dataset_name == 'ARKitScenes':
            for depth_cnt, depth_frame_name in enumerate(depth_img_list):
                depth_img_path_new = os.path.join(self.parent, data_path, 'scans', scene_name, scene_name + '_frames',
                                                  'lowres_depth', depth_frame_name)
                depth_img = load_depth_img(depth_img_path_new)
                depth_imgs.append(depth_img)

        depth_imgs_ary = np.asarray(depth_imgs)
        depth_imgs_ary = np.expand_dims(depth_imgs_ary, axis=1)
        depth_sensor = torch.from_numpy(depth_imgs_ary).to(self.device)
        depth_sensor = F.interpolate(depth_sensor,
                                     scale_factor=(
                                         self.config.getfloat('img_scale'), self.config.getfloat('img_scale')),
                                     )
        depth_sensor = depth_sensor.permute((0, 2, 3, 1)).to(self.device)
        depth_sensor = torch.repeat_interleave(depth_sensor,
                                               (self.num_scales *
                                                len(self.rotations)), dim=0)
        mask_depth_valid_sensor = torch.zeros_like(depth_sensor).to(self.device)
        mask_depth_valid_sensor[depth_sensor > 0] = 1

        depth_gt_tensor = None
        depth_bg_tensor = None
        mask_gt_tensor = None
        mask_depth_valid_render_gt_tensor = None
        mesh_bg_list = []
        for view_cnt, (R_cnt,T_cnt) in enumerate(zip(R,T)):
            R_cnt = np.expand_dims(R_cnt,axis=0)
            T_cnt = np.expand_dims(T_cnt,axis=0)

            renderer_scene = initialize_renderer(1, self.config.getfloat('img_scale'), R_cnt, T_cnt, intrinsics,
                                                 self.config.getint('batch_size'),
                                                 1,
                                                 self.device,
                                                 self.config_general.getfloat('img_height'),
                                                 self.config_general.getfloat('img_width'))

            depth_gt, depth_bg, mask_gt, mask_depth_valid_render_gt, mesh_bg = self.prepare_GT_data(scene_mesh,
                                                                                                    mesh_bg,
                                                                                                    renderer_scene,
                                                                                                        1,
                                                                                                    self.device)
            del renderer_scene

            if depth_gt_tensor is None:
                depth_gt_tensor = copy.deepcopy(depth_gt)
                depth_bg_tensor = copy.deepcopy(depth_bg)
                mask_gt_tensor = copy.deepcopy(mask_gt)
                mask_depth_valid_render_gt_tensor = copy.deepcopy(mask_depth_valid_render_gt)
            else:
                depth_gt_tensor = torch.cat([depth_gt_tensor, depth_gt], dim=0)
                depth_bg_tensor = torch.cat([depth_bg_tensor, depth_bg], dim=0)
                mask_gt_tensor = torch.cat([mask_gt_tensor, mask_gt], dim=0)
                mask_depth_valid_render_gt_tensor = torch.cat([mask_depth_valid_render_gt_tensor,
                                                               mask_depth_valid_render_gt], dim=0)

            mesh_bg_list.append(mesh_bg)

        del scene_mesh
        torch.cuda.empty_cache()

        mesh_bg_tensor = join_meshes_as_batch(mesh_bg_list)
        max_depth_gt = torch.max(depth_gt_tensor)

        renderer = initialize_renderer(n_views, self.config.getfloat('img_scale'), R, T, intrinsics,
                                       self.config.getint('batch_size'),
                                       len(self.rotations) * self.num_scales,
                                       self.device,
                                       self.config_general.getfloat('img_height'),
                                       self.config_general.getfloat('img_width'))

        return n_views, mesh_bg_tensor, renderer, depth_gt_tensor, depth_bg_tensor, mask_gt_tensor, \
            mask_depth_valid_render_gt_tensor, \
            max_depth_gt, mesh_obj, depth_sensor, mask_depth_valid_sensor
