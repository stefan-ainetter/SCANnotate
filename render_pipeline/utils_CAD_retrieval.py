import copy
import os

import cv2
import numpy as np
import open3d as o3d
import torch
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.transforms import Transform3d

from misc.utils import SEMANTIC_IDX2NAME, COLOR_DETECTRON2


def normalize_mesh(verts_init, faces, device):
    mesh = Meshes(
        verts=[verts_init],
        faces=[faces],
    )
    bbox = mesh.get_bounding_boxes().squeeze(dim=0)
    bbox = bbox.cpu().detach().numpy()

    center = torch.tensor(bbox.mean(1)).float().to(device)
    vector_x = np.array([bbox[0, 1] - bbox[0, 0], 0, 0])
    vector_y = np.array([0, bbox[1, 1] - bbox[1, 0], 0])
    vector_z = np.array([0, 0, bbox[2, 1] - bbox[2, 0]])

    coeff_x = np.linalg.norm(vector_x)
    coeff_y = np.linalg.norm(vector_y)
    coeff_z = np.linalg.norm(vector_z)

    mesh = mesh.offset_verts(-center)
    transform_func = Transform3d().scale(x=(1 / coeff_x), y=(1 / coeff_y), z=1 / coeff_z).to(device)
    tverts = transform_func.transform_points(mesh.verts_list()[0]).unsqueeze(dim=0)

    mesh = Meshes(
        verts=[tverts.squeeze(dim=0)],
        faces=[faces]
    )

    return mesh, tverts


def compose_cad_transforms(box_item, rotations_list, num_scales):
    transform_dict_list = []
    transform_list = []
    transform_cnt = 0
    scale_func_list = []

    if num_scales not in [1,2]:
        raise ValueError('Wrong value for num_scales. Valid values [1,2]. Current value: ' + str(num_scales) +
                         '. Change value in config file!')


    scale_func_xyz = Transform3d().scale(x=torch.as_tensor(box_item.scale[0]).float(),
                                         y=torch.as_tensor(box_item.scale[1]).float(),
                                         z=torch.as_tensor(box_item.scale[2]).float()
                                         )
    scale_func_list.append(scale_func_xyz)

    # Swapping x-z; important for 90 degree rotations to ensure that the rotated CAD model still fits into the
    # given 3d bounding box
    if num_scales == 2:
        scale_func_zyx = Transform3d().scale(x=torch.as_tensor(box_item.scale[2]).float(),
                                             y=torch.as_tensor(box_item.scale[1]).float(),
                                             z=torch.as_tensor(box_item.scale[0]).float()
                                             )
        scale_func_list.append(scale_func_zyx)

    translation_func = Transform3d().translate(x=torch.as_tensor(box_item.center[0]).float(),
                                               y=torch.as_tensor(box_item.center[1]).float(),
                                               z=torch.as_tensor(box_item.center[2]).float()
                                               )

    rotate_func_tmp = Transform3d().rotate(R=torch.as_tensor(box_item.basis).float())

    for scale_func_id, scale_func in enumerate(scale_func_list):
        for rot_idx, rot in enumerate(rotations_list):
            transform_dict_tmp = {}
            if scale_func_id == 1:
                rotation_transform = Transform3d().rotate_axis_angle(angle=rot + 90, axis='Y', degrees=True)
            else:
                rotation_transform = Transform3d().rotate_axis_angle(angle=rot, axis='Y', degrees=True)

            rotate_func_init = rotation_transform.compose(rotate_func_tmp)

            final_transform = scale_func.compose(rotate_func_init, translation_func)
            transform_dict_tmp['scale_transform'] = scale_func
            transform_dict_tmp['rotate_transform'] = rotate_func_init
            transform_dict_tmp['translate_transform'] = translation_func
            transform_dict_list.append(transform_dict_tmp)

            transform_list.append(final_transform)
            transform_cnt += 1

    return transform_list, transform_dict_list


def load_depth_img(path):
    depth_image = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) / 1000.
    return depth_image


def save_depth_img(depth, depth_out_path, filename=None):
    if depth.ndim == 4:
        depth = depth[:, :, :, 0]

    depth_im_out = None

    for cnt, depth_im in enumerate(depth):
        depth_norm = ((depth_im / np.max(depth_im)) * 255).astype('uint8')
        if depth_im_out is None:
            depth_im_out = depth_norm
        else:
            depth_im_out = np.vstack((depth_im_out, depth_norm))

    if filename is None:
        cv2.imwrite(os.path.join(depth_out_path, 'depth_rendered.png'), depth_im_out)
    else:
        cv2.imwrite(os.path.join(depth_out_path, filename + '.png'), depth_im_out)
    return


def cut_meshes(mesh_o3d, indices_list):
    mesh_o3d_obj = copy.deepcopy(mesh_o3d)
    mesh_o3d_obj = mesh_o3d_obj.select_by_index(indices_list)
    mesh_o3d.remove_vertices_by_index(indices_list)

    face_list_bg = np.asarray(mesh_o3d.triangles)
    face_list_obj = np.asarray(mesh_o3d_obj.triangles)

    mesh_bg = Meshes(
        verts=[torch.tensor(np.asarray(mesh_o3d.vertices)).float()],
        faces=[torch.tensor(np.asarray(face_list_bg))]
    )

    mesh_obj = Meshes(
        verts=[torch.tensor(np.asarray(mesh_o3d_obj.vertices)).float()],
        faces=[torch.tensor(np.asarray(face_list_obj))]
    )
    return mesh_bg, mesh_obj


def load_textured_cad_model(model_path, cad_transform_base, cls_name, device='cpu'):
    try:
        sem_id = list(SEMANTIC_IDX2NAME.keys())[
            list(SEMANTIC_IDX2NAME.values()).index(cls_name)]
    except:
        sem_id = 0
    mesh_color = COLOR_DETECTRON2[sem_id]

    verts, faces_, _ = load_obj(model_path, load_textures=False, device=device)
    faces = faces_.verts_idx

    _, tverts = normalize_mesh(verts, faces, device)

    tverts_final = cad_transform_base.transform_points(tverts)

    if device == 'cpu':
        tverts_final_ary = tverts_final.squeeze(dim=0).detach().numpy()
        faces_ary = faces.detach().numpy().astype(np.int32)

    else:
        tverts_final_ary = tverts_final.squeeze(dim=0).cpu().detach().numpy()
        faces_ary = faces.cpu().detach().numpy().astype(np.int32)

    cad_model_o3d = o3d.geometry.TriangleMesh()
    cad_model_o3d.vertices = o3d.utility.Vector3dVector(tverts_final_ary)
    cad_model_o3d.triangles = o3d.utility.Vector3iVector(faces_ary)
    vertex_n = np.array(cad_model_o3d.vertices).shape[0]
    vertex_colors = np.ones((vertex_n, 3)) * mesh_color
    cad_model_o3d.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    return cad_model_o3d
