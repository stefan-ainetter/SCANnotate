import numpy as np
import torch
import torch.nn as nn
from pytorch3d.structures import Meshes
from pytorch3d.transforms import *

from losses import loss_pose_refine


class PoseRefineModel(nn.Module):
    def __init__(self, bg_mesh, obj_mesh,
                 scale_func_init,
                 rotate_func_init,
                 translate_func_init,
                 renderer, device,
                 num_views, depth_gt, mask_gt, depth_bg, max_depth_gt,
                 depth_sensor, mask_depth_valid_sensor, mask_depth_valid_render_gt, depth_out_path):
        super().__init__()
        self.mesh_bg = bg_mesh
        self.obj_mesh = obj_mesh
        self.scale_func_init = scale_func_init
        self.rotate_func_init = rotate_func_init
        self.translate_func_init = translate_func_init
        self.device = device
        self.renderer = renderer
        self.num_views = num_views
        self.refined_trans_func = None
        self.max_depth_GT = max_depth_gt
        self.register_buffer('depth_GT', depth_gt)
        self.register_buffer('mask_GT', mask_gt)
        self.register_buffer('depth_bg', depth_bg)

        self.register_buffer('depth_sensor', depth_sensor)
        self.register_buffer('mask_depth_valid_sensor', mask_depth_valid_sensor)
        self.register_buffer('mask_depth_valid_render_GT', mask_depth_valid_render_gt)
        self.depth_out_path = depth_out_path
        self.mask_depth_valid_render_pred = None

        self.translate_func_refined = None
        self.rot_func_refined = None
        self.scale_func_refined = None
        self.transform_refine = None

        self.scale_offset = None
        self.rotate_offest = None
        self.translate_offset = None

        tmp_params = torch.from_numpy(np.array([0., 0., 0., 1., 1., 1., 0., 0., 0.], dtype=np.float32)).to(self.device)
        self.transform_params = nn.Parameter(tmp_params, requires_grad=True)

    def reset_model(self, obj_mesh_, scale_func_init_, rotate_func_init_, translate_func_init_):
        self.scale_func_init = scale_func_init_
        self.rotate_func_init = rotate_func_init_
        self.translate_func_init = translate_func_init_
        self.obj_mesh = obj_mesh_
        tmp_params = torch.from_numpy(np.array([0., 0., 0., 1., 1., 1., 0., 0., 0.], dtype=np.float32)).to(self.device)

        self.transform_params = nn.Parameter(tmp_params, requires_grad=True)

        self.translate_func_refined = None
        self.rot_func_refined = None
        self.scale_func_refined = None
        self.transform_refine = None

        self.scale_offset = None
        self.rotate_offest = None
        self.translate_offset = None

    def forward(self):

        transform_parmas_np = self.transform_params.detach().squeeze().cpu().numpy()
        if np.any(np.isnan(transform_parmas_np)):
            print('-----------------')
            print('NanNs in transform parameters')
            print(transform_parmas_np)
            return None, None, None, None, None

        roll = self.transform_params[0]
        pitch = self.transform_params[1]
        yaw = self.transform_params[2]

        scale_x = self.transform_params[3]
        scale_y = self.transform_params[4]
        scale_z = self.transform_params[5]

        trans_x = self.transform_params[6]
        trans_y = self.transform_params[7]
        trans_z = self.transform_params[8]

        # 1) base translate
        # 2) translate offset
        translate_func_offset = Transform3d(device=self.device).translate(x=trans_x, y=trans_y, z=trans_z)
        translate_func = self.translate_func_init.compose(translate_func_offset)
        self.translate_func_refined = translate_func
        self.translate_offset = translate_func_offset

        # 3) base scale
        # 4) scale offset
        scale_func_offset = Transform3d(device=self.device).scale(x=scale_x, y=scale_y, z=scale_z)
        scale_func = self.scale_func_init.compose(scale_func_offset)
        self.scale_func_refined = scale_func
        self.scale_offset = scale_func_offset

        # 5)  base rotation (orientation + scene rot)
        # 6) rotation offset
        rot_func_offset = Transform3d(device=self.device).rotate_axis_angle(angle=roll, axis='X',
                                                                            degrees=False).rotate_axis_angle(
            angle=pitch, axis='Y',
            degrees=False).rotate_axis_angle(angle=yaw, axis='Z', degrees=False)
        rot_func = self.rotate_func_init.compose(rot_func_offset)

        self.rot_func_refined = rot_func
        self.rotate_offest = rot_func_offset

        transform_refine = scale_func.compose(rot_func.compose(translate_func))
        self.transform_refine = transform_refine

        tverts = transform_refine.transform_points(self.obj_mesh.verts_list()[0])
        faces = self.obj_mesh.faces_list()[0]

        tmesh = Meshes(
            verts=[tverts.to(self.device)],
            faces=[faces.to(self.device)]
        )

        mesh_joint = tmesh.extend(self.num_views)

        fragments = self.renderer(meshes_world=mesh_joint)
        depth_pred = fragments.zbuf
        depth_pred[fragments.pix_to_face == -1] = self.max_depth_GT
        if self.mask_depth_valid_render_pred is None:
            self.mask_depth_valid_render_pred = torch.zeros_like(depth_pred).to(self.device)
            self.mask_depth_valid_render_pred[fragments.pix_to_face != -1] = 1

        mask_pred = torch.zeros_like(depth_pred)
        mask_depth_bg = torch.zeros_like(depth_pred)

        mask_pred[depth_pred < self.depth_bg] = 1.
        mask_depth_bg[depth_pred >= self.depth_bg] = 1.

        mask_combined = torch.zeros_like(mask_pred)
        mask_combined[mask_pred == 1] = 1.
        mask_combined[self.mask_GT == 1] = 1.

        depth_final = depth_pred * mask_pred + self.depth_bg * mask_depth_bg

        loss_sil, loss_depth, loss_sensor = loss_pose_refine(mask_pred, self.mask_GT, self.depth_GT,
                                                             depth_final, mask_combined, self.depth_sensor,
                                                             self.mask_depth_valid_sensor,
                                                             self.mask_depth_valid_render_GT,
                                                             self.mask_depth_valid_render_pred, self.device)

        return loss_sil, loss_depth, loss_sensor, tmesh, depth_final
