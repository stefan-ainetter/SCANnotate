import torch
from pytorch3d.renderer import RasterizationSettings, MeshRasterizer
from pytorch3d.renderer.cameras import PerspectiveCameras

from retrieval_pipeline.PyTorch3DRenderer.pytorch3d_rasterizer_custom import MeshRendererScannet
from .SimpleShader import SimpleShader


def initialize_renderer(n_views, img_scale, R, T, intrinsics, batch_size, num_orientations_per_mesh, device, img_height,
                        img_width, bin_size=None):
    raster_settings = RasterizationSettings(
        image_size=(int(img_height * img_scale), int(img_width * img_scale)),
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=bin_size,
        perspective_correct=True,
        clip_barycentric_coords=False,
        cull_backfaces=False
    )

    R_world_to_cam = R
    T_world_to_cam = T

    R = torch.as_tensor(R_world_to_cam).to(device)
    R = R.repeat_interleave(repeats=num_orientations_per_mesh * batch_size, dim=0)

    T = torch.as_tensor(T_world_to_cam).to(device)
    T = T.repeat_interleave(repeats=num_orientations_per_mesh * batch_size, dim=0)

    px, py = (intrinsics[0, 2] * img_scale), (intrinsics[1, 2] * img_scale)
    principal_point = torch.as_tensor([px, py])[None].type(torch.FloatTensor).to(device)
    principal_point = principal_point.repeat(n_views, 1)
    principal_point = principal_point.repeat_interleave(repeats=num_orientations_per_mesh * batch_size, dim=0)

    fx, fy = ((intrinsics[0, 0] * img_scale)), ((intrinsics[1, 1] * img_scale))
    focal_length = torch.as_tensor([fx, fy])[None].type(torch.FloatTensor).to(device)
    focal_length = focal_length.repeat(n_views, 1)
    focal_length = focal_length.repeat_interleave(repeats=num_orientations_per_mesh * batch_size, dim=0)

    cameras = PerspectiveCameras(
        focal_length=focal_length,
        principal_point=principal_point,
        in_ndc=False,
        device=device, T=T, R=R,
        image_size=((int(img_height * img_scale), int(img_width * img_scale)),))

    renderer = MeshRendererScannet(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SimpleShader(
            device=device,
            cameras=cameras
        )
    )

    del R, T, principal_point, focal_length
    return renderer


def prepare_GT_data(scene_mesh, mesh_bg, renderer, device):
    fragments_gt = renderer(meshes_world=scene_mesh.to(device))
    depth_gt = fragments_gt.zbuf

    mask_depth_valid_render_gt = torch.zeros_like(depth_gt)
    mask_depth_valid_render_gt[fragments_gt.pix_to_face != -1] = 1

    depth_gt[fragments_gt.pix_to_face == -1] = torch.max(depth_gt[fragments_gt.pix_to_face > 0])
    max_depth_gt = torch.max(depth_gt)

    fragments_bg = renderer(meshes_world=mesh_bg.to(device))
    depth_bg = fragments_bg.zbuf
    depth_bg[fragments_bg.pix_to_face == -1] = max_depth_gt

    mask_gt = torch.zeros_like(depth_gt)
    mask_bg = torch.zeros_like(depth_gt)
    mask_gt[depth_gt < depth_bg] = 1.
    mask_bg[depth_gt >= depth_bg] = 1.

    del fragments_gt, fragments_bg
    return depth_gt, depth_bg, mask_gt, mask_depth_valid_render_gt, max_depth_gt
