import torch
import torch.nn as nn
from pytorch3d.renderer.blending import (
    BlendParams,
    hard_rgb_blend
)
from pytorch3d.renderer.cameras import OpenGLPerspectiveCameras


class SimpleShader(nn.Module):
    def __init__(self, device="cpu", cameras=None):
        super().__init__()

        self.cameras = (
            cameras if cameras is not None else OpenGLPerspectiveCameras(device=device)

        )
        self.blend_params = BlendParams(background_color=[0, 0, 0])

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)

        texels = meshes.sample_textures(fragments)

        images = hard_rgb_blend(texels, fragments, self.blend_params)

        return images  # (N, H, W, 3) RGBA image


class UVsCorrespondenceShader(nn.Module):
    """
    UV correspondence shader will render the model with a custom texture map as it's input.
    No lightning or blending will be applied
    To use the default values, simply initialize the shader with the desired
    device e.g.
    .. code-block::
        shader = UVsCorrespondenceShader(
                blend_params=bp,
                device=device,
                cameras=cameras,
                colormap_padded=colormap_padded
    """

    def __init__(
            self, device="cpu", cameras=None, blend_params=None, colormap=None
    ):
        super().__init__()

        self.cameras = cameras
        self.colormap = colormap
        self.blend_params = blend_params if blend_params is not None else BlendParams(background_color=[0, 0, 0])

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        colormap = kwargs.get("colormap", self.colormap)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of SoftPhongShader"
            raise ValueError(msg)
        texels = meshes.sample_textures(fragments)
        images = hard_rgb_blend(texels, fragments, self.blend_params)
        return images
