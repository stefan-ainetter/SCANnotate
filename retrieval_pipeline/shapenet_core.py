import copy
import os
from typing import Dict

import torch
from pytorch3d.datasets.shapenet_base import ShapeNetBase
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.structures.meshes import join_meshes_as_batch


class ShapeNetCore_SCANnotate(ShapeNetBase):  # pragma: no cover
    """
    This class loads ShapeNetCore from a given directory into a Dataset object.
    ShapeNetCore is a subset of the ShapeNet dataset and can be downloaded from
    https://www.shapenet.org/.
    """

    def __init__(
            self,
            data_dir,
            load_textures: bool = True,
            synsets=None,
            texture_resolution: int = 4,
            transform=None,
            scale_func=None,
            rotations=[0],
            num_views=1,
            cad_transformations=None,
            num_sampled_points=None,
            cls_name=None
    ) -> None:
        """
        """
        super().__init__()
        self.shapenet_dir = data_dir
        self.load_textures = load_textures
        self.texture_resolution = texture_resolution
        self.model_paths = []
        self.param_file_paths = []
        self.transform = transform
        self.scale_func = scale_func
        self.rotations = rotations
        self.num_views = num_views
        self.cad_transformations = cad_transformations
        self.num_sampled_points = num_sampled_points
        self.cls_label = cls_name
        self.synset_id = synsets

        self.CAD_paths, self.obj_ids = self._load_paths()

    def _load_paths(self):
        cls_path = os.path.join(self.shapenet_dir, self.synset_id)
        obj_ids = os.listdir(cls_path)
        paths = [os.path.join(cls_path, x.strip(), 'models', 'model_normalized.obj') for x in obj_ids]

        obj_ids_valid = []
        paths_valid = []
        for obj_id, path in zip(obj_ids, paths):
            if os.path.exists(path):
                paths_valid.append(path)
                obj_ids_valid.append(obj_id)

        return paths_valid, obj_ids_valid

    def __getitem__(self, idx: int) -> Dict:
        """
        Read a model by the given index.

        Args:
            idx: The idx of the model to be retrieved in the dataset.

        Returns:
            dictionary with following keys:
            - verts: FloatTensor of shape (V, 3).
            - faces: LongTensor of shape (F, 3) which indexes into the verts tensor.
            - synset_id (str): synset id
            - model_id (str): model id
            - label (str): synset label.
        """
        model = {}
        model_path = self.CAD_paths[idx]
        verts, faces, _ = self._load_mesh(model_path)

        if not torch.isfinite(verts).all():
            print(model_path)
            raise ValueError("Meshes contain nan or inf.")

        mesh_list = []
        pcl_list = []

        for transform_id, transform in enumerate(self.cad_transformations):
            verts_new = copy.deepcopy(verts)
            tverts = transform.transform_points(verts_new)
            mesh = Meshes(
                verts=[tverts.squeeze(dim=0)],
                faces=[faces],
            )

            if not torch.isfinite(mesh.verts_packed()).all():
                raise ValueError("Meshes contain nan or inf.")

            points_sampled, normals_sampled = sample_points_from_meshes(mesh, num_samples=self.num_sampled_points,
                                                                        return_normals=True,
                                                                        return_textures=False)
            pcl = Pointclouds(points=points_sampled)
            pcl_list.append(pcl)
            mesh_list.append(mesh)

        mesh_all = join_meshes_as_batch(mesh_list, include_textures=False)
        mesh_list_all = []
        for i in range(self.num_views):
            mesh_list_all.append(mesh_all)

        mesh_joint = join_meshes_as_batch(mesh_list_all, include_textures=False)
        model['path'] = model_path
        model['cls_label'] = self.cls_label
        model['obj_id'] = self.obj_ids[idx]
        model['id'] = idx

        model['mesh_list'] = mesh_joint
        model['pcl_list'] = pcl_list

        return model

    def __len__(self) -> int:
        return len(self.CAD_paths)
