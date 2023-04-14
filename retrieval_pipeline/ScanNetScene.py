class ScanNetScene(object):
    def __init__(self, scene_name, bbox3d_list, tmesh, sem_seg_3d, inst_labels, points, scene_type):
        self.scene_name = scene_name
        self.bbox3d_list = bbox3d_list
        self.scene_mesh = tmesh
        self.sem_seg_3d = sem_seg_3d
        self.inst_seg_3d = inst_labels
        self.points_3d = points
        self.scene_type = scene_type


class Box3D(object):
    def __init__(self, center, basis, scale, cls_name, inst_seg_id, view_params, scan2cad_obj_id=None):
        self.center = center
        self.basis = basis
        self.scale = scale

        self.cls_name = cls_name
        self.inst_seg_id = inst_seg_id
        self.view_params = view_params

        self.scan2cad_obj_id = scan2cad_obj_id

        self.transform3d_list = None
        self.transform_dict_list = None
        self.obj_id_retrieval_list = None

        self.obj_id_similarity_list = None

        self.transform3d_refine_list = None
        self.transform_refine_dict_list = None
        self.obj_id_refined_list = None

    def add_obj_id_retrieval_list(self, obj_id_list):
        self.obj_id_retrieval_list = obj_id_list

    def add_obj_id_similarity(self, obj_id_list):
        self.obj_id_similarity_list = obj_id_list

    def add_obj_id_refined(self, obj_id_list):
        self.obj_id_refined_list = obj_id_list

    def add_transform3D(self, transform3d_list, transform_dict_list):
        self.transform3d_list = transform3d_list
        self.transform_dict_list = transform_dict_list

    def add_transform3D_refined(self, transform3d_refined):
        self.transform3d_refine_list = transform3d_refined
