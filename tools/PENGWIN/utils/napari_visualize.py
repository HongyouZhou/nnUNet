try:
    import pyrootutils
    pyrootutils.setup_root(__file__, indicator="project-root", pythonpath=True)
except:
    pass
import napari
from challenge_pengwin.utils.utils import MedVol
from os.path import join


def napari_visualize(data_dir, dataset, exp_instance_name, name):
    image = MedVol(join(data_dir, "imagesTr", f"{name}_0000.nii.gz"))
    gt_instance = MedVol(join(data_dir, "labelsTr_instances", f"{name}.nii.gz"))
    pred_border_core = MedVol(join(data_dir, "predsTr_fold0", "instance", dataset, exp_instance_name, "border_core", f"{name}.nii.gz"))
    pred_instance = MedVol(join(data_dir, "predsTr_fold0", "instance", dataset, exp_instance_name, "instance", f"{name}.nii.gz"))
    pred_semantic = MedVol(join(data_dir, "predsTr_fold0", "instance", dataset, exp_instance_name, "semantic", f"{name}.nii.gz"))
    pred_remapped_instance = MedVol(join(data_dir, "predsTr_fold0", "instance", dataset, exp_instance_name, "remapped_instance", f"{name}.nii.gz"))

    viewer = napari.Viewer()
    image_layer = viewer.add_image(image.array, name=f"Image ({name})")
    gt_instance_layer = viewer.add_labels(gt_instance.array, name="GT - Instance", opacity=1, visible=True)
    pred_border_core_layer = viewer.add_labels(pred_border_core.array, name="Pred - Border Core", opacity=1, visible=False)
    pred_instance_layer = viewer.add_labels(pred_instance.array, name="Pred - Instance", opacity=1, visible=False)
    pred_semantic_layer = viewer.add_labels(pred_semantic.array, name="Pred - Semantic", opacity=1, visible=True)
    pred_remapped_instance_layer = viewer.add_labels(pred_remapped_instance.array, name="Pred - Remapped Instance", opacity=1, visible=False)
    pred_instance_layer.contour = 2
    pred_semantic_layer.contour = 2
    pred_remapped_instance_layer.contour = 2
    napari.run()


if __name__ == '__main__':
    data_dir = "/home/k539i/Documents/network_drives/E132-Projekte/Projects/2024_Pengwin_Challenge/border_core"
    dataset = "Dataset2002_Pengwin_challenge_2024"
    exp_instance_name = "nnUNetMultiTalentCTPlans_b2"
    name = "087"

    napari_visualize(data_dir, dataset, exp_instance_name, name)