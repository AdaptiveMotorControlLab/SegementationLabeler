import glob
import os
import cv2
import numpy as np
import napari
from magicgui import magicgui


def get_image_and_mask_paths(data_path):
    """Get sorted lists of image and mask paths."""
    image_paths = sorted(glob.glob(os.path.join(data_path, "frames", "*")))
    mask_paths = sorted(glob.glob(os.path.join(data_path, "masks", "*")))
    return image_paths, mask_paths


def get_unlabeled_image_ids(image_paths, mask_paths):
    """Extracts IDs of images that do not have a corresponding mask."""
    extract_id = lambda path: int(os.path.basename(path).split("_")[1].split(".")[0])
    image_ids = list(map(extract_id, image_paths))
    mask_ids = list(map(extract_id, mask_paths))
    return sorted(list(set(image_ids) - set(mask_ids)))


def preprocess_image(image_path):
    """Reads and crops an image from the given path."""
    image = cv2.imread(image_path)[70:-300, 320:-500]
    return image


def create_or_load_mask(data_path, current_id, step):
    """Creates a new mask or loads an existing one based on the current ID."""
    if current_id == 0:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
    else:
        previous_id = current_id - step
        mask_path = os.path.join(data_path, "masks", f"frame_{previous_id:04d}.npy")
        mask = np.load(mask_path)
    return mask


def setup_viewer(image, mask, current_id):
    """Sets up the napari viewer with the image and mask layers."""
    viewer = napari.Viewer()
    viewer.add_image(image, name=f"image{current_id:04d}")
    viewer.add_labels(mask, name=f"mask{current_id:04d}")
    return viewer


def add_save_button_to_viewer(viewer, data_path, current_id):
    """Adds a save button to the napari viewer."""
    save_button = magicgui(
        save_annotated_mask,
        call_button="Save",
        viewer={"visible": False, "value": viewer},
        data_path={"value": data_path},
        ind={"value": current_id},
    )
    viewer.window.add_dock_widget(save_button)


def save_annotated_mask(viewer: napari.Viewer, data_path: str, ind: int):
    layer_name = f"mask{ind:04d}"
    mask = viewer.layers[layer_name].data
    save_path = os.path.join(data_path, "masks", f"frame_{ind:04d}.npy")
    np.save(save_path, mask)
    viewer.close()


def toggle_modes(viewer: napari.Viewer, ind: int):

    modes = ["paint", "fill"]
    layer_name = f"mask{ind:04d}"
    current_mode = viewer.layers[layer_name].mode

    if current_mode in modes:
        next_mode_index = (modes.index(current_mode) + 1) % len(modes)
        next_mode = modes[next_mode_index]
    else:
        next_mode = modes[0]

    viewer.layers[layer_name].mode = next_mode
    print(f"Switched to {next_mode} mode")


def set_label_to(viewer: napari.Viewer, ind: int, label_value: int):
    layer_name = f"mask{ind:04d}"
    if layer_name in viewer.layers:
        layer = viewer.layers[layer_name]
        if isinstance(layer, napari.layers.Labels):
            layer.selected_label = label_value
            print(f"Switched to label: {label_value}")
        else:
            print(f"The layer '{layer_name}' is not a Labels layer.")
    else:
        print(f"No layer named '{layer_name}' found.")
