"""
Module that contains functions for the processing from lyft raw to "data" format
"""
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

Image.MAX_IMAGE_PIXELS = None

VEHICLE_ROAD_RGB = [128, 128, 128]  # Straße
PEDESTRIAN_ZEBRA_RGB = [250, 235, 215]  # Fußgängerübergang Zebrastreifen
PEDESTRIAN_GREEN_RGB = [0, 201, 131]  # Fußgängerübergang Grüne Ampel
PEDESTRIAN_ONLY_RGB = [211, 211, 211]  # Fußgängerweg

# Lyft Map ist im RGB Format gegeben. Ich habe per Hand die Farben der Straßenbereiche ermittelt, die sind nicht gegeben.
# Für die Gesamte befahrbare Fahrfläche werden alle Flächen außer "PEDESTRIAN_ONLY_RGB" verwendet".


def lyft_category2env_category(old_category, attribute, env):
    """
    adjust names of agent classes to "env" format
    """
    skip_bool = False
    ego_category = env.NodeType.VEHICLE
    if old_category == "pedestrian":
        new_category = env.NodeType.PEDESTRIAN
    elif (
        old_category == "vehicle"
        or old_category == "truck"
        and "parked" not in attribute
    ):
        new_category = env.NodeType.VEHICLE
    else:
        new_category = None
        skip_bool = True
    return skip_bool, new_category, ego_category


def process_lyft_map(map_path, args):
    """
    Arguments
    ---------
    map_path : str
    map_config : dict()

    Returns
    -------
    map_mask : np.array[1, y, x]
    """
    # interpolation der originalen map zu der erforderlichen Rastergöße
    img = Image.open(map_path)
    native_resolution = 0.1
    size_x = int(img.size[0] / args.fraction * native_resolution)
    size_y = int(img.size[1] / args.fraction * native_resolution)
    Image.MAX_IMAGE_PIXELS = None
    img = img.resize((size_x, size_y), resample=Image.NEAREST)
    rgb_map = np.array(img)

    # Extract Layers from RGB by color
    area_colours = {
        "VEHICLE_ROAD_RGB": VEHICLE_ROAD_RGB,
        "PEDESTRIAN_ZEBRA_RGB": PEDESTRIAN_ZEBRA_RGB,
        "PEDESTRIAN_GREEN_RGB": PEDESTRIAN_GREEN_RGB,
    }
    map_mask = extract_colours_from_map(rgb_map, area_colours)

    # Die Fahrbahn hat eine andere Farbe als die Spur.
    # Um die gesamte befahrbare Fläche zu erhalten, wird die Map mit einem Pooling layer bearbeitet,
    # der die Lücken durch die Fahrspur schließt.
    map_mask = fill_road_gaps(map_mask)

    # Anpassung der Ausrichtung und Dimension
    map_mask = flip_upside_down(map_mask)
    map_mask = np.expand_dims(map_mask, 0)
    return map_mask


def extract_one_colour_from_map(rgb_map, colour):
    """Extract boolean mask of map of a single rgb colour

    Arguments
    ---------
    rgb_map : np.array[y,x,3]
    colour : list(int, int, int)

    Returns
    -------
    colour_mask : np.array[y,x]
    """
    return np.all((rgb_map - np.array(colour)) == 0, axis=-1)


def fill_road_gaps(road):
    """Lanes are removed from original map, this functions pools over the area.
    Arguments
    ---------
    road : np.array[y,x]

    Returns
    -------
    road : np.array[y,x]
    """
    road = F.max_pool2d(
        input=torch.from_numpy(road).unsqueeze(0).float(),
        kernel_size=(2, 2),
        stride=1,
        padding=1,
    ).squeeze(0)
    return road.numpy()[1:, 1:]


def flip_upside_down(map_mask):
    """
    Arguments
    ---------
    map_mask : [y,x]

    Returns
    -------
    map_mask : [y,x]
    """
    return map_mask[::-1]


def extract_colours_from_map(rgb_map, area_colours_dict):
    """
    Arguments
    ---------
    rgb_map : np.array[y,x,3] 0-255

    Returns
    -------
    total_area : np.area[y,x] 0-1
    """
    area_dict = {}
    for area_name, area_colour in area_colours_dict.items():
        area_mask = extract_one_colour_from_map(rgb_map, area_colour)
        area_dict[area_name] = area_mask

    total_area = np.stack(
        [area_mask for (_, area_mask) in area_dict.items()], axis=0
    ).max(axis=0)
    return total_area
