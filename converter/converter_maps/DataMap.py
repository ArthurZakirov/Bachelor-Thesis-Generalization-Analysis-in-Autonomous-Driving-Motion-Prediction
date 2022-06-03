"""
This Module contains the Map class for the converters
"""
import numpy as np


class DataMap(object):
    def __init__(self, map_mask, args):
        self.fraction = args.fraction
        self.map_buffer = args.map_buffer
        self.scene_buffer = args.scene_buffer

        y_max, x_max = tuple(np.array(map_mask.shape[1:]) * self.fraction)
        y_min, x_min = 0, 0
        self.map_boundaries = x_min, x_max, y_min, y_max

        self.map_mask_full = map_mask
        self.padded_map_mask_full = self._create_padded_map_mask_full()

    @classmethod
    def from_openDD(cls, map_mask, map_boundaries, map_name, args):
        self = cls.__new__(cls)

        self.fraction = args.fraction
        self.map_buffer = args.map_buffer
        self.scene_buffer = args.scene_buffer

        self.map_boundaries = map_boundaries
        self.map_mask_full = map_mask
        self.padded_map_mask_full = map_mask
        self.map_name = map_name
        return self

    def _create_padded_map_mask_full(self):
        stretched_map_buffer = int(self.map_buffer / self.fraction)

        if self.map_buffer == 0:
            padded_mask = self.map_mask_full
        elif self.map_buffer > 0:
            padded_mask = np.zeros(
                (
                    self.map_mask_full.shape[0],
                    self.map_mask_full.shape[1] + 2 * stretched_map_buffer,
                    self.map_mask_full.shape[2] + 2 * stretched_map_buffer,
                )
            )
            padded_mask[
                :,
                stretched_map_buffer:-stretched_map_buffer,
                stretched_map_buffer:-stretched_map_buffer,
            ] = self.map_mask_full
        return padded_mask

    def _round_to_fraction(self, x, fraction):
        return np.round(x / fraction) * fraction

    def _stretch_boundaries(self, boundaries, fraction):
        stretched_boundaries = tuple(
            (self._round_to_fraction(np.array(boundaries), fraction) / fraction).astype(
                int
            )
        )
        return stretched_boundaries

    def extract_scene_mask_from_map_mask(self, scene_boundaries):
        stretched_map_buffer = int(self.map_buffer / self.fraction)
        stretched_scene_buffer = int(self.scene_buffer / self.fraction)

        # size map
        (x_min, x_max, y_min, y_max) = self._stretch_boundaries(
            self.map_boundaries, self.fraction
        )

        # size scene
        (x_min_scene, x_max_scene, y_min_scene, y_max_scene) = self._stretch_boundaries(
            scene_boundaries, self.fraction
        )

        # extract scene from full mask

        x_rel_min = x_min_scene - x_min + stretched_map_buffer
        x_rel_max = x_max_scene - x_min + stretched_map_buffer
        y_rel_min = y_min_scene - y_min + stretched_map_buffer
        y_rel_max = y_max_scene - y_min + stretched_map_buffer

        scene_mask = self.padded_map_mask_full[
            :, y_rel_min:y_rel_max, x_rel_min:x_rel_max
        ]
        return scene_mask
