import os
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import sqlite3
from matplotlib.path import Path
import matplotlib.pyplot as plt


class OpenDDMap:
    def __init__(self, map_path, map_name, args):
        self.map_name = map_name

        # import map connfig
        self.fraction = args.fraction
        self.lane_radius = args.lane_radius
        self.scene_buffer = args.scene_buffer
        self.map_buffer = args.map_buffer

        df = self.load_df(map_path)

        # load lane/bound data
        (
            self.lane_frame,
            self.border_frame,
            self.lanes_boundaries,
        ) = self.get_lanes_from_sql(df, dataformat="df")

        # load area
        (
            self.drivable_area_arrays,
            self.non_drivable_area_arrays,
            self.areas_boundaries,
        ) = self.get_areas_from_sql(df, dataformat="arrays")

        # map boundaries
        self.map_boundaries = self.determine_total_map_boundaries(
            self.areas_boundaries, self.lanes_boundaries
        )

        # create mask
        self.map_mask_full = self.create_map_mask_full(plot=False)
        self.padded_map_mask_full = self.create_padded_map_mask_full(plot=False)
        self.padded_map_mask_full = self.padded_map_mask_full[:2].max(
            axis=0, keepdims=True
        )
        self.add_lanes_outside_map()

    #################################################################################################################
    ###   STACK MASKS   #############################################################################################
    #################################################################################################################

    def create_map_mask_of_areas(self):

        (x_min_map, x_max_map, y_min_map, y_max_map) = self.stretch_boundaries(
            self.map_boundaries, self.fraction
        )

        empty_map_mask_layer = np.zeros(
            (y_max_map - y_min_map + 1, x_max_map - x_min_map + 1)
        )
        map_mask_areas = self.create_drivable_area_mask(
            self.map_boundaries,
            empty_map_mask_layer,
            self.drivable_area_arrays,
            self.non_drivable_area_arrays,
        )
        return map_mask_areas

    def create_map_mask_of_lanes(self, df):

        df_lanes = self.interpolate_lanes_df_to_fraction(df)
        united_lanes_array = self.unite_lane_segments(df_lanes)
        map_mask_lanes = self.create_mask_from_coordinates(
            united_lanes_array, self.map_boundaries, self.fraction
        )
        return map_mask_lanes

    def create_map_mask_full(self, plot):
        map_mask_lanes = self.create_map_mask_of_lanes(
            self.lane_frame.loc[:, ["x", "y"]]
        )
        map_mask_border = self.create_map_mask_of_lanes(
            self.border_frame.loc[:, ["x", "y"]]
        )
        map_mask_lanes = self.thicken_lanes(map_mask_lanes, self.lane_radius)
        map_mask_areas = self.create_map_mask_of_areas()

        map_mask_full = np.stack(
            (map_mask_lanes, map_mask_areas, map_mask_border), axis=0
        )

        if plot:
            plt.figure(1)
            plt.title("Drivable Area")
            plt.imshow(map_mask_areas)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid()
            print("Drivable Areas:", map_mask_areas.shape)

            plt.figure(2)
            plt.title("Borderlines")
            plt.imshow(map_mask_border)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid()
            print("Borderlines:   ", map_mask_border.shape)

            plt.figure(3)
            plt.title("Lanes")
            plt.imshow(map_mask_lanes)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid()
            print("Lanes:         ", map_mask_lanes.shape)

        return map_mask_full

    def create_padded_map_mask_full(self, plot):
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

        if plot:
            plt.figure(4)
            plt.title("Padded Mask")
            plt.imshow(padded_mask[0])
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid()
            plt.show()
        return padded_mask

    def add_lanes_outside_map(self):
        blacklist = ["rdb2"]
        if not (self.map_name in blacklist):
            M = self.padded_map_mask_full[0]
            y, x = np.argwhere(M).T
            edge_range = 4  # 7 nuScenes
            interval_l = x.min() + edge_range
            interval_r = M.shape[1] - x.max() + edge_range
            interval_b = y.min() + edge_range
            interval_t = M.shape[0] - y.max() + edge_range

            bottom = M[:interval_b, :]
            top = M[-interval_t:, :]
            left = M[:, :interval_l]
            right = M[:, -interval_r:]

            y_b, x_b = np.argwhere(bottom).T
            y_t, x_t = np.argwhere(top).T
            y_l, x_l = np.argwhere(left).T
            y_r, x_r = np.argwhere(right).T

            M[:interval_b, x_b.min() : x_b.max()] = 1
            M[-interval_t:, x_t.min() : x_t.max()] = 1
            M[y_l.min() : y_l.max(), :interval_l] = 1
            M[y_r.min() : y_r.max(), -interval_r:] = 1
            self.padded_map_mask_full[0] = M

    ###################################################################################################################
    ###    CREATE LANE/ BORDERLINE MASK   #############################################################################
    ##################################################################################################################

    def interpolate_lanes_df_to_fraction(self, df):
        df_interpolated = df.apply(
            lambda row: self.interpolate_lane_segment_to_fraction(row, self.fraction),
            axis=1,
            result_type="expand",
        )
        df_interpolated.columns = ["x", "y"]

        return df_interpolated

    def interpolate_lane_segment_to_fraction(self, row, fraction):
        x = row.x
        y = row.y

        y_min = self.round_to_fraction(y.min(), fraction)
        y_max = self.round_to_fraction(y.max(), fraction)
        x_min = self.round_to_fraction(x.min(), fraction)
        x_max = self.round_to_fraction(x.max(), fraction)

        y_is_const = y_max == y_min
        x_is_const = x_max == x_min

        y_right_order = np.all(np.diff(y) > 0)
        x_right_order = np.all(np.diff(x) > 0)

        if self.swap_axes_necessary(x):
            if not y_right_order:
                x = np.flip(x)
                y = np.flip(y)
            if y_is_const:
                x_interp = np.arange(x_min, x_max, 1 / 3)
                y_interp = y_max * np.ones(len(x_interp))
            else:
                y_interp = np.arange(y_min, y_max, 1 / 3)
                x_interp = self.round_to_fraction(np.interp(y_interp, y, x), fraction)

        else:
            if not x_right_order:
                x = np.flip(x)
                y = np.flip(y)
            if x_is_const:
                y_interp = np.arange(y_min, y_max, 1 / 3)
                x_interp = x_max * np.ones(len(y_interp))
            else:
                x_interp = np.arange(x_min, x_max, 1 / 3)
                y_interp = self.round_to_fraction(np.interp(x_interp, x, y), fraction)

        return [x_interp, y_interp]

    def swap_axes_necessary(self, x):

        idx_x_loc_min = argrelextrema(x, np.less)[0]
        idx_x_loc_max = argrelextrema(x, np.greater)[0]

        optimum_exists = not (len(idx_x_loc_min) == 0 and len(idx_x_loc_max) == 0)

        if optimum_exists:
            swap_axes = True
        else:
            swap_axes = False

        return swap_axes

    def unite_lane_segments(self, df):
        x = df.x.values
        y = df.y.values
        x_vec = x[0]
        y_vec = y[0]
        for lane_segment in range(1, len(df)):
            x_vec = np.concatenate((x_vec, x[lane_segment]), axis=0)
            y_vec = np.concatenate((y_vec, y[lane_segment]), axis=0)
        return x_vec, y_vec

    def create_mask_from_coordinates(self, coordinates, boundaries, fraction):

        (x, y) = self.stretch_boundaries(coordinates, fraction)
        (x_min, x_max, y_min, y_max) = self.stretch_boundaries(boundaries, fraction)

        x_normalized = x - x_min
        y_normalized = y - y_min

        canvas_shape = (y_max - y_min + 1, x_max - x_min + 1)
        mask = np.zeros(canvas_shape)

        for i in range(len(x_normalized)):
            position = (y_normalized[i], x_normalized[i])
            mask[position] = 1

        return mask

    def thicken_lanes(self, mask, radius):
        # add padding
        stretched_radius = int(radius / self.fraction)
        padding = stretched_radius
        padded_mask_shape = (mask.shape[0] + 2 * padding, mask.shape[1] + 2 * padding)
        padded_mask = np.zeros(padded_mask_shape)
        padded_mask[padding:-padding, padding:-padding] = mask

        # find True mask entry
        for y in range(padding, padded_mask.shape[0] - padding):
            for x in range(padding, padded_mask.shape[1] - padding):
                if padded_mask[y, x] == 1:

                    # iterate over square around position
                    for distance_y in range(-stretched_radius, stretched_radius + 1):
                        for distance_x in range(
                            -stretched_radius, stretched_radius + 1
                        ):

                            relative_position = np.array([distance_x, distance_y])
                            relative_distance = np.linalg.norm(relative_position)

                            # find points, that are within a circle
                            if relative_distance <= stretched_radius:
                                padded_mask[y + distance_y, x + distance_x] = 2

        # remove padding
        new_mask = padded_mask[padding:-padding, padding:-padding]
        return new_mask / 2

    ###################################################################################################################
    ###   UTILS   #####################################################################################################
    ###################################################################################################################

    def stretch_boundaries(self, boundaries, fraction):
        stretched_boundaries = tuple(
            (self.round_to_fraction(np.array(boundaries), fraction) / fraction).astype(
                int
            )
        )
        return stretched_boundaries

    def round_to_fraction(self, x, fraction):
        return np.round(x / fraction) * fraction

    def determine_total_map_boundaries(self, areas_boundaries, lanes_boundaries):

        x_min_lanes, x_max_lanes, y_min_lanes, y_max_lanes = lanes_boundaries
        x_min_areas, x_max_areas, y_min_areas, y_max_areas = areas_boundaries
        x_min = min(x_min_areas, x_min_lanes)
        x_max = max(x_max_areas, x_max_lanes)
        y_min = min(y_min_areas, y_min_lanes)
        y_max = max(y_max_areas, y_max_lanes)

        map_boundaries = x_min, x_max, y_min, y_max
        return map_boundaries

    ###################################################################################################################
    ###   CREATE AREA MASK   ##########################################################################################
    ###################################################################################################################
    def create_area_mask_from_polygon(self, polygon):

        # ACHTUNG: für 'Path' ist y die erste koordinate, deswegen flip()
        polygon_arr = np.array(polygon)
        polygon_arr = np.flip(polygon_arr, axis=1)

        polygon_min = polygon_arr.min(axis=0)
        polygon_max = polygon_arr.max(axis=0)
        polygon_size = polygon_max - polygon_min
        polygon_size_stretched = self.stretch_boundaries(polygon_size, self.fraction)

        area_boundaries = polygon_min[1], polygon_max[1], polygon_min[0], polygon_max[0]

        polygon_arr_norm = polygon_arr - polygon_min
        polygon_arr_norm_stretched = (
            self.round_to_fraction(polygon_arr_norm, self.fraction) / self.fraction
        )

        poly_path = Path(polygon_arr_norm_stretched)

        height, width = polygon_size_stretched
        x, y = np.mgrid[:height, :width]
        coors = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))

        mask_flatten = poly_path.contains_points(coors)
        mask = mask_flatten.reshape(height, width)

        return mask, area_boundaries

    def insert_area_into_map_mask(
        self, map_boundaries, map_mask, area_boundaries, area_mask, drivable
    ):

        x_min_map, x_max_map, y_min_map, y_max_map = self.stretch_boundaries(
            map_boundaries, self.fraction
        )
        x_min_area, x_max_area, y_min_area, y_max_area = self.stretch_boundaries(
            area_boundaries, self.fraction
        )

        x_min_area_rel = x_min_area - x_min_map
        x_max_area_rel = x_min_area_rel + area_mask.shape[1]
        y_min_area_rel = y_min_area - y_min_map
        y_max_area_rel = y_min_area_rel + area_mask.shape[0]

        if drivable:
            map_mask[y_min_area_rel:y_max_area_rel, x_min_area_rel:x_max_area_rel] += (
                1 * area_mask
            )
        elif not drivable:
            map_mask[y_min_area_rel:y_max_area_rel, x_min_area_rel:x_max_area_rel] -= (
                5 * area_mask
            )

        return map_mask

    def create_drivable_area_mask(
        self, map_boundaries, map_mask, drivable_areas, non_drivable_areas
    ):

        blacklist = []  # [1,2,3]

        for i, polygon in enumerate(drivable_areas):
            if i in blacklist:
                continue
            (area_mask, area_boundaries) = self.create_area_mask_from_polygon(polygon)
            map_mask = self.insert_area_into_map_mask(
                map_boundaries, map_mask, area_boundaries, area_mask, drivable=True
            )

        for i, polygon in enumerate(non_drivable_areas):
            area_mask, area_boundaries = self.create_area_mask_from_polygon(polygon)
            map_mask = self.insert_area_into_map_mask(
                map_boundaries, map_mask, area_boundaries, area_mask, drivable=False
            )

        map_mask = map_mask > 0

        return map_mask

    ###################################################################################################################
    ###   SCENE MAP PATCH   ###########################################################################################
    ###################################################################################################################

    def extract_scene_mask_from_map_mask(self, scene_boundaries):

        stretched_map_buffer = int(self.map_buffer / self.fraction)
        stretched_scene_buffer = int(self.scene_buffer / self.fraction)

        # size map
        (x_min, x_max, y_min, y_max) = self.stretch_boundaries(
            self.map_boundaries, self.fraction
        )

        # size scene
        (x_min_scene, x_max_scene, y_min_scene, y_max_scene) = self.stretch_boundaries(
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

    ###################################################################################################################
    ###   CREATE LANES   ##############################################################################################
    ###################################################################################################################

    def load_df(self, sql_path):
        dir, file = os.path.split(sql_path)
        table_name = file.split(".")[0].split("_")[1]
        con = sqlite3.connect(sql_path)
        df = pd.read_sql_query(f"SELECT * from {table_name}", con)
        con.close()
        return df

    def get_lanes_from_sql(self, df, dataformat):

        lanes_df = df[(df.type == "trafficLane") | (df.type == "boundary")][
            ["type", "geometry"]
        ].dropna()

        # prepare different output dataformats
        lanes_lists = list()
        lanes_arrays = list()
        lanes_dataframe = pd.DataFrame({"x": [], "y": []})

        bound_lists = list()
        bound_arrays = list()
        bound_dataframe = pd.DataFrame({"x": [], "y": []})

        # initialize star min/max values
        x_min_lanes = 1e8
        x_max_lanes = 0
        y_min_lanes = 1e8
        y_max_lanes = 0
        lanes_id = 0

        lanes_ids = range(len(lanes_df))
        for lane_id in lanes_ids:

            # convert string data to float
            linestring_list = list()
            lane_type = lanes_df.type.iloc[lane_id]
            linestring_strings = (
                lanes_df.geometry.iloc[lane_id].split("(")[1].split(")")[0].split(",")
            )
            point_ids = range(len(linestring_strings))

            for point_id in point_ids:
                x = float(linestring_strings[point_id].split()[0])
                y = float(linestring_strings[point_id].split()[1])
                linestring_list.append((x, y))

            linestring_arr = np.array(linestring_list)

            linestring_df = pd.DataFrame(
                {"x": [linestring_arr[:, 0]], "y": [linestring_arr[:, 1]]}
            )
            # determine max and min
            x_min_linestring, y_min_linestring = tuple(linestring_arr.min(axis=0))
            x_max_linestring, y_max_linestring = tuple(linestring_arr.max(axis=0))

            if x_min_linestring < x_min_lanes:
                x_min_lanes = x_min_linestring
            if y_min_linestring < y_min_lanes:
                y_min_lanes = y_min_linestring
            if x_max_linestring > x_max_lanes:
                x_max_lanes = x_max_linestring
            if y_max_linestring > y_max_lanes:
                y_max_lanes = y_max_linestring

                # separate lane types
            if lane_type == "trafficLane":
                lanes_lists.append(linestring_list)
                lanes_arrays.append(linestring_arr)
                lanes_dataframe = lanes_dataframe.append(linestring_df)

            elif lane_type == "boundary":
                bound_lists.append(linestring_list)
                bound_arrays.append(linestring_arr)
                bound_dataframe = bound_dataframe.append(linestring_df)

        # determine output data format
        if dataformat == "arrays":
            lanes = lanes_arrays
            bound = bound_arrays

        elif dataformat == "lists":
            lanes = lanes_lists
            bound = bound_lists

        elif dataformat == "df":
            lanes = lanes_dataframe.reset_index(drop=True)
            bound = bound_dataframe.reset_index(drop=True)

        lanes_boundaries = x_min_lanes, x_max_lanes, y_min_lanes, y_max_lanes

        return lanes, bound, lanes_boundaries

    def get_areas_from_sql(self, df, dataformat):

        areas_df = df[df.type == "area"]

        # prepare different output dataformats
        drivable_areas_lists = list()
        drivable_areas_arrays = list()
        drivable_areas_dataframe = pd.DataFrame({"x": [], "y": []})

        non_drivable_areas_lists = list()
        non_drivable_areas_arrays = list()
        non_drivable_areas_dataframe = pd.DataFrame({"x": [], "y": []})

        # initialize star min/max values
        x_min_areas = 1e8
        x_max_areas = 0
        y_min_areas = 1e8
        y_max_areas = 0

        area_ids = range(len(areas_df))
        for area_id in area_ids:

            # convert string data to float
            polygon_list = list()
            area_type = areas_df.areaType.iloc[area_id]
            polygon_strings = (
                areas_df.geometry.iloc[area_id].split("((")[1].split("))")[0].split(",")
            )
            point_ids = range(len(polygon_strings))

            for point_id in point_ids:
                x = float(polygon_strings[point_id].split()[0])
                y = float(polygon_strings[point_id].split()[1])
                polygon_list.append((x, y))

            polygon_arr = np.array(polygon_list)
            polygon_df = pd.DataFrame(
                {"x": [polygon_arr[:, 0]], "y": [polygon_arr[:, 1]]}
            )

            # determine max and min
            x_min_polygon, y_min_polygon = tuple(polygon_arr.min(axis=0))
            x_max_polygon, y_max_polygon = tuple(polygon_arr.max(axis=0))

            if x_min_polygon < x_min_areas:
                x_min_areas = x_min_polygon
            if y_min_polygon < y_min_areas:
                y_min_areas = y_min_polygon
            if x_max_polygon > x_max_areas:
                x_max_areas = x_max_polygon
            if y_max_polygon > y_max_areas:
                y_max_areas = y_max_polygon

            # separate area types
            if area_type in ["COMPLETE"]:
                drivable_areas_lists.append(polygon_list)
                drivable_areas_arrays.append(polygon_arr)
                drivable_areas_dataframe = drivable_areas_dataframe.append(polygon_df)

            elif area_type in ["NEVER", "EMERGENCY"]:
                non_drivable_areas_lists.append(polygon_list)
                non_drivable_areas_arrays.append(polygon_arr)
                non_drivable_areas_dataframe = non_drivable_areas_dataframe.append(
                    polygon_df
                )

        # determine output dataformat
        if dataformat == "arrays":
            drivable_areas = drivable_areas_arrays
            non_drivable_areas = non_drivable_areas_arrays

        elif dataformat == "lists":
            drivable_areas = drivable_areas_lists
            non_drivable_areas = non_drivable_areas_lists

        elif dataformat == "df":
            drivable_areas = drivable_areas_dataframe
            non_drivable_areas = non_drivable_areas_dataframe

        areas_boundaries = x_min_areas, x_max_areas, y_min_areas, y_max_areas

        return drivable_areas, non_drivable_areas, areas_boundaries
