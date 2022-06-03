import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap
import cv2
import math
import pdb
from tqdm.notebook import tqdm

# colormap
colors = [(0, 0, 0), (0.87, 0.87, 0.87), (0.54, 0.54, 0.54), (0.29, 0.57, 0.25)]
cmap_name = 'scene_list'
cm = LinearSegmentedColormap.from_list(
    cmap_name, colors, N=4)

class TrackDataset(data.Dataset):
    """
    Dataset class for KITTI.
    The building class is merged into the background class
    0:background 1:street 2:sidewalk, 3:building 4: vegetation ---> 0:background 1:street 2:sidewalk, 3: vegetation
    """
    def __init__(self, tracks, len_past=6, len_future=12, dt=0.5, dim_clip=180):
        self.num_scene_layers = 2
        dataset_name = tracks['title']
        
        self.tracks = tracks['data']           # dataset dict
        self.dim_clip = dim_clip        # dim_clip*2 is the pixel-dimension of scene (pixel)

        for possible_dataset in ['nuScenes', 'openDD', 'lyft']:
            if possible_dataset in dataset_name:
                tracks = pd.DataFrame(tracks['data'])
                self.index =          list(tracks['index'])
                self.pasts =          list(tracks['pasts'])
                self.futures =        list(tracks['futures'])
                self.presents =       list(tracks['presents'])
                self.angle_presents = list(tracks['angle_presents'])
                self.video_track =    list(tracks['video_track'])
                self.vehicles =       list(tracks['vehicles'])
                self.number_vec =     list(tracks['number_vec'])
                self.scene =          list(tracks['scene'])
                self.scene_crop =     list(tracks['scene_crop'])

                self.index = np.array(self.index)
                self.pasts = torch.FloatTensor(self.pasts)
                self.futures = torch.FloatTensor(self.futures)
                self.presents = torch.FloatTensor(self.presents)
                self.video_track = np.array(self.video_track)
                self.vehicles = np.array(self.vehicles)
                self.number_vec = np.array(self.number_vec)
                self.scene = np.array(self.scene)
                self.scene_crop = np.array(self.scene_crop)
            
        if 'Kitti' in dataset_name:
            tracks = tracks['data']
            self.video_track = []     # '0001'
            self.vehicles = []        # 'Car'
            self.number_vec = []      # '4'
            self.index = []           # '50'
            self.pasts = []           # [len_past, 2]
            self.presents = []        # position in complete scene
            self.angle_presents = []  # trajectory angle in complete scene
            self.futures = []         # [len_future, 2]
            self.scene = []           # [dim_clip, dim_clip, 1], scene fot qualitative examples
            self.scene_crop = []      # [dim_clip, dim_clip, 4], input to IRM

            num_total = len_past + len_future

            # Preload data
            for video in tqdm(tracks.keys(), leave=True):
                vehicles = self.tracks[video].keys()
                video_id = video[-9:-5]
                print('video: ' + video_id)
                maps_dir = '../../../datasets/processed/Mantra_format/Kitti_road_classes/maps'
                path_scene = os.path.join(maps_dir, '2011_09_26__2011_09_26_drive_' + video_id + '_sync_map.png')
                scene_track = cv2.imread(path_scene, 0)
                scene_track_onehot = scene_track.copy()

                # Remove building class
                scene_track[np.where(scene_track_onehot >= 2)] = 0
                scene_track_onehot[np.where(scene_track_onehot >= 2)] = 0
                for vec in vehicles:
                    class_vec = tracks[video][vec]['cls']
                    num_vec = vec.split('_')[1]
                    start_frame = tracks[video][vec]['start']
                    points = np.array(tracks[video][vec]['trajectory']).T
                    len_track = len(points)
                    for count in range(0, len_track, 1):
                        if len_track - count > num_total:

                            temp_past = points[count:count + len_past].copy()
                            temp_future = points[count + len_past:count + num_total].copy()
                            origin = temp_past[-1]
                            # filter out noise for non-moving vehicles
                            if np.var(temp_past[:, 0]) < dt and np.var(temp_past[:, 1]) < dt:
                                temp_past = np.zeros((len_past, 2))
                            else:
                                temp_past = temp_past - origin

                            if np.var(temp_past[:, 0]) < dt and np.var(temp_past[:, 1]) < dt:
                                temp_future = np.zeros((len_future, 2))
                            else:
                                temp_future = temp_future - origin

                            scene_track_clip = scene_track[
                                               int(origin[1]) * 2 - self.dim_clip:int(origin[1]) * 2 + self.dim_clip,
                                               int(origin[0]) * 2 - self.dim_clip:int(origin[0]) * 2 + self.dim_clip]

                            scene_track_onehot_clip = scene_track_onehot[
                                                      int(origin[1]) * 2 - self.dim_clip:int(origin[1]) * 2 + self.dim_clip,
                                                      int(origin[0]) * 2 - self.dim_clip:int(origin[0]) * 2 + self.dim_clip]

                            # rotation invariance
                            unit_y_axis = torch.Tensor([0, -1])
                            vector = temp_past[-1]
                            if vector[0] > 0.0:
                                angle = np.rad2deg(self.angle_vectors(vector, unit_y_axis))
                            else:
                                angle = -np.rad2deg(self.angle_vectors(vector, unit_y_axis))
                            matRot_track = cv2.getRotationMatrix2D((0, 0), angle, 1)
                            matRot_scene = cv2.getRotationMatrix2D((self.dim_clip, self.dim_clip), angle, 1)

                            past_rot = cv2.transform(temp_past.reshape(-1, 1, 2), matRot_track).squeeze()
                            future_rot = cv2.transform(temp_future.reshape(-1, 1, 2), matRot_track).squeeze()
                            scene_track_onehot_clip = cv2.warpAffine(scene_track_onehot_clip, matRot_scene,
                                               (scene_track_onehot_clip.shape[0], scene_track_onehot_clip.shape[1]),
                                               borderValue=0,
                                               flags=cv2.INTER_NEAREST)  # (1, 0, 0, 0)

                            self.index.append(count + len_past -1 + start_frame)
                            self.pasts.append(past_rot)
                            self.futures.append(future_rot)
                            self.presents.append(origin)
                            self.angle_presents.append(angle)
                            self.video_track.append(video_id)
                            self.vehicles.append(class_vec)
                            self.number_vec.append(num_vec)
                            self.scene.append(scene_track_clip)
                            self.scene_crop.append(scene_track_onehot_clip)

            self.index = np.array(self.index)
            self.pasts = torch.FloatTensor(self.pasts)
            self.futures = torch.FloatTensor(self.futures)
            self.presents = torch.FloatTensor(self.presents)
            self.video_track = np.array(self.video_track)
            self.vehicles = np.array(self.vehicles)
            self.number_vec = np.array(self.number_vec)
            self.scene = np.array(self.scene)
            self.scene_crop = np.array(self.scene_crop)

    @classmethod
    def from_df(cls, df_samples, scene_mask_dict, len_past=4, len_future=12, dt=0.5, dim_clip=175):
        self = cls.__new__(cls)
        super(TrackDataset, self).__init__()
        if df_samples.empty:
            return None
        self.num_scene_layers = 2
        self.dataset_name = df_samples.columns._levels[0].values.item()
        self.dim_clip = dim_clip  # dim_clip*2 is the dimension of scene (pixel)
        self.len_past = len_past
        self.len_future = len_future
        self.dt = dt

        tracks = df_samples[self.dataset_name].copy()
        self.index = list(tracks['index'])
        self.pasts = list(tracks['pasts'])
        self.futures = list(tracks['futures'])
        self.presents = list(tracks['presents'])
        self.angle_presents = list(tracks['angle_presents'])
        self.video_track = list(tracks['video_track'])
        self.vehicles = list(tracks['vehicles'])
        self.number_vec = list(tracks['number_vec'])
        self.scene_id = list(tracks['scene_id'])


        self.index = np.array(self.index)
        self.pasts = torch.FloatTensor(self.pasts)
        self.futures = torch.FloatTensor(self.futures)
        self.presents = torch.FloatTensor(self.presents)
        self.video_track = np.array(self.video_track)
        self.vehicles = np.array(self.vehicles)
        self.number_vec = np.array(self.number_vec)
        self.scene_mask_dict = scene_mask_dict
        return self


    def save_dataset(self, folder_save):
        for i in range(self.pasts.shape[0]):
            video = self.video_track[i]
            vehicle = self.vehicles[i]
            number = self.number_vec[i]
            past = self.pasts[i]
            future = self.futures[i]
            scene_track = self.scene_crop[i]

            saving_list = ['only_tracks', 'only_scenes', 'tracks_on_scene']
            for sav in saving_list:
                folder_save_type = folder_save + sav + '/'
                if not os.path.exists(folder_save_type + video):
                    os.makedirs(folder_save_type + video)
                video_path = folder_save_type + video + '/'
                if not os.path.exists(video_path + vehicle + number):
                    os.makedirs(video_path + vehicle + number)
                vehicle_path = video_path + '/' + vehicle + number + '/'
                if sav == 'only_tracks':
                    self.draw_track(past, future, index_tracklet=self.index[i], path=vehicle_path)
                if sav == 'only_scenes':
                    self.draw_scene(scene_track, index_tracklet=self.index[i], path=vehicle_path)
                if sav == 'tracks_on_scene':
                    self.draw_track_in_scene(past, scene_track, index_tracklet=self.index[i], future=future, path=vehicle_path)

    def draw_track(self, past, future, index_tracklet, path):
        plt.plot(past[:, 0], -past[:, 1], c='blue', marker='o', markersize=1)
        if future is not None:
            future = future.cpu().numpy()
            plt.plot(future[:, 0], -future[:, 1], c='green', marker='o', markersize=1)
        plt.axis('equal')
        plt.savefig(path + str(index_tracklet) + '.png')
        plt.close()

    def draw_scene(self, scene_track, index_tracklet, path):
        # print semantic map
        cv2.imwrite(path + str(index_tracklet) + '.png', scene_track)

    def draw_track_in_scene(self, story, scene_track, index_tracklet, future=None, path=''):
        plt.imshow(scene_track, cmap=cm)
        plt.plot(story[:, 0] * 2 + self.dim_clip, story[:, 1] * 2 + self.dim_clip, c='blue', marker='o', markersize=1)
        plt.plot(future[:, 0] * 2 + self.dim_clip, future[:, 1] * 2 + self.dim_clip, c='green', marker='o', markersize=1)
        plt.savefig(path + str(index_tracklet) + '.png')
        plt.close()

    @staticmethod
    def get_desire_track_files(train):
        """ Get videos only from the splits defined in DESIRE: https://arxiv.org/abs/1704.04394
        Splits obtained from the authors:
        all: [1, 2, 5, 9, 11, 13, 14, 15, 17, 18, 27, 28, 29, 32, 48, 51, 52, 56, 57, 59, 60, 70, 84, 91]
        train: [5, 9, 11, 13, 14, 17, 27, 28, 48, 51, 56, 57, 59, 60, 84, 91]
        test: [1, 2, 15, 18, 29, 32, 52, 70]
        """
        if train:
            desire_ids = [5, 9, 11, 13, 14, 17, 27, 28, 48, 51, 56, 57, 59, 60, 84, 91]
        else:
            desire_ids = [1, 2, 15, 18, 29, 32, 52, 70]

        tracklet_files = ['video_2011_09_26__2011_09_26_drive_' + str(x).zfill(4) + '_sync'
                          for x in desire_ids]
        return tracklet_files, desire_ids

    def unit_vector(self, vector):
        """ Returns the unit vector of the vector.  """
        return vector / (np.linalg.norm(vector) + 0.0001)

    def angle_vectors(self, v1, v2):
        """ Returns angle between two vectors.  """
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        if math.isnan(angle):
            return 0.0
        else:
            return angle

    def __len__(self):
        return self.pasts.shape[0]

    def __getitem__(self, idx):
        index = self.index[idx]
        pasts = self.pasts[idx]
        futures = self.futures[idx]
        presents = self.presents[idx]
        angle_presents = self.angle_presents[idx]
        video_track = self.video_track[idx]
        vehicles = self.vehicles[idx]
        number_vec = self.number_vec[idx]
        scene_id = self.scene_id[idx]
        scene_mask = self.scene_mask_dict[scene_id]

        origin = presents
        angle = angle_presents
        dim_clip = self.dim_clip
        fraction = 0.5
        x_current = int(origin[0] / fraction)
        y_current = int(origin[1] / fraction)

        scene_track_clip = scene_mask[
                           y_current - dim_clip: y_current + dim_clip,
                           x_current - dim_clip: x_current + dim_clip]
        matRot_scene = cv2.getRotationMatrix2D(center=(dim_clip, dim_clip), angle=-angle, scale=1)
        scene_track_onehot_clip = cv2.warpAffine(
            scene_track_clip,
            matRot_scene,
            (2 * dim_clip, 2 * dim_clip),
            borderValue=0,
            flags=cv2.INTER_NEAREST).astype(int)
        scene_crop = scene_track_onehot_clip
        scene_crop = np.eye(self.num_scene_layers, dtype=np.float32)[scene_crop]
        scene = torch.FloatTensor(scene_track_clip)

        return index, pasts, futures, presents, angle_presents, video_track, vehicles, number_vec, scene, scene_crop

    def torch_sample(self, idx):
        sample = self.__getitem__(idx)
        return [torch.from_numpy(element) if (type(element) == np.ndarray) else element for element in sample]


