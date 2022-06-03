import numpy as np
import matplotlib.pyplot as plt

# def plot_scene_post_train(time, scene, trajectron):
#     eval_stg.predict(scene,
#                      timesteps,
#                      ph_eval,
#                      num_samples=k,
#                      min_future_timesteps=ph_eval,
#                      z_mode=False,
#                      gmm_mode=True,
#                      full_dist=False,
#                      all_z_sep=False,
#                      min_k=True)


def plot_scene_pre_train(time_interval, scene):
    """
    Arguments
    ---------
    time_interval : tuple(int, int)
    scene : Scene
    """
    t_start, t_end = time_interval
    homography = 3
    cut_off = 1 * homography
    fig, ax = plt.subplots(figsize=(20, 20))
    for node in scene.nodes:
        xy = homography * node.get(np.array([t_start, t_end]), {'position': ['x', 'y']}).T
        xy -= cut_off
        x, y = xy

        if node.is_robot:
            ax.scatter(x[0], y[0], color='green', s=200)
            ax.plot(x, y, color='green', linewidth=10)

        elif str(node.type) == 'PEDESTRIAN':
            ax.scatter(x[0], y[0], color='blue', s=70)
            ax.plot(x, y, color='blue', linewidth=4)

        elif str(node.type) == 'VEHICLE':
            ax.scatter(x[0], y[0], color='red', s=70)
            ax.plot(x, y, color='red', linewidth=4)

    ax.plot(0, 0, color='red', label='Vehicle')
    ax.plot(0, 0, color='blue', label='Pedestrian')
    ax.plot(0, 0, color='green', label='Robot')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=30)
    scene_mask = scene.map['VEHICLE'].torch_map('cpu').numpy()
    ax.imshow(~scene_mask[0, cut_off:-cut_off, cut_off:-cut_off].transpose(), cmap='Greys', interpolation='nearest')