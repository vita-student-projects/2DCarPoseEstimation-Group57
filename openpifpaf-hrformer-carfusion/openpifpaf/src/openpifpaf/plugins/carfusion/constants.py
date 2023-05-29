import numpy as np

import os
try:
    import matplotlib.cm as mplcm
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    pass

import openpifpaf


CARFUSION_KEYPOINTS = [
    'Right_Front_wheel',        # 0
    'Left_Front_wheel',         # 1
    'Right_Back_wheel',         # 2
    'Left_Back_wheel',          # 3
    'Right_Front_Headlight',    # 4
    'Left_Front_Headlight',     # 5
    'Right_Back_Headlight',     # 6
    'Left_Back_Headlight',      # 7
    'Exhaust',                  # 8
    'Right_Front_Top',          # 9
    'Left_Front_Top',           # 10
    'Right_Back_Top',           # 11
    'Left_Back_Top',            # 12
    'Center',                   # 13
]

CARFUSION_SKELETON =[
    (0, 1), (1, 3), (2, 3), (0, 2), # wheels
    (9, 10), (10, 12), (11, 12), (9, 11), # roof
    (0, 4), (4, 9), (1, 5), (5, 10), (4, 5), # Front
    (2, 6), (6, 11), (3, 7), (7, 12), (6, 7) # Back
]
CARFUSION_SIGMAS = [0.05]*len(CARFUSION_KEYPOINTS)

split, error = divmod(len(CARFUSION_KEYPOINTS), 4)
CARFUSION_SCORE_WEIGHTS = [10.0] * split + [3.0] * split + \
    [1.0] * split + [0.1] * split + [0.1] * error
assert len(CARFUSION_SCORE_WEIGHTS) == len(CARFUSION_KEYPOINTS)



HFLIP = {
    'Right_Front_wheel': 'Left_Front_wheel',
    'Left_Front_wheel': 'Right_Front_wheel',
    'Right_Back_wheel': 'Left_Back_wheel',
    'Left_Back_wheel': 'Right_Back_wheel',
    'Right_Front_Headlight': 'Left_Front_Headlight',
    'Left_Front_Headlight': 'Right_Front_Headlight',
    'Right_Back_Headlight': 'Left_Back_Headlight',
    'Left_Back_Headlight': 'Right_Back_Headlight',
    'Right_Front_Top': 'Left_Front_Top',
    'Left_Front_Top': 'Right_Front_Top',
    'Right_Back_Top': 'Left_Back_Top',
    'Left_Back_Top': 'Right_Back_Top'
}


CAR_CATEGORIES = ['car']

p = 0.25
FRONT = -6.0
BACK = 4.5

# CAR POSE is used for joint rescaling. x = [-3, 3] y = [0,4]
CAR_POSE = np.array([
    [-3.2, 0.2, FRONT * 0.7],  # 'Right_Front_wheel',
    [3.2, 0.2, FRONT * 0.7],  # 'Left_Front_wheel',
    [-3.0, 0.3, BACK * 0.7],  # 'Right_Back_wheel'
    [3.0, 0.3, BACK * 0.7],   # 'Left_Back_wheel'
    [-2.0, 2.0, FRONT],  # 'Right_Front_Headlight',
    [2.0, 2.0, FRONT],  # 'Left_Front_Headlight',
    [-2.5, 2.2, BACK],  # 'Right_Back_Headlight',
    [2.5, 2.2, BACK],   # 'Left_Back_Headlight',
    [-2.9, 4.0, FRONT * 0.5],  # 'Right_Front_Top',
    [2.9, 4.0, FRONT * 0.5],   # 'Left_Front_Top',
    [-2.4, 4.3, BACK * 0.35],  # 'Right_Back_Top'
    [2.4, 4.3, BACK * 0.35],  # 'Left_Back_Top',
    [0, 0.1, 1], #Center
    [0, 2.2, BACK], #Exhaust
])

CAR_POSE_FRONT = np.array([
    [-2 - p, 0.0 - p / 2, 1.0],  # 'Right_Front_wheel', # 20
    [2.0 + p, 0.1 - p / 2, 1.0],  # 'Left_Front_wheel', # 8
    [-2, 0.0, 0.0],  # 'Right_Back_wheel'               # 19
    [2, 0.1, 0.0],   # 'Left_Back_wheel'                # 9
    [-1.3, 2.0, 2.0],  # 'Right_Front_Headlight',       # 3
    [1.3, 2.0, 2.0],  # 'Left_Front_Headlight',         # 4
    [-2.1, 1.9, 0.0],  # 'Right_Back_Headlight',        # 14
    [2.1, 1.9, 0.0],   # 'Left_Back_Headlight',         # 13
    [-2.0, 4.0, 2.0],  # 'Right_Front_Top',             # 1
    [2.0, 4.0, 2.0],   # 'Left_Front_Top',              # 2
    [-2.0, 4.0, 0.0],  # 'Right_Back_Top'               # 12
    [2.0, 4.1, 0.0],  # 'Left_Back_Top',                # 11
    [0, 0.1, 0.5], #Center
    [0, 1.9, 0.0], #Exhaust
])

CAR_POSE_REAR = np.array([
    [-2, 0.0, 0.0],  # 'Right_Front_wheel', # 20
    [2, 0.0, 0.0],  # 'Left_Front_wheel', # 8
    [-2, 0.0, 0.0],  # 'Right_Back_wheel'               # 19
    [2, 0.0, 0.0],   # 'Left_Back_wheel'                # 9
    [-1.3, 2.0, 0.0],  # 'Right_Front_Headlight',       # 3
    [1.3, 2.0, 0.0],  # 'Left_Front_Headlight',         # 4
    [1.6, 2.2, 2.0],  # 'Right_Back_Headlight',        # 14
    [-1.6, 2.2, 2.0],   # 'Left_Back_Headlight',         # 13
    [-2.0, 4.0, 0.0],  # 'Right_Front_Top',             # 1
    [2.0, 4.0, 0.0],   # 'Left_Front_Top',              # 2
    [2.0, 4.0, 2.0],  # 'Right_Back_Top'               # 12
    [-2.0, 4.0, 2.0],  # 'Left_Back_Top',                # 11
    [0, 0.0, 0.0], #Center
    [0, 2.2, 2.0], #Exhaust
])

CAR_POSE_LEFT = np.array([
    [-2, 0.0, 0.0],  # 'Right_Front_wheel', # 20
    [-4, 0.0, 2.0],  # 'Left_Front_wheel', # 8
    [-2, 0.0, 0.0],  # 'Right_Back_wheel'               # 19
    [4, 0.0, 2.0],   # 'Left_Back_wheel'                # 9
    [-1.3, 2.0, 0.0],  # 'Right_Front_Headlight',       # 3
    [1.3, 2.0, 0.0],  # 'Left_Front_Headlight',         # 4
    [1.6, 2.2, 0.0],  # 'Right_Back_Headlight',        # 14
    [5 + p, 2 + p, 1.0],   # 'Left_Back_Headlight',         # 13
    [-2.0, 4.0, 0.0],  # 'Right_Front_Top',             # 1
    [0 - 5 * p, 4.0 - p / 2, 2.0],   # 'Left_Front_Top',              # 2
    [2.0, 4.0, 0.0],  # 'Right_Back_Top'               # 12
    [0 + 5 * p, 4.0 - p / 2, 2.0],  # 'Left_Back_Top',                # 11
    [1, 0.0, 1.0], #Center
    [3.5, 2.2, 0.5], #Exhaust
])


CAR_POSE_RIGHT = np.array([
    [4, 0.0, 2.0],  # 'Right_Front_wheel', # 20
    [-4, 0.0, 0.0],  # 'Left_Front_wheel', # 8
    [-4, 0.0, 2.0],  # 'Right_Back_wheel'               # 19
    [4, 0.0, 0.0],   # 'Left_Back_wheel'                # 9
    [-1.3, 2.0, 0.0],  # 'Right_Front_Headlight',       # 3
    [1.3, 2.0, 0.0],  # 'Left_Front_Headlight',         # 4
    [-5 - p, 2.0 + p, 2.0],  # 'Right_Back_Headlight',        # 14
    [5, 2, 0.0],   # 'Left_Back_Headlight',         # 13
    [0 + 5 * p, 4.0 - p / 2, 2.0],  # 'Right_Front_Top',             # 1
    [0, 4.0, 0.0],   # 'Left_Front_Top',              # 2
    [0 - 5 * p, 4.0 - p / 2, 2.0],  # 'Right_Back_Top'               # 12
    [0 + 5, 4.0, 0.0],  # 'Left_Back_Top',                # 11
    [4, 0.0, 1.0], #Center
    [0, 2.2, 1.0], #Exhaust
])


def draw_ann(ann, *, keypoint_painter, filename=None, margin=0.5, aspect=None, **kwargs):
    from openpifpaf import show  # pylint: disable=import-outside-toplevel

    bbox = ann.bbox()
    xlim = bbox[0] - margin, bbox[0] + bbox[2] + margin
    ylim = bbox[1] - margin, bbox[1] + bbox[3] + margin
    if aspect == 'equal':
        fig_w = 5.0
    else:
        fig_w = 5.0 / (ylim[1] - ylim[0]) * (xlim[1] - xlim[0])

    with show.canvas(filename, figsize=(fig_w, 5), nomargin=True, **kwargs) as ax:
        ax.set_axis_off()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        if aspect is not None:
            ax.set_aspect(aspect)

        keypoint_painter.annotation(ax, ann)


def draw_skeletons(pose):
    import openpifpaf  # pylint: disable=import-outside-toplevel
    openpifpaf.show.KeypointPainter.show_joint_scales = True
    keypoint_painter = openpifpaf.show.KeypointPainter()

    scale = np.sqrt(
        (np.max(pose[:, 0]) - np.min(pose[:, 0]))
        * (np.max(pose[:, 1]) - np.min(pose[:, 1]))
    )

    ann = openpifpaf.Annotation(keypoints=CARFUSION_KEYPOINTS,
                                skeleton=CARFUSION_SKELETON,
                                score_weights=CARFUSION_SCORE_WEIGHTS)
    ann.set(pose, np.array(CARFUSION_SIGMAS) * scale)

    os.makedirs('docs', exist_ok=True)
    draw_ann(ann, filename='docs/skeleton_car.png', keypoint_painter=keypoint_painter)


def plot3d_red(ax_2D, p3d, skeleton):
    skeleton = [(bone[0] - 1, bone[1] - 1) for bone in skeleton]

    rot_p90_x = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    p3d = p3d @ rot_p90_x

    fig = ax_2D.get_figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.set_axis_off()
    ax_2D.set_axis_off()

    ax.view_init(azim=-90, elev=20)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    max_range = np.array([p3d[:, 0].max() - p3d[:, 0].min(),
                          p3d[:, 1].max() - p3d[:, 1].min(),
                          p3d[:, 2].max() - p3d[:, 2].min()]).max() / 2.0
    mid_x = (p3d[:, 0].max() + p3d[:, 0].min()) * 0.5
    mid_y = (p3d[:, 1].max() + p3d[:, 1].min()) * 0.5
    mid_z = (p3d[:, 2].max() + p3d[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)  # pylint: disable=no-member

    for ci, bone in enumerate(skeleton):
        c = mplcm.get_cmap('tab20')((ci % 20 + 0.05) / 20)  # Same coloring as Pifpaf preds
        ax.plot(p3d[bone, 0], p3d[bone, 1], p3d[bone, 2], color=c)

    def animate(i):
        ax.view_init(elev=10., azim=i)
        return fig

    return FuncAnimation(fig, animate, frames=360, interval=100)


def print_associations():
    print("\nAssociations of the car skeleton")
    for j1, j2 in CARFUSION_SKELETON:
        print(CARFUSION_KEYPOINTS[j1 - 1], '-', CARFUSION_KEYPOINTS[j2 - 1])


if __name__ == '__main__':
    print_associations()

    draw_skeletons(CAR_POSE)
    with openpifpaf.show.Canvas.blank(nomargin=True) as ax_2D:
        anim_24 = plot3d_red(ax_2D, CAR_POSE, CARFUSION_SKELETON)
        anim_24.save('openpifpaf/plugins/carfusion/docs/CAR_14_Pose.gif', fps=30)

