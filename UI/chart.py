import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

def plot_enlarge(x_true, x_ekf, index, dB, modelname):
    fig, axs = plt.subplot_mosaic([['big','.'],['big','small']])
    fig.set_size_inches(10,7)
    x_true = x_true.detach().cpu().numpy()
    x_ekf = x_ekf.detach().cpu().numpy()
    ax = axs["big"]
    ax.plot(x_true[0,:,0], x_true[0,:,1], 'r-')
    ax.plot(x_ekf[0, :, 0], x_ekf[0, :, 1], 'b-')
    ax.plot(x_true[0,0,0], x_true[0,0,1], 'ro')
    ax.set_xlabel("x/m")
    ax.set_ylabel("y/m")
    ax.legend(["ground truth", "kalmanNet","start point"])

    dist = np.linalg.norm(x_true[0, :, :] - x_ekf[0,:,0:2], axis=1)
    max_idx = np.argmax(dist[3:]) + 3  # 计算距离最大的位置，shape为(batch_size,)

    # ax_zoom = zoomed_inset_axes(ax, 5,loc='lower right', bbox_to_anchor=Bbox([[10,10],[200,200]]))  # 15是放大倍数
    ax_zoom = axs["small"]
    ax_zoom.plot(x_true[0,:,0], x_true[0,:,1], 'r-')
    ax_zoom.plot(x_ekf[0, :, 0], x_ekf[0, :, 1], 'b-')
    # 设置放大轨迹的坐标范围
    point_num = 1
    min_x = np.min([x_true[0,max_idx-point_num,0], x_true[0,max_idx+point_num,0]])
    max_x = np.max([x_true[0,max_idx-point_num,0], x_true[0,max_idx+point_num,0]])
    min_y = np.min([x_true[0,max_idx-point_num,1], x_true[0,max_idx+point_num,1]])
    max_y = np.max([x_true[0,max_idx-point_num,1], x_true[0,max_idx+point_num,1]])
    ax_zoom.set_xlim([min_x, max_x])
    ax_zoom.set_ylim([min_y, max_y])
    ax_zoom.set_xlabel("x/m")
    ax_zoom.set_ylabel("y/m")
    ax_zoom.set_title("Partial Enlarged View")
    # 在原始轨迹中标记放大轨迹的区域
    mark_inset(ax, ax_zoom, loc1=1, loc2=3, fc="none", ec="0.5")
    fig.suptitle(f"{dB:.2f}dB")
    # 显示图像
    fig.savefig(f"./Result/{modelname}Test/{index}_{dB:.2f}dB.png")