import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import ListedColormap
from config import *

def plot_heatmap(data, output_file=None, dpi=600, vertical_lines=None, horizontal_lines=None,v_labels=None, h_labels=None, offset=0):
    fig, ax = plt.subplots()


    img = ax.imshow(data, cmap='jet', interpolation='nearest',  vmin=0.3, vmax=0.8)  #origin='lower',

    cbar = plt.colorbar(img)

    cbar.outline.set_linewidth(2)
    cbar.ax.tick_params(labelsize=12, width=2)
    cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontweight='bold')
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # update 
    nb_residues = data.shape[0]

    disc = 40
    start_tick = (offset // disc) * disc
    if start_tick < offset:
        start_tick += disc

    x_tick_labels = np.arange(start_tick, nb_residues+offset, disc)
    y_tick_labels = np.arange(start_tick, nb_residues+offset, disc)

    x_positions = np.arange(start_tick - offset, nb_residues, disc)
    y_positions = np.arange(start_tick - offset, nb_residues, disc)
    # x_tikcs = np.arange(0, nb_residues, 20)
    # y_ticks = np.arange(0, nb_residues, 20)

    plt.xticks(x_positions, x_tick_labels, fontweight='bold', fontsize=12)
    plt.yticks(y_positions, y_tick_labels, fontweight='bold', fontsize=12)

    for spine in plt.gca().spines.values():
        spine.set_linewidth(2)
    plt.tick_params(width=2)

    # cbar.ax.set_position([0.745, 0.11, 0.05, 0.77])

    # plt.title('Correlation Matrix', fontweight='bold', fontsize=20)
    plt.xlabel('Residue Index', fontweight='bold', fontsize=14)
    plt.ylabel('Residue Index', fontweight='bold', fontsize=14)

    # draw vertial virtual line
    if vertical_lines is not None and v_labels is not None:
        for i, x in enumerate(vertical_lines):
            if i == len(vertical_lines) - 1:
                start_pos = 0
            else:
                start_pos = (vertical_lines[i + 1] - x) / 2
            ax.axvline(x, linestyle='--', color='#4169E1', linewidth=0.5)
            ax.text(x + start_pos, -1.5, v_labels[i], horizontalalignment='center', fontweight='bold', fontsize=10,
                    rotation=5)

        # Draw horizontal dashed lines
    if horizontal_lines is not None and h_labels is not None:
        for i, y in enumerate(horizontal_lines):
            if i == len(horizontal_lines) - 1:
                start_pos = 0
            else:
                start_pos = (horizontal_lines[i + 1] - y) / 2
            ax.axhline(y, linestyle='--', color='#4169E1', linewidth=0.5)
            ax.text(nb_residues - 0.5, y + start_pos, h_labels[i], verticalalignment='center', fontweight='bold',
                    fontsize=9, rotation=85)

    if output_file:
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    else:
        plt.show()


def plot_heatmap2(data, output_file=None, dpi=600, vertical_lines=None, horizontal_lines=None, offset=0):


    #custom_colors = ['#619DB8', '#AECDD7', '#E3EEEF', '#FAE7D9', '#F0B79A', '#C85D4D']
    custom_colors = ['#E3EEEF', '#F6C63C', '#EFA143', '#D96558','#B43970','#692F7C']
    custom_cmap = ListedColormap(custom_colors)

    fig, ax = plt.subplots()

    img = ax.imshow(data, cmap=custom_cmap, interpolation='nearest', vmin=0.2, vmax=0.8)  # origin='lower',


    cbar = plt.colorbar(img)

    cbar.outline.set_linewidth(2)
    cbar.ax.tick_params(labelsize=12, width=2)
    cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontweight='bold')
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # update
    nb_residues = data.shape[0]

    disc = 40
    start_tick = (offset // disc) * disc
    if start_tick < offset:
        start_tick += disc

    x_tick_labels = np.arange(start_tick, nb_residues + offset, disc)
    y_tick_labels = np.arange(start_tick, nb_residues + offset, disc)

    x_positions = np.arange(start_tick - offset, nb_residues, disc)
    y_positions = np.arange(start_tick - offset, nb_residues, disc)
    # x_tikcs = np.arange(0, nb_residues, 20)
    # y_ticks = np.arange(0, nb_residues, 20)

    plt.xticks(x_positions, x_tick_labels, fontweight='bold', fontsize=12)
    plt.yticks(y_positions, y_tick_labels, fontweight='bold', fontsize=12)

    for spine in plt.gca().spines.values():
        spine.set_linewidth(2)
    plt.tick_params(width=2)

    # cbar.ax.set_position([0.745, 0.11, 0.05, 0.77])

    # plt.title('Correlation Matrix', fontweight='bold', fontsize=20)
    plt.xlabel('Residue Index', fontweight='bold', fontsize=14)
    plt.ylabel('Residue Index', fontweight='bold', fontsize=14)

    # draw vertial virtual line
    if vertical_lines is not None:
        for x in vertical_lines:
            plt.axvline(x, linestyle='--', color='gray', linewidth=1)

    # draw horizontal virtual line
    if horizontal_lines is not None:
        for y in horizontal_lines:
            plt.axhline(y, linestyle='--', color='gray', linewidth=1)

    if output_file:
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    else:
        plt.show()

if __name__ == '__main__':
    # 示例数据和输出文件
    data = np.loadtxt('m3.dat')
    output_file = "correlation_matrix.png"

    # 调用封装函数绘制相关性矩阵图
    plot_heatmap(data, output_file=output_file, dpi=600)