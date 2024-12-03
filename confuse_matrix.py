import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(confusion_matrix, labels, save_path):
    #plt.imshow(confusion_matrix, cmap='Blues', vmin=0, vmax=1)
    #plt.colorbar()
    fig, ax = plt.subplots()

    img = ax.imshow(confusion_matrix,  cmap='Blues', vmin=0, vmax=1, interpolation='nearest')  # origin='lower',

    cbar = plt.colorbar(img)

    cbar.outline.set_linewidth(2)
    cbar.ax.tick_params(labelsize=12, width=2)
    cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontweight='bold')


    # 在图中的每个方格中添加对应的值
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, "{:.5f}".format(confusion_matrix[i, j]),
                    horizontalalignment="center",
                    color="white" if confusion_matrix[i, j] > (confusion_matrix.max() / 2) else "black")

    # 设置轴标签、刻度和标题
    tick_marks = np.arange(len(labels))
    #plt.xticks(tick_marks, labels, rotation=45)
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)

    for label in ax.get_xticklabels():
        label.set_fontsize(10)  # 设置字体大小
        label.set_fontweight('bold')  # 设置字体粗体

    # 设置 y 轴刻度字体大小和粗体
    for label in ax.get_yticklabels():
        label.set_fontsize(10)  # 设置字体大小
        label.set_fontweight('bold')  # 设置字体粗体

    plt.xlabel('Predicted Class', fontweight='bold', fontsize=16)
    plt.ylabel('True Class', fontweight='bold', fontsize=16)
    # plt.title('Confusion Matrix', fontweight='bold', fontsize=16)

    # 在图中绘制网格线，以更好地区分方格
    plt.grid(False)

    # 自适应图像剪裁边界
    plt.tight_layout()

    # 显示图像
    # plt.show()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.ion()
    plt.pause(1)
    plt.close()



if __name__ == '__main__':
    # 示例混淆矩阵和标签
    confusion_matrix = np.array([[10, 2, 0],
                                [3, 8, 1],
                                [4, 6, 5]])
    labels = ['Class 1', 'Class 2', 'Class 3']

    # 绘制混淆矩阵图
    plot_confusion_matrix(confusion_matrix, labels)