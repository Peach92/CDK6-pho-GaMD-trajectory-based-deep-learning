import os
import cv2
import numpy as np

folder_path = './results'  # 替换为目标文件夹的路径
image_list = os.listdir(folder_path)  # 获取文件夹中的所有文件名

image_sum = None  # 存储所有图像的像素总和

threshold_value_scale = 0.5
threshold_value = threshold_value_scale * 255  # 设置像素阈值
pixels_above_threshold = []  # 存储大于阈值的像素坐标

for image_name in image_list:
    image_path = os.path.join(folder_path, image_name)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE )

    if image_sum is None:
        image_sum = np.array(image, dtype=np.uint64)
    else:
        image_sum += np.array(image, dtype=np.uint64)

num_images = len(image_list)
average_image = (image_sum // num_images).astype(np.uint8)  # 计算平均值图像

above_threshold = np.argwhere(average_image > threshold_value)  # 获取大于阈值的像素坐标
print(above_threshold)

print(*above_threshold + 1)

save_path = './residues_cdk6d_lqq.txt'  # 保存坐标的文件路径
np.savetxt(save_path, above_threshold, fmt='%d')

save_image_path = './average_cdk6d_lqq.jpg'  # 保存平均值图像的路径和文件名
cv2.imwrite(save_image_path, average_image)

print("Coordinates saved to", save_path)
print("Average image saved to", save_image_path)