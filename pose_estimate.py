import numpy as np
from estimater import EstimatePose
import cv2

# 读取 FastSAM 传来的数据
mask = np.load("/home/zddd/tmp/mask.npy")
rgb  = np.load("/home/zddd/tmp/rgb.npy")

# 相机内参（你需要自己写对）
K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
], dtype=np.float32)

# 裁剪区域
ys, xs = np.where(mask > 0)
x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()

crop_rgb = rgb[y1:y2, x1:x2, :]
crop_mask = mask[y1:y2, x1:x2]

# 修正内参
K_crop = K.copy()
K_crop[0,2] -= x1
K_crop[1,2] -= y1

# 加载 FoundationPose
estimator = EstimatePose(
    model_dir="/home/zddd/FoundationPose/checkpoints/my_object/",
    object_name="my_object",
    cam_intrinsics=K_crop
)

pose = estimator.run(crop_rgb, None, crop_mask)

R = pose["R"]   # 旋转矩阵
t = pose["t"]   # 平移向量

print("R =\n", R)
print("t =\n", t)
