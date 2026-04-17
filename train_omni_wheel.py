import os
import sys

# 把 FoundationPose 根目录加入 Python 路径
sys.path.append("/home/zddd/FoundationPose")

# 正确导入训练配置与训练器（基于你实际的文件结构）
from learning.training.training_config import TrainingConfig
from learning.training.predict_pose_refine import Trainer


def main():
    obj_dir = "/home/zddd/FoundationPose/objects/omni_wheel/"
    out_dir = "/home/zddd/FoundationPose/training/omni_wheel/"

    os.makedirs(out_dir, exist_ok=True)

    # 创建训练配置（字段名与你的 training_config.py 完全对应）
    cfg = TrainingConfig(
        object_model_path=os.path.join(obj_dir, "object.obj"),
        output_dir=out_dir,
        batch_size=32,
        total_iterations=5000,
        render_image_size=256,
        use_texture=False,              # 你没有纹理，必须 False
        use_photometric_loss=False,     # 无纹理必须关闭
        camera_distance_range=(0.2, 0.6),
        elevation_range=(-30, 60),
        azimuth_range=(0, 360),
        lr=1e-4
    )

    print("开始训练 omni_wheel...")
    trainer = Trainer(cfg)
    trainer.train()
    print("训练完成！模型保存在：", out_dir)


if __name__ == "__main__":
    main()
