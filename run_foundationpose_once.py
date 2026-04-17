import os
import argparse
import numpy as np
import trimesh
import nvdiffrast.torch as dr

from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
from Utils import get_torch_device

def load_single_mesh(mesh_file: str) -> trimesh.Trimesh:
    """
    确保从 obj 中拿到一个 Trimesh，而不是 Scene
    """
    # 尝试直接按 mesh 加载
    mesh = trimesh.load(mesh_file, force='mesh')
    # 有些情况下 force='mesh' 仍然会返回 Scene，这里再判断一次
    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 0:
            raise ValueError(f"No geometry found in mesh file: {mesh_file}")
        # 把 Scene 里的所有几何合并成一个 Trimesh
        mesh = trimesh.util.concatenate(
            [g for g in mesh.geometry.values()]
        )
    return mesh

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    args = parser.parse_args()

    device = get_torch_device(args.device)
    base_dir = "/home/zddd/tmp"

    rgb = np.load(os.path.join(base_dir, "rgb.npy"))    # H,W,3, BGR
    depth = np.load(os.path.join(base_dir, "depth.npy"))  # H,W, float32, meter
    mask = np.load(os.path.join(base_dir, "mask.npy"))    # H,W, 0/1
    K = np.load(os.path.join(base_dir, "K.npy")).astype(np.float32)

    # 如果内部假设 rgb 是 RGB，可根据需要转换
    rgb_rgb = rgb[..., ::-1]   # BGR -> RGB

    # 加载你的 omni_wheel mesh
    mesh_file = "/home/zddd/FoundationPose/objects/omni_wheel/object.obj"
    mesh = load_single_mesh(mesh_file)

    print("Loaded mesh type:", type(mesh))
    print("Vertices:", mesh.vertices.shape, "Faces:", mesh.faces.shape)

    scorer = ScorePredictor(device=device)
    refiner = PoseRefinePredictor(device=device)
    glctx = dr.RasterizeCudaContext(str(device)) if device.type == 'cuda' else dr.RasterizeGLContext()

    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
                        # symmetry_tfs 如果有特殊对称变换再传
        scorer=scorer,
        refiner=refiner,
        glctx=glctx,
        device=device,
        debug=2,
        debug_dir="/home/zddd/tmp/fp_debug"
    )

    os.makedirs("/home/zddd/tmp/fp_debug", exist_ok=True)

    print("Running FoundationPose.register ...")
    best_pose = est.register(
        K=K,
        rgb=rgb_rgb,
        depth=depth,
        ob_mask=mask.astype(np.uint8),
        iteration=5
    )

    

    print("Estimated pose 4x4:\n", best_pose)

if __name__ == "__main__":
    main()