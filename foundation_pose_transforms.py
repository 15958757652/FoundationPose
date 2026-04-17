import torch
import torch.nn.functional as F

# ---------------- SO3 ----------------

def hat(v):
    # v: (3,)
    return torch.tensor([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ], dtype=v.dtype, device=v.device)

def so3_exp_map(axis_angle):
    # axis_angle: (..., 3)
    angle = axis_angle.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    axis = axis_angle / angle
    half = angle * 0.5
    sin_half = torch.sin(half)
    qw = torch.cos(half)
    qxyz = axis * sin_half
    return quaternion_to_matrix(torch.cat([qw, qxyz], dim=-1))

def so3_log_map(R):
    # R: (..., 3, 3)
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = (trace - 1) / 2
    cos_theta = cos_theta.clamp(-1 + 1e-7, 1 - 1e-7)
    theta = torch.acos(cos_theta)
    w = torch.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1],
    ], dim=-1)
    return theta.unsqueeze(-1) * w / (2 * torch.sin(theta).unsqueeze(-1))

# ------------- Quaternion utils ----------------

def quaternion_to_matrix(q):
    # q: (..., 4)
    w, x, y, z = q.unbind(-1)
    B = q.shape[:-1]
    R = torch.empty(B + (3,3), device=q.device, dtype=q.dtype)
    R[...,0,0] = 1 - 2*(y*y + z*z)
    R[...,0,1] = 2*(x*y - z*w)
    R[...,0,2] = 2*(x*z + y*w)

    R[...,1,0] = 2*(x*y + z*w)
    R[...,1,1] = 1 - 2*(x*x + z*z)
    R[...,1,2] = 2*(y*z - x*w)

    R[...,2,0] = 2*(x*z - y*w)
    R[...,2,1] = 2*(y*z + x*w)
    R[...,2,2] = 1 - 2*(x*x + y*y)
    return R

# ------------- SE3 ----------------

def se3_exp_map(se3_vec):
    # (..., 6)
    rot = se3_vec[..., :3]
    trans = se3_vec[..., 3:]

    R = so3_exp_map(rot)
    return R, trans

def se3_log_map(R, t):
    aa = so3_log_map(R)
    return torch.cat([aa, t], dim=-1)

# ------------- Euler / 6D ----------------

def euler_angles_to_matrix(euler, convention="XYZ"):
    x, y, z = euler.unbind(-1)
    Rx = torch.stack([
        torch.ones_like(x), torch.zeros_like(x), torch.zeros_like(x),
        torch.zeros_like(x), torch.cos(x), -torch.sin(x),
        torch.zeros_like(x), torch.sin(x), torch.cos(x)
    ], dim=-1).reshape(euler.shape[:-1] + (3,3))

    Ry = torch.stack([
        torch.cos(y), torch.zeros_like(y), torch.sin(y),
        torch.zeros_like(y), torch.ones_like(y), torch.zeros_like(y),
        -torch.sin(y), torch.zeros_like(y), torch.cos(y)
    ], dim=-1).reshape(euler.shape[:-1] + (3,3))

    Rz = torch.stack([
        torch.cos(z), -torch.sin(z), torch.zeros_like(z),
        torch.sin(z), torch.cos(z), torch.zeros_like(z),
        torch.zeros_like(z), torch.zeros_like(z), torch.ones_like(z)
    ], dim=-1).reshape(euler.shape[:-1] + (3,3))

    return Rz @ Ry @ Rx

def matrix_to_euler_angles(R, convention="XYZ"):
    sy = torch.sqrt(R[...,0,0]**2 + R[...,1,0]**2)
    x = torch.atan2(R[...,2,1], R[...,2,2])
    y = torch.atan2(-R[...,2,0], sy)
    z = torch.atan2(R[...,1,0], R[...,0,0])
    return torch.stack([x, y, z], dim=-1)

def matrix_to_axis_angle(R):
    return so3_log_map(R)

def rotation_6d_to_matrix(rot_6d):
    a1 = F.normalize(rot_6d[..., 0:3], dim=-1)
    a2 = F.normalize(rot_6d[..., 3:6], dim=-1)
    a3 = torch.cross(a1, a2, dim=-1)
    return torch.stack([a1, a2, a3], dim=-1)
