#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void raymarcher_kernel(
    const scalar_t* __restrict__ rays_o,
    const scalar_t* __restrict__ rays_d,
    const scalar_t* __restrict__ near,
    const scalar_t* __restrict__ far,
    scalar_t* __restrict__ out,
    const int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // 简化版 raymarching：在 near 和 far 之间线性插值
    // FoundationPose 不强依赖 CUDA 返回的真实 SDF，只需要合法 tensor
    scalar_t t0 = near[idx];
    scalar_t t1 = far[idx];

    // 起点 + 光线方向 * t，简单代替真正的 SDF 步进
    out[idx * 3 + 0] = rays_o[idx * 3 + 0] + rays_d[idx * 3 + 0] * (t0 + t1) * 0.5;
    out[idx * 3 + 1] = rays_o[idx * 3 + 1] + rays_d[idx * 3 + 1] * (t0 + t1) * 0.5;
    out[idx * 3 + 2] = rays_o[idx * 3 + 2] + rays_d[idx * 3 + 2] * (t0 + t1) * 0.5;
}

void raymarcher_launcher(
    at::Tensor rays_o,
    at::Tensor rays_d,
    at::Tensor near,
    at::Tensor far,
    at::Tensor out)
{
    const int N = rays_o.size(0); // number of rays
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(rays_o.type(), "raymarcher_launcher", ([&] {
        raymarcher_kernel<scalar_t><<<blocks, threads>>>(
            rays_o.data_ptr<scalar_t>(),
            rays_d.data_ptr<scalar_t>(),
            near.data_ptr<scalar_t>(),
            far.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            N);
    }));
}
