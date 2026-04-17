#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void bbox_kernel(
    const scalar_t* __restrict__ in,
    scalar_t* __restrict__ out,
    const int N)
{
    // 这里的 kernel 是占位示例逻辑（按实际需求可以调整）
    // 简单地把 in 拷贝到 out（或做一些简单计算）
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    out[idx] = in[idx] * 1.0; // 占位操作
}

void bbox_launcher(at::Tensor in, at::Tensor out)
{
    const int N = in.numel();
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(in.type(), "bbox_launcher", ([&] {
        bbox_kernel<scalar_t><<<blocks, threads>>>(
            in.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            N);
    }));
}
