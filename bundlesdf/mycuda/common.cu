#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// External launcher declarations
void bbox_launcher(at::Tensor in, at::Tensor out);
void raymarcher_launcher(
    at::Tensor rays_o,
    at::Tensor rays_d,
    at::Tensor near,
    at::Tensor far,
    at::Tensor out
);

// =========================================================================
// Simple example op
// =========================================================================
template <typename scalar_t>
__global__ void common_kernel(
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ out,
    const int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    out[idx] = a[idx] + b[idx];
}

void common_launcher(at::Tensor a, at::Tensor b, at::Tensor out)
{
    const int N = a.numel();
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "common_launcher", ([&] {
        common_kernel<scalar_t><<<blocks, threads>>>(
            a.data_ptr<scalar_t>(),
            b.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            N);
    }));
}

// =========================================================================
// PyBind registration
// =========================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("common", &common_launcher, "common kernel");
    m.def("bbox", &bbox_launcher, "bbox kernel");
    m.def("raymarcher", &raymarcher_launcher, "raymarcher kernel");
}
