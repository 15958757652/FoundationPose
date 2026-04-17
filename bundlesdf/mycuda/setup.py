from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='common',
    ext_modules=[
        CUDAExtension(
            name='common',
            sources=[
                'common.cu',
                'bbox.cu',
                'raymarcher.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math'
                ],
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
