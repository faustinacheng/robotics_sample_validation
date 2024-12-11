import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import time
import random
import math


class SampleValidationCUDA:
    def __init__(self):
        self.current_obstacle = "NONE"
        random.seed(time.time())

        self.cuda_kernel_code = """
        #include <curand_kernel.h>

        __device__ bool is_state_valid_cuda(float *q_seg) {
            curandState state;
            curand_init((unsigned long long)clock() + blockIdx.x * blockDim.x + threadIdx.x, 0, 0, &state);

            float x = curand_uniform(&state);
            return x > 0.015;
        }

        extern "C" {
        __global__ void validate_segment(float *q_start, float *q_end, bool *result, int num_segs, int num_elements) {
            extern __shared__ int shared_result[];
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (threadIdx.x == 0) {
                shared_result[0] = 0;
            }
            __syncthreads();
            if (idx < num_segs && shared_result[0] == 0) {
                for (int i = idx; i < num_segs + 1; i += blockDim.x) {
                    float q_seg[6];
                    float t = (float)i / num_segs;
                    for (int j = 0; j < num_elements; ++j) {
                        q_seg[j] = q_start[j] + t * (q_end[j] - q_start[j]);
                    }
                    if (!is_state_valid_cuda(q_seg)) {
                        shared_result[0] = 1;  // Mark as invalid
                        result[0] = false;
                    }
                }
            }
            __syncthreads();
            if (threadIdx.x == 0 && shared_result[0] == 0) {
                result[0] = true;
            }
        }
        }
        """

        self.mod = SourceModule(self.cuda_kernel_code, no_extern_c=True)
        self.validate_segment_kernel = self.mod.get_function("validate_segment")

    def is_segment_valid(self, q_start, q_end):
        if self.current_obstacle == "NONE":
            step_size = 0.05
        elif self.current_obstacle == "SIMPLE":
            step_size = 0.02
        elif self.current_obstacle == "HARD":
            step_size = 0.01
        elif self.current_obstacle == "SUPER":
            step_size = 0.005
        else:
            step_size = 0.05

        direction = [qe - qs for qs, qe in zip(q_start, q_end)]
        length = math.sqrt(sum(d**2 for d in direction))
        if length == 0:
            return self.is_state_valid(q_start)

        num_segs = max(1, int(length / step_size))

        q_start_np = np.array(q_start, dtype=np.float32)
        q_end_np = np.array(q_end, dtype=np.float32)
        q_result_np = np.array([True], dtype=np.bool_)

        threadsperblock = 128

        self.validate_segment_kernel(
            cuda.In(q_start_np),
            cuda.In(q_end_np),
            cuda.Out(q_result_np),
            np.int32(num_segs),
            np.int32(len(q_start)),
            block=(threadsperblock, 1, 1),
            grid=(1, 1),
            shared=4,
        )

        return q_result_np[0]
