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

        __device__ float* step(float *q_near, float *q_rand, int num_elements, float step_size, float *direction, float *steps) {
            float length = 0.0f;

            for (int i = 0; i < num_elements; ++i) {
                direction[i] = q_rand[i] - q_near[i];
                length += direction[i] * direction[i];
            }
            length = sqrtf(length);
            if (length == 0.0f) {
                return q_near;
            }

            for (int i = 0; i < num_elements; ++i) {
                steps[i] = q_near[i] + (direction[i] / length) * step_size;
                steps[i] = fmaxf(fminf(steps[i], M_PI), -M_PI);
            }
            return steps;
        }


        __device__ bool is_state_valid_cuda(float *q_seg) {
            curandState state;
            curand_init((unsigned long long)clock() + blockIdx.x * blockDim.x + threadIdx.x, 0, 0, &state);

            float x = curand_uniform(&state);
            //printf("x: %f\\n", x);
            return x > 0.015;
        }

        extern "C" {
        __global__ void validate_segment(float *q_start, float *q_end, float *direction, float *steps, bool *result, float step_size, int num_segs, int num_elements, int segments_per_thread) {
            extern __shared__ int shared_result[];
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (threadIdx.x == 0) {
                shared_result[0] = 0;
            }
            __syncthreads();
            if (idx < num_segs && shared_result[0] == 0) {
                float *res;
                for (int i = idx; i < num_segs + 1; i += segments_per_thread) {
                    float q_seg[6];
                    float t = (float)i / num_segs;
                    for (int j = 0; j < num_elements; ++j) {
                        q_seg[j] = q_start[j] + t * (q_end[j] - q_start[j]);
                    }
                    //res = step(q_start, q_end, num_elements, i * step_size, direction, steps);
                    if (!is_state_valid_cuda(res)) {
                        //printf("Invalid segment at %d\\n", idx);
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
        # self.cuda_kernel_code = """
        # #include <curand_kernel.h>

        # __device__ bool is_state_valid_cuda(float *q_seg) {
        #     curandState state;
        #     curand_init((unsigned long long)clock() + threadIdx.x, 0, 0, &state);

        #     float x = curand_uniform(&state);
        #     return x > 0.015;
        # }

        # __global__ void validate_segment(float *q_start, float *q_end, bool *result, float step_size, int num_segs) {
        #     extern __shared__ int shared_result;
        #     int idx = blockIdx.x * blockDim.x + threadIdx.x;
        #     if (idx < num_segs && shared_result == 0) {
        #         float t = (float)idx / (num_segs - 1);
        #         float q_seg[3];
        #         for (int i = 0; i < 3; ++i) {
        #             q_seg[i] = q_start[i] + t * (q_end[i] - q_start[i]);
        #         }
        #         // Here you would call a CUDA version of is_state_valid, which we assume exists for simplicity
        #         if (!is_state_valid_cuda(q_seg)) {
        #             printf("Invalid segment at %d\\n", idx);
        #             shared_result = 1;  // Mark as invalid
        #             result[0] = false;
        #         }
        #     }
        # }
        # """
        self.mod = SourceModule(self.cuda_kernel_code, no_extern_c=True)
        self.validate_segment_kernel = self.mod.get_function("validate_segment")

    def trajectory_sample(self):
        q_rand = [random.uniform(-math.pi, math.pi) for _ in range(self.num_joints)]
        return q_rand

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
        direction_np = np.zeros(len(q_start), dtype=np.float32)
        steps_np = np.zeros(len(q_start), dtype=np.float32)

        # Prepare data for GPU
        # q_start_gpu = cuda.mem_alloc(q_start_np.nbytes)
        # q_end_gpu = cuda.mem_alloc(q_end_np.nbytes)
        # result_gpu = cuda.mem_alloc(np.bool_().itemsize)

        # cuda.memcpy_htod(q_start_gpu, np.array(q_start, dtype=np.float32))
        # cuda.memcpy_htod(q_end_gpu, np.array(q_end, dtype=np.float32))
        # cuda.memcpy_htod(result_gpu, np.array([True], dtype=np.bool_))

        # Launch kernel
        threadsperblock = 256
        # blockspergrid = (num_segs + threadsperblock - 1) // threadsperblock
        blockspergrid = 1
        segments_per_thread = math.ceil(num_segs / (threadsperblock * blockspergrid))
        print(f"num_segs: {num_segs}, segments_per_thread: {segments_per_thread}")
        self.validate_segment_kernel(
            cuda.In(q_start_np),
            cuda.In(q_end_np),
            cuda.In(direction_np),
            cuda.In(steps_np),
            cuda.Out(q_result_np),
            np.float32(step_size),
            np.int32(num_segs),
            np.int32(len(q_start)),
            np.int32(segments_per_thread),
            block=(threadsperblock, 1, 1),
            grid=(blockspergrid, 1),
            shared=4,
        )

        # Copy result back
        # result = np.array([True], dtype=np.bool_)
        # result[0] = False
        # cuda.memcpy_dtoh(result, result_gpu)

        # Clean up
        # q_start_gpu.free()
        # q_end_gpu.free()
        # result_gpu.free()
        # print(q_result_np[0])
        return q_result_np[0]
