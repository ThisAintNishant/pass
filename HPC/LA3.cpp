#include <iostream>
#include <climits>
#include <cuda_runtime.h>
using namespace std;

#define BLOCK_SIZE 256

_global_ void reduceMin(int* input, int* output, int size) {
    _shared_ int sdata[BLOCK_SIZE];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < size) ? input[i] : INT_MAX;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = min(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

_global_ void reduceMax(int* input, int* output, int size) {
    _shared_ int sdata[BLOCK_SIZE];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < size) ? input[i] : INT_MIN;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = max(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

_global_ void reduceSum(int* input, int* output, int size) {
    _shared_ int sdata[BLOCK_SIZE];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < size) ? input[i] : 0;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

_global_ void reduceAverage(int* input, float* output, int size) {
    _shared_ float sdata[BLOCK_SIZE];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < size) ? static_cast<float>(input[i]) : 0.0f;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0] / static_cast<float>(size);
    }
}

float measureKernelTime(void (kernel)(int, int*, int), int* d_input, int* d_output, int size, int blocks, int threads) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    kernel<<<blocks, threads>>>(d_input, d_output, size);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return milliseconds;
}

int main() {
    const int array_size = 256;
    int input[array_size];
    for (int i = 0; i < array_size; ++i) {
        input[i] = i + 1;
    }

    int* d_input;
    int *d_output_min, *d_output_max, *d_output_sum;
    float* d_output_avg;

    cudaMalloc(&d_input, sizeof(int) * array_size);
    cudaMalloc(&d_output_min, sizeof(int) * array_size);
    cudaMalloc(&d_output_max, sizeof(int) * array_size);
    cudaMalloc(&d_output_sum, sizeof(int) * array_size);
    cudaMalloc(&d_output_avg, sizeof(float) * array_size);

    cudaMemcpy(d_input, input, sizeof(int) * array_size, cudaMemcpyHostToDevice);

    int threads_per_block = BLOCK_SIZE;
    int blocks_per_grid = (array_size + threads_per_block - 1) / threads_per_block;

    cudaEvent_t start, stop;
    float time_ms;

    // Min
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduceMin<<<blocks_per_grid, threads_per_block>>>(d_input, d_output_min, array_size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    cout << "Min kernel time: " << time_ms << " ms" << endl;

    // Max
    cudaEventRecord(start);
    reduceMax<<<blocks_per_grid, threads_per_block>>>(d_input, d_output_max, array_size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    cout << "Max kernel time: " << time_ms << " ms" << endl;

    // Sum
    cudaEventRecord(start);
    reduceSum<<<blocks_per_grid, threads_per_block>>>(d_input, d_output_sum, array_size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    cout << "Sum kernel time: " << time_ms << " ms" << endl;

    // Average
    cudaEventRecord(start);
    reduceAverage<<<blocks_per_grid, threads_per_block>>>(d_input, d_output_avg, array_size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    cout << "Average kernel time: " << time_ms << " ms" << endl;

    int min_result, max_result, sum_result;
    float avg_result;

    cudaMemcpy(&min_result, d_output_min, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&max_result, d_output_max, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&sum_result, d_output_sum, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&avg_result, d_output_avg, sizeof(float), cudaMemcpyDeviceToHost);

    cout << "Minimum value: " << min_result << endl;
    cout << "Maximum value: " << max_result << endl;
    cout << "Sum: " << sum_result << endl;
    cout << "Average: " << avg_result << endl;

    cudaFree(d_input);
    cudaFree(d_output_min);
    cudaFree(d_output_max);
    cudaFree(d_output_sum);
    cudaFree(d_output_avg);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}