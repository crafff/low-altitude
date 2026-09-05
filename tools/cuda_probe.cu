// Tiny native CUDA smoke probe. Compile and execute only through tools/lab.py.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void check(cudaError_t code, const char* operation) {
    if (code != cudaSuccess) {
        std::printf("{\"ok\":false,\"operation\":\"%s\",\"cuda_error\":%d,"
                    "\"message\":\"%s\"}\n",
                    operation, static_cast<int>(code), cudaGetErrorString(code));
        std::exit(1);
    }
}

__global__ void affine(int* output, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) output[i] = 3 * i + 7;
}

int main() {
    int devices = 0, driver = 0, runtime = 0;
    check(cudaGetDeviceCount(&devices), "device_count");
    if (devices == 0) {
        std::puts("{\"ok\":false,\"message\":\"no CUDA device\"}");
        return 1;
    }
    check(cudaDriverGetVersion(&driver), "driver_version");
    check(cudaRuntimeGetVersion(&runtime), "runtime_version");
    check(cudaSetDevice(0), "set_device");
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, 0), "device_properties");
    size_t free_bytes = 0, total_bytes = 0;
    check(cudaMemGetInfo(&free_bytes, &total_bytes), "memory_info");
    constexpr int count = 4096;
    int* device_output = nullptr;
    check(cudaMalloc(&device_output, count * sizeof(int)), "allocate");
    affine<<<16, 256>>>(device_output, count);
    check(cudaGetLastError(), "kernel_launch");
    check(cudaDeviceSynchronize(), "kernel_synchronize");
    std::vector<int> output(count);
    check(cudaMemcpy(output.data(), device_output, count * sizeof(int),
                     cudaMemcpyDeviceToHost), "copy_to_host");
    check(cudaFree(device_output), "free");
    int mismatch = 0;
    for (int i = 0; i < count; ++i) mismatch += output[i] != 3 * i + 7;
    std::printf("{\"ok\":%s,\"device_count\":%d,\"selected_device\":0,"
                "\"compute_major\":%d,\"compute_minor\":%d,"
                "\"driver_api_version\":%d,\"runtime_version\":%d,"
                "\"free_bytes_before_allocation\":%zu,\"total_bytes\":%zu,"
                "\"elements\":%d,\"allocation_bytes\":%zu,\"mismatches\":%d}\n",
                mismatch == 0 ? "true" : "false", devices,
                properties.major, properties.minor, driver, runtime,
                free_bytes, total_bytes, count, count * sizeof(int), mismatch);
    return mismatch == 0 ? 0 : 1;
}
