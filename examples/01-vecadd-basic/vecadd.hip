#include <hip/hip_runtime.h>
#include <iostream>

// HIP kernel，实现向量相加
__global__
void vecAdd(const float* A, const float* B, float* C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

int main()
{
    int N = 1 << 12; // 元素个数（4096），保持 gate 友好的非压力规模
    size_t bytes = N * sizeof(float);

    // 分配主机内存
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];

    // 初始化输入向量
    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(2 * i);
    }

    // 分配设备内存
    float *d_A, *d_B, *d_C;
    hipMalloc(&d_A, bytes);
    hipMalloc(&d_B, bytes);
    hipMalloc(&d_C, bytes);

    // 拷贝输入数据到设备
    hipMemcpy(d_A, h_A, bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, bytes, hipMemcpyHostToDevice);

    // 设置 kernel 每个线程块和网格大小
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // 启动 HIP kernel
    hipLaunchKernelGGL(vecAdd, dim3(gridSize), dim3(blockSize), 0, 0, d_A, d_B, d_C, N);

    // 把结果拷贝回主机
    hipMemcpy(h_C, d_C, bytes, hipMemcpyDeviceToHost);

    // 简单地验证结果
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            correct = false;
            std::cout << "Mismatch at i=" << i << std::endl;
            break;
        }
    }

    if(correct) std::cout << "vecadd validation ok" << std::endl;
    else std::cout << "Vector addition failed!" << std::endl;

    // 清理内存
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
