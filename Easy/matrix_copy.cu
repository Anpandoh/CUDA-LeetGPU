#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>

__global__ void copy_matrix_kernel(const float* A, float* B, int N) {
    int indexX = threadIdx.x + blockDim.x*blockIdx.x;
    // int indexY = threadIdx.y + blockDim.y*blockIdx.y;

    if (indexX < N*N) {
        B[indexX] = A[indexX];
        // B[indexX][indexY] = A[indexX][indexY];
    }

}

// A, B are device pointers (NxN matrices)
extern "C" void solve(const float* A, float* B, int N) {
    int totalElements = N * N;
    int threadsPerBlock = 256;
    int blocksPerGrid = (totalElements + threadsPerBlock - 1) / threadsPerBlock;

    copy_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, N);
    cudaDeviceSynchronize();
}

void generate_test_arrays(float** A_cases, float** B_cases, int* sizes, int cases) {
    for (int i = 0; i < cases; i++) {
        int N = rand() % 50 + 10;
        sizes[i] = N;
        
        A_cases[i] = new float[N * N];
        B_cases[i] = new float[N * N];
        
        for (int j = 0; j < N * N; j++) {
            A_cases[i][j] = static_cast<float>(rand() % 100) / 10.0f;
        }
    }
}

int main(int argc, char* argv[]) {
    int cases = 10;
    float** A_cases = new float*[cases];
    float** B_cases = new float*[cases];
    int* sizes = new int[cases];

    generate_test_arrays(A_cases, B_cases, sizes, cases);

    for (int i = 0; i < cases; i++) {
        int N = sizes[i];

        float *d_A, *d_B;
        cudaMalloc(&d_A, N * N * sizeof(float));
        cudaMalloc(&d_B, N * N * sizeof(float));

        cudaMemcpy(d_A, A_cases[i], N * N * sizeof(float), cudaMemcpyHostToDevice);

        solve(d_A, d_B, N);

        cudaMemcpy(B_cases[i], d_B, N * N * sizeof(float), cudaMemcpyDeviceToHost);

        // Verify copy
        bool pass = true;
        for (int j = 0; j < N * N; j++) {
            if (fabs(A_cases[i][j] - B_cases[i][j]) > 1e-5) {
                pass = false;
                break;
            }
        }

        std::cout << "Case " << i << " (N=" << N << ", " << N*N << " elements): ";
        std::cout << "First element: A=" << A_cases[i][0] << ", B=" << B_cases[i][0];
        std::cout << (pass ? " [PASS]" : " [FAIL]") << std::endl;

        cudaFree(d_A);
        cudaFree(d_B);
    }

    for (int i = 0; i < cases; i++) {
        delete[] A_cases[i];
        delete[] B_cases[i];
    }
    delete[] A_cases;
    delete[] B_cases;
    delete[] sizes;

    return 0;
}