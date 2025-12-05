#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {


    int idx = threadIdx.x + blockIdx.x * blockDim.x;


    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }


}

// A, B, C are device pointers
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}

void generate_test_arrays(float** A_cases, float** B_cases, float** C_cases, int* sizes, int cases) {
    for (int i = 0; i < cases; i++) {
        int N = rand() % 1000 + 100;
        sizes[i] = N;
        
        A_cases[i] = new float[N];
        B_cases[i] = new float[N];
        C_cases[i] = new float[N];
        
        for (int j = 0; j < N; j++) {
            A_cases[i][j] = static_cast<float>(rand() % 100) / 10.0f;
            B_cases[i][j] = static_cast<float>(rand() % 100) / 10.0f;
        }
    }
}

int main(int argc, char* argv[]) {
    int cases = 10;
    float** A_cases = new float*[cases];
    float** B_cases = new float*[cases];
    float** C_cases = new float*[cases];
    int* sizes = new int[cases];

    generate_test_arrays(A_cases, B_cases, C_cases, sizes, cases);

    for (int i = 0; i < cases; i++) {
        int N = sizes[i];

        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, N * sizeof(float));
        cudaMalloc(&d_B, N * sizeof(float));
        cudaMalloc(&d_C, N * sizeof(float));

        cudaMemcpy(d_A, A_cases[i], N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B_cases[i], N * sizeof(float), cudaMemcpyHostToDevice);

        solve(d_A, d_B, d_C, N);

        cudaMemcpy(C_cases[i], d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

        // Verify addition
        bool pass = true;
        for (int j = 0; j < N; j++) {
            float expected = A_cases[i][j] + B_cases[i][j];
            if (fabs(C_cases[i][j] - expected) > 1e-5) {
                pass = false;
                break;
            }
        }

        std::cout << "Case " << i << " (N=" << N << "): ";
        std::cout << A_cases[i][0] << " + " << B_cases[i][0] << " = " << C_cases[i][0];
        std::cout << (pass ? " [PASS]" : " [FAIL]") << std::endl;

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    for (int i = 0; i < cases; i++) {
        delete[] A_cases[i];
        delete[] B_cases[i];
        delete[] C_cases[i];
    }
    delete[] A_cases;
    delete[] B_cases;
    delete[] C_cases;
    delete[] sizes;

    return 0;
}