#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int idxCol = threadIdx.x + blockIdx.x * blockDim.x;
    int idxRow = threadIdx.y + blockIdx.y * blockDim.y;

    if (idxCol < K and idxRow < M){
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
                    //For A
                    //i iterates through the  cols as we are multiplying
                    //the idxRow is just for selection of row
                    //For B
                    //idxCol iterates through the row as we are multiplying
                    //the i is just fo selection of col
                    //the idxRow and idxCol dont overflow because they are bound by the if statement earlier
            sum += A[idxRow * N + i] * B[i * K + idxCol];
        }

        //                                        IdxCol      0     1
        //idxRow * k + indxCol is just finding the position [xxxx, xxXx]
        C[idxRow * K + idxCol] = sum;
    }

}

// A, B, C are device pointers. A is MxN, B is NxK, C is MxK
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

void generate_test_arrays(float** A_cases, float** B_cases, float** C_cases, 
                          int* M_arr, int* N_arr, int* K_arr, int cases) {
    for (int i = 0; i < cases; i++) {
        int M = rand() % 50 + 10;
        int N = rand() % 50 + 10;
        int K = rand() % 50 + 10;
        M_arr[i] = M;
        N_arr[i] = N;
        K_arr[i] = K;
        
        A_cases[i] = new float[M * N];
        B_cases[i] = new float[N * K];
        C_cases[i] = new float[M * K];
        
        for (int j = 0; j < M * N; j++) {
            A_cases[i][j] = static_cast<float>(rand() % 10) / 10.0f;
        }
        for (int j = 0; j < N * K; j++) {
            B_cases[i][j] = static_cast<float>(rand() % 10) / 10.0f;
        }
    }
}

int main(int argc, char* argv[]) {
    int cases = 10;
    float** A_cases = new float*[cases];
    float** B_cases = new float*[cases];
    float** C_cases = new float*[cases];
    int* M_arr = new int[cases];
    int* N_arr = new int[cases];
    int* K_arr = new int[cases];

    generate_test_arrays(A_cases, B_cases, C_cases, M_arr, N_arr, K_arr, cases);

    for (int i = 0; i < cases; i++) {
        int M = M_arr[i];
        int N = N_arr[i];
        int K = K_arr[i];

        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, M * N * sizeof(float));
        cudaMalloc(&d_B, N * K * sizeof(float));
        cudaMalloc(&d_C, M * K * sizeof(float));

        cudaMemcpy(d_A, A_cases[i], M * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B_cases[i], N * K * sizeof(float), cudaMemcpyHostToDevice);

        solve(d_A, d_B, d_C, M, N, K);

        cudaMemcpy(C_cases[i], d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost);

        // CPU verification of first element C[0][0]
        float cpu_c00 = 0.0f;
        for (int j = 0; j < N; j++) {
            cpu_c00 += A_cases[i][j] * B_cases[i][j * K];
        }
        bool pass = fabs(C_cases[i][0] - cpu_c00) < 1e-3;

        std::cout << "Case " << i << " (M=" << M << ", N=" << N << ", K=" << K << "): ";
        std::cout << "C[0][0]: GPU=" << C_cases[i][0] << ", CPU=" << cpu_c00;
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
    delete[] M_arr;
    delete[] N_arr;
    delete[] K_arr;

    return 0;
}