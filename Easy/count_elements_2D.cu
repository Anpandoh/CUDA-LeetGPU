#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

__global__ void count_2d_equal_kernel(const int* input, int* output, int N, int M, int K) {
    int indexX = threadIdx.x + blockDim.x*blockIdx.x;
    int indexY = threadIdx.y + blockDim.y*blockIdx.y;


    if (indexX < N && indexY < M) {
        if (input[indexX + indexY * N] == K) {
            atomicAdd(output, 1);
        } 
    }

}

// input, output are device pointers
extern "C" void solve(const int* input, int* output, int N, int M, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    count_2d_equal_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, M, K);
    cudaDeviceSynchronize();
}

void generate_test_arrays(int** input_cases, int* N_arr, int* M_arr, int* K_arr, int cases) {
    for (int i = 0; i < cases; i++) {
        int N = rand() % 100 + 10;
        int M = rand() % 100 + 10;
        int K = rand() % 10;
        N_arr[i] = N;
        M_arr[i] = M;
        K_arr[i] = K;
        
        input_cases[i] = new int[N * M];
        for (int j = 0; j < N * M; j++) {
            input_cases[i][j] = rand() % 10;
        }
    }
}

int main(int argc, char* argv[]) {
    int cases = 10;
    int** input_cases = new int*[cases];
    int* N_arr = new int[cases];
    int* M_arr = new int[cases];
    int* K_arr = new int[cases];

    generate_test_arrays(input_cases, N_arr, M_arr, K_arr, cases);

    for (int i = 0; i < cases; i++) {
        int N = N_arr[i];
        int M = M_arr[i];
        int K = K_arr[i];

        int *d_input, *d_output;
        int h_output = 0;
        
        cudaMalloc(&d_input, N * M * sizeof(int));
        cudaMalloc(&d_output, sizeof(int));
        
        cudaMemcpy(d_input, input_cases[i], N * M * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_output, &h_output, sizeof(int), cudaMemcpyHostToDevice);

        solve(d_input, d_output, N, M, K);

        cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);

        // CPU verification
        int cpu_count = 0;
        for (int j = 0; j < N * M; j++) {
            if (input_cases[i][j] == K) cpu_count++;
        }

        std::cout << "Case " << i << " (N=" << N << ", M=" << M << ", K=" << K << "): ";
        std::cout << "GPU count=" << h_output << ", CPU count=" << cpu_count;
        std::cout << (h_output == cpu_count ? " [PASS]" : " [FAIL]") << std::endl;

        cudaFree(d_input);
        cudaFree(d_output);
    }

    for (int i = 0; i < cases; i++) {
        delete[] input_cases[i];
    }
    delete[] input_cases;
    delete[] N_arr;
    delete[] M_arr;
    delete[] K_arr;

    return 0;
}