#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

__global__ void count_equal_kernel(const int* input, int* output, int N, int K) {
    int index = threadIdx.x + blockDim.x*blockIdx.x;

    if (index < N) {
        if (input[index] == K) {
            //making it atomic so that no concurrency issues
            atomicAdd(output, 1);
        }
    }
    

}

// input, output are device pointers
extern "C" void solve(const int* input, int* output, int N, int K) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    count_equal_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, K);
    cudaDeviceSynchronize();
}

void generate_test_arrays(int** input_cases, int* sizes, int* K_arr, int cases) {
    for (int i = 0; i < cases; i++) {
        int N = rand() % 1000 + 100;
        int K = rand() % 10;
        sizes[i] = N;
        K_arr[i] = K;
        
        input_cases[i] = new int[N];
        for (int j = 0; j < N; j++) {
            input_cases[i][j] = rand() % 10;
        }
    }
}

int main(int argc, char* argv[]) {
    int cases = 10;
    int** input_cases = new int*[cases];
    int* sizes = new int[cases];
    int* K_arr = new int[cases];

    generate_test_arrays(input_cases, sizes, K_arr, cases);

    for (int i = 0; i < cases; i++) {
        int N = sizes[i];
        int K = K_arr[i];

        int *d_input, *d_output;
        int h_output = 0;
        
        cudaMalloc(&d_input, N * sizeof(int));
        cudaMalloc(&d_output, sizeof(int));
        
        cudaMemcpy(d_input, input_cases[i], N * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_output, &h_output, sizeof(int), cudaMemcpyHostToDevice);

        solve(d_input, d_output, N, K);

        cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);

        // CPU verification
        int cpu_count = 0;
        for (int j = 0; j < N; j++) {
            if (input_cases[i][j] == K) cpu_count++;
        }

        std::cout << "Case " << i << " (N=" << N << ", K=" << K << "): ";
        std::cout << "GPU count=" << h_output << ", CPU count=" << cpu_count;
        std::cout << (h_output == cpu_count ? " [PASS]" : " [FAIL]") << std::endl;

        cudaFree(d_input);
        cudaFree(d_output);
    }

    for (int i = 0; i < cases; i++) {
        delete[] input_cases[i];
    }
    delete[] input_cases;
    delete[] sizes;
    delete[] K_arr;

    return 0;
}