#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

__device__ unsigned int fnv1a_hash(int input) {
    const unsigned int FNV_PRIME = 16777619;
    const unsigned int OFFSET_BASIS = 2166136261;
    
    unsigned int hash = OFFSET_BASIS;
    
    for (int byte_pos = 0; byte_pos < 4; byte_pos++) {
        unsigned char byte = (input >> (byte_pos * 8)) & 0xFF;
        hash = (hash ^ byte) * FNV_PRIME;
    }
    
    return hash;
}

__global__ void rainbow_table_kernel(const int* input, unsigned int* output, int N) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (index < N) {
        output[index] = fnv1a_hash(input[index]);
    }
}

// input, output are device pointers
extern "C" void solve(const int* input, unsigned int* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    rainbow_table_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}

// CPU version for verification
unsigned int fnv1a_hash_cpu(int input) {
    const unsigned int FNV_PRIME = 16777619;
    const unsigned int OFFSET_BASIS = 2166136261;
    
    unsigned int hash = OFFSET_BASIS;
    
    for (int byte_pos = 0; byte_pos < 4; byte_pos++) {
        unsigned char byte = (input >> (byte_pos * 8)) & 0xFF;
        hash = (hash ^ byte) * FNV_PRIME;
    }
    
    return hash;
}

void generate_test_arrays(int** input_cases, unsigned int** output_cases, int* sizes, int cases) {
    for (int i = 0; i < cases; i++) {
        int N = rand() % 1000 + 100;
        sizes[i] = N;
        
        input_cases[i] = new int[N];
        output_cases[i] = new unsigned int[N];
        
        for (int j = 0; j < N; j++) {
            input_cases[i][j] = rand();
        }
    }
}

int main(int argc, char* argv[]) {
    int cases = 10;
    int** input_cases = new int*[cases];
    unsigned int** output_cases = new unsigned int*[cases];
    int* sizes = new int[cases];

    generate_test_arrays(input_cases, output_cases, sizes, cases);

    for (int i = 0; i < cases; i++) {
        int N = sizes[i];

        int *d_input;
        unsigned int *d_output;
        cudaMalloc(&d_input, N * sizeof(int));
        cudaMalloc(&d_output, N * sizeof(unsigned int));

        cudaMemcpy(d_input, input_cases[i], N * sizeof(int), cudaMemcpyHostToDevice);

        solve(d_input, d_output, N);

        cudaMemcpy(output_cases[i], d_output, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);

        // Verify first few hashes
        bool pass = true;
        for (int j = 0; j < std::min(5, N); j++) {
            unsigned int expected = fnv1a_hash_cpu(input_cases[i][j]);
            if (output_cases[i][j] != expected) {
                pass = false;
                break;
            }
        }

        std::cout << "Case " << i << " (N=" << N << "): ";
        std::cout << "Sample: input=" << input_cases[i][0] << " -> hash=" << output_cases[i][0];
        std::cout << (pass ? " [PASS]" : " [FAIL]") << std::endl;

        cudaFree(d_input);
        cudaFree(d_output);
    }

    for (int i = 0; i < cases; i++) {
        delete[] input_cases[i];
        delete[] output_cases[i];
    }
    delete[] input_cases;
    delete[] output_cases;
    delete[] sizes;

    return 0;
}