#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>

__global__ void reverse_array(float* input, float* output, int N) {

    int index = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (index < N) {
        output[N - index - 1] = input[index];
    }
}

// input, output are device pointers
extern "C" void solve(float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}

void generate_test_arrays(float** input_cases, float** output_cases, int* sizes, int cases) {
    for (int i = 0; i < cases; i++) {
        int N = rand() % 1000 + 100;
        sizes[i] = N;
        
        input_cases[i] = new float[N];
        output_cases[i] = new float[N];
        
        for (int j = 0; j < N; j++) {
            input_cases[i][j] = static_cast<float>(rand() % 100);
        }
    }
}

int main(int argc, char* argv[]) {
    int cases = 10;
    float** input_cases = new float*[cases];
    float** output_cases = new float*[cases];
    int* sizes = new int[cases];

    generate_test_arrays(input_cases, output_cases, sizes, cases);

    for (int i = 0; i < cases; i++) {
        int N = sizes[i];

        float *d_input, *d_output;
        cudaMalloc(&d_input, N * sizeof(float));
        cudaMalloc(&d_output, N * sizeof(float));

        cudaMemcpy(d_input, input_cases[i], N * sizeof(float), cudaMemcpyHostToDevice);

        solve(d_input, d_output, N);

        cudaMemcpy(output_cases[i], d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

        // Verify reversal
        bool pass = true;
        for (int j = 0; j < N; j++) {
            if (fabs(input_cases[i][j] - output_cases[i][N - j - 1]) > 1e-5) {
                pass = false;
                break;
            }
        }

        std::cout << "Case " << i << " (N=" << N << "): ";
        std::cout << "First=" << input_cases[i][0] << " -> Last=" << output_cases[i][N-1];
        std::cout << ", Last=" << input_cases[i][N-1] << " -> First=" << output_cases[i][0];
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