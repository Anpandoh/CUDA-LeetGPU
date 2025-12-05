#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>

__global__ void swiglu_kernel(const float* input, float* output, int halfN) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < halfN) {
        output[index] = input[halfN + index] * (input[index] * (1/(1+exp(-input[index]))));
    }

}

// input is device pointer with size N (2*halfN), output has size halfN
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}

void generate_test_arrays(float** input_cases, float** output_cases, int* sizes, int cases) {
    for (int i = 0; i < cases; i++) {
        int halfN = rand() % 500 + 50;
        int N = halfN * 2;
        sizes[i] = N;
        
        input_cases[i] = new float[N];
        output_cases[i] = new float[halfN];
        
        for (int j = 0; j < N; j++) {
            input_cases[i][j] = static_cast<float>(rand() % 200 - 100) / 10.0f;
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
        int halfN = N / 2;

        float *d_input, *d_output;
        cudaMalloc(&d_input, N * sizeof(float));
        cudaMalloc(&d_output, halfN * sizeof(float));

        cudaMemcpy(d_input, input_cases[i], N * sizeof(float), cudaMemcpyHostToDevice);

        solve(d_input, d_output, N);

        cudaMemcpy(output_cases[i], d_output, halfN * sizeof(float), cudaMemcpyDeviceToHost);

        // Verify a few values (SwiGLU: gate * SiLU(x))
        // output[i] = input[halfN + i] * (input[i] * sigmoid(input[i]))
        bool pass = true;
        for (int j = 0; j < std::min(5, halfN); j++) {
            float x = input_cases[i][j];
            float gate = input_cases[i][halfN + j];
            float expected = gate * (x * (1.0f / (1.0f + expf(-x))));
            if (fabs(output_cases[i][j] - expected) > 1e-4) pass = false;
        }

        std::cout << "Case " << i << " (N=" << N << ", halfN=" << halfN << "): ";
        std::cout << "Sample: x=" << input_cases[i][0] << ", gate=" << input_cases[i][halfN];
        std::cout << " -> output=" << output_cases[i][0];
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