#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
    int input_size, int kernel_size) {

int index = threadIdx.x + blockDim.x * blockIdx.x;

if (index < input_size - kernel_size + 1) {
float val = 0;
for (int i = 0; i < kernel_size; i++) {
val += input[index + i] * kernel[i];
}
output[index] = val;
}
//cache convolution ideally with cudaMemcpyToSymbol, I assume you can't do it here bc kernel_Size is not given and we would need
//to define the kernel as global constant

}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_size, kernel_size);
    cudaDeviceSynchronize();
}

void generate_test_arrays(float** input_cases, float** kernel_cases, float** output_cases, 
                          int* input_sizes, int* kernel_sizes, int cases) {
    for (int i = 0; i < cases; i++) {
        int input_size = rand() % 500 + 50;
        int kernel_size = rand() % 10 + 3;
        if (kernel_size > input_size) kernel_size = input_size;
        
        input_sizes[i] = input_size;
        kernel_sizes[i] = kernel_size;
        
        int output_size = input_size - kernel_size + 1;
        
        input_cases[i] = new float[input_size];
        kernel_cases[i] = new float[kernel_size];
        output_cases[i] = new float[output_size];
        
        for (int j = 0; j < input_size; j++) {
            input_cases[i][j] = static_cast<float>(rand() % 100) / 10.0f;
        }
        for (int j = 0; j < kernel_size; j++) {
            kernel_cases[i][j] = static_cast<float>(rand() % 10) / 10.0f;
        }
    }
}

int main(int argc, char* argv[]) {
    int cases = 10;
    float** input_cases = new float*[cases];
    float** kernel_cases = new float*[cases];
    float** output_cases = new float*[cases];
    int* input_sizes = new int[cases];
    int* kernel_sizes = new int[cases];

    generate_test_arrays(input_cases, kernel_cases, output_cases, input_sizes, kernel_sizes, cases);

    for (int i = 0; i < cases; i++) {
        int input_size = input_sizes[i];
        int kernel_size = kernel_sizes[i];
        int output_size = input_size - kernel_size + 1;

        float *d_input, *d_kernel, *d_output;
        cudaMalloc(&d_input, input_size * sizeof(float));
        cudaMalloc(&d_kernel, kernel_size * sizeof(float));
        cudaMalloc(&d_output, output_size * sizeof(float));

        cudaMemcpy(d_input, input_cases[i], input_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_kernel, kernel_cases[i], kernel_size * sizeof(float), cudaMemcpyHostToDevice);

        solve(d_input, d_kernel, d_output, input_size, kernel_size);

        cudaMemcpy(output_cases[i], d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

        std::cout << "Case " << i << " (input_size=" << input_size << ", kernel_size=" << kernel_size << "): ";
        std::cout << "First 5 outputs: ";
        for (int j = 0; j < 5 && j < output_size; j++) {
            std::cout << output_cases[i][j] << " ";
        }
        std::cout << std::endl;

        cudaFree(d_input);
        cudaFree(d_kernel);
        cudaFree(d_output);
    }

    for (int i = 0; i < cases; i++) {
        delete[] input_cases[i];
        delete[] kernel_cases[i];
        delete[] output_cases[i];
    }
    delete[] input_cases;
    delete[] kernel_cases;
    delete[] output_cases;
    delete[] input_sizes;
    delete[] kernel_sizes;

    return 0;
}