#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    int colIndex = threadIdx.x + blockDim.x * blockIdx.x;
    int rowIndex = threadIdx.y + blockDim.y * blockIdx.y;

    if (colIndex < cols && rowIndex < rows) {
        output[colIndex * rows + rowIndex] = input[rowIndex * cols + colIndex];
    }

}

// input is rows x cols, output is cols x rows (device pointers)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}

void generate_test_arrays(float** input_cases, float** output_cases, int* rows_arr, int* cols_arr, int cases) {
    for (int i = 0; i < cases; i++) {
        int rows = rand() % 50 + 10;
        int cols = rand() % 50 + 10;
        rows_arr[i] = rows;
        cols_arr[i] = cols;
        
        input_cases[i] = new float[rows * cols];
        output_cases[i] = new float[cols * rows];
        
        for (int j = 0; j < rows * cols; j++) {
            input_cases[i][j] = static_cast<float>(rand() % 100) / 10.0f;
        }
    }
}

int main(int argc, char* argv[]) {
    int cases = 10;
    float** input_cases = new float*[cases];
    float** output_cases = new float*[cases];
    int* rows_arr = new int[cases];
    int* cols_arr = new int[cases];

    generate_test_arrays(input_cases, output_cases, rows_arr, cols_arr, cases);

    for (int i = 0; i < cases; i++) {
        int rows = rows_arr[i];
        int cols = cols_arr[i];

        float *d_input, *d_output;
        cudaMalloc(&d_input, rows * cols * sizeof(float));
        cudaMalloc(&d_output, cols * rows * sizeof(float));

        cudaMemcpy(d_input, input_cases[i], rows * cols * sizeof(float), cudaMemcpyHostToDevice);

        solve(d_input, d_output, rows, cols);

        cudaMemcpy(output_cases[i], d_output, cols * rows * sizeof(float), cudaMemcpyDeviceToHost);

        // Verify transpose: output[col][row] should equal input[row][col]
        bool pass = true;
        for (int r = 0; r < rows && pass; r++) {
            for (int c = 0; c < cols && pass; c++) {
                float in_val = input_cases[i][r * cols + c];
                float out_val = output_cases[i][c * rows + r];
                if (fabs(in_val - out_val) > 1e-5) {
                    pass = false;
                }
            }
        }

        std::cout << "Case " << i << " (" << rows << "x" << cols << " -> " << cols << "x" << rows << "): ";
        std::cout << "input[0][0]=" << input_cases[i][0] << " -> output[0][0]=" << output_cases[i][0];
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
    delete[] rows_arr;
    delete[] cols_arr;

    return 0;
}