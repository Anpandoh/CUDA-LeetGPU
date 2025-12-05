#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int pixelIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (pixelIndex < width * height) {
        int idx = pixelIndex * 4;
        image[idx + 0] = 255 - image[idx + 0];
        image[idx + 1] = 255 - image[idx + 1];
        image[idx + 2] = 255 - image[idx + 2];
        
    }
}

// image is device pointer (RGBA format)
extern "C" void solve(unsigned char* image, int width, int height) {
    int totalPixels = width * height;
    int threadsPerBlock = 256;
    int blocksPerGrid = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}

void generate_test_images(unsigned char** image_cases, int* widths, int* heights, int cases) {
    for (int i = 0; i < cases; i++) {
        int width = rand() % 100 + 10;
        int height = rand() % 100 + 10;
        widths[i] = width;
        heights[i] = height;
        
        int size = width * height * 4; // RGBA
        image_cases[i] = new unsigned char[size];
        
        for (int j = 0; j < size; j++) {
            image_cases[i][j] = rand() % 256;
        }
    }
}

int main(int argc, char* argv[]) {
    int cases = 10;
    unsigned char** image_cases = new unsigned char*[cases];
    int* widths = new int[cases];
    int* heights = new int[cases];

    generate_test_images(image_cases, widths, heights, cases);

    for (int i = 0; i < cases; i++) {
        int width = widths[i];
        int height = heights[i];
        int size = width * height * 4;

        // Store original first pixel for verification
        unsigned char orig_r = image_cases[i][0];
        unsigned char orig_g = image_cases[i][1];
        unsigned char orig_b = image_cases[i][2];

        unsigned char* d_image;
        cudaMalloc(&d_image, size * sizeof(unsigned char));
        cudaMemcpy(d_image, image_cases[i], size * sizeof(unsigned char), cudaMemcpyHostToDevice);

        solve(d_image, width, height);

        cudaMemcpy(image_cases[i], d_image, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);

        std::cout << "Case " << i << " (" << width << "x" << height << "): ";
        std::cout << "First pixel [R,G,B]: [" << (int)orig_r << "," << (int)orig_g << "," << (int)orig_b << "] -> ";
        std::cout << "[" << (int)image_cases[i][0] << "," << (int)image_cases[i][1] << "," << (int)image_cases[i][2] << "]" << std::endl;

        cudaFree(d_image);
    }

    for (int i = 0; i < cases; i++) {
        delete[] image_cases[i];
    }
    delete[] image_cases;
    delete[] widths;
    delete[] heights;

    return 0;
}