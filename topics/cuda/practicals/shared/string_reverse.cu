#include <cstdlib>
#include <cstdio>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include "util.hpp"

// TODO : implement a kernel that reverses a string of length n in place
__global__ void reverse_string_kernel(char* str, int n) {
    auto i = threadIdx.x;
    if (i * 2 < n) {
        auto tmp = str[i];
        str[i] = str[n - i - 1];
        str[n - i - 1] = tmp;
    }
}

__global__ void reverse_string2(char* str, int n) {
    auto i = threadIdx.x;
    if (i < n) {
        char buffer = str[n - i - 1];
        __syncthreads();
        str[i] = buffer;
    }
}

__host__ void reverse_string(char* str, int n) {
    reverse_string_kernel <<<1, n / 2 >>> (str, n);
}

int main(int argc, char** argv) {
    // check that the user has passed a string to reverse
    if(argc<2) {
        std::cout << "useage : ./string_reverse \"string to reverse\"\n" << std::endl;
        exit(0);
    }

    // determine the length of the string, and copy in to buffer
    auto n = strlen(argv[1]);
    auto string = malloc_managed<char>(n+1);
    std::copy(argv[1], argv[1]+n, string);
    string[n] = 0; // add null terminator

    std::cout << "string to reverse:\n" << string << "\n";

    // TODO : call the string reverse function
    //reverse_string(string, n);
    reverse_string2<<<1, n>>>(string, n);

    // print reversed string
    cudaDeviceSynchronize();
    std::cout << "reversed string:\n" << string << "\n";

    // free memory
    cudaFree(string);

    return 0;
}

