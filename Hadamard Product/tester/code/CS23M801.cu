#include <chrono>
#include <fstream>
#include <iostream>
#include <cuda.h>

using std::cin;
using std::cout;


__global__
void CalculateHadamardProduct(long int* A, long int* B, int N) {

    // TODO: Write your kernel here
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int row=i/N;
    int col=i%N;
    if (i < N * N) {
        A[row*N+col] *= B[col*N+row];
    }
}

__global__
void FindWeightMatrix(long int* A, long int* B, int N) {

    // TODO: Write your kernel here
    int i=blockIdx.x*blockDim.x*blockDim.y + threadIdx.x*blockDim.y+threadIdx.y;
    if (i < N * N) {
        A[i] = (A[i] > B[i]) ? A[i] : B[i];
    }
}

__global__
void CalculateFinalMatrix(long int* A, long int* B, int N) {

    // TODO: Write your kernel here
    int i=blockIdx.x*gridDim.y*blockDim.x*blockDim.y + blockIdx.y*blockDim.x*blockDim.y+threadIdx.x*blockDim.y+threadIdx.y;
    int row = (i/(2*N))%N;
    int col = i%(N);
    if(i<4*N*N){
        B[i]*=A[(row*N+col)];
    }
}


int main(int argc, char** argv) {


    int N;
    cin >> N;
    long int* A = new long int[N * N];
    long int* B = new long int[N * N];
    long int* C = new long int[N * N];
    long int* D = new long int[2 * N * 2 * N];


    for (long int i = 0; i < N * N; i++) {
        cin >> A[i];
    }

    for (long int i = 0; i < N * N; i++) {
        cin >> B[i];
    }

    for (long int i = 0; i < N * N; i++) {
        cin >> C[i];
    }

    for (long int i = 0; i < 2 * N * 2 * N; i++) {
        cin >> D[i];
    }

    /**
     *
     * DO NOT CHANGE ANYTHING ABOVE THIS LINE
     *
    */

    long int* d_A;
    long int* d_B;
    long int* d_C;
    long int* d_D;

    cudaMalloc((void**)&d_A, N * N * sizeof(long int));
    cudaMalloc((void**)&d_B, N * N * sizeof(long int));
    cudaMalloc((void**)&d_C, N * N * sizeof(long int));
    cudaMalloc((void**)&d_D, 2 * N * 2 * N * sizeof(long int));

    cudaMemcpy(d_A, A, N * N * sizeof(long int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(long int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, N * N * sizeof(long int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_D, D, 4*N * N * sizeof(long int), cudaMemcpyHostToDevice);


    dim3 threadsPerBlock(1024, 1, 1);
    dim3 blocksPerGrid(ceil(N * N / 1024.0), 1, 1);


    auto start = std::chrono::high_resolution_clock::now();
    CalculateHadamardProduct<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, N);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end - start;


    threadsPerBlock = dim3(32, 32, 1);
    blocksPerGrid = dim3(ceil(N * N / 1024.0), 1, 1);


    start = std::chrono::high_resolution_clock::now();
    FindWeightMatrix<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C, N);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed2 = end - start;


    threadsPerBlock = dim3(32, 32, 1);
    blocksPerGrid = dim3(ceil(2 * N / 32.0), ceil(2 * N / 32.0), 1);


    start = std::chrono::high_resolution_clock::now();
    CalculateFinalMatrix<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_D, N);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed3 = end - start;

    // Make sure your final output from the device is stored in d_D.

    /**
     *
     * DO NOT CHANGE ANYTHING BELOW THIS LINE
     *
    */

    cudaMemcpy(D, d_D, 2 * N * 2 * N * sizeof(long int), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    std::ofstream file("cuda.out");
    if (file.is_open()) {
        for (long int i = 0; i < 2 * N; i++) {
            for (long int j = 0; j < 2 * N; j++) {
                file << D[i * 2 * N + j] << " ";
            }
            file << "\n";
        }
        file.close();
    } else {
        std::cout << "Unable to open file";
    }

    std::ofstream file2("cuda_timing.out");
    if(file2.is_open()) {
        file2 << elapsed1.count() << "\n";
        file2 << elapsed2.count() << "\n";
        file2 << elapsed3.count() << "\n";
        file2.close();
    } else {
        std::cout << "Unable to open file";
    }

    return 0;
}