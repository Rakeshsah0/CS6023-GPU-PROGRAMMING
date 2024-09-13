
/**
*   CS6023: GPU Programming
*   Assignment 2
*
*   Please don't change any existing code in this file.
*
*   Please add necessary memory APIs for your implementation. Use cudaFree()
*   to free up memory as soon as you're done with an allocation.
*   This will ensure that you don't run out of memory while running
*   large test cases. Use the minimum required memory for your
*   implementation. DO NOT change the kernel configuration parameters.
*/

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <cuda.h>

using namespace std;

using std::cin;
using std::cout;

typedef long long ll;

__global__ void calculateConvolution(int m,int n,int k,long int *matrix,long int *filter,long int *result)
{
  extern __shared__ long int arr[];
  long int id = blockIdx.x * blockDim.x + threadIdx.x;
  int chunkSize=ceil((1.0*k*k)/n);
  for(int i=0;i<chunkSize;++i)
  {
    int sharedIdx=threadIdx.x+i*n;
    if(sharedIdx < k*k)
    {
       arr[sharedIdx]=filter[sharedIdx];
    }
  }
  __syncthreads();
  long int row=id/n;
  long int col=id%n;
  for(int i=0;i<k;++i)
  {
    for(int j=0;j<k;++j)
    {
      int n_row=row-k/2+i;
      int n_col=col-k/2+j;
      if(n_row>=0 && n_row<m && n_col>=0 && n_col<n)
        result[id]+=filter[i*k+j]*matrix[n_row*n+n_col];
    }
  } 
}


int main(int argc, char** argv) {

  
    int m,n,k;
    cin>>m>>n>>k;


    long int* h_mat = new long int[m * n];
    long int* h_filter = new long int[k * k];


    long int* h_ans = new long int[m * n];


    for (long int i = 0; i < m * n; i++) {
        cin>>h_mat[i];
    }

    for (long int i = 0; i < k * k; i++) {
        cin>>h_filter[i];
    }

    /**
     *
     * DO NOT CHANGE ANYTHING ABOVE THIS LINE
     *
    **/

    long int *gpumatrix;
    long int *gpufilter;
    long int *gpuresult;

    cudaMalloc(&gpumatrix,sizeof(long int)*m*n);
    cudaMalloc(&gpuresult,sizeof(long int)*m*n);
    cudaMalloc(&gpufilter,sizeof(long int)*k*k);
    cudaMemcpy(gpumatrix,h_mat,sizeof(long int)*m*n,cudaMemcpyHostToDevice);
    cudaMemcpy(gpufilter,h_filter,sizeof(long int)*k*k,cudaMemcpyHostToDevice);
    cudaMemset(gpuresult,0,sizeof(long int)*m*n);


    

    /****************************************************Start Here***********************************************************/

    auto start = std::chrono::high_resolution_clock::now();//keep it just before the kernel launch

    calculateConvolution<<<m,n,sizeof(long int)*k*k>>>(m,n,k,gpumatrix,gpufilter,gpuresult);
    auto end = std::chrono::high_resolution_clock::now();//keep it just after the kernel launch
    cudaMemcpy(h_ans,gpuresult,sizeof(long int)*m*n,cudaMemcpyDeviceToHost);


     cudaFree(gpumatrix);
     cudaFree(gpufilter);
     cudaFree(gpuresult);

     
    /*$$$$$$$$$$$$$$$$$$$$$$$$Make sure your final output from the device is stored in h_ans.$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/
    std::chrono::duration<double> elapsed1 = end - start;
    /**
     *
     * DO NOT CHANGE ANYTHING BELOW THIS LINE
     *
    */



    std::ofstream file("cuda.out");
    if (file.is_open()) {
        for (long int i = 0; i < m; i++) {
            for (long int j = 0; j < n; j++) {
                file << h_ans[i * n + j] << " ";
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
        file2.close();
    } else {
        std::cout << "Unable to open file";
    }

    return 0;
}
