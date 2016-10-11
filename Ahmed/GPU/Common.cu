#include "Common.h"

// Allocates an array with random float entries.
void RandomInit(uint *data, int n, uint lower_limit = 0, uint upper_limit = 1024)
{
    for (int i = 0; i < n; ++i)
    {
        data[i] = rand() % upper_limit + lower_limit;
    }
}

// warmup 函数，用于计时时 warmup GPU，实际是一个 vector 相加
__global__ void
vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
        C[i] = A[i] + B[i];
}
void warmup()
{
    int numElements = 1024;
    size_t size = numElements * sizeof(float);

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    float *d_A = NULL;
    cudaMalloc((void **)&d_A, size);

    float *d_B = NULL;
    cudaMalloc((void **)&d_B, size);

    float *d_C = NULL;
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 32;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

}

// This is just a linear search through the array, since the error_id's are not
// always ocurring consecutively
const char *getCudaDrvErrorString(CUresult error_id)
{
    int index = 0;

    while (sCudaDrvErrorString[index].error_id != error_id &&
           sCudaDrvErrorString[index].error_id != -1)
    {
        index++;
    }

    if (sCudaDrvErrorString[index].error_id == error_id)
        return (const char *)sCudaDrvErrorString[index].error_string;
    else
        return (const char *)"CUDA_ERROR not found!";
}

// These are the inline versions for all of the SDK helper functions
void __check_cuda_errors(cudaError err, const char *file, const int line)
{
    if (cudaSuccess != err)
    {
        fflush(stdout);
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",file, line, (int)err, cudaGetErrorString(err));
        
        cudaDeviceReset();
        exit(-1);
    }
}

void __getLastCudaError(const char *errorMessage, const char *file, const int line)
{
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
                file, line, errorMessage, (int)err, cudaGetErrorString(err));
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

void BubbleSort(uint* array,int start,int end){
    for(int i = 0;i < end - start;i++)
        for(int j = 0;j < end - start;j++){
            if(array[j] > array[j + 1])
                swap(array[j],array[j + 1]);
        }
}

void swap(uint* arr, int i, int j) {
  uint tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
}

void quicksort(uint* arr, int st, int end) {
  if(st == end) return;

  int i, sep = st;
  for(i = st + 1; i < end; i++) {
    if(arr[i] < arr[st]) swap(arr, ++sep, i);
  }

  swap(arr, st, sep);
  quicksort(arr, st, sep);
  quicksort(arr, sep + 1, end);
}