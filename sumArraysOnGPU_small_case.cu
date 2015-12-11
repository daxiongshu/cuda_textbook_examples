#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdio.h>

__global__ void checkIndex(void){
   printf("threadIdx:(%d,%d,%d) blockIdx:(%d,%d,%d) blockDim:(%d,%d,%d) "
        "gridDim: (%d,%d,%d)\n", threadIdx.x,threadIdx.y,threadIdx.z,
        blockIdx.x,blockIdx.y,blockIdx.z,blockDim.x,blockDim.y,
        blockDim.z,gridDim.x,gridDim.y,gridDim.z
   );

}

void sumArraysOnHost(float *A, float *B, float *C, const int N){
   for(int idx=0; idx<N;idx++){
      C[idx]=A[idx]+B[idx];
   }
   // no need to return C
}

void initialData(float *ip, int size){
   time_t t;
   srand((unsigned int) time(&t));
   for (int i=0; i<size; i++){
      ip[i] = (float) ( rand() & 0xFF )/10.0f;
   }
}

int main(int argc, char **argv){
   int nElem = 6;//1024;
   size_t nBytes = nElem * sizeof(float);

   /*******************CPU part********************/
   
   float *h_A, *h_B, *h_C;
   h_A = (float *)malloc(nBytes);
   h_B = (float *)malloc(nBytes);
   h_C = (float *)malloc(nBytes);

   initialData(h_A,nElem);
   initialData(h_B,nElem);

   sumArraysOnHost(h_A,h_B,h_C,nElem);

   /*******************CPU part********************/

   /*******************GPU part********************/
   dim3 block(3);
   dim3 grid((nElem+block.x-1)/block.x); // how the grid is calculated?

   printf("grid.x %d grid.y %d grid.z %d \n",grid.x,grid.y,grid.z);
   printf("block.x %d block.y %d block.z %d \n",block.x,block.y,block.z);

   // understand index
   checkIndex <<<grid,block>>>();
   cudaDeviceReset();

   float *d_A, *d_B, *d_C;
   cudaMalloc((float**)&d_A, nBytes);
   cudaMalloc((float**)&d_B, nBytes);
   cudaMalloc((float**)&d_C, nBytes);

   cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
   cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

   //cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);




   free(h_A);
   free(h_B);
   free(h_C);

   cudaFree(d_A);
   cudaFree(d_B);
   cudaFree(d_C);

   return(0);

}
