#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdio.h>
#include "common.h"



__global__ void checkIndex(void){
   printf("threadIdx:(%d,%d,%d) blockIdx:(%d,%d,%d) blockDim:(%d,%d,%d) "
        "gridDim: (%d,%d,%d)\n", threadIdx.x,threadIdx.y,threadIdx.z,
        blockIdx.x,blockIdx.y,blockIdx.z,blockDim.x,blockDim.y,
        blockDim.z,gridDim.x,gridDim.y,gridDim.z
   );

}

__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N)
{  
   // why no boundry check?
   int i=threadIdx.x+blockIdx.x*blockDim.x;
   if (i<N)
      C[i]=A[i]+B[i];

}
int checkResult(float *hostRef, float *gpuRef, const int N){
   
   double epsilon= 1.0E-8;
   bool match = 1;
   int error=-1;
   for(int i=0;i<N;i++){
      if (abs(hostRef[i]-gpuRef[i])>epsilon){
         match=0;
         error=i;
         printf("Arrays don't match!\n");
         printf("host %5.2f gpu %5.2f at current %d\n",hostRef[i],gpuRef[i],i);
         break;
      }
   }
   if (match) printf("Arrays match. \n\n");
   return error;
}

void sumArraysOnHost(float *A, float *B, float *C, const int N){
   // CPU version of the kernel
   for(int idx=0; idx<N;idx++){
      C[idx]=A[idx]+B[idx];
   }
   // no need to return C
}

void initialData(float *ip, int size){
   time_t t;
   srand((unsigned int) time(&t)); 
   // initialize random number with seed of time
   // time(&t) <==> t=time(NULL); assign the current time to t


   for (int i=0; i<size; i++){
      ip[i] = (float) ( rand() & 0xFF )/10.0f; // rand() returns a random number between 0 and RAND_MAX
      //0xFF is 255
   }
}

int main(int argc, char **argv){
   printf("%s Starting ...\n",argv[0]);

   int nElem = 32*1e3;//1024;
   printf("Vector size is %d\n",nElem);

   // allocate memory
   size_t nBytes = nElem * sizeof(float);
   //printf("%d,%d\n",0xFF,RAND_MAX);
   /*******************CPU part********************/
   
   float *h_A, *h_B, *h_C, *gpuRef;
   // h_C = h_A + h_B
   // d_C copy to gpuRef
   h_A = (float *)malloc(nBytes);
   h_B = (float *)malloc(nBytes);
   h_C = (float *)malloc(nBytes);
   gpuRef = (float *)malloc(nBytes);

   initialData(h_A,nElem);
   initialData(h_B,nElem);
   memset(h_C,0,nBytes);
   memset(gpuRef,-1,nBytes);

   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   float cpumilliseconds = 0;

   double iStart,iElaps;
   iStart=cpuSecond();
   cudaEventRecord(start);

   sumArraysOnHost(h_A,h_B,h_C,nElem);

   cudaEventRecord(stop);
   cudaEventElapsedTime(&cpumilliseconds, start, stop);

   iElaps=cpuSecond()-iStart;
   printf("\nCPU timer\n");
   printf("sumArraysOnHost Time Elapsed %f\n",iElaps);
   /*******************CPU part********************/

   


   /*******************GPU part********************/
   int dev=0;
   cudaSetDevice(dev); // use the device_id=0 GPU;

   dim3 block(1024);
   dim3 grid((nElem+block.x-1)/block.x); // how the grid is calculated?

   // understand index
   //checkIndex <<<grid,block>>>();
   //cudaDeviceReset();

   float *d_A, *d_B, *d_C;
   cudaMalloc((float**)&d_A, nBytes);
   cudaMalloc((float**)&d_B, nBytes);
   cudaMalloc((float**)&d_C, nBytes);

   CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
   CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));

   float beforesync,gpumilliseconds,gpumillisecondsbeforesync;
   iStart=cpuSecond();
   cudaEventRecord(start);
   sumArraysOnGPU<<<grid,block>>>(d_A, d_B, d_C,nElem);
   cudaEventRecord(stop);
   cudaEventSynchronize(stop); // the later cudaDeviceSynchronize doesn't achieve the same 
   cudaEventElapsedTime(&gpumillisecondsbeforesync, start, stop);

   beforesync=cpuSecond()-iStart;
   CHECK(cudaDeviceSynchronize());
   iElaps=cpuSecond()-iStart;
   cudaEventRecord(stop);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&gpumilliseconds, start, stop);   

   printf("sumArraysOnGPU Time Elapsed %f before sync %f\n\n",iElaps,beforesync);

   printf("\nGPU timer\n");
   printf("sumArraysOnHost Time Elapsed %f\n",cpumilliseconds);
   printf("sumArraysOnGPU Time Elapsed %f before sync %f\n\n",gpumilliseconds,gpumillisecondsbeforesync);

   printf("Kernel configuration: (%d,%d,%d),(%d,%d,%d)\n",grid.x,grid.y,grid.z,block.x,block.y,block.z);
   cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

   int error=checkResult(h_C,gpuRef,nElem);
   //printf("%f+%f=%f\n",h_A[error],h_B[error],h_A[error]+h_B[error]);




   free(h_A);
   free(h_B);
   free(h_C);
   free(gpuRef);

   cudaFree(d_A);
   cudaFree(d_B);
   cudaFree(d_C);
   return(0);
}
