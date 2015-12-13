#include <stdio.h>
#include "common.h"

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


__global__ void sumMatrixOnGPU(float *A, float *B, float *C, const int nx, const int ny){
   unsigned int ix_start = threadIdx.x*nx/blockDim.x;
   unsigned int ix_stop = threadIdx.x*nx/blockDim.x+nx/blockDim.x;
   unsigned int iy = blockIdx.y;
   unsigned int idx;
   for (unsigned int i=ix_start;i<ix_stop;i++){
       if (i<nx)
       {
          idx=iy*nx+i;
          C[idx]=A[idx]+B[idx];
       }
   }
}

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny){
   float *ia=A;
   float *ib=B;
   float *ic=C;
   
   for (int iy=0;iy<ny;iy++){
      for(int ix=0;ix<nx;ix++){
         int idx=iy*nx+ix;
         ic[idx]=ia[idx]+ib[idx];
      }
   }
}

int main(int argc, char **argv){
   printf("%s Starting ...\n",argv[0]);
   
   int dev=0;
   cudaDeviceProp deviceProp;
   CHECK(cudaGetDeviceProperties(&deviceProp, dev));
   printf("Using Device %d: %s\n",dev,deviceProp.name);
   CHECK(cudaSetDevice(dev));

   /*******************CPU part********************/
   int nx = 1<<14;
   int ny = 1<<14;
   int nxy = nx * ny;
   int nBytes = nxy*sizeof(float); // can also use size_t
   printf("Matrix size: (%d,%d)\n",nx,ny);

   
   float *h_A, *h_B, *cpuRef, *gpuRef;
   h_A = (float *) malloc(nBytes);
   h_B = (float *) malloc(nBytes);
   cpuRef = (float *) malloc(nBytes);
   gpuRef = (float *) malloc(nBytes);
   
      
   double iStart = cpuSecond();
   initialData (h_A,nxy);
   initialData (h_B,nxy);
   double iElapse = cpuSecond()-iStart;

   memset(cpuRef,0,nBytes);
   memset(gpuRef,0,nBytes);

   iStart = cpuSecond();
   sumMatrixOnHost(h_A, h_B,cpuRef,nx,ny);
   iElapse = cpuSecond()-iStart;
   printf("\nCPU timer sumMatrixOnHost %f\n",iElapse);
   /*******************CPU part********************/


   /*******************GPU part********************/
   float *d_A, *d_B, *d_C;
   cudaMalloc((void **)&d_A,nBytes);
   cudaMalloc((void **)&d_B,nBytes);
   cudaMalloc((void **)&d_C,nBytes);
   
   cudaMemcpy(d_A,h_A,nBytes,cudaMemcpyHostToDevice);
   cudaMemcpy(d_B,h_B,nBytes,cudaMemcpyHostToDevice);  

   int dimx=1024;
   int dimy=1;
   dim3 block(dimx,dimy);
   dim3 grid(1,ny);
   
   iStart = cpuSecond();
   sumMatrixOnGPU<<<grid,block>>>(d_A, d_B,d_C,nx,ny);   
   CHECK(cudaDeviceSynchronize()); // this cannot catch the error that kernel is not launched successfully
   iElapse = cpuSecond()-iStart;
   printf("\nCPU timer sumMatrixOnGPU <<<(%d,%d),(%d,%d)>>> %f\n\n",grid.x,grid.y,block.x,block.y,iElapse);
   cudaMemcpy(gpuRef,d_C,nBytes,cudaMemcpyDeviceToHost);   
   /*******************GPU part********************/

   checkResult(cpuRef,gpuRef,nxy);
   
   cudaFree(d_A);
   cudaFree(d_B);
   cudaFree(d_C);
   free(h_A);
   free(h_B);
   free(cpuRef);
   free(gpuRef); 
   cudaDeviceReset();
   return(0);
}
