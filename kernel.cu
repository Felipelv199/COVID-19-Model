#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"

#include "kernel.h"

#define THREADS_N 1024
#define BLOCKS_N 10

using namespace std;

__global__ void setup_kernel (curandState* state, unsigned long seed)
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    curand_init ( seed, idx, 0, &state[idx] );
} 

__global__ void inicializacion_GPU(Agent* A, curandState* globalState, int PQ) 
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    curandState localState = globalState[idx];
    Agent newAgent;
    newAgent.X = (int)(0 + curand_uniform(&localState) * (PQ - 0));
    newAgent.Y = (int)(0 + curand_uniform(&localState) * (PQ - 0));
    newAgent.Pcon = (2 + curand_uniform(&localState) * (3 - 2)) / 100.0;
    newAgent.Pext = (2 + curand_uniform(&localState) * (3 - 2)) / 100.0;
    newAgent.Pfat = (7 + curand_uniform(&localState) * (70 - 7)) / 1000.0;
    newAgent.Pmov = (3 + curand_uniform(&localState) * (5 - 3)) / 10.0;
    newAgent.Psmo = (7 + curand_uniform(&localState) * (9 - 7)) / 10.0;
    newAgent.Tinc = 5 + curand_uniform(&localState) * (6 - 5);
    A[idx] = newAgent;
    globalState[idx] = localState;
}

__global__ void contagio_GPU(Agent* A, curandState* globalState, int r, int n)
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    curandState localState = globalState[idx];
    Agent ai = A[idx];
    int sd = ai.S;

    if(sd != 0){
        return;
    }

    int beta = 0;
    int sigma = 0;

    for(int i=0; i<n; i++)
    {
        if(i != idx)
        {
            Agent aj = A[i];
            if (aj.S > 0)
            {
                beta = 1;
            }
            else
            {
                beta = 0;
            }
            
            int x = ai.X;
            int y = ai.Y;
            double distance = sqrt((float)(x * x) + (float)(y * y));

            if (distance <= r)
            {
                sigma += distance * beta;
            }
        }
    }

    int alfa = 0;

    if (sigma >= 1)
    {
        alfa = 1;
    }

    float random = (curand_uniform(&localState) * 100) / 100.0;
    int Pcond = ai.Pcon;

    if (random <= Pcond)
    {
        ai.S = random * alfa;
        A[idx] = ai;
    }
}

__host__ void check_CUDA_error(const char* msj) {
	cudaError_t error;
	cudaDeviceSynchronize();
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("Error: %d %s (%s)\n", error, cudaGetErrorString(error), msj);
	}
}

__host__ void inicializacion(int n, int pq, Agent *host_agents){
    Agent* dev_agents;
    curandState* dev_states;

    cudaMalloc((void**)&dev_states, n*sizeof(curandState));
    check_CUDA_error("Error en cudaMalloc dev_states");
    cudaMalloc((void**)&dev_agents, n*sizeof(Agent));
    check_CUDA_error("Error en cudaMalloc dev_agents");

    dim3 block(THREADS_N);
    dim3 grid(BLOCKS_N);

    setup_kernel<<<grid,block>>>(dev_states, time(NULL));
    check_CUDA_error("Error en kernel setup_kernel");
    cudaDeviceSynchronize();
    inicializacion_GPU<<<grid, block>>>(dev_agents, dev_states, pq);
    check_CUDA_error("Error en kernel dev_agents");
    cudaDeviceSynchronize();

    cudaMemcpy(host_agents, dev_agents, n*sizeof(Agent), cudaMemcpyDeviceToHost);
    check_CUDA_error("Error en cudaMemcpy dev_agents-->host_agents");

    cudaFree(dev_agents);
    cudaFree(dev_states);
}

__host__ void contagio(Agent *host_agents, Simulacion *host_simulacion, int n){
    Agent* dev_agents;
    curandState* dev_states;

    cudaMalloc((void**)&dev_agents, n*sizeof(Agent));
    check_CUDA_error("Error en cudaMalloc dev_agents");
    cudaMalloc((void**)&dev_states, n*sizeof(curandState));
    check_CUDA_error("Error en cudaMalloc dev_states");

    cudaMemcpy(dev_agents, host_agents, n*sizeof(Agent), cudaMemcpyHostToDevice);
    check_CUDA_error("Error en cudaMalloc host_agents-->dev_agents");

    dim3 block(THREADS_N);
    dim3 grid(BLOCKS_N);

    setup_kernel<<<grid,block>>>(dev_states, time(NULL));
    check_CUDA_error("Error en kernel setup_kernel");
    cudaDeviceSynchronize();
    contagio_GPU<<<grid, block>>>(dev_agents, dev_states, host_simulacion->R, host_simulacion->N);
    check_CUDA_error("Error en kernel contagio_GPU");
    cudaDeviceSynchronize();

    cudaMemcpy(host_agents, dev_agents, n*sizeof(Agent), cudaMemcpyDeviceToHost);
    check_CUDA_error("Error en cudaMalloc host_agents-->dev_agents");

    cudaFree(dev_agents);
    cudaFree(dev_states);
}

__host__ void printAgent(Agent ai)
{
    printf("X: %d, Y: %d, S: %d, Pcon: %f, Pext: %f, Pfat: %f, Pmov: %f, Psmo: %f, Tinc: %d\n", ai.X, ai.Y, ai.S, ai.Pcon, ai.Pext, ai.Pfat, ai.Pmov, ai.Psmo, ai.Tinc);
}

int main(){
    const int N = 10240;
    const int DAYS = 100;
    Simulacion simulacion;
    simulacion.N = N;
    simulacion.dmax = DAYS;
    int mM = simulacion.Mmax;
    Agent* agents;

    agents = (Agent*)malloc(N*sizeof(Agent));

    inicializacion(N, simulacion.PQ, agents);

    for(int i=0; i<DAYS; i++)
    {
        for (int j = 0; j < mM; j++)
        {   
            contagio(agents, &simulacion, N);
        }
    } 

    free(agents);
    return 0;
}