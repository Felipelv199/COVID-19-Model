#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"

#include "kernel.h"

using namespace std;

__global__ void setup_kernel ( curandState * state, unsigned long seed )
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

__host__ void check_CUDA_error(const char* msj) {
	cudaError_t error;
	cudaDeviceSynchronize();
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("Error: %d %s (%s)\n", error, cudaGetErrorString(error), msj);
	}
}

__host__ void inicializacion(int n, int pq, Agent *host_agents){
    curandState* dev_states;
    Agent* dev_agents;

    cudaMalloc((void**)&dev_states, n*sizeof(curandState));
    check_CUDA_error("Error en cudaMalloc dev_states");
    cudaMalloc((void**)&dev_agents, n*sizeof(Agent));
    check_CUDA_error("Error en cudaMalloc dev_agents");

    dim3 block(1024);
    dim3 grid(10);

    setup_kernel<<<grid,block>>>(dev_states, time(NULL));
    check_CUDA_error("Error en kernel setup_kernel");
    cudaDeviceSynchronize();
    inicializacion_GPU<<<grid, block>>>(dev_agents, dev_states, pq);
    check_CUDA_error("Error en kernel dev_agents");
    cudaDeviceSynchronize();

    cudaMemcpy(host_agents, dev_agents, n*sizeof(Agent), cudaMemcpyDeviceToHost);
    check_CUDA_error("Error en cudaMemcpy dev_agents-->host_agents");
    
    for(int i=0; i<n; i++){
        Agent ai = host_agents[n];
        printf("X:%d Y:%d Pcon:%f Pext:%f Pfat:%f Pmov:%f Psmo:%f Tinc:%d\n", ai.X, ai.Y, ai.Pcon, ai.Pext, ai.Pfat, ai.Pmov, ai.Psmo, ai.Tinc);
    } 

    cudaFree(dev_agents);
    cudaFree(dev_states);
}

int main(){
    const int N = 10240;
    const int DAYS = 100;
    Simulacion sim;
    sim.N = N;
    sim.dmax = DAYS;
    Agent* agents;

    agents = (Agent*)malloc(N*sizeof(Agent));

    inicializacion(N, sim.PQ, agents);

    free(agents);
    return 0;
}