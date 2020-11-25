#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"
#include "structures.h"

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
    newAgent.X = (0 + curand_uniform(&localState) * (PQ - 0));
    newAgent.Y = (0 + curand_uniform(&localState) * (PQ - 0));
    newAgent.Pcon = (2 + curand_uniform(&localState) * (3 - 2)) / 100.0;
    newAgent.Pext = (2 + curand_uniform(&localState) * (3 - 2)) / 100.0;
    newAgent.Pfat = (7 + curand_uniform(&localState) * (70 - 7)) / 1000.0;
    newAgent.Pmov = (3 + curand_uniform(&localState) * (5 - 3)) / 10.0;
    newAgent.Psmo = (7 + curand_uniform(&localState) * (9 - 7)) / 10.0;
    newAgent.Tinc = 5 + curand_uniform(&localState) * (6 - 5);
    A[idx] = newAgent;
    globalState[idx] = localState;
}

__global__ void contagio_GPU(Agent* A, curandState* globalState, float r, int n, int *nuevos)
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    curandState localState = globalState[idx];
    nuevos[idx] = 0;
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
            
            if (aj.S == 1)
            {
                beta = 1;
            }
            else
            {
                beta = 0;
            }
            
            float x = aj.X - ai.X;
            float y = aj.Y - ai.Y;
            double distance = sqrt((float)((x * x) + (y * y)));
            
            if (distance <= r)
            {
                sigma += beta;
            }
        }
    }

    int alfa = 0;

    if (sigma >= 1)
    {
        alfa = 1;
    }

    float random = curand_uniform(&localState)/10;
    float Pcond = ai.Pcon;
    //printf("random: %f Pcond: %f\n", random, Pcond);
    if (random <= Pcond)
    {
        ai.S = alfa;
        A[idx] = ai;
        if(alfa == 1)
        {
            nuevos[idx]++;
        }
        
        
        
    }
}

__global__ void movilidad_GPU(Agent* A, curandState* globalState, float pq, float lMax)
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    curandState localState = globalState[idx];
    Agent ai = A[idx];
    int sd = ai.S;

    if (sd == -2)
    {
        return;
    }

    float pSmod = ai.Psmo;
    int delta = 0;

    if (curand_uniform(&localState) <= pSmod)
    {
        delta = 1;
    }

    float p = pq;
    float q = pq;
    float xd = ai.X;
    float yd = ai.Y;

    float X_2 = p*curand_uniform(&localState)*(1-delta);
    float X = ((xd + (2*curand_uniform(&localState)-1))*lMax)*delta + X_2;
    
    float Y_2 = q*curand_uniform(&localState)*(1-delta);
    float Y = ((yd + (2*curand_uniform(&localState)-1))*lMax)*delta + Y_2;
    
    int gamma = 0;
    float pMovd = ai.Pmov;

    if (curand_uniform(&localState) <= pMovd)
    {
        gamma = 1;
    }

    float xd1 = X;
    float yd1 = Y;

    if (xd1 > pq)
    {
        xd1 = pq - 1;
    }
    else if (xd1 < 0)
    {
        xd1 = 0;
    }

    if (yd1 > pq)
    {
        yd1 = pq - 1;
    }
    else if (yd1 < 0)
    {
        yd1 = 0;
    }

    if (gamma != 0)
    {
        ai.X = xd1;
        ai.Y = yd1;
    }
    else
    {
        ai.X = xd;
        ai.Y = yd;
    }

    A[idx] = ai;
}

__global__ void contagioExterno_GPU(Agent* A, curandState* globalState, int n, int* R)
{
    
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    R[idx] = 0;
    curandState localState = globalState[idx];
    Agent ai = A[idx];
    int sd = ai.S;

    if(sd == 0)
    {
        int epsilon = 1;

        if (sd != 0)
        {
            epsilon = 0;
        }

        int sd1 = sd;
        float pExtd = ai.Pext;
        float random = curand_uniform(&localState);
        
        if( random <= pExtd)
        {
            if (epsilon > 0)
            {
                //printf("%f random %fPext\n", random, pExtd);
                sd1 = 1;
            }
        }
        
        ai.S = sd1;
        A[idx] = ai;
        if(A[idx].S == 1)
        {
            R[idx]++;
        }
    }

    
}

__global__ void tiempoIncSinCurRec_GPU(Agent* A, curandState* globalState, Results* R)
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    Agent ai = A[idx];
    int sd = ai.S;

    if(sd == 2 || sd == -2)
    {
        return;
    }

    int trecd1 = ai.Trec;
    

    if(sd == -1)
    {
        trecd1 -= 1;
        if (trecd1 == 0)
        {
            ai.S = 2;
            R->cRecupXDia++;
            
        }
        ai.Trec = trecd1;
        A[idx] = ai;
    }
    else if(sd == 1)
    {
        int tincd = ai.Tinc;

        tincd -= 1;
        

        int sd1 = -1;

        if (tincd > 0)
        {
            sd1 = sd;
        }
        
        
        ai.S = sd1;
        ai.Tinc = tincd;
        A[idx] = ai;
    }

    
    

    
}

__global__ void casosFatales_GPU(Agent* A, curandState* globalState, Results* R)
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    curandState localState = globalState[idx];
    Agent ai = A[idx];
    int sd = ai.S;

    if (sd == 2 || sd == -2)
    {
        return;
    }

    int rho = 0;

    if (sd < 0)
    {
        rho = 1;
    }

    int sd1 = sd;
    float random = curand_uniform(&localState);
    float pFatd = ai.Pfat;

    if (random <= pFatd)
    {
        if (rho > 0)
        {
            sd1 = -2;
            R->cFatXDia++;
        }
    }

    ai.S = sd1;
    A[idx] = ai;
}

__host__ void check_CUDA_error(const char* msj) {
	cudaError_t error;
	cudaDeviceSynchronize();
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("Error: %d %s (%s)\n", error, cudaGetErrorString(error), msj);
	}
}

__host__ void printAgent(Agent ai)
{
    printf("X: %f, Y: %f, S: %d, Pcon: %f, Pext: %f, Pfat: %f, Pmov: %f, Psmo: %f, Tinc: %d, Trec: %d\n", ai.X, ai.Y, ai.S, ai.Pcon, ai.Pext, ai.Pfat, ai.Pmov, ai.Psmo, ai.Tinc, ai.Trec);
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



__host__ int contagio(Agent *host_agents, Simulacion *host_simulacion){
    Agent* dev_agents;
    curandState* dev_states;
    int n = host_simulacion->N;
    float r = host_simulacion->R;
    int *cXdia;

    cXdia = (int*)malloc(n*sizeof(int));

    int * devcXDia;
    int nuevos = 0;

    cudaMalloc((void**)&dev_agents, n*sizeof(Agent));
    check_CUDA_error("Error en cudaMalloc dev_agents");
    cudaMalloc((void**)&dev_states, n*sizeof(curandState));
    check_CUDA_error("Error en cudaMalloc dev_states");
    cudaMalloc((void**)&devcXDia, n*sizeof(int));
    check_CUDA_error("Error en cudaMalloc devDia");

    cudaMemcpy(dev_agents, host_agents, n*sizeof(Agent), cudaMemcpyHostToDevice);
    check_CUDA_error("Error en cudaMalloc host_agents-->dev_agents");
    cudaMemcpy(devcXDia, cXdia, n*sizeof(int), cudaMemcpyHostToDevice);
    check_CUDA_error("Error en cudaMalloc cXdia-->devcXDia");

    dim3 block(THREADS_N);
    dim3 grid(BLOCKS_N);

    setup_kernel<<<grid,block>>>(dev_states, time(NULL));
    cudaDeviceSynchronize();
    check_CUDA_error("Error en kernel setup_kernel");
    
    contagio_GPU<<<grid, block>>>(dev_agents, dev_states, r, n, devcXDia);
    cudaDeviceSynchronize();
    check_CUDA_error("Error en kernel contagio_GPU");
    
    cudaMemcpy(cXdia, devcXDia, n*sizeof(int), cudaMemcpyDeviceToHost);
    check_CUDA_error("Error en cudaMalloc devcXDia-->cXDia");
    cudaMemcpy(host_agents, dev_agents, n*sizeof(Agent), cudaMemcpyDeviceToHost);
    check_CUDA_error("Error en cudaMalloc host_agents-->dev_agents");

    for(int i =0; i < n; i++)
    {
        nuevos+= cXdia[i];
        //printAgent(host_agents[i]);
    }

    cudaFree(dev_agents);
    cudaFree(dev_states);
    cudaFree(devcXDia);
    free(cXdia);

    return nuevos;
}

__host__ void movilidad(Agent *host_agents, Simulacion *host_simulacion)
{
    Agent* dev_agents;
    curandState* dev_states;
    Results* devResults;
    int n = host_simulacion->N;
    float PQ = host_simulacion->PQ;
    float lmax = host_simulacion->lmax;

    cudaMalloc((void**)&dev_agents, n*sizeof(Agent));
    check_CUDA_error("Error en cudaMalloc dev_agents");
    cudaMalloc((void**)&dev_states, n*sizeof(curandState));
    check_CUDA_error("Error en cudaMalloc dev_states");
    cudaMalloc((void**)&devResults, sizeof(Results));
    check_CUDA_error("Error en cudaMalloc devResults");

    cudaMemcpy(dev_agents, host_agents, n*sizeof(Agent), cudaMemcpyHostToDevice);
    check_CUDA_error("Error en cudaMalloc host_agents-->dev_agents");
    cudaMemcpy(devResults, &host_simulacion->results, sizeof(Results), cudaMemcpyHostToDevice);
    check_CUDA_error("Error en cudaMalloc host_results-->devResults");

    dim3 block(THREADS_N);
    dim3 grid(BLOCKS_N);

    setup_kernel<<<grid,block>>>(dev_states, time(NULL));
    cudaDeviceSynchronize();

    check_CUDA_error("Error en kernel setup_kernel");
    movilidad_GPU<<<grid, block>>>(dev_agents, dev_states, PQ, lmax);
    cudaDeviceSynchronize();

    check_CUDA_error("Error en kernel movilidad_GPU");

    cudaMemcpy(host_agents, dev_agents, n*sizeof(Agent), cudaMemcpyDeviceToHost);
    check_CUDA_error("Error en cudaMalloc host_agents-->dev_agents");
    cudaMemcpy(&host_simulacion->results, devResults, sizeof(Results), cudaMemcpyDeviceToHost);
    check_CUDA_error("Error en cudaMalloc devResults-->host_results");

    cudaFree(dev_agents);
    cudaFree(dev_states);
    cudaFree(devResults);
}

__host__ int contagioExterno(Agent *host_agents, Simulacion *host_simulacion)
{
    Agent* dev_agents;
    curandState* dev_states;
    int n = host_simulacion->N;
    int *cXdia;

    cXdia = (int*)malloc(n*sizeof(int));

    int * devcXDia;
    int nuevos = 0;

    cudaMalloc((void**)&dev_agents, n*sizeof(Agent));
    check_CUDA_error("Error en cudaMalloc dev_agents");
    cudaMalloc((void**)&dev_states, n*sizeof(curandState));
    check_CUDA_error("Error en cudaMalloc dev_states");
    cudaMalloc((void**)&devcXDia, n*sizeof(int));
    check_CUDA_error("Error en cudaMalloc devDia");

    cudaMemcpy(dev_agents, host_agents, n*sizeof(Agent), cudaMemcpyHostToDevice);
    check_CUDA_error("Error en cudaMalloc host_agents-->dev_agents");
    cudaMemcpy(devcXDia, cXdia, n*sizeof(int), cudaMemcpyHostToDevice);
    check_CUDA_error("Error en cudaMalloc cXdia-->devcXDia");

    dim3 block(THREADS_N);
    dim3 grid(BLOCKS_N);

    setup_kernel<<<grid,block>>>(dev_states, time(NULL));
    check_CUDA_error("Error en kernel setup_kernel");
    cudaDeviceSynchronize();
    contagioExterno_GPU<<<grid,block>>>(dev_agents, dev_states, n, devcXDia);
    check_CUDA_error("Error en kernel contagioExterno_GPU");
    cudaDeviceSynchronize();

    cudaMemcpy(host_agents, dev_agents, n*sizeof(Agent), cudaMemcpyDeviceToHost);
    check_CUDA_error("Error en cudaMalloc dev_agents-->host_agents");
    cudaMemcpy(cXdia, devcXDia, n*sizeof(int), cudaMemcpyDeviceToHost);
    check_CUDA_error("Error en cudaMalloc devcXDia-->cXDia");

    for(int i =0; i < n; i++)
    {
        nuevos+= cXdia[i];
        //printAgent(host_agents[i]);
    }

    cudaFree(dev_agents);
    cudaFree(dev_states);
    cudaFree(devcXDia);
    free(cXdia);

    return nuevos;
}

__host__ void tiempoIncSinCurRec(Agent *host_agents, Simulacion *host_simulacion)
{
    Agent* dev_agents;
    curandState* dev_states;
    Results* devResults;
    int n = host_simulacion->N;

    cudaMalloc((void**)&dev_agents, n*sizeof(Agent));
    check_CUDA_error("Error en cudaMalloc dev_agents");
    cudaMalloc((void**)&dev_states, n*sizeof(curandState));
    check_CUDA_error("Error en cudaMalloc dev_states");
    cudaMalloc((void**)&devResults, sizeof(Results));
    check_CUDA_error("Error en cudaMalloc devResults");

    cudaMemcpy(dev_agents, host_agents, n*sizeof(Agent), cudaMemcpyHostToDevice);
    check_CUDA_error("Error en cudaMalloc host_agents-->dev_agents");
    cudaMemcpy(devResults, &host_simulacion->results, sizeof(Results), cudaMemcpyHostToDevice);
    check_CUDA_error("Error en cudaMalloc host_results-->devResults");

    dim3 block(THREADS_N);
    dim3 grid(BLOCKS_N);

    setup_kernel<<<grid,block>>>(dev_states, time(NULL));
    check_CUDA_error("Error en kernel setup_kernel");
    cudaDeviceSynchronize();
    tiempoIncSinCurRec_GPU<<<grid,block>>>(dev_agents, dev_states, devResults);
    check_CUDA_error("Error en kernel tiempoIncSinCurRec_GPU");
    cudaDeviceSynchronize();

    cudaMemcpy(host_agents, dev_agents, n*sizeof(Agent), cudaMemcpyDeviceToHost);
    check_CUDA_error("Error en cudaMalloc dev_agents-->host_agents");
    cudaMemcpy(&host_simulacion->results, devResults, sizeof(Results), cudaMemcpyDeviceToHost);
    check_CUDA_error("Error en cudaMalloc devResults-->host_results");

    cudaFree(dev_agents);
    cudaFree(dev_states);
    cudaFree(devResults);
}

__host__ void casosFatales(Agent *host_agents, Simulacion *host_simulacion)
{
    Agent* dev_agents;
    curandState* dev_states;
    Results* devResults;
    int n = host_simulacion->N;

    cudaMalloc((void**)&dev_agents, n*sizeof(Agent));
    check_CUDA_error("Error en cudaMalloc dev_agents");
    cudaMalloc((void**)&dev_states, n*sizeof(curandState));
    check_CUDA_error("Error en cudaMalloc dev_states");
    cudaMalloc((void**)&devResults, sizeof(Results));
    check_CUDA_error("Error en cudaMalloc devResults");

    cudaMemcpy(dev_agents, host_agents, n*sizeof(Agent), cudaMemcpyHostToDevice);
    check_CUDA_error("Error en cudaMalloc host_agents-->dev_agents");
    cudaMemcpy(devResults, &host_simulacion->results, sizeof(Results), cudaMemcpyHostToDevice);
    check_CUDA_error("Error en cudaMalloc host_results-->devResults");

    dim3 block(THREADS_N);
    dim3 grid(BLOCKS_N);

    setup_kernel<<<grid,block>>>(dev_states, time(NULL));
    check_CUDA_error("Error en kernel setup_kernel");
    cudaDeviceSynchronize();
    casosFatales_GPU<<<grid,block>>>(dev_agents, dev_states, devResults);
    check_CUDA_error("Error en kernel casosFatales_GPU");
    cudaDeviceSynchronize();

    cudaMemcpy(host_agents, dev_agents, n*sizeof(Agent), cudaMemcpyDeviceToHost);
    check_CUDA_error("Error en cudaMalloc dev_agents-->host_agents");
    cudaMemcpy(&host_simulacion->results, devResults, sizeof(Results), cudaMemcpyDeviceToHost);
    check_CUDA_error("Error en cudaMalloc devResults-->host_results");

    cudaFree(dev_agents);
    cudaFree(dev_states);
    cudaFree(devResults);
}

__host__ ResultsDays nuevoDia(int movimientos, Agent *host_agents, Simulacion *host_simulacion)
{

    ResultsDays results;

    Agent* dev_agents;
    curandState* dev_states;
    int * devcXDia;

    int n = host_simulacion->N;
    int nuevos = 0;

    float r = host_simulacion->R;
    float PQ = host_simulacion->PQ;
    float lmax = host_simulacion->lmax;

    int *cXdia;
    cXdia = (int*)malloc(n*sizeof(int));

    cudaMalloc((void**)&dev_agents, n*sizeof(Agent));
    check_CUDA_error("Error en cudaMalloc dev_agents");
    cudaMalloc((void**)&dev_states, n*sizeof(curandState));
    check_CUDA_error("Error en cudaMalloc dev_states");
    cudaMalloc((void**)&devcXDia, n*sizeof(int));
    check_CUDA_error("Error en cudaMalloc devDia");

    cudaMemcpy(dev_agents, host_agents, n*sizeof(Agent), cudaMemcpyHostToDevice);
    check_CUDA_error("Error en cudaMalloc host_agents-->dev_agents");
    cudaMemcpy(devcXDia, cXdia, n*sizeof(int), cudaMemcpyHostToDevice);
    check_CUDA_error("Error en cudaMalloc cXdia-->devcXDia");

    dim3 block(THREADS_N);
    dim3 grid(BLOCKS_N);

    setup_kernel<<<grid,block>>>(dev_states, time(NULL));
    cudaDeviceSynchronize();
    check_CUDA_error("Error en kernel setup_kernel");
    
    contagio_GPU<<<grid, block>>>(dev_agents, dev_states, r, n, devcXDia);
    cudaDeviceSynchronize();
    check_CUDA_error("Error en kernel contagio_GPU");

    movilidad_GPU<<<grid, block>>>(dev_agents, dev_states, PQ, lmax);
    cudaDeviceSynchronize();

    check_CUDA_error("Error en kernel movilidad_GPU");
    cudaMemcpy(host_agents, dev_agents, n*sizeof(Agent), cudaMemcpyDeviceToHost);
    check_CUDA_error("Error en cudaMalloc host_agents-->dev_agents");

    cudaMemcpy(cXdia, devcXDia, n*sizeof(int), cudaMemcpyDeviceToHost);
    check_CUDA_error("Error en cudaMalloc devcXDia-->cXDia");
    cudaMemcpy(host_agents, dev_agents, n*sizeof(Agent), cudaMemcpyDeviceToHost);
    check_CUDA_error("Error en cudaMalloc host_agents-->dev_agents");

    for(int i =0; i < n; i++)
    {
        nuevos+= cXdia[i];
        //printAgent(host_agents[i]);
    }

    cudaFree(dev_agents);
    cudaFree(dev_states);
    cudaFree(devcXDia);
    free(cXdia);


    return results;
}

int main(){
    const int N = THREADS_N * BLOCKS_N;
    const int DAYS = 31;
    Simulacion simulacion;
    simulacion.N = N;
    simulacion.dmax = DAYS;
    int mM = simulacion.Mmax;
    Agent* agents;

    agents = (Agent*)malloc(N*sizeof(Agent));

    inicializacion(N, simulacion.PQ, agents);

    for(int i=1; i<=DAYS; i++)
    {
        simulacion.results.cXDia = 0;
        simulacion.results.cRecupXDia = 0;
        simulacion.results.cFatXDia = 0;

        for (int j = 0; j < mM; j++)
        {   
            simulacion.results.cXDia += contagio(agents, &simulacion);
            movilidad(agents, &simulacion);
        }
        simulacion.results.cXDia += contagioExterno(agents, &simulacion);
        tiempoIncSinCurRec(agents, &simulacion);
        casosFatales(agents, &simulacion);
        
        
        simulacion.results.cAcum += simulacion.results.cXDia;
        simulacion.results.cAcumAgRecup += simulacion.results.cRecupXDia;
        simulacion.results.cFatAcum += simulacion.results.cFatXDia;
        if (simulacion.results.cAcum == simulacion.results.cXDia && simulacion.results.cAcum > 0)
        {
            simulacion.results.cZero = i;
        }
        //printf("%d es aqui, %d,%d\n", i, simulacion.results.cAcumAgRecup, simulacion.results.cRecupXDia);
        if (simulacion.results.cAcumAgRecup == simulacion.results.cRecupXDia && simulacion.results.cAcumAgRecup > 0)
        {
            
            simulacion.results.recupPrim = i;
        }
        if (N / 2 == simulacion.results.cAcumAgRecup)
        {
            simulacion.results.recup50per = i;
        }
        if (N == simulacion.results.cAcumAgRecup)
        {
            simulacion.results.recup100per = i;
        }

        if (simulacion.results.cFatAcum == simulacion.results.cFatXDia && simulacion.results.cFatAcum > 0)
        {
            //printf("%d es aqui tambien\n", i);
            simulacion.results.cFatPrim = i;
        }
        if (N / 2 == simulacion.results.cFatAcum)
        {
            simulacion.results.cFat50per = i;
        }
        else if (N == simulacion.results.cFatAcum)
        {
            simulacion.results.cFat100per = i;
        }
        
        printf("Dia %d\n", i);
        printf("    Numero de nuevos casos positivos por dia: %d\n", simulacion.results.cXDia);
        printf("    Numero de casos recuperados por dia: %d\n", simulacion.results.cRecupXDia);
        printf("    Numero de casos fatales por dia: %d\n", simulacion.results.cFatXDia);
        
        printf("------------------------\n");
        
        

        

    } 

    printf("Resultados Finales\n");
    printf("    Numero de casos acumulados de agentes contagiados: %d\n", simulacion.results.cAcum);
    printf("    Numero de casos acumulados de agentes recuperados: %d\n", simulacion.results.cAcumAgRecup);
    printf("    Numero de casos fatales acumulados: %d\n", simulacion.results.cFatAcum);
    printf("    Dia en que se contagio el primer agente: %d\n", simulacion.results.cZero);
    printf("    Dia en que se contagio el 50%% de los agentes contagiados: %d\n", simulacion.results.c50per);
    printf("    Dia en que se contagio el 100%% de los agentes contagiados: %d\n", simulacion.results.c100per);
    printf("    Dia en que se recupero el primer agente: %d\n", simulacion.results.recupPrim);
    printf("    Dia en que se recupero el 50%% de los agentes recuperados: %d\n", simulacion.results.recup50per);
    printf("    Dia en que se recupero el 100%% de los agentes recuperados: %d\n", simulacion.results.recup100per);
    printf("    Dia en que ocurrio el primer caso fatal: %d\n", simulacion.results.cFatPrim);
    printf("    Dia en que ocurrio el 50%% de los casos fatales: %d\n", simulacion.results.cFat50per);
    printf("    Dia en que ocurrio el 100%% de los casos fatales: %d\n", simulacion.results.cFat100per);

    free(agents);
    
    return 0;
}
