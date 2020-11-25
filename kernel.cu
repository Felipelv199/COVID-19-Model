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

__global__ void contagio_GPU(Agent* A, float r, int n, int *nuevos, float *random)
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
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
                //printf("aiX: %f aiY: %f ajX: %f ajY: %f distance: %f r:%f\n",ai.X, ai.Y,aj.X, aj.Y, distance, r);

                sigma += beta;
            }
        }
    }

    int alfa = 0;

    if (sigma >= 1)
    {
        alfa = 1;
    }
    float Pcond = ai.Pcon;
    //printf("random: %f Pcond: %f\n", random, Pcond);
    if (random[idx] <= Pcond)
    {
        ai.S = alfa;
        A[idx] = ai;
        if(alfa == 1)
        {
            //printf("%d entro\n", idx);
            nuevos[idx]++;
        }
        
        
        
    }
}

__global__ void movilidad_GPU(Agent* A, float pq, float lMax,
float *random1, float *random2,float *random3,float *random4,float *random5,float *random6)
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    Agent ai = A[idx];
    int sd = ai.S;

    if (sd == -2)
    {
        return;
    }

    float pSmod = ai.Psmo;
    int delta = 0;

    if (random1[idx] <= pSmod)
    {
        delta = 1;
    }

    float p = pq;
    float q = pq;
    float xd = ai.X;
    float yd = ai.Y;

    float X_2 = p*random2[idx]*(1-delta);
    float X = xd + (((2*random3[idx]-1)*lMax)*delta) + X_2;
    
    float Y_2 = q*random4[idx]*(1-delta);
    float Y =  yd + (((2*random5[idx]-1)*lMax)*delta) + Y_2;
    
    //printf("X: %f Y: %f\n", X, Y);

    int gamma = 0;
    float pMovd = ai.Pmov;

    if (random6[idx] <= pMovd)
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

__global__ void contagioExterno_GPU(Agent* A, int* R, float* random)
{
    
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
   
    R[idx] = 0;
    Agent ai = A[idx];
    int sd = ai.S;

    if(sd == 0)
    {

        int sd1 = sd;
        float pExtd = ai.Pext;
        
        //printf("id: %d %f random %fPext\n", idx, random, pExtd);

        if( random[idx] <= pExtd)
        {

            sd1 = 1;
            
        }
        
        ai.S = sd1;
        A[idx] = ai;
        if(A[idx].S == 1)
        {
            //printf("%d entro\n", idx);
            R[idx]++;
        }
    }

    
}

__global__ void tiempoIncSinCurRec_GPU(Agent* A, int *R )
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    R[idx] = 0;
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
            R[idx]++;
            
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

__global__ void casosFatales_GPU(Agent* A, int* R, float *random)
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    R[idx] = 0;
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
   
    float pFatd = ai.Pfat;

    if (random[idx] <= pFatd)
    {
        if (rho > 0)
        {
            sd1 = -2;
            R[idx]++;
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

__host__ float rangeRandom(int min, int max)
{
    return (rand() % 10001 / 10000.0) ;
}

__host__ ResultsDays newDay(int movements, Agent *host_agents, Simulacion *host_simulacion)
{

    ResultsDays results;

    Agent* dev_agents;
    int * devcXDia;
    int * devrecupXDia;
    int * devfatXDia;

    float* devrandContagios;

    float* devRandMov1;
    float* devRandMov2;
    float* devRandMov3;
    float* devRandMov4;
    float* devRandMov5;
    float* devRandMov6;
    
    float* devrandContagiosExt;
    float* devrandFat;

    int n = host_simulacion->N;

    float r = host_simulacion->R;
    float PQ = host_simulacion->PQ;
    float lmax = host_simulacion->lmax;

    int *cXdia;
    cXdia = (int*)malloc(n*sizeof(int));
    int *recupXdia;
    recupXdia = (int*)malloc(n*sizeof(int));
    int *fatXdia;
    fatXdia = (int*)malloc(n*sizeof(int));

    float* randomContagios;
    randomContagios = (float*)malloc(n*sizeof(float));

    float* randomMov1;
    randomMov1 = (float*)malloc(n*sizeof(float));
    float* randomMov2;
    randomMov2 = (float*)malloc(n*sizeof(float));
    float* randomMov3;
    randomMov3 = (float*)malloc(n*sizeof(float));
    float* randomMov4;
    randomMov4 = (float*)malloc(n*sizeof(float));
    float* randomMov5;
    randomMov5 = (float*)malloc(n*sizeof(float));
    float* randomMov6;
    randomMov6 = (float*)malloc(n*sizeof(float));

    float* randomContagiosExt;
    randomContagiosExt = (float*)malloc(n*sizeof(float));
    float* randomFat;
    randomFat = (float*)malloc(n*sizeof(float));

    dim3 block(THREADS_N);
    dim3 grid(BLOCKS_N);

    cudaMalloc((void**)&dev_agents, n*sizeof(Agent));
    check_CUDA_error("Error en cudaMalloc dev_agents");
    cudaMalloc((void**)&devcXDia, n*sizeof(int));
    check_CUDA_error("Error en cudaMalloc devDia");
    cudaMalloc((void**)&devrecupXDia, n*sizeof(int));
    check_CUDA_error("Error en cudaMalloc devrecupXDia");
    cudaMalloc((void**)&devfatXDia, n*sizeof(int));
    check_CUDA_error("Error en cudaMalloc devfatXDia");
    
    cudaMalloc((void**)&devrandContagios, n*sizeof(float));
    check_CUDA_error("Error en cudaMalloc devrandContagios");

    cudaMalloc((void**)&devRandMov1, n*sizeof(float));
    check_CUDA_error("Error en cudaMalloc devRandMov1");
    cudaMalloc((void**)&devRandMov2, n*sizeof(float));
    check_CUDA_error("Error en cudaMalloc devRandMov2");
    cudaMalloc((void**)&devRandMov3, n*sizeof(float));
    check_CUDA_error("Error en cudaMalloc devRandMov3");
    cudaMalloc((void**)&devRandMov4, n*sizeof(float));
    check_CUDA_error("Error en cudaMalloc devRandMov4");
    cudaMalloc((void**)&devRandMov5, n*sizeof(float));
    check_CUDA_error("Error en cudaMalloc devRandMov5");
    cudaMalloc((void**)&devRandMov6, n*sizeof(float));
    check_CUDA_error("Error en cudaMalloc devRandMov6");

    cudaMalloc((void**)&devrandContagiosExt, n*sizeof(float));
    check_CUDA_error("Error en cudaMalloc devrandContagios");
    cudaMalloc((void**)&devrandFat, n*sizeof(float));
    check_CUDA_error("Error en cudaMalloc devrandContagios"); 

    for (int j = 0; j < movements; j++)
    {
        cudaMemcpy(dev_agents, host_agents, n*sizeof(Agent), cudaMemcpyHostToDevice);
        check_CUDA_error("Error en cudaMalloc host_agents-->dev_agents");
        cudaMemcpy(devcXDia, cXdia, n*sizeof(float), cudaMemcpyHostToDevice);
        check_CUDA_error("Error en cudaMalloc cXdia-->devcXDia");

        for(int i = 0; i < n; i++)
        {
            randomContagios[i]  = rangeRandom(0, 1);
            randomMov1[i] = rangeRandom(0, 1);
            randomMov2[i] = rangeRandom(0, 1);
            randomMov3[i] = rangeRandom(0, 1);
            randomMov4[i] = rangeRandom(0, 1);
            randomMov5[i] = rangeRandom(0, 1);
            randomMov6[i] = rangeRandom(0, 1);
        }

        cudaMemcpy(devrandContagios, randomContagios, n*sizeof(float), cudaMemcpyHostToDevice);
        check_CUDA_error("Error en cudaMalloc randomContagios-->devrandContagios");

        cudaMemcpy(devRandMov1, randomMov1, n*sizeof(float), cudaMemcpyHostToDevice);
        check_CUDA_error("Error en cudaMalloc randomContagios-->devrandContagios");
        cudaMemcpy(devRandMov2, randomMov2, n*sizeof(float), cudaMemcpyHostToDevice);
        check_CUDA_error("Error en cudaMalloc randomContagios-->devrandContagios");
        cudaMemcpy(devRandMov3, randomMov3, n*sizeof(float), cudaMemcpyHostToDevice);
        check_CUDA_error("Error en cudaMalloc randomContagios-->devrandContagios");
        cudaMemcpy(devRandMov4, randomMov4, n*sizeof(float), cudaMemcpyHostToDevice);
        check_CUDA_error("Error en cudaMalloc randomContagios-->devrandContagios");
        cudaMemcpy(devRandMov5, randomMov5, n*sizeof(float), cudaMemcpyHostToDevice);
        check_CUDA_error("Error en cudaMalloc randomContagios-->devrandContagios");
        cudaMemcpy(devRandMov6, randomMov6, n*sizeof(float), cudaMemcpyHostToDevice);
        check_CUDA_error("Error en cudaMalloc randomContagios-->devrandContagios");

        contagio_GPU<<<grid, block>>>(dev_agents, r, n, devcXDia, devrandContagios);
        check_CUDA_error("Error en kernel contagio_GPU");
        cudaDeviceSynchronize();

        movilidad_GPU<<<grid, block>>>(dev_agents, PQ, lmax, devRandMov1, devRandMov2, devRandMov3, devRandMov4, devRandMov5, devRandMov6);
        check_CUDA_error("Error en kernel movilidad_GPU");
        cudaDeviceSynchronize();

        cudaMemcpy(cXdia, devcXDia, n*sizeof(int), cudaMemcpyDeviceToHost);
        check_CUDA_error("Error en cudaMalloc devcXDia-->cXDia");
        for(int j =0; j < n; j++)
        {
            results.c+= cXdia[j];
            //printf("casos:%d\n", cXdia[j]);
        }
        cudaMemcpy(host_agents, dev_agents, n*sizeof(Agent), cudaMemcpyDeviceToHost);
        check_CUDA_error("Error en cudaMalloc devAgents-->host_agents");

    }
    cudaMemcpy(dev_agents, host_agents, n*sizeof(Agent), cudaMemcpyHostToDevice);
    check_CUDA_error("Error en cudaMalloc host_agents-->dev_agents");
    cudaMemcpy(devcXDia, cXdia, n*sizeof(int), cudaMemcpyHostToDevice);
    check_CUDA_error("Error en cudaMalloc cXdia-->devcXDia");

    cudaMemcpy(devrecupXDia, recupXdia, n*sizeof(int), cudaMemcpyHostToDevice);
    check_CUDA_error("Error en cudaMalloc recupXdia-->devrecupXDia");
    cudaMemcpy(devfatXDia, fatXdia, n*sizeof(int), cudaMemcpyHostToDevice);
    check_CUDA_error("Error en cudaMalloc fatXdia-->devfatXDia");

    for(int i = 0; i < n; i++)
    {
        randomContagiosExt[i]  = rangeRandom(0, 1);
        randomFat[i]  = rangeRandom(0, 1);
    }    
    
    cudaMemcpy(devrandContagiosExt, randomContagiosExt, n*sizeof(float), cudaMemcpyHostToDevice);
    check_CUDA_error("Error en cudaMalloc randomContagios-->devrandContagios");
    cudaMemcpy(devrandFat, randomFat, n*sizeof(float), cudaMemcpyHostToDevice);
    check_CUDA_error("Error en cudaMalloc randomContagios-->devrandContagios");

    contagioExterno_GPU<<<grid,block>>>(dev_agents, devcXDia, devrandContagiosExt);
    check_CUDA_error("Error en kernel contagioExterno_GPU");
    cudaDeviceSynchronize();
    cudaMemcpy(cXdia, devcXDia, n*sizeof(int), cudaMemcpyDeviceToHost);
    check_CUDA_error("Error en cudaMalloc devcXDia-->cXDia");

    cudaMemcpy(host_agents, dev_agents, n*sizeof(Agent), cudaMemcpyDeviceToHost);
    check_CUDA_error("Error en cudaMalloc devAgents-->host_agents");
    
    tiempoIncSinCurRec_GPU<<<grid,block>>>(dev_agents, devrecupXDia);
    check_CUDA_error("Error en kernel tiempoIncSinCurRec_GPU");
    cudaDeviceSynchronize();

    casosFatales_GPU<<<grid,block>>>(dev_agents, devfatXDia, devrandFat);
    check_CUDA_error("Error en kernel casosFatales_GPU");
    cudaDeviceSynchronize();

    cudaMemcpy(cXdia, devcXDia, n*sizeof(int), cudaMemcpyDeviceToHost);
    check_CUDA_error("Error en cudaMalloc devcXDia-->cXDia");
    cudaMemcpy(recupXdia, devrecupXDia, n*sizeof(int), cudaMemcpyDeviceToHost);
    check_CUDA_error("Error en cudaMalloc devcXDia-->cXDia");
    cudaMemcpy(fatXdia, devfatXDia, n*sizeof(int), cudaMemcpyDeviceToHost);
    check_CUDA_error("Error en cudaMalloc devcXDia-->cXDia");
    cudaMemcpy(host_agents, dev_agents, n*sizeof(Agent), cudaMemcpyDeviceToHost);
    check_CUDA_error("Error en cudaMalloc devAgents-->host_agents");
    
    for(int j =0; j < n; j++)
    {
        results.c+= cXdia[j];
        results.cRecup += recupXdia[j];
        results.cFat += fatXdia[j];
    //printf("casos:%d\n", cXdia[j]);
    }
    

    cudaFree(dev_agents);

    cudaFree(devrandContagios);
    free(randomContagios);

    cudaFree(devRandMov1);
    free(randomMov1);
    cudaFree(devRandMov2);
    free(randomMov2);
    cudaFree(devRandMov3);
    free(randomMov3);
    cudaFree(devRandMov4);
    free(randomMov4);
    cudaFree(devRandMov5);
    free(randomMov5);
    cudaFree(devRandMov6);
    free(randomMov6);

    cudaFree(devrandContagiosExt);
    free(randomContagiosExt);
    cudaFree(devrandFat);
    free(randomFat);

    cudaFree(devcXDia);
    free(cXdia);

    cudaFree(devrecupXDia);
    free(recupXdia);

    cudaFree(devfatXDia);
    free(fatXdia);

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

        ResultsDays results = newDay(mM, agents, &simulacion);
        
        simulacion.results.cAcum += results.c;
        simulacion.results.cAcumAgRecup += results.cRecup;
        simulacion.results.cFatAcum += results.cFat;
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
        
        printf("    Numero de nuevos casos positivos por dia: %d\n", results.c);
        printf("    Numero de casos recuperados por dia: %d\n", results.cRecup);
        printf("    Numero de casos fatales por dia: %d\n", results.cFat);
        
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
