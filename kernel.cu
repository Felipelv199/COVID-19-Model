#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"
#include "kernel.h"
#include <iostream>
#include <time.h>
#include <math.h>
#include <stdlib.h>

__host__ void check_CUDA_error(const char* msj) {
	cudaError_t error;
	cudaDeviceSynchronize();
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("Error: %d %s (%s) \n", error, cudaGetErrorString(error), msj);
	}
}
// olv

__global__ void setup_kernel ( curandState* state, unsigned long seed )
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    curand_init ( seed, idx, 0, &state[idx] );
} 
__global__ void InitializeCUDA(int PQ, Agent *A, curandState* globalState) 
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    curandState localState = globalState[idx];
    Agent newAgent;
    newAgent.X = (int)(0 + curand_uniform( &localState) * ((PQ) - 0));
    newAgent.Y = (int)(0 + curand_uniform( &localState) * ((PQ) - 0));

    newAgent.Pcon = (2 + curand_uniform( &localState) * ((3) - 2)) / 100.0;
    newAgent.Pext = (2 + curand_uniform( &localState) * ((3) - 2)) / 100.0;
    newAgent.Pfat = (7 + curand_uniform( &localState) * ((70) - 7)) / 1000.0;
    newAgent.Pmov = (3 + curand_uniform( &localState) * ((5) - 3)) / 10.0;
    newAgent.Psmo = (7 + curand_uniform( &localState) * ((9) - 7)) / 10.0;
    newAgent.Tinc = 5 + curand_uniform( &localState) * ((6+1) - 5);

    A[idx] = newAgent;

    globalState[idx] = localState;
}
/*
void Initialize(Simulacion *S, Agent *A)
{
    curandState* devStates;

    Agent * devAgents;

    cudaMalloc((void**)&devStates, S->N*sizeof(curandState));
    cudaMalloc((void**)&devAgents, S->N*sizeof(Agent));

    dim3 block(1024); 
    dim3 grid(10); 

    setup_kernel<<<grid,block>>>(devStates, time(NULL));
    cudaDeviceSynchronize();
    generate<<<grid, block>>>(S->PQ, devAgents, devStates, 3, 2);
    cudaDeviceSynchronize();

    cudaMemcpy(A, devAgents, S->N*sizeof(Agent), cudaMemcpyDeviceToHost);
    
    for(int i=0;i<S->N;i++)
    {
        printf("X:%d Y:%d Pcon:%f Pext:%f Pfat:%f Pmov:%f Psmo:%f Tinc:%d\n", A[i].X, A[i].Y, A[i].Pcon, A[i].Pext, A[i].Pfat, A[i].Pmov, A[i].Psmo, A[i].Tinc);
    }
    
    cudaFree(devAgents);
    cudaFree(devStates);
}
*/

int rangeRandom(int min, int max)
{
    return min + rand() % ((max + 1) - min);
}

void inicializacion(Simulacion *S, Agent *A)
{
    int y = 0;
    int x = 0;

    for (int i = 0; i < S->N; i++)
    {
        y = rand() % S->PQ;
        x = rand() % S->PQ;
        Agent newAgent;
        newAgent.X = x;
        newAgent.Y = y;
        newAgent.Pcon = rangeRandom(2, 3) / 100.0;
        newAgent.Pext = rangeRandom(2, 3) / 100.0;
        newAgent.Pfat = rangeRandom(7, 70) / 1000.0;
        newAgent.Pmov = rangeRandom(3, 5) / 10.0;
        newAgent.Psmo = rangeRandom(7, 9) / 10.0;
        newAgent.Tinc = rangeRandom(5, 6);
        A[i] = newAgent;
    }
}

double distance(int x0, int y0, int x1, int y1)
{
    int x = x1 - x0;
    int y = y1 - y0;
    return sqrt((x * x) + (y * y));
}

__global__ void contagioCUDA(int n, int r, Agent *A, Results *R, int day, curandState* globalState)
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    
    curandState localState = globalState[idx];
    int x = A[idx].X;
    int y = A[idx].Y;
    int beta = 0;
    int sigma = 0;

    if (A[idx].S != 0)
    {
        return;
    }

    for (int j = 0; j < n; j++)
    {
        if (j != idx)
        {
            Agent aj = A[j];
            //printf("a: %d\n",j);
            if (aj.S != 2)
            {
                if (aj.S > 0)
                {
                    beta = 1;
                }
                else
                {
                    beta = 0;
                }

                double d = sqrt((float)((aj.X - x) * (aj.X - x)) + ((aj.Y - y) * (aj.Y - y))); 
                

                if (d <= (double)r)
                {
                    sigma += d * beta;
                }
            }
        }
        //printf("beta: %d\n", beta);
    }

    int alfa = 0;

    if (sigma >= 1)
    {
        alfa = 1;
    }

    float random = curand_uniform( &localState);

    if (random <= A[idx].Pcon)
    {

        A[idx].S = random * alfa;
        //printf("entro, %f %d %d\n", (rand() % 101) / 100.0, alfa, ai->S);
        if (A[idx].S == 1)
        {
            R->cAcum++;
            R->cXDia++;
            if (R->cAcum == 1)
            {
                R->cZero = day;
            }
            else if (n / 2 == R->cAcum)
            {
                R->c50per = day;
            }
            else if (n == R->cAcum)
            {
                R->c100per = day;
            }
        }
    }
}

void contagio(int n, int r, int x, int y, int i, int Pcon, Results *R, Agent *A, Agent *ai, int day)
{
    int beta = 0;
    int sigma = 0;

    if (ai->S != 0 )
    {
        return;
    }

    for (int j = 0; j < n; j++)
    {
        if (j != i)
        {
            Agent aj = A[j];
            if (aj.S != 2)
            {
                if (aj.S > 0)
                {
                    beta = 1;
                }
                else
                {
                    beta = 0;
                }

                double d = distance(x, y, aj.X, aj.Y);
                if (d <= r)
                {
                    sigma += d * beta;
                }
            }
        }
        //printf("beta: %d\n", beta);
    }

    int alfa = 0;

    if (sigma >= 1)
    {
        alfa = 1;
    }

    float random = (rand() % 101) / 100.0;

    if (random <= Pcon)
    {

        ai->S = random * alfa;
        //printf("entro, %f %d %d\n", (rand() % 101) / 100.0, alfa, ai->S);
        if (ai->S == 1)
        {
            R->cAcum++;
            R->cXDia++;
            if (R->cAcum == 1)
            {
                R->cZero = day;
            }
            else if (n / 2 == R->cAcum)
            {
                R->c50per = day;
            }
            else if (n == R->cAcum)
            {
                R->c100per = day;
            }
        }
    }
}

__global__ void movilidadCUDA(Simulacion S, Agent *A, curandState* globalState)
{

    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    curandState localState = globalState[idx];
    if (A[idx].S == -2)
    {
        return;
    }
    
    int delta = 0;

    if ((curand_uniform( &localState) * (100)) / 100.0 <= A[idx].Psmo)
    {
        delta = 1;
    }

    int X = 0;
    int Y = 0;
    int p = S.PQ;
    int q = S.PQ;
    int xd = A[idx].X;
    int yd = A[idx].Y;

    X = ((xd + (((2 * (curand_uniform( &localState))) - 1) * S.lmax)) * delta) + (p * (curand_uniform( &localState)) * (1 - delta));
    Y = ((yd + (((2 * (curand_uniform( &localState))) - 1) * S.lmax)) * delta) + (q * (curand_uniform( &localState)) * (1 - delta));

    int gamma = 0;

    if (((curand_uniform( &localState) * ((100))) / 100.0) <= A[idx].Pmov)
    {
        gamma = 1;
    }

    int yd1 = Y;
    int xd1 = X;

    if (yd1 > S.PQ)
        yd1 = S.PQ - 1;
    else if (yd1 < 0)
    {
        yd1 = 0;
    }

    if (xd1 > S.PQ)
        xd1 = S.PQ - 1;
    else if (xd1 < 0)
    {
        xd1 = 0;
    }

    if (gamma != 0)
    {
        A[idx].X = xd1;
        A[idx].Y = yd1;
    }
    else
    {
        A[idx].X = xd;
        A[idx].Y = yd;
    }
    
    
}

void movilidad(Simulacion *S, Agent *ai)
{
    if (ai->S == -2)
    {
        return;
    }
    int delta = 0;

    if (rand() % 2 <= ai->Psmo)
    {
        delta = 1;
    }

    int X = 0;
    int Y = 0;
    int p = S->PQ;
    int q = S->PQ;
    int xd = ai->X;
    int yd = ai->Y;

    X = ((xd + (((2 * ((rand() % 101) / 100.0)) - 100) * S->lmax)) * delta) + (p * ((rand() % 101) / 100.0) * (1 - delta));
    Y = ((yd + (((2 * ((rand() % 101) / 100.0)) - 100) * S->lmax)) * delta) + (q * ((rand() % 101) / 100.0) * (1 - delta));

    int gamma = 0;

    if (((rand() % 101) / 100.0) <= ai->Pmov)
    {
        gamma = 1;
    }

    int yd1 = Y;
    int xd1 = X;

    if (yd1 > S->PQ)
        yd1 = S->PQ - 1;
    else if (yd1 < 0)
    {
        yd1 = 0;
    }

    if (xd1 > S->PQ)
        xd1 = S->PQ - 1;
    else if (xd1 < 0)
    {
        xd1 = 0;
    }

    if (gamma != 0)
    {
        ai->X = xd1;
        ai->Y = yd1;
    }
    else
    {
        ai->X = xd;
        ai->Y = yd;
    }
}

__global__ void contagioExternoCUDA(Agent *A, Results *R, int n, int day, curandState* globalState)
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    curandState localState = globalState[idx];

    int sd = A[idx].S;
    if (sd == 2 || sd == -2)
    {
        return;
    }
    int epsilon = 1;

    if (sd != 0)
    {
        epsilon = 0;
    }

    int sd1 = sd;
    float pext = A[idx].Pext;
    float random = curand_uniform( &localState);
    printf("random: %f Pext:%f\n" , random, pext);
    if ( (random<= pext) * epsilon > 0)
    {
        
        sd1 = 1;
        R->cAcum++;
        R->cXDia++;
        if (R->cAcum == 1)
        {
            R->cZero = day;
        }
        else if (n / 2 == R->cAcum)
        {
            R->c50per = day;
        }
        else if (n == R->cAcum)
        {
            R->c100per = day;
        }
    }

    A[idx].S = sd1;
}

void contagioExterno(Agent *ai, Results *R, int n, int day)
{
    int sd = ai->S;
    if (sd == 2 || sd == -2)
    {
        return;
    }
    int epsilon = 1;

    if (sd != 0)
    {
        epsilon = 0;
    }

    int sd1 = sd;
    int pext = ai->Pext;

    if ((((rand() % 101) / 100.0) <= pext) * epsilon > 0)
    {
        sd1 = 1;
        R->cAcum++;
        R->cXDia++;
        if (R->cAcum == 1)
        {
            R->cZero = day;
        }
        else if (n / 2 == R->cAcum)
        {
            R->c50per = day;
        }
        else if (n == R->cAcum)
        {
            R->c100per = day;
        }
    }

    ai->S = sd1;
}

void tiempoIncSinCurRec(Agent *ai, Results *R, int n, int day)
{
    int Sd = ai->S;
    int Trecd = ai->Trec;
    int Trecd1 = Trecd;

    if (Sd == 2 || Sd == -2)
    {
        return;
    }

    if (Sd < 0)
    {
        Trecd1 -= 1;
        if (Trecd1 == 0)
        {
            ai->S = 2;
            R->cAcumAgRecup++;
            R->cRecupXDia++;

            if (R->cAcumAgRecup == 1)
            {
                R->recupPrim = day;
            }
            else if (n / 2 == R->cAcumAgRecup)
            {
                R->recup50per = day;
            }
            else if (n == R->cAcumAgRecup)
            {
                R->recup100per = day;
            }

            return;
        }
    }

    int Tincd = ai->Tinc;
    int Sd1 = -1;

    if (Tincd > 0)
    {
        Sd1 = Sd;
    }

    int Tincd1 = Tincd;

    if (Sd > 0)
    {
        Tincd1 -= 1;
    }

    ai->Trec = Trecd1;
    ai->S = Sd1;
    ai->Tinc = Tincd1;
}

void casosFatales(Agent *ai, Results *R, int n, int day)
{
    int rho = 0;
    int sd = ai->S;

    if (sd == 2 || sd == -2)
    {
        return;
    }

    if (sd < 0)
    {
        rho = 1;
    }

    int sd1 = sd;
    float random = (rand() % 101) / 100.0;

    if (random <= ai->Pfat)
    {
        if (random * rho > 0)
        {
            sd1 = -2;
            R->cFatAcum++;
            R->cFatXDia++;

            if (R->cFatAcum == 1)
            {
                R->cFatPrim = day;
            }
            else if (n / 2 == R->cFatAcum)
            {
                R->cFat50per = day;
            }
            else if (n == R->cFatAcum)
            {
                R->cFat100per = day;
            }
        }
    }

    ai->S = sd1;
}

int main()
{
    const int N = 32;
    Simulacion sim;
    sim.N = N;
    Agent agents[N];
    Results results;
    //inicializacion(&sim, agents);
    //Initialize(&sim, agents); ... asi estaba cuando trono haha
    curandState* devStates;

    Agent * devAgents;

    cudaMalloc((void**)&devStates, sim.N*sizeof(curandState));
    cudaMalloc((void**)&devAgents, sim.N*sizeof(Agent));

    dim3 block(N); 
    dim3 grid(1); 

    setup_kernel<<<grid,block>>>(devStates, time(NULL));
    cudaDeviceSynchronize();
    check_CUDA_error("Error en cudaError setupKernel");
    InitializeCUDA<<<grid, block>>>(sim.PQ, devAgents, devStates);
    cudaDeviceSynchronize();
    check_CUDA_error("Error en cudaError Initialize");


    /*
    for(int i=0;i<sim.N;i++)
    {
        printf("X:%d Y:%d Pcon:%f Pext:%f Pfat:%f Pmov:%f Psmo:%f Tinc:%d\n", agents[i].X, agents[i].Y, agents[i].Pcon, agents[i].Pext, agents[i].Pfat, agents[i].Pmov, agents[i].Psmo, agents[i].Tinc);
    }
    ahi esta
    */
    
    
    int dM = sim.dmax;
    int mM = sim.Mmax;
    srand(time(NULL));
    for (int i = 1; i <= dM; i++)
    {
 


        printf("Dia %d\n", i);
        for (int j = 0; j < mM; j++)
        {

            contagioCUDA<<<grid, block>>>(sim.N, sim.R, devAgents, &results, i, devStates);
            cudaDeviceSynchronize();
            check_CUDA_error("Error en cudaError contagio");
            
            
            movilidadCUDA<<<grid, block>>>(sim, devAgents, devStates);
            cudaDeviceSynchronize();
            check_CUDA_error("Error en cudaError movilidad");

        }

        contagioExternoCUDA<<<grid,block>>>(devAgents, &results, sim.N, i, devStates);
        cudaDeviceSynchronize();
        //antes me salia si lo que era, esta pasando un error al copiar again al host creo
        check_CUDA_error("Error en cudaError externo");

        results.cXDia = 0;
        results.cFatXDia = 0;
        results.cRecupXDia = 0;
        
    }
    cudaMemcpy(agents, devAgents, sim.N*sizeof(Agent), cudaMemcpyDeviceToHost);

    cudaFree(devAgents);
    cudaFree(devStates);
    
    return 0;
    
    
}
