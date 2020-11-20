#include <iostream>
#include <time.h>
#include <math.h>

using namespace std;

class Agent
{
public:
    float Pcon;
    float Pext;
    float Pfat;
    float Pmov;
    float Psmo;
    int Tinc;
    int Trec = 14;
    int S = 0;
    int X = 0;
    int Y = 0;

    Agent(int x, int y)
    {
        X = x;
        Y = y;
    }
};

class Simulacion
{
public:
    int N;
    int dmax = 15;
    int Mmax = 10;
    int lmax = 500;
    int R = 100;
    int PQ = 50000;

    Simulacion(int n)
    {
        N = n;
    }
};

class Results
{
public:
    int cAcum = 0;
    int cXDia = 0;
};

int rangeRandom(int min, int max)
{
    return min + rand() % ((max + 1) - min);
}

void inicializacion(Simulacion *S, Agent *A[])
{
    int y = 0;
    int x = 0;

    for (int i = 0; i < S->N; i++)
    {
        y = rand() % S->PQ;
        x = rand() % S->PQ;
        Agent *newAgent = new Agent(x, y);
        newAgent->Pcon = rangeRandom(2, 3) / 100.0;
        newAgent->Pext = rangeRandom(2, 3) / 100.0;
        newAgent->Pfat = rangeRandom(7, 70) / 1000.0;
        newAgent->Pmov = rangeRandom(3, 5) / 10.0;
        newAgent->Psmo = rangeRandom(7, 9) / 10.0;
        newAgent->Tinc = rangeRandom(5, 6);
        A[i] = newAgent;
    }
}

double distance(int x0, int y0, int x1, int y1)
{
    int x = x1 - x0;
    int y = y1 - y0;
    return sqrt((x * x) + (y * y));
}

void contagio(Simulacion *S, Agent *A[], Agent *ai, int i, Results *R)
{
    int beta = 0;
    int sigma = 0;
    int y = ai->Y;
    int x = ai->X;

    for (int j = 0; j < S->N; j++)
    {
        if (j != i)
        {
            Agent *aj = A[j];
            if (aj->S > 0)
            {
                beta = 1;
            }
            else
            {
                beta = 0;
            }

            double d = distance(x, y, aj->X, aj->Y);
            if (d <= S->R)
            {
                sigma += d * beta;
            }
        }
    }
    int alfa = 0;
    if (sigma >= 1)
    {
        alfa = 1;
    }

    int r = rand() % 2;

    if (r <= ai->Pcon)
    {
        ai->S = r * alfa;
    }
}

void movilidad(Simulacion *S, Agent *ai)
{

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

    X = ((xd + (((2 * (rand() % 2)) - 1) * S->lmax)) * delta) + (p * (rand() % 2) * (1 - delta));
    Y = ((yd + (((2 * (rand() % 2)) - 1) * S->lmax)) * delta) + (q * (rand() % 2) * (1 - delta));
    int gamma = 0;

    if (rand() % 2 <= ai->Pmov)
    {
        gamma = 1;
    }

    int yd1 = Y;

    if (yd1 > S->PQ)
        yd1 = S->PQ - 1;
    else if (yd1 < 0)
    {
        yd1 = 0;
    }

    int xd1 = X;

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
}

void contagioExterno(Agent *ai, Results *R)
{
    int sd = ai->S;
    int epsilon = 1;

    if (sd != 0)
    {
        epsilon = 0;
    }

    int sd1 = sd;
    int pext = ai->Pext;

    if (rand() % 2 <= pext)
    {
        if ((rand() % 2 <= pext) * epsilon > 0)
        {
            sd1 = 1;
            R->cAcum++;
            R->cXDia++;
        }
    }

    ai->S = sd1;
}

void tiempoIncSinCurRec(Agent *ai)
{
    int Sd = ai->S;
    int Trecd = ai->Trec;
    int Trecd1 = Trecd;

    if (Sd < 0)
    {
        Trecd1 -= 1;
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

void casosFatales(Agent *ai)
{
    int rho = 0;
    int sd = ai->S;

    if (sd < 0)
    {
        rho = 1;
    }

    int sd1 = sd;
    int r = rand() % 2;

    if (r <= ai->Pfat)
    {
        if (r * rho > 0)
        {
            sd1 = -2;
        }
    }

    ai->S = sd1;
}

int main()
{
    const int N = 1000;
    Simulacion *sim = new Simulacion(N);
    Agent *agents[N] = {0};
    Results *results = new Results();
    inicializacion(sim, agents);

    int dM = sim->dmax;
    int mM = sim->Mmax;

    for (int i = 0; i < dM; i++)
    {
        printf("Dia %d\n", i + 1);
        for (int j = 0; j < mM; j++)
        {
            for (int k = 0; k < N; k++)
            {
                Agent *agent = agents[k];
                contagio(sim, agents, agent, k, results);
                movilidad(sim, agent);
            }
        }
        printf("    Casos Nuevos por Dia: %d\n", results->cXDia);
        for (int i = 0; i < N; i++)
        {
            Agent *agent = agents[i];
            contagioExterno(agent, results);
            tiempoIncSinCurRec(agent);
            casosFatales(agent);
        }
        printf("    Casos Acumulados: %d\n", results->cAcum);
        printf("    Casos Nuevos por Dia: %d\n", results->cXDia);
        printf("--------------------------\n");
        results->cXDia = 0;
    }

    return 0;
}
