#include <iostream>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include "kernel.h"

using namespace std;

int rangeRandom(int min, int max)
{
    return min + rand() % ((max + 1) - min);
}

void printAgent(Agent ai)
{
    printf("X: %d, Y: %d, S: %d, Pcon: %f, Pext: %f, Pfat: %f, Pmov: %f, Psmo: %f, Tinc: %d\n", ai.X, ai.Y, ai.S, ai.Pcon, ai.Pext, ai.Pfat, ai.Pmov, ai.Psmo, ai.Tinc);
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

void contagio(int n, int r, int x, int y, int i, float Pcon, Results *R, Agent *A, Agent *ai, int day)
{
    int beta = 0;
    int sigma = 0;

    if (ai->S != 0)
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
        }
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
    const int N = 1024;
    const int DAYS = 30;
    Simulacion sim;
    sim.N = N;
    sim.dmax = DAYS;
    Agent agents[N];
    ResultsDays resultsDays[DAYS];
    Results results;
    inicializacion(&sim, agents);

    int dM = sim.dmax;
    int mM = sim.Mmax;
    srand(time(NULL));
    for (int i = 1; i <= dM; i++)
    {
        for (int j = 0; j < mM; j++)
        {
            for (int k = 0; k < N; k++)
            {
                Agent agent = agents[k];
                contagio(sim.N, sim.R, agent.X, agent.Y, k, agent.Pcon, &results, agents, &agent, i);
                movilidad(&sim, &agent);
                agents[k] = agent;
            }
        }
        for (int j = 0; j < N; j++)
        {
            Agent agent = agents[j];
            contagioExterno(&agent, &results, N, i);
            tiempoIncSinCurRec(&agent, &results, N, i);
            casosFatales(&agent, &results, N, i);
            agents[j] = agent;
        }
        ResultsDays newDay;
        newDay.c = results.cXDia;
        newDay.cFat = results.cFatXDia;
        newDay.cRecup = results.cRecupXDia;
        resultsDays[i] = newDay;

        results.cXDia = 0;
        results.cFatXDia = 0;
        results.cRecupXDia = 0;
    }

    int counterC = 0;
    int counterRecup = 0;
    int counterFat = 0;

    for (int i = 1; i <= DAYS; i++)
    {
        ResultsDays resultsDay = resultsDays[i];

        counterC += resultsDay.c;
        if (counterC > 0 && counterC < results.cAcum / 2 && results.cZero == 0)
        {
            results.cZero = i;
        }
        else if (counterC >= results.cAcum / 2 && counterC < results.cAcum && results.c50per == 0)
        {
            results.c50per = i;
        }
        else if (counterC == results.cAcum && results.c100per == 0)
        {
            results.c100per = i;
        }

        counterRecup += resultsDay.cRecup;
        if (counterRecup > 0 && counterRecup < results.cAcumAgRecup / 2 && results.recupPrim == 0)
        {
            results.recupPrim = i;
        }
        else if (counterRecup >= results.cAcumAgRecup / 2 && counterRecup < results.cAcumAgRecup && results.recup50per == 0)
        {
            results.recup50per = i;
        }
        else if (counterRecup == results.cAcumAgRecup && results.recup100per == 0)
        {
            results.recup100per = i;
        }

        counterFat += resultsDay.cFat;
        if (counterFat > 0 && counterFat < results.cFatAcum / 2 && results.cFatPrim == 0)
        {
            results.cFatPrim = i;
        }
        else if (counterFat >= results.cFatAcum / 2 && counterFat < results.cFatAcum && results.cFat50per == 0)
        {
            results.cFat50per = i;
        }
        else if (counterFat == results.cFatAcum && results.cFat100per == 0)
        {
            results.cFat100per = i;
        }

        printf("Dia %d\n", i);
        printf("    Numero de nuevos casos positivos por dia: %d\n", resultsDay.c);
        printf("    Numero de casos recuperados por dia: %d\n", resultsDay.cRecup);
        printf("    Numero de casos fatales por dia: %d\n", resultsDay.cFat);
        printf("------------------------------------------------------\n");
    }

    printf("Resultados Finales\n");
    printf("    Numero de casos acumulados de agentes contagiados: %d\n", results.cAcum);
    printf("    Numero de casos acumulados de agentes recuperados: %d\n", results.cAcumAgRecup);
    printf("    Numero de casos fatales acumulados: %d\n", results.cFatAcum);
    printf("    Dia en que se contagio el primer agente: %d\n", results.cZero);
    printf("    Dia en que se contagio el 50%% de los agentes contagiados: %d\n", results.c50per);
    printf("    Dia en que se contagio el 100%% de los agentes contagiados: %d\n", results.c100per);
    printf("    Dia en que se recupero el primer agente: %d\n", results.recupPrim);
    printf("    Dia en que se recupero el 50%% de los agentes recuperados: %d\n", results.recup50per);
    printf("    Dia en que se recupero el 100%% de los agentes recuperados: %d\n", results.recup100per);
    printf("    Dia en que ocurrio el primer caso fatal: %d\n", results.cFatPrim);
    printf("    Dia en que ocurrio el 50%% de los casos fatales: %d\n", results.cFat50per);
    printf("    Dia en que ocurrio el 100%% de los casos fatales: %d\n", results.cFat100per);
    return 0;
}
