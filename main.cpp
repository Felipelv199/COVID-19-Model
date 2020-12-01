#include <iostream>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include "structures.h"

using namespace std;

float rangeRandom(int min, int max)
{
    return min + rand() % ((max + 1) - min);
}

void printAgent(Agent ai)
{
    printf("X: %f, Y: %f, S: %d, Pcon: %f, Pext: %f, Pfat: %f, Pmov: %f, Psmo: %f, Tinc: %d\n", ai.X, ai.Y, ai.S, ai.Pcon, ai.Pext, ai.Pfat, ai.Pmov, ai.Psmo, ai.Tinc);
}

void inicializacion(Simulacion *S, Agent *A)
{
    float y = 0;
    float x = 0;

    for (int i = 0; i < S->N; i++)
    {
        y = rand() % ((int)S->PQ + 1);
        x = rand() % ((int)S->PQ + 1);
        Agent newAgent;
        newAgent.X = x;
        newAgent.Y = y;
        newAgent.Pcon = rangeRandom(2000, 3000) / 100000.0;
        newAgent.Pext = rangeRandom(2000, 3000) / 100000.0;
        newAgent.Pfat = rangeRandom(70000, 7000) / 10000000.0;
        newAgent.Pmov = rangeRandom(3000, 5000) / 10000.0;
        newAgent.Psmo = rangeRandom(7000, 9000) / 10000.0;
        newAgent.Tinc = rangeRandom(5, 6);
        A[i] = newAgent;
    }
}

double distance(float x, float y)
{
    return sqrt((x * x) + (y * y));
}

void contagio(int n, float r, float x, float y, int i, float Pcon, Results *R, Agent *A, Agent *ai, int day)
{
    int sd = ai->S;

    if (sd != 0)
    {
        return;
    }

    int beta = 0;
    int sigma = 0;

    for (int j = 0; j < n; j++)
    {
        if (j != i)
        {
            Agent aj = A[j];
            if (aj.S == 1)
            {
                beta = 1;
            }
            else
            {
                beta = 0;
            }
            double d = distance(aj.X - x, aj.Y - y);
            if (d <= r)
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

    float random = rand() % 1001 / 1000.0;

    if (random <= Pcon)
    {

        ai->S = alfa;
        if (ai->S == 1)
        {
            R->cAcum++;
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
    float random = rand() % 1001 / 1000.0;
    if (random <= ai->Psmo)
    {
        delta = 1;
    }

    float X = 0;
    float Y = 0;
    float pq = S->PQ;
    float xd = ai->X;
    float yd = ai->Y;

    X = ((xd + (((2 * (rand() % 1001 / 1000.0)) - 1) * S->lmax)) * delta) + (pq * (rand() % 1001 / 1000.0) * (1 - delta));
    Y = ((yd + (((2 * (rand() % 1001 / 1000.0)) - 1) * S->lmax)) * delta) + (pq * (rand() % 1001 / 1000.0) * (1 - delta));

    int gamma = 0;
    random = rand() % 1001 / 1000.0;

    if (random <= ai->Pmov)
    {
        gamma = 1;
    }

    float yd1 = Y;
    float xd1 = X;

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

void contagioExterno(Agent *ai, Results *R)
{
    int sd = ai->S;
    if (sd == 0)
    {
        int sd1 = sd;
        float pext = ai->Pext;
        float random = rand() % 1001 / 1000.0;
        if (random <= pext)
        {
            sd1 = 1;
        }

        ai->S = sd1;

        if (ai->S == 1)
        {
            R->cAcum++;
        }
    }
}

void tiempoIncSinCurRec(Agent *ai, Results *R)
{
    int sd = ai->S;

    if (sd == 2 || sd == -2)
    {
        return;
    }

    int trecd1 = ai->Trec;
    if (sd == -1)
    {
        trecd1 -= 1;
        if (trecd1 == 0)
        {
            ai->S = 2;
            R->cAcumAgRecup++;
        }
        ai->Trec = trecd1;
    }
    else if (sd == 1)
    {
        int Tincd = ai->Tinc;
        Tincd -= 1;
        int Sd1 = -1;

        if (Tincd > 0)
        {
            Sd1 = sd;
        }

        ai->S = Sd1;
        ai->Tinc = Tincd;
    }
}

void casosFatales(Agent *ai, Results *R)
{
    int sd = ai->S;

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
    float random = rand() % 1001 / 1000.0;
    if (random <= ai->Pfat)
    {
        if (rho > 0)
        {
            sd1 = -2;
            R->cFatAcum++;
        }
    }

    ai->S = sd1;
}

int main()
{
    const int N = 10240;
    const int DAYS = 31;
    Simulacion sim;
    sim.N = N;
    sim.dmax = DAYS;
    Agent agents[N];
    ResultsDays resultsDays[DAYS];
    Results results;

    float elapsedTime = 0;
    clock_t start = clock();
    inicializacion(&sim, agents);
    clock_t end = clock();
    elapsedTime += end - start;
    int dM = sim.dmax;
    int mM = sim.Mmax;
    srand(time(NULL));

    int t_cAcum = 0;
    int t_cAcumAgRecup = 0;
    int t_cFatAcum = 0;

    for (int i = 1; i <= dM; i++)
    {
        t_cAcum = results.cAcum;
        t_cAcumAgRecup = results.cAcumAgRecup;
        t_cFatAcum = results.cFatAcum;
        for (int j = 0; j < mM; j++)
        {
            for (int k = 0; k < N; k++)
            {
                Agent agent = agents[k];
                start = clock();
                contagio(sim.N, sim.R, agent.X, agent.Y, k, agent.Pcon, &results, agents, &agent, i);
                movilidad(&sim, &agent);
                end = clock();
                elapsedTime += end - start;
                agents[k] = agent;
            }
        }
        for (int j = 0; j < N; j++)
        {
            Agent agent = agents[j];
            start = clock();
            contagioExterno(&agent, &results);
            tiempoIncSinCurRec(&agent, &results);
            casosFatales(&agent, &results);
            end = clock();
            elapsedTime += end - start;
            agents[j] = agent;
        }

        ResultsDays newDay;
        newDay.c = results.cAcum - t_cAcum;
        newDay.cRecup = results.cAcumAgRecup - t_cAcumAgRecup;
        newDay.cFat = results.cFatAcum - t_cFatAcum;
        resultsDays[i] = newDay;

        printf("Dia %d\n", i);
        printf("recup: %d\n", results.cAcumAgRecup);
        printf("    Numero de nuevos casos positivos por dia: %d\n", newDay.c);
        printf("    Numero de casos recuperados por dia: %d\n", newDay.cRecup);
        printf("    Numero de casos fatales por dia: %d\n", newDay.cFat);
        printf("------------------------------------------------------\n");

        if (results.cZero == 0 && newDay.c > 0)
        {
            results.cZero = i;
        }
        if (results.recupPrim == 0 && newDay.cRecup > 0)
        {
            results.recupPrim = i;
        }
        if (results.cFatPrim == 0 && newDay.cFat > 0)
        {
            results.cFatPrim = i;
        }
    }

    int counterC = 0;
    int counterRecup = 0;
    int counterFat = 0;

    for (int i = 1; i <= DAYS; i++)
    {
        ResultsDays resultsDay = resultsDays[i];
        counterC += resultsDay.c;
        counterRecup += resultsDay.cRecup;
        counterFat += resultsDay.cFat;
        if (results.c50per == 0 && counterC >= results.cAcum / 2)
        {
            results.c50per = i;
        }
        if (results.c100per == 0 && counterC == results.cAcum)
        {
            results.c100per = i;
        }

        if (results.recup50per == 0 && counterRecup >= results.cAcumAgRecup / 2)
        {
            results.recup50per = i;
        }
        if (results.recup100per == 0 && counterRecup == results.cAcumAgRecup)
        {
            results.recup100per = i;
        }

        if (results.cFat50per == 0 && counterFat >= results.cFatAcum / 2)
        {
            results.cFat50per = i;
        }
        if (results.cFat100per == 0 && counterFat == results.cFatAcum)
        {
            results.cFat100per = i;
        }
    }
    printf("Resultados Finales\n");
    printf("    Numero de casos acumulados de agentes contagiados: %d\n", results.cAcum);
    printf("    -> Dia en que se contagio el primer agente: %d\n", results.cZero);
    printf("    -> Dia en que se contagio el 50%% de los agentes contagiados: %d\n", results.c50per);
    printf("    -> Dia en que se contagio el 100%% de los agentes contagiados: %d\n", results.c100per);
    printf("    Numero de casos acumulados de agentes recuperados: %d\n", results.cAcumAgRecup);
    printf("    -> Dia en que se recupero el primer agente: %d\n", results.recupPrim);
    printf("    -> Dia en que se recupero el 50%% de los agentes recuperados: %d\n", results.recup50per);
    printf("    -> Dia en que se recupero el 100%% de los agentes recuperados: %d\n", results.recup100per);
    printf("    Numero de casos fatales acumulados: %d\n", results.cFatAcum);
    printf("    -> Dia en que ocurrio el primer caso fatal: %d\n", results.cFatPrim);
    printf("    -> Dia en que ocurrio el 50%% de los casos fatales: %d\n", results.cFat50per);
    printf("    -> Dia en que ocurrio el 100%% de los casos fatales: %d\n", results.cFat100per);
    printf("Tiempo transcurrido: %f milisegundos\n", elapsedTime);
    printf("------------------------\n");
    return 0;
}
