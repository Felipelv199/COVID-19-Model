struct Agent
{
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
};



struct Results
{
    int cAcum = 0;
    int cXDia = 0;
    int cAcumAgRecup = 0;
    int cRecupXDia = 0;
    int cFatAcum = 0;
    int cFatXDia = 0;

    int cZero = 0;
    int c50per = 0;
    int c100per = 0;

    int recupPrim = 0;
    int recup50per = 0;
    int recup100per = 0;

    int cFatPrim = 0;
    int cFat50per = 0;
    int cFat100per = 0;

    int timeCPu = 0;
    int timeGPU = 0;
};
struct Simulacion
{
    int N;
    int dmax;
    int Mmax = 10;
    int lmax = 500;
    int R = 100;
    int PQ = 5000;
    Results results;
};

struct ResultsDays
{
    int c = 0;
    int cRecup = 0;
    int cFat = 0;
};