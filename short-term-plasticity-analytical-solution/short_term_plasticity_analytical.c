//This code reproduce the pre-print "Analytical solutions for the short-term plasticity - bioRxiv"
// P. R. Protachevicz, A. M. Batista, I. L. Caldas, M. S. Baptista
#include<math.h>
#include<stdio.h>
#include<stdlib.h>
//****** Parametros do gerador aleatorio ********************//
#define IA 16807
#define IM 2147483647
#define AM (1.0/IM)
#define IQ 127773
#define IR 2836
#define NTAB 32
#define NDIV (1+(IM-1)/NTAB)
#define EPS 1.2e-7
#define RNMX (1.0-EPS)
#define tol1 0.001   /////// tolerancia para identificar os regimes
#define tol2 0.001
//******   Parameters ********************//
#define N 4             // number of equations
#define h 0.001         // integration step
#define nn 8000  ///// ms
#define n nn/h // total number of steps nn*h

float ran1(long *idum);
#define NR_END 1
#define FREE_ARG char*

void nrerror(char error_text[]);
int *vector(long nl,long nh);
int **imatrix(long nrl, long nrh, long ncl, long nch);
void free_vector(int *v, long nl, long nh);
void free_imatrix(int **m, long nrl, long nrh, long ncl, long nch);
double *dvector(long nl,long nh);
void free_dvector(double *v, long nl, long nh);
void derivs(double y[],double df[]);
double tau_rec,tau_fac, tau_ina, U,z1[5],taux,map[5],map_new[5];

FILE *o,*p,*q,*s,*ooo;

int main(void)
{
  double *df,*y,*a,*b,*c,*x,A,B,C,D1,T;

int i,j,k,t,max_spike,cont,contD;
double tempo,tspike,freq,yampl,ymax,yant,yant2,iay,u,D,ID,umax,zreset,ymax2;
double tsp,ureset,u_analytic,xxx,yyy,zzz,uuu,xreset,y_max_analytics,ymax_map,y_max_analytics2;
double y_max_analytics3,y_max_analytics4,y_ampl[500];
int facil,depre,regime,cont_spike,first_facil,first_depre,y0,conty,tspmax,tspmax_map;

long idum;

idum=-123456789; /// seed for random number generator

y=dvector(1,N+2);
df=dvector(1,N+2);
x=dvector(1,N+2);
a=dvector(1,N+2);
b=dvector(1,N+2);
c=dvector(1,N+2);

ooo=fopen("Solucao.dat","wt");
o=fopen("Dinamics1_f9_U0.01.dat","wt");
p=fopen("Dinamics2_f9_U0.01.dat","wt");
q=fopen("Dinamics3_f9_U0.01.dat","wt");
s=fopen("Parameter_space.dat","wt");

if(o==NULL)
    {
     puts ("erro no arquivo");
     getchar();
     exit(1);
    }
    if(p==NULL)
    {
     puts ("erro no arquivo");
     getchar();
     exit(1);
    }
    if(q==NULL)
    {
     puts ("erro no arquivo");
     getchar();
     exit(1);
    }

   //////////////////////// Parameters ////////////////
   tau_rec=800.0;  /////////// recovery time constant
   tau_fac=1000.0;  ///////////facilitation time constant
   tau_ina=3.0;  ////////// inativation  time constant
   U=0.01;
   freq=9.0;
   ID=1.0;   /// spike variability
   ID=0.0;
// for(U=0;U<1;U=U+0.025)
// for(freq=0.01;freq<=10.01;freq=freq+0.25)
  //    for(U=0;U<=1;U=U+0.01)
 //  for(freq=0.01;freq<=10.01;freq=freq+0.1)
   {
	conty=0;
   //////////////////////////////////////////////////
   ////                                          ////
   ////          Spike protocoll                 ////
   ////                                          ////
   //////////////////////////////////////////////////

 T=1000/freq;
 A=exp(-T/tau_ina);
 B=exp(-T/tau_rec);
 C=exp(-T/tau_fac);
 uuu=U*C/(1+(U-1)*C);
 D1=A*B*(2*uuu-1)+(A+B)*(1-uuu)-1;
 xxx=(A+B+A*B -1)/D1;
 yyy= A*(B-1)*uuu/D1;
 zzz=B*(A-1)*uuu/D1;
 tspmax=0;tspmax_map=0;
 ymax_map=0;
// fprintf(ooo,"%f %f %f %f %f %f\n", 0.00000, xxx, yyy, zzz, uuu,xxx*uuu);
//  fprintf(ooo,"%f %f %f %f %f %f\n", 20000.00000, xxx, yyy, zzz, uuu,xxx*uuu);
//****************  Initials Conditions from the rest  **************************
x[1]=1.0; //// x avaliable     x + y + z = 1  !!!! always !!!
x[2]=0.0; //// y active       = 0 !
x[3]=0.0; //// z inactive     = 0 !
x[4]=0.0; ///U; //// fraction of avaliable vesicules which become actives  = U!
k=0;
ymax=0;  yant=0.0;
facil=0;    depre=0; first_facil=0; first_depre=0;
regime=0;cont_spike=0; y0=0;
iay=0;contD=0;
tspike=0; taux=9999999;
//********************* LOOP DO TEMPO  ***************************
tempo=0.0;  tspmax=0;
  /////////////// Initial Conditions - Analytical Solution
    z1[1]=1.0;
    z1[2]=0.0;
    z1[3]=0.0;
    z1[4]=0.0;
    map[1]=1.0;
 	map[2]=0.0;
 	map[3]=0.0;
 	map[4]=0.0;
    map_new[1]=0.0;
 	map_new[2]=0.0;
 	map_new[3]=0.0;
 	map_new[4]=0.0;
 	ymax_map=0;
    xreset=z1[1];

    for(t=1;t<=n;t++)
	{
	tempo=tempo+h;      //in milliseconds
	yant2=yant;         //// amplitute of active neurontransmittes before the antepenult spike of the presynaptic spike.

	if(contD==0) { D= ID*(1-2*ran1(& idum)); contD=1;}
	//////////////////////////// update time spike ////////////////////
	if(tempo>=tspike+1000/freq && tempo<tspike+1000/freq+h)
	{

	contD=0;
	u =	x[1]*x[4];          //// amount of neurotransmitters change in the state due spike
	x[1]= x[1] - u;   //// x
	x[2]= x[2] + u;   //// y
	//x[3]=x[3] ;             //// z
	x[4]=x[4] +U*(1-x[4]);	  //// u

	//  printf("%f %f \n", tempo, tspike[j]);

	///// Reseting analytical solution due spike
	u_analytic=z1[1]*z1[4];
	z1[1] = z1[1]-u_analytic;
	z1[2] = z1[2]+u_analytic;
	z1[3] = z1[3];
	z1[4] = z1[4]+U*(1.0-z1[4]);

	y_ampl[conty]=map[1]*map[4];

	tsp=tempo;
	umax=u;
	ymax2=z1[2];
	ureset=z1[4];
	zreset=z1[3];
	xreset=z1[1];

	//// Analyses of Synaptic Regimes
	tspike=tempo;

	yampl=x[2];

	if(x[2]>0.01) y0=1;

	if(ymax<=yampl) {ymax = yampl; tspmax=conty;  }

	//printf("%f %f %f %f %f \n", tempo, yampl,yant,iay,u);

	if(yant>0)
	{
		iay  = yampl-yant;
		if(iay>tol1) facil=1;
		if(iay<-tol1) depre=1;

	//printf("%f %f %f %f \n", tempo, yampl,yant,iay);

		if(facil==1 & depre==0)
			{
			first_facil=1;
			}

		if(facil==0 & depre==1)
			{
			first_depre=1;
		   }
	}
	yant=x[2];

////////////////////////////////////////////////////////////////////////////
//////////// Evolution of the systhem by the obtained map///////////////////
////////////////////////////////////////////////////////////////////////////
fprintf(o,"%d %f %f %f %f %f\n",conty,map[1],map[2],map[3],map[4],map[1]*map[4]);
fflush(o);

//map_new[1]= (map[1]-map[1]*map[4]) -(map[2]+map[1]*map[4])*((B-1.0)-((tau_ina)/(tau_ina+tau_rec))*A*B) -map[3]*(B-1); //// Original Solution
map_new[1]=(map[1]+map[2])-map[3]*(B-1.0)-(map[2]+map[1]*map[4])*B*(1-((tau_ina)/(tau_rec+tau_ina))*A);   /// Approximation Solution
map_new[2]=(map[2]+map[1]*map[4])*A;  //Y
map_new[3]=(map[3]+(map[2]+map[1]*map[4])*(1-A))*B; //Z
map_new[4]=(map[4]+U*(1.0-map[4]))*C; //U

map[1]=map_new[1];
map[2]=map_new[2];
map[3]=map_new[3];
map[4]=map_new[4];

if(ymax_map<map[1]*map[4]) {ymax_map = map[1]*map[4];tspmax_map=conty+1;}

conty++;

	cont_spike++;
	   }

	   taux=tempo-tsp;

/////////////////////////////////////////////////////////////////////////
////////////Evolution of the System by the analytical solution //////////
/////////////////////////////////////////////////////////////////////////
	z1[2]=ymax2*exp(-taux/tau_ina);
	z1[3]=(zreset+ymax2*(1 - exp(-1.0*taux/tau_ina)))*exp(-1.0*taux/tau_rec);
	z1[4]=ureset*exp(-taux*1.0/tau_fac);
	z1[1]= xreset-ymax2*((exp(-taux*1.0/tau_rec)-1.0)- ((tau_ina)/(tau_rec +tau_ina))*exp(-taux*((1.0)/(tau_ina) + (1.0)/(tau_rec)))  )-zreset*(exp(-taux*1.0/tau_rec)-1);

if(tempo>20000) if(iay<tol1 && iay> -tol1) break;    ///// Importante Break

	 // ------------ Numerical Integrator Runge-Kutta 4th order---------------
      for(i=1;i<=N;i++) y[i]=x[i];

	  derivs(y,df);
	  for(i=1;i<=N;i++)
	    {
	     a[i]=h*df[i];
	     y[i]=x[i]+a[i]/2.0;
	    }
	  derivs(y,df);
	  for(i=1;i<=N;i++)
	    {
	     b[i]=h*df[i];
	     y[i]=x[i]+b[i]/2.0;
	    }
	  derivs(y,df);
	  for(i=1;i<=N;i++)
	    {
	     c[i]=h*df[i];
	     y[i]=x[i]+c[i];
	    }
	  derivs(y,df);
	  for(i=1;i<=N;i++)
	    x[i]=x[i]+(a[i]+h*df[i])/6.0+(b[i]+c[i])/3.0;
	  //--------------------------------------------------------------------

if(cont>200) {
cont=0;
	}
cont++;

if(x[1]+x[2]+x[3]>=1.001) { printf("Error 1, Neurotransmitter conservation is not satisfied!!\n"); break;}

if(x[1]+x[2]+x[3]<=0.999)
	{
	printf("Error 2, Neurotransmitter conservation is not satisfied !!!! Soma= %f\n",x[1]+x[2]+x[3]); break;
	}
	}  // ending of time loop
	//// Synaptic Regimes
    if(facil==1 && depre==0) regime = 1; ////// Facilitation
	if(facil==0 && depre==1) regime = 2; ////// Depression
	if(facil==1 && depre==1) regime = 4; ////// Biphasic

	if(facil==1 && depre==1)
	{
    if(sqrt((yampl-ymax)*(yampl-ymax))<tol2) regime = 3; ///// pseudo-linear
	if(sqrt((yant2-ymax)*(yant2-ymax))<tol2) regime = 3; ///// pseudo-linear
	}

if(regime==0)
printf("Frequencia=%f, U=%f, Regime = Nada?\n",freq,U);

if(regime==1)
printf("Frequencia=%f, U=%f, Regime = Facilitation\n",freq,U);

if(regime==2)
printf("Frequencia=%f U=%f, Regime = Depression \n",freq,U);

if(regime==3)
printf("Frequencia=%f U=%f, Regime = Pseudo-linear\n",freq,U);

if(regime==4)
printf("Frequencia=%f U=%f, Regime = Biphasic\n",freq,U);

if(y0==0) regime=-1;

if(regime==2){
y_max_analytics=U*C;
y_max_analytics2=U*C;
y_max_analytics3=U*C;
y_max_analytics4=U*C;
}

if(regime==4)
{
y_max_analytics2=C*U*(2+C-U-2*C*U+C*C*U*U)*(1-B*C*U*(2+C*(1-(1+B*(1+C*(1-U)))*U)));
y_max_analytics=U*(2*C+U*C*C*(-1-2*B+C*B));
y_max_analytics3=y_ampl[4];
y_max_analytics4=y_ampl[5];
}
if(regime==1)
{
y_max_analytics= xxx*uuu;
y_max_analytics2= xxx*uuu;
y_max_analytics3= xxx*uuu;
y_max_analytics4= xxx*uuu;
}

if(regime!=4) tspmax=0;
fprintf(s,"%f %f %d %f %f %f %.12f %.12f %.12f %.12f %d %d\n",freq,U,regime,ymax,yampl,y_max_analytics, ymax-y_max_analytics,ymax-y_max_analytics2,ymax-y_max_analytics3,ymax-y_max_analytics4,tspmax,tspmax_map);
fflush(s);
if(y0==0) printf("Alerta, y nÃ£o sai de zero!!\n");

}
  free_dvector(y,1,N+2);
  free_dvector(df,1,N+2);
  free_dvector(x,1,N+2);
  free_dvector(a,1,N+2);
  free_dvector(b,1,N+2);
  free_dvector(c,1,N+2);

  fclose(o);
  fclose(p);
  fclose(q);
  return 0;
}

void derivs(double y[],double df[]) // Equacoes diferenciais acopladas
{
/////                           x[1] = x ,     x[2] = y,     x[3] = z,     x[4] = u

      df[1]= y[3]/tau_rec;    //// dx/dt   				fraction of avaliable NTs
      df[2]= -y[2]/tau_ina;    //// dy/dt  				fraction of active NTs
      df[3]= y[2]/tau_ina-y[3]/tau_rec;   //// dz/dt   	recovering
      df[4]= -y[4]/tau_fac;     //// dz/dt 				fraction of avaliable NT using in the next spike
}

float ran1(long *idum)
{
 int j;
 long k;
 static long iy=0;
 static long iv[NTAB];
 float temp;

 if(*idum<=0 || !iy)
   {
     if(-(*idum)<1) *idum=1;
     else *idum = -(*idum);
    for(j=NTAB+7;j>=0;j--)
      {
       k=(*idum)/IQ;
       *idum=IA*(*idum-k*IQ)-IR*k;
       if(*idum<0) *idum +=IM;
       if(j<NTAB) iv[j]=*idum;
      }
      iy=iv[0];
   }
   k=(*idum)/IQ;
   *idum=IA*(*idum-k*IQ)-IR*k;
   if(*idum<0) *idum += IM;
   j=iy/NDIV;
   iy=iv[j];
   iv[j]=*idum;
   if((temp=AM*iy)>RNMX) return RNMX;
   else return temp;
}

double *dvector(long nl,long nh)
{
   double *v;

   v=(double *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(double)));
   if (!v) nrerror("allocation failure in dvector()");
   return v-nl+NR_END;
}

void free_dvector(double *v, long nl, long nh)
{
   free((FREE_ARG) (v+nl-NR_END));
}

void nrerror(char error_text[])
{
  fprintf(stderr,"Numerical Recipes run-time error...\n");
  fprintf(stderr,"%s\n",error_text);
  fprintf(stderr,"...now exiting to system...\n");
  exit(1);
}

int *vector(long nl,long nh)
{
   int *v;

   v=(int *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(int)));
   if (!v) nrerror("allocation failure in dvector()");
   return v-nl+NR_END;
}

int **imatrix(long nrl, long nrh, long ncl, long nch)
{
   long i, nrow=nrh-nrl+1,ncol=nch-ncl+1;
   int **m;

   m=(int **) malloc((size_t)((nrow+NR_END)*sizeof(int*)));
   if (!m) nrerror("allocation failure 1 in imatrix()");
   m += NR_END;
   m -= nrl;

   m[nrl]=(int *) malloc((size_t)((nrow*ncol+NR_END)*sizeof(int)));
   if (!m[nrl]) nrerror("allocation failure 2 in imatrix()");
   m[nrl] += NR_END;
   m[nrl] -= ncl;

   for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;

   return m;
}

void free_vector(int *v, long nl, long nh)
{
   free((FREE_ARG) (v+nl-NR_END));
}

void free_imatrix(int **m, long nrl, long nrh, long ncl, long nch)
{
   free((FREE_ARG) (m[nrl]+ncl-NR_END));
   free((FREE_ARG) (m+nrl-NR_END));
}
