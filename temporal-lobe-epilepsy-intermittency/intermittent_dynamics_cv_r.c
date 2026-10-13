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
#define PI acos(-1.0)
#define n_n 110000

//******   Parametros do sistema ********************//
#define NN 1000          // numero of IF
#define N 2             // numero de equacoes
#define h 0.01          // passo de integracao
#define transient 100    // ms
#define NMD 1000000 //numero maximo de disparo de um neuronio: 10^4
#define conexMAX NN   // each neuron max conections number
#define P_exc_ini 0.8  //porcentagem de conexÃµes excitatÃ³rias

//*******gerador de numeros aleatorios******//
float ran1(long *idum);
#define NR_END 1
#define FREE_ARG char*

//*******ponteiros para alocar memoria*****///
void nrerror(char error_text[]);
int *vector(long nl,long nh);
int **imatrix(long nrl, long nrh, long ncl, long nch);
void free_vector(int *v, long nl, long nh);
void free_imatrix(int **m, long nrl, long nrh, long ncl, long nch);
double *dvector(long nl,long nh);
void free_dvector(double *v, long nl, long nh);
double **dmatrix(long nrl, long nrh, long ncl, long nch);
void free_dmatrix(double **m, long nrl, long nrh, long ncl, long nch);


void derivs(double y[],double df[],double *I,double *aa,double g_exc,double g_ini,double *Iext,double *Gexc,double *Gini);

FILE *o;
FILE *o2;
FILE *o3;
FILE *o4;


int main(void)
{
  double *df,*y,*I,*a,*b,*c,*x,*aa,*tpeak,*Iext,*cont_Iext,*Gexc,*Gini,tauex,tauin,gARC,contUP,contDOWN,tempoUP,tempoDOWN,contUP2;

  int i,j,cont,t,n,contISI,icontUP,icontDOWN;
  int **listaexc,*conextotalex,**listaini,*conextotalin;
  double tempo,b_IF,g_exc,g_ini,V_reset;
  double kinicial,kfinal,R,real,compl,Rmedio,**nk,*phi;
  int contR,*k,*kmax;
  double ISI,ISI2,CV,desviopadrao,KM,g;
  int numerodeconexoes,sum,auxx,jj;

  long idum;

  I=dvector(1,NN+1);
  Iext=dvector(1,NN+1);
  cont_Iext=dvector(1,NN+1);
  aa=dvector(1,NN+1);

  y=dvector(1,N*NN+1);
  df=dvector(1,N*NN+1);
  x=dvector(1,N*NN+1);
  a=dvector(1,N*NN+1);
  b=dvector(1,N*NN+1);
  c=dvector(1,N*NN+1);
  k=vector(1,NN+1);
  kmax=vector(1,NN+1);
  phi=dvector(1,NN+1);
  nk=dmatrix(1,NMD+2,1,NN+2);
  tpeak=dvector(1,NN+1);
  Gexc=dvector(1,NN+1);
  Gini=dvector(1,NN+1);

  conextotalex=vector(1,NN+2);
  listaexc=imatrix(1,NN+1,1,conexMAX+2);

  conextotalin=vector(1,NN+2);
  listaini=imatrix(1,NN+1,1,conexMAX+2);

  o=fopen("raster_gexc0.5_g4_10p_gARC1.31.dat","wt");
  o2=fopen("CV_gexc0.5_g4_10p_gARC1.31.dat","wt");
  o3=fopen("R_gexc0.5_g4_10p_gARC1.31.dat","wt");
  o4=fopen("Vt_gexc0.5_g4_10p_gARC1.31.dat","wt");



//-------Grau medio de conexao KM----------//
for(KM=99.9;KM<=99.9;KM=KM+2.0)  // range [2:NN-1] //-90.0
 {
 idum=-1234567899;
  numerodeconexoes=NN*KM;//rede aleatÃ³ria com KM //se KM=(NN-1); //global network

 //****************** Criando lista com as conexoes *********************
//-----------zerando tudo------------//
  for(i=1;i<=NN;i++)
    {
      conextotalex[i]=0.0;
      conextotalin[i]=0.0;
      for(j=1;j<=conexMAX;j++)
              {
	   listaexc[i][j]=0.0;    //// listaexc[pÃ³s-sinaptico][numero da conexao] = prÃ©-sinaptico
           listaini[i][j]=0.0;
             }
    }

//--------pelo menos uma conexao para cada neuronio------//
  for(i=1;i<=NN;i++)
    {
      conextotalex[i]=1.0;
      listaexc[i][1]=(int)NN*ran1(& idum);
      conextotalin[i]=1.0;
      listaini[i][1]=(int)NN*ran1(& idum);
    }

//------- criando lista de conexoes exc---------//
  sum=0.0;
  while(sum<numerodeconexoes) ///i neurÃ´nio pÃ³s sinaptico FIXO
    {
      i=(int)NN*ran1(& idum); //i neurÃ´nio pÃ³s sinaptico
      j=(int)NN*ran1(& idum);// j neurÃ´nio prÃ© sinaptico

      if(i==0)
	i=NN;
      if(j==0)
	j=NN;

      auxx=0.0;

      for(jj=1;jj<=conextotalex[j];jj++) //conta apenas novas conexoes
	if(listaexc[j][jj]==i)
	  auxx=1.0;

      for(jj=1;jj<=conextotalin[j];jj++) //conta apenas novas conexoes
	if(listaini[j][jj]==i)
	  auxx=1.0;

      if(auxx==0.0)
	if(j!=i)
	  {
                if(j<=P_exc_ini*NN)
                  {
		conextotalex[j]=conextotalex[j]+1;
		listaexc[j][conextotalex[j]]=i;
                  }
                else
                  {
		conextotalin[j]=conextotalin[j]+1;
		listaini[j][conextotalin[j]]=i;
                  }
         }

      sum=0.0;
      for(i=1;i<=NN;i++)
	     sum=sum+conextotalex[i]+conextotalin[i];
    }


 //*****************************************************************
for(gARC=1.31;gARC<=1.310001;gARC=gARC+0.025)
{
	 g=4.0;
     n=n_n; // total number of steps n*h
	 tauex=2.728; //ms synaptic conductances time constants
     tauin=tauex;

  //****************  Condicoes iniciais  **************************
   for(i=1;i<=NN;i++)
	{
	 x[1+(i-1)*N]=-70.6+20.0*ran1(& idum); //V
	 x[2+(i-1)*N]= 0.0+80.0*ran1(& idum);  //w
  	 Gexc[i]=0.0;
	 Gini[i]=0.0;
     tpeak[i]=n*h-100.0;	//tempo do Ãºltimo disparo
	 I[i]=512.4;            //corrente externa
     aa[i]=2.0;
    }

for(g_exc=0.5;g_exc<=0.500002;g_exc=g_exc+0.1)
   {
    g_ini=g*g_exc;
	V_reset=-58.0;
    b_IF=70.0;

     ISI=0;
     ISI2=0;
     contISI=0;
     real=transient;

   //********************* LOOP DO TEMPO  ***************************
   for(i=1;i<=NN;i++)
	{
     tpeak[i]=tpeak[i]-n*h;	//tempo do Ãºltimo disparo
	 k[i]=0.0;               // contador do numero de disparos de um neuronio
	 kmax[i]=0.0;	         // contador do total de disparos de um neuronio
	 nk[1][i]=0.0;           // tempo em que ocorrem os disparos de um neuronio
	 Iext[i]=0.0;
	 cont_Iext[i]=0.0;
    }
      tempo=0.0;

      for(t=1;t<=n;t++)
	{
	  tempo=tempo+h;      //em milisegundos

	  for(i=1;i<=NN;i++)  // tempo anterior
	    {
             Gexc[i]=Gexc[i]*exp(-h/tauex);  // condutancia recebida
	         Gini[i]=Gini[i]*exp(-h/tauin);

	      if(x[1+(i-1)*N]>-35.0) // modelo de aIEF
			{
			   if(tempo>transient)
					fprintf(o,"%.2f %d\n",tempo,i); // raster plot

               if(tempo>transient && k[i]<NMD)     //salvando os disparos e o tempo em que ocorrem para cada neuronio
                  {
				   k[i]=k[i]+1;
				   nk[k[i]][i]=tempo;

		           if(tempo<transient+500.0)
				    if(k[i]==1.0)
				     kinicial=tempo;
				  }

	          tpeak[i]=tempo;

               if(k[i]>=NMD)
                 t=n+1;

              x[1+(i-1)*N]=V_reset; //V
		      x[2+(i-1)*N]=x[2+(i-1)*N]+b_IF; //w
			}

	      y[1+(i-1)*N]=x[1+(i-1)*N];
	      y[2+(i-1)*N]=x[2+(i-1)*N];

	      if(tempo>tpeak[i] && tempo<=tpeak[i]+h) //delay=h
            {
			if(i<=P_exc_ini*NN)
		       for(j=1;j<=conextotalex[i];j++)
		        if(listaexc[i][j]<=0.1*NN) //P_ARC = 0.1
		         Gexc[listaexc[i][j]]=Gexc[listaexc[i][j]]+gARC*1.0; // condutancia exc recebida
		        else
		         Gexc[listaexc[i][j]]=Gexc[listaexc[i][j]]+1.0; // condutancia exc recebida
		     else
		       for(j=1;j<=conextotalin[i];j++)
		         Gini[listaini[i][j]]=Gini[listaini[i][j]]+1.0;
		    }
	    }


	  // ------------ Integrador numÃ©rico Runge-Kutta 4Âª ordem---------------
	  derivs(y,df,I,aa,g_exc,g_ini,Iext,Gexc,Gini);
	  for(i=1;i<=N*NN;i++)
	    {
	     a[i]=h*df[i];
	     y[i]=x[i]+a[i]/2.0;
	    }
	  derivs(y,df,I,aa,g_exc,g_ini,Iext,Gexc,Gini);
	   for(i=1;i<=N*NN;i++)
	    {
	     b[i]=h*df[i];
	     y[i]=x[i]+b[i]/2.0;
	    }
	 derivs(y,df,I,aa,g_exc,g_ini,Iext,Gexc,Gini);
	for(i=1;i<=N*NN;i++)
	    {
	     c[i]=h*df[i];
	     y[i]=x[i]+c[i];
	    }
    derivs(y,df,I,aa,g_exc,g_ini,Iext,Gexc,Gini);
	   for(i=1;i<=N*NN;i++)
	    x[i]=x[i]+(a[i]+h*df[i])/6.0+(b[i]+c[i])/3.0;
	  //--------------------------------------------------------------------

    }

// CALCULO DO PARAMETRO DE ORDEM //

tempo=0.0;
Rmedio=0.0;
contR=0.0;
kfinal=n*h;

for(i=1;i<=NN;i=i+1)
{
  phi[i]=0.0;
  kmax[i]=k[i];
  k[i]=1.0;

  if(kmax[i]>1)
  if(kfinal>nk[kmax[i]][i] && kfinal>kinicial)
    kfinal=nk[kmax[i]][i];
}

for(tempo=transient;tempo<=n*h;tempo=tempo+10.0*h)
{
  for(i=1;i<=NN;i=i+1)
    if(kmax[i]>1)
      if(tempo>nk[k[i]][i] && tempo<nk[kmax[i]][i])
      {
	if(tempo<=nk[k[i]+1][i])
	phi[i]=2*PI*(tempo-nk[k[i]][i])/(nk[k[i]+1][i]-nk[k[i]][i]); //+2*PI*k[i]

	if(tempo>=nk[k[i]+1][i])
	k[i]=k[i]+1;
      }


  if(tempo>=kinicial && tempo<=kfinal)
  {
    real=0.0;
    compl=0.0;
    cont=0;
    for(i=1;i<=NN;i++)
    if(kmax[i]>1)
      {
      real=real+cos(phi[i]);
      compl=compl+sin(phi[i]);
      cont=cont+1;
      }

    real=real/cont;
    compl=compl/cont;

    R=sqrt(real*real+compl*compl);
    Rmedio=Rmedio+R;
    contR=contR+1.0;

	fprintf(o3,"%f %f\n",tempo,R);
  }

}

// CALCULO DO PARAMETRO DA CV (t)//
tempo=0.0;
Rmedio=0.0;
contR=0.0;
icontUP=0.0;
icontDOWN=0.0;
contUP=0.0;
contUP2=0.0;
contDOWN=0.0;
tempoUP=0.0;
tempoDOWN=0.0;

for(i=1;i<=NN;i=i+1)
{
  phi[i]=0.0;
  kmax[i]=k[i];
  k[i]=1.0;
}

for(tempo=transient;tempo<=n*h;tempo=tempo+10.0*h)
{
  ISI=0;
  ISI2=0;
  contISI=0;

   for(i=1;i<=0.1*NN;i=i+1)
     if(tempo>nk[k[i]][i] && tempo<nk[kmax[i]][i])
      {
	      if(k[i]>5 && k[i]<kmax[i]-5)
			for(j=4;j>=-5;j=j-1)
			 {
			  ISI=ISI+(nk[k[i]+1+j][i]-nk[k[i]+j][i]);
			  ISI2=ISI2+(nk[k[i]+1+j][i]-nk[k[i]+j][i])*(nk[k[i]+1+j][i]-nk[k[i]+j][i]);
			  contISI=contISI+1;
		     }

		if(tempo>=nk[k[i]+1][i])
		k[i]=k[i]+1;
	  }

      if(contISI>0)
      {
		desviopadrao=sqrt(ISI2/contISI - (ISI/contISI)*(ISI/contISI));
		CV = desviopadrao/(ISI/contISI); //CV
	    fprintf(o2,"%f %f\n",tempo,CV);


	    if(CV>=0.5)
	      {
	       jj=0;

			  if(contUP2>0)//acrescenta ao UP se oscilar entre CV [0.4 e 0.5]
			  {
			      contUP=contUP+contUP2;
				  tempoUP=tempoUP+contUP2;
			      contDOWN=contDOWN-contUP2;
			      tempoDOWN=tempoDOWN-contUP2;
	              contUP2=0.0;
	              jj=1;
			  }

	      contUP=contUP+10.0*h;
		  tempoUP=tempoUP+10.0*h;
		  }
	      else //CV<0.5
	      {
			 if(contUP>0.0 && CV>0.4) //CV>0.4 && CV<0.5
		      contUP2=contUP2+10.0*h;

			 if(contUP>0.0 && CV<=0.4)//CV<=0.4 && CV<0.5
			 {
				icontUP=icontUP+1;

	            contUP=0.0;
	            contUP2=0.0; //	zera se CV nÃ£o chegar a 0.5 antes de 0.4
		     }
		  }

	      if(CV<0.5)
	      {
	      contDOWN=contDOWN+10.0*h;
	      tempoDOWN=tempoDOWN+10.0*h;
		  }
	      else//CV>=0.5
	      {
			 if(contDOWN>0.0 && jj==0)
			 {
		       	icontDOWN=icontDOWN+1;

	            contDOWN=0.0;
		     }
		  }
	  }
}

	printf("%.3f %.3f %.3f %f %d %d %.3f %.3f \n",gARC,tempoUP,tempoDOWN,tempoUP/tempoDOWN,icontUP,icontDOWN,tempoUP/icontUP,tempoDOWN/icontDOWN);
} // fim loop de g_exc e/ou b_IF e/ou V_reset
} // fim de loop
} // fim loop de KM

  free_dvector(y,1,N*NN+1);
  free_dvector(df,1,N*NN+1);
  free_dvector(x,1,N*NN+1);
  free_dvector(a,1,N*NN+1);
  free_dvector(b,1,N*NN+1);
  free_dvector(c,1,N*NN+1);
  free_dvector(I,1,NN+1);
  free_dvector(Iext,1,NN+1);
  free_dvector(cont_Iext,1,NN+1);
  free_dvector(aa,1,NN+1);
  free_vector(k,1,NN+1);
  free_vector(kmax,1,NN+1);
  free_dvector(phi,1,NN+1);
  free_dmatrix(nk,1,NMD+2,1,NN+2);
  free_dvector(tpeak,1,NN+1);
  free_dvector(Gexc,1,NN+1);
  free_dvector(Gini,1,NN+1);

  free_vector(conextotalex,1,NN+2);
  free_imatrix(listaexc,1,NN+1,1,conexMAX+2);

  fclose(o);
  fclose(o2);
  fclose(o3);
  fclose(o4);

  return 0;
}


void derivs(double y[],double df[],double *I,double *aa, double g_exc,double g_ini,double *Iext,double *Gexc,double *Gini) // Equacoes diferenciais acopladas
{
  int i;
  double C,gL,VT,DeltaT,tauw,Vr_exc,Vr_ini,EL;

  C=200.0;    //pF
  gL=12.0;    //nS
  VT=-50.0;   //mV
  DeltaT=2.0; //mV
  tauw=300.0; //ms
  EL=-70.0;
  Vr_exc=0.0;      // mV Potencial reverso exc
  Vr_ini=-80.0;      // mV Potencial reverso inh

  //////////////////////////EQUAÃ‡OES ACOPLADAS////////////////////*(Vr_exc-y[1+(i-1)*N])

  for(i=1;i<=NN;i++)
    {
      df[1+(i-1)*N]=(1.0/C)*(-gL*(y[1+(i-1)*N]-EL)+ gL*DeltaT*exp((y[1+(i-1)*N]-VT)/DeltaT)
                                         - y[2+(i-1)*N] +Iext[i]
                                            + g_exc*Gexc[i]*(Vr_exc-y[1+(i-1)*N])
                                             + g_ini*Gini[i]*(Vr_ini-y[1+(i-1)*N])
                                                +I[i]
                                                  );

      df[2+(i-1)*N]=(1.0/tauw)*(aa[i]*(y[1+(i-1)*N]-EL) - y[2+(i-1)*N]);
       }
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


double **dmatrix(long nrl, long nrh, long ncl, long nch)
{
   long i, nrow=nrh-nrl+1,ncol=nch-ncl+1;
   double **m;

   m=(double **) malloc((size_t)((nrow+NR_END)*sizeof(double*)));
   if (!m) nrerror("allocation failure 1 in matrix()");
   m += NR_END;
   m -= nrl;

   m[nrl]=(double *) malloc((size_t)((nrow*ncol+NR_END)*sizeof(double)));
   if (!m[nrl]) nrerror("allocation failure 2 in matrix()");
   m[nrl] += NR_END;
   m[nrl] -= ncl;

   for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;

   return m;
}

void free_dmatrix(double **m, long nrl, long nrh, long ncl, long nch)
{
   free((FREE_ARG) (m[nrl]+ncl-NR_END));
   free((FREE_ARG) (m+nrl-NR_END));
}
