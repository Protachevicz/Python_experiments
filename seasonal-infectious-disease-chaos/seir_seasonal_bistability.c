//Modelo SEIR com sazonalidade
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define N 100000000
#define NP 1000//tamanho da populaÃ§Ã£o
#define PI 3.14159265359
#define b 0.02 //taxa de nascimento
#define d 0.02 //taxa de mortalidade
#define alfa 130.98221 //taxa de e -> i
//~ #define alfacritico 130.98221//taxa de e -> i
#define gama 100 //taxa de i -> r
#define omega 2.0*PI
#define delta 0.25
#define beta1 0.28
//~ #define beta0 1800.0
#define transiente 30000000
FILE *input;

double sus(double t, double s, double e, double i, double r); //susceptible
double expo(double t, double s, double e, double i, double r); //exposed
double inf(double t, double s, double e, double i, double r); //infective
double per(double t, double s, double e, double i, double r); //periodo

int main(){
	int n,conts, contr, conti,conte,atrator;;
	double s,e,i,r,t, h,ds,de,di,T,dT,beta0;
	double s0,e0,i0,r0, t0,T0;
	double ks2,ks3,ks4, ke2,ke3,ke4, ki2,ki3,ki4, kT2,kT3,kT4;

	input = fopen("caos_p1_alfa130.88200.dat","wt");

	//~ int variable;
	//~ char output_filename[200];
	//~ scanf("%d", &variable);
	//~ sprintf(output_filename,"131.00000-_%d.dat",variable);
	//~ input=fopen(output_filename,"wt");

	h = 0.001;
	t0 = 0.0;
	T0 = 0.0;
	//condiÃ§Ã£o inicial do caÃ³tico
	e0 = 145.000000;
	i0 = 185.000000;
	r0 = 595.000000;
	s0 = 75.000000;
	//condiÃ§Ã£o inicial do periÃ³dico
	//~ e0 = 144.000000;
	//~ i0 = 186.000000;
	//~ r0 = 596.000000;
	//~ s0 = 74.000000;


	e = e0;
	i = i0;
	r = r0;
	s = s0;
	t = t0;
	T = T0;
	//~ fprintf(input,"%.4f %.4f %.4f %.4f %.4f \n", t0, s0, e0, i0, r0);

	for(beta0=270.0;beta0<=270.0;beta0=beta0+0.01){

	//~ fprintf(input,"%.4f %.4f %.4f %.4f %.4f \n", t, s, e, i, r);

	double sus(double t, double s, double e, double i){
		return(b*NP  - d*s - beta0*(1.0 + beta1*cos(T))*s*i/NP + delta*r);
	}

	double expo(double t, double s, double e, double i){
		return(beta0*(1.0 + beta1*cos(T))*s*i/NP - (alfa+d)*e);
	}

	double inf(double t, double s, double e, double i){
		return(alfa*e - (gama + d)*i);
	}

	double per(double t, double s, double e, double i){
		return(omega);
	}

	for(n=1;n<=N;n++){
		/**inicio das soluÃ§Ãµes**/
		ds = sus(t, s, e, i);
		de = expo(t, s, e, i);
		di = inf(t, s, e, i);
		dT = per(t, s, e, i);

		ks2 = sus(t + h/2.0, s + h*ds/2.0, e + h*de/2.0, i + h*di/2.0);
		ke2 = expo(t + h/2.0, s + h*ds/2.0, e + h*de/2.0, i + h*di/2.0);
		ki2 = inf(t + h/2.0, s + h*ds/2.0, e + h*de/2.0, i + h*di/2.0);
		kT2 = per(t + h/2.0, s + h*ds/2.0, e + h*de/2.0, i + h*di/2.0);

		ks3 = sus(t + h/2.0, s + h*ks2/2.0, e + h*ke2/2.0, i + h*ki2/2.0);
		ke3 = expo(t + h/2.0, s + h*ks2/2.0, e + h*ke2/2.0, i + h*ki2/2.0);
		ki3 = inf(t + h/2.0, s + h*ks2/2.0, e + h*ke2/2.0, i + h*ki2/2.0);
		kT3 = per(t + h/2.0, s + h*ks2/2.0, e + h*ke2/2.0, i + h*ki2/2.0);

		ks4 = sus(t + h, s + h*ks3, e + h*ke3, i + h*ki3);
		ke4 = expo(t + h, s + h*ks3, e + h*ke3, i + h*ki3);
		ki4 = inf(t + h, s + h*ks3, e + h*ke3, i + h*ki3);
		kT4 = per(t + h, s + h*ks3, e + h*ke3, i + h*ki3);

		//solucoes
		s = s + h*(ds + 2.0*ks2 + 2.0*ks3 + ks4)/6.0;
		e = e + h*(de + 2.0*ke2 + 2.0*ke3 + ke4)/6.0;
		i = i + h*(di + 2.0*ki2 + 2.0*ki3 + ki4)/6.0;
		T = T + h*(dT + 2.0*kT2 + 2.0*kT3 + kT4)/6.0;
		r = NP - s - e - i;

		t = t + h;

		if(n>=transiente){
			if(cos(T)>0.9999950000){
				fprintf(input,"%.6f %.6f %.6f %.6f\n", s, e, i, r);
				}
		}

		/**final das soluÃ§Ãµes**/

		}

	}
	fclose(input);
}
