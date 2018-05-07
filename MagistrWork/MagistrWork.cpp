// MagistrWork.cpp : Defines the entry point for the console application.
//
//решаем уравнение для цепочки джозефсоновских контактов (синус гордон)
//реализованы схемы кренка-николсона и явная схема (leapfrog) в качестве решения выдает 2 файла,
//где послойно (по времени) записаны значение функции в узлах по координате
//добавлен шум в оба метода, записывает результаты методов, вычисляет производную по координате,используем гсч из мкл
//
#include "stdafx.h"
#include<cmath>
#include<direct.h>
#include <locale>
#include<iostream>
#include<string>
#include <fstream>
#include <cstdio>
#include <ctime>
#include <omp.h>
#include <mkl_vsl.h>
using namespace std;

#define BRNG    VSL_BRNG_MT2203
#define METHOD  VSL_RNG_METHOD_GAUSSIAN_ICDF
#define PI 3.14159265358979323

#pragma region auxiliary functions
template<typename T>
void New(T **&f, int N, int M)
{
	if (f == NULL)
	{
		f = new T*[N];
		for (int i = 0; i < N; i++)
		{
			f[i] = new T[M];
			for (int j = 0; j < M; j++)
				f[i][j] = (T)0;
		}
	}
	else
		for (int i = 0; i < N; i++)
			for (int j = 0; j < M; j++)
				f[i][j] = (T)0;

}

template<typename T>
void Delete(T **&f, int N)
{
	for (int i = 0; i < N; i++)
	{
		delete[] f[i];
	}
	delete[]f;
}

void raz(double** result, double **a, double**b, int N, int M)
{
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < M; j++)
		{
			result[i][j] = a[i][j] - b[i][j];
		}
	}
}

void Print(string FileName, double** f, int _n, int _m, double dx, double dt)
{
	FILE* out;
	out = fopen(FileName.c_str(), "w");
	fprintf(out, "%.10f ", 0.0);
	for (int i = 0; i < _n; i++)
	{
		fprintf(out, "%.10f ", i*dx);
	}
	fprintf(out, "\n");
	for (int j = 0; j < _m; j++)
	{
		fprintf(out, "%.10f ", j*dt);
		for (int i = 0; i < _n; i++)
		{
			fprintf(out, "%.10f ", f[i][j]);
		}
		fprintf(out, "\n");
	}
	fclose(out);
}

void Print(string FileName, double* f, int _m, double dt)
{
	FILE* out;
	out = fopen(FileName.c_str(), "w");
	for (int j = 0; j < _m; j++)
	{
		fprintf(out, "%.10f %.10f\n", j*dt, f[j]);
	}
	fclose(out);
}

void d_dt(double** f, int _n, int _m, double _dt)
{
	for (int j = 0; j < _m - 1; j++)
	{
		for (int i = 0; i < _n; i++)
		{
			f[i][j] = (f[i][j + 1] - f[i][j]) / _dt;
		}
	}
}
void d_dt(double* f, int _m, double _dt)
{
	for (int j = 0; j < _m - 1; j++)
	{
		f[j] = -(f[j + 1] - f[j]) / _dt;
	}
}
void d_dx(double** f, int _n, int _m, double _dx)
{
	for (int j = 0; j < _m; j++)
	{
		for (int i = 0; i < _n - 1; i++)
		{
			f[i][j] = (f[i + 1][j] - f[i][j]) / _dx;
		}
	}
}
#pragma endregion

#pragma region Crank_Nicholson method
void CHM2_Crank_Nucolson(double **f, double I, double G1, double G2, double alpha, double V, double dt, double dx,
	int N, int M, double x0, double d, bool nu0 = true)
{
	int seed = time(NULL);
#pragma region mkl
	float *r = new float[N*M];
	int index = 0;
	VSLStreamStatePtr stream;
	vslNewStream(&stream, BRNG, seed + omp_get_thread_num());
	vsRngGaussian(METHOD, stream, N*M, r, 0.0, 1.0);
#pragma endregion
#pragma region koef
	double ksi = sqrt(2.0*d*alpha / (dt*dx));
	double Ai = -0.5 / (dx*dx), Ci = -(1.0 / (dt*dt) + 0.5*alpha / dt + 1.0 / (dx*dx)), Bi = -0.5 / (dx*dx), kap = 1.0;
	double knu = 1.0 / sqrt(1.0 - V*V);
	double* al, *bet;
	al = new double[N];
	bet = new double[N];
	double psi;
	double t2 = 1.0 / (dt*dt);
	double x2 = 0.5 / (dx*dx);
	double k1 = 0.5*alpha / dt;
#pragma endregion
	for (int i = 0; i < N; i++)
	{
		f[i][0] = (nu0) ? 0.0 : 4.0*atan(exp((i*dx - x0)*knu));
		f[i][1] = (nu0) ? 0.0 : 4.0*atan(exp(((i*dx - x0) - V*dt)*knu));
	}
	f[0][0] = f[1][0];
	f[0][1] = f[1][1];
	f[N - 1][0] = f[N - 2][0];
	f[N - 1][1] = f[N - 2][1];
	for (int j = 1; j < M - 1; j++)
	{
		al[1] = kap; bet[1] = 0.0;
		for (int i = 1; i < N - 1; i++)
		{
			psi = -(ksi*r[index] + I - sin(f[i][j]) + k1*f[i][j - 1] + x2*(f[i - 1][j - 1] - 2.0*f[i][j - 1] + f[i + 1][j - 1]) - t2*(-2.0*f[i][j] + f[i][j - 1]));
			al[i + 1] = Bi / (Ci - al[i] * Ai);
			bet[i + 1] = (psi + Ai*bet[i]) / (Ci - al[i] * Ai);
			index++;
		}
		f[N - 1][j + 1] = bet[N - 1] / (1 - al[N - 1]);
		for (int i = N - 1; i > 0; i--)
		{
			f[i - 1][j + 1] = al[i] * f[i][j + 1] + bet[i];
		}
	}

	vslDeleteStream(&stream);
	delete[]r;

}

void CHM2_Crank_Nucolson_Prob(double **P, double I, double G1, double G2, double alpha, double V, double dt, double dx,
	int N, int M, double x0, double d, int NAver, int NTH, int Niter, bool nu0 = true)
{
	int Nvect = (int)(NAver / Niter);
	Niter = (100 < NAver) ? (100) : NAver;
	for (int l = 0; l < Nvect; l++)
	{
		int seed = time(NULL);
		omp_set_num_threads(NTH);
#pragma omp parallel shared(P)
		{
#pragma region mkl
			int size_r = (int)(N*M*Niter / NTH) + 1;
			float *r = new float[size_r];
			int index = 0;
			VSLStreamStatePtr stream;
			vslNewStream(&stream, BRNG, seed + omp_get_thread_num());
			vsRngGaussian(METHOD, stream, size_r, r, 0.0, 1.0);
#pragma endregion
#pragma region koef
			double ksi = sqrt(2.0*d*alpha / (dt*dx));
			double Ai = -0.5 / (dx*dx), Ci = -(1.0 / (dt*dt) + 0.5*alpha / dt + 1.0 / (dx*dx)), Bi = -0.5 / (dx*dx), kap = 1.0;
			double knu = 1.0 / sqrt(1.0 - V*V);
			double* al, *bet;
			al = new double[N];
			bet = new double[N];
			double psi;
			double t2 = 1.0 / (dt*dt);
			double x2 = 0.5 / (dx*dx);
			double k1 = 0.5*alpha / dt;
			double invNaver = 1.0 / NAver;
			double **f = NULL;
			New(f, N, 3);
#pragma endregion
#pragma omp for
			for (int s = 0; s < Niter; s++) {
				for (int i = 0; i < N; i++)
				{
					f[i][0] = (nu0) ? 0.0 : 4.0*atan(exp((i*dx - x0)*knu));//NU

					if ((-PI < f[i][0]) && (f[i][0] < PI))
					{
						P[i][0] += invNaver;
					}
					f[i][1] = (nu0) ? 0.0 : 4.0*atan(exp(((i*dx - x0) - V*dt)*knu));//NU
					if ((-PI < f[i][1]) && (f[i][1] < PI))
					{
						P[i][1] += invNaver;
					}
				}
				f[0][0] = f[1][0];
				f[0][1] = f[1][1];
				f[N - 1][0] = f[N - 2][0];
				f[N - 1][1] = f[N - 2][1];
				for (int j = 1; j < M - 1; j++)
				{
					al[1] = kap; bet[1] = 0.0;
					for (int i = 1; i < N - 1; i++)
					{
						psi = -(ksi*r[index] + I - sin(f[i][1]) + k1*f[i][0] + x2*(f[i - 1][0] - 2.0*f[i][0] + f[i + 1][0]) - t2*(-2.0*f[i][1] + f[i][0]));
						al[i + 1] = Bi / (Ci - al[i] * Ai);
						bet[i + 1] = (psi + Ai*bet[i]) / (Ci - al[i] * Ai);
						index++;
					}
					f[N - 1][2] = bet[N - 1] / (1 - al[N - 1]);
					if ((-PI < f[N - 1][2]) && (f[N - 1][2] < PI))
					{
						P[N - 1][j + 1] += invNaver;
					}
					for (int i = N - 1; i > 0; i--)
					{
						f[i - 1][2] = al[i] * f[i][2] + bet[i];
						if ((-PI < f[i - 1][2]) && (f[i - 1][2] < PI))
						{
							P[i - 1][j + 1] += invNaver;
						}
						f[i][0] = f[i][1];
						f[i][1] = f[i][2];
					}
					f[0][0] = f[0][1];
					f[0][1] = f[0][2];
				}
			}
			vslDeleteStream(&stream);
			Delete(f, N);
			delete[]r;
		}
	}
}

void CHM2_Crank_Nucolson_Prob(double *P, double I, double G1, double G2, double alpha, double V, double dt, double dx,
	int N, int M, double x0, double d, int NAver, int NTH, int Niter)
{
	int Nvect = (int)(NAver / Niter);
	Niter = (Niter < NAver) ? (Niter) : NAver;
	for (int l = 0; l < Nvect; l++)
	{
		int seed = time(NULL);
		omp_set_num_threads(NTH);
#pragma omp parallel shared(P)
		{
#pragma region mkl
			int size_r = (int)(N*M*Niter / NTH) + 1;
			float *r = new float[size_r];
			int index = 0;
			VSLStreamStatePtr stream;
			vslNewStream(&stream, BRNG, seed + omp_get_thread_num());
			vsRngGaussian(METHOD, stream, size_r, r, 0.0, 1.0);
#pragma endregion
#pragma region koef
			double ksi = sqrt(2.0*d*alpha / (dt*dx));
			double Ai = -0.5 / (dx*dx), Ci = -(1.0 / (dt*dt) + 0.5*alpha / dt + 1.0 / (dx*dx)), Bi = -0.5 / (dx*dx), kap = 1.0;
			double knu = 1.0 / sqrt(1.0 - V*V);
			double* al, *bet;
			al = new double[N];
			bet = new double[N];
			double psi;
			double t2 = 1.0 / (dt*dt);
			double x2 = 0.5 / (dx*dx);
			double k1 = 0.5*alpha / dt;
			double invNaver = 1.0 / (N*NAver);
			double **f = NULL;
			New(f, N, 3);
#pragma endregion
#pragma omp for
			for (int s = 0; s < Niter; s++) {
				for (int i = 0; i < N; i++)
				{
					f[i][0] = 4.0*atan(exp((i*dx - x0)*knu));//NU

					if ((-PI < f[i][0]) && (f[i][0] < PI))
					{
						P[0] += invNaver;
					}
					f[i][1] = 4.0*atan(exp(((i*dx - x0) - V*dt)*knu));//NU
					if ((-PI < f[i][1]) && (f[i][1] < PI))
					{
						P[1] += invNaver;
					}
				}
				f[0][0] = f[1][0];
				f[0][1] = f[1][1];
				f[N - 1][0] = f[N - 2][0];
				f[N - 1][1] = f[N - 2][1];
				for (int j = 1; j < M - 1; j++)
				{
					al[1] = kap; bet[1] = 0.0;
					for (int i = 1; i < N - 1; i++)
					{
						psi = -(ksi*r[index] + I - sin(f[i][1]) + k1*f[i][0] + x2*(f[i - 1][0] - 2.0*f[i][0] + f[i + 1][0]) - t2*(-2.0*f[i][1] + f[i][0]));
						al[i + 1] = Bi / (Ci - al[i] * Ai);
						bet[i + 1] = (psi + Ai*bet[i]) / (Ci - al[i] * Ai);
						index++;
					}
					f[N - 1][2] = bet[N - 1] / (1 - al[N - 1]);
					if ((-PI < f[N - 1][2]) && (f[N - 1][2] < PI))
					{
						P[j + 1] += invNaver;
					}
					for (int i = N - 1; i > 0; i--)
					{
						f[i - 1][2] = al[i] * f[i][2] + bet[i];
						if ((-PI < f[i - 1][2]) && (f[i - 1][2] < PI))
						{
							P[j + 1] += invNaver;
						}
						f[i][0] = f[i][1];
						f[i][1] = f[i][2];
					}
					f[0][0] = f[0][1];
					f[0][1] = f[0][2];
				}
			}
			vslDeleteStream(&stream);
			Delete(f, N);
			delete[]r;
		}
	}
}

void CHM2_Crank_Nucolson_Prob_DynamicI(double *P, double v_I, double G1, double G2, double alpha, double V, double dt, double dx,
	int N, int M, double x0, double d, int NAver, int NTH, int Niter)
{
	int Nvect = (int)(NAver / Niter);
	Niter = (Niter < NAver) ? (Niter) : NAver;
	for (int l = 0; l < Nvect; l++)
	{
		int seed = time(NULL);
		omp_set_num_threads(NTH);
#pragma omp parallel shared(P)
		{
#pragma region mkl
			int size_r = (int)(N*M*Niter / NTH) + 1;
			float *r = new float[size_r];
			int index = 0;
			VSLStreamStatePtr stream;
			vslNewStream(&stream, BRNG, seed + omp_get_thread_num());
			vsRngGaussian(METHOD, stream, size_r, r, 0.0, 1.0);
#pragma endregion
#pragma region koef
			double ksi = sqrt(2.0*d*alpha / (dt*dx));
			double Ai = -0.5 / (dx*dx), Ci = -(1.0 / (dt*dt) + 0.5*alpha / dt + 1.0 / (dx*dx)), Bi = -0.5 / (dx*dx), kap = 1.0;
			double knu = 1.0 / sqrt(1.0 - V*V);
			double* al, *bet;
			al = new double[N];
			bet = new double[N];
			double psi;
			double I = 0.0;
			double t2 = 1.0 / (dt*dt);
			double x2 = 0.5 / (dx*dx);
			double k1 = 0.5*alpha / dt;
			double invNaver = 1.0 / (N*NAver);
			double **f = NULL;
			New(f, N, 3);
#pragma endregion
#pragma omp for
			for (int s = 0; s < Niter; s++) {
				for (int i = 0; i < N; i++)
				{
					f[i][0] = 0.0; // 4.0*atan(exp((i*dx - x0)*knu));//NU

					if ((-PI < f[i][0]) && (f[i][0] < PI))
					{
						P[0] += invNaver;
					}
					f[i][1] = 0.0;// 4.0*atan(exp(((i*dx - x0) - V*dt)*knu));//NU
					if ((-PI < f[i][1]) && (f[i][1] < PI))
					{
						P[1] += invNaver;
					}
				}
				f[0][0] = f[1][0];
				f[0][1] = f[1][1];
				f[N - 1][0] = f[N - 2][0];
				f[N - 1][1] = f[N - 2][1];
				for (int j = 1; j < M - 1; j++)
				{
					al[1] = kap; bet[1] = 0.0;
					I = v_I*dt*j;
					for (int i = 1; i < N - 1; i++)
					{
						psi = -(ksi*r[index] + I - sin(f[i][1]) + k1*f[i][0] + x2*(f[i - 1][0] - 2.0*f[i][0] + f[i + 1][0]) - t2*(-2.0*f[i][1] + f[i][0]));
						al[i + 1] = Bi / (Ci - al[i] * Ai);
						bet[i + 1] = (psi + Ai*bet[i]) / (Ci - al[i] * Ai);
						index++;
					}
					f[N - 1][2] = bet[N - 1] / (1 - al[N - 1]);
					if ((-PI < f[N - 1][2]) && (f[N - 1][2] < PI))
					{
						P[j + 1] += invNaver;
					}
					for (int i = N - 1; i > 0; i--)
					{
						f[i - 1][2] = al[i] * f[i][2] + bet[i];
						if ((-PI < f[i - 1][2]) && (f[i - 1][2] < PI))
						{
							P[j + 1] += invNaver;
						}
						f[i][0] = f[i][1];
						f[i][1] = f[i][2];
					}
					f[0][0] = f[0][1];
					f[0][1] = f[0][2];
				}
			}
			vslDeleteStream(&stream);
			Delete(f, N);
			delete[]r;
		}
	}
}
#pragma endregion

#pragma region Leapfrog method
void CHM1_explicit(double **f, double I, double G1, double G2, double alpha, double V, double dt, double dx,
	int N, int M, double x0, double d)
{
	int seed = time(NULL);
#pragma region mkl
	float *r = new float[N*M];
	int index = 0;
	VSLStreamStatePtr stream;
	vslNewStream(&stream, BRNG, seed + omp_get_thread_num());
	vsRngGaussian(METHOD, stream, N*M, r, 0.0, 1.0);
#pragma endregion
#pragma region koef
	double knu = 1.0 / sqrt(1.0 - V*V);
	double t2 = 1.0 / (dt*dt);
	double x2 = 1.0 / (dx*dx);
	double k1 = 2.0*dt*dt / (2.0 + alpha*dt);
	double k2 = 0.5*alpha / dt;
	double ksi = sqrt(2.0*d*alpha / (dt*dx));
#pragma endregion
	for (int i = 0; i < N; i++)
	{
		f[i][0] = 0.0;//4.0*atan(exp((i*dx - x0)*knu));
		f[i][1] = 0.0;//4.0*atan(exp(((i*dx - x0) - V*dt)*knu));
	}
	f[0][0] = f[1][0];
	f[0][1] = f[1][1];
	f[N - 1][0] = f[N - 2][0];
	f[N - 1][1] = f[N - 2][1];
	for (int j = 1; j < M - 1; j++)
	{
		f[0][j] = f[1][j];
		f[N - 1][j] = f[N - 2][j];
		for (int i = 1; i < N - 1; i++)
		{
			f[i][j + 1] = (ksi*(double)r[index] + I + x2*(f[i + 1][j] - 2.0*f[i][j] + f[i - 1][j]) - sin(f[i][j]) + k2*f[i][j - 1] - t2*(-2.0*f[i][j] + f[i][j - 1]))*k1;
			index++;
		}
	}
	f[0][M - 1] = f[1][M - 1];
	f[N - 1][M - 1] = f[N - 2][M - 1];
	vslDeleteStream(&stream);
	delete[]r;
}

void CHM1_explicit_Prob(double **P, double I, double G1, double G2, double alpha, double V, double dt, double dx,
	int N, int M, double x0, double d, int NAver, int NTH, int Niter)
{
	int Nvect = (int)(NAver / Niter);
	Niter = (100 < NAver) ? (100) : NAver;
	for (int l = 0; l < Nvect; l++)
	{
		int seed = time(NULL);
		omp_set_num_threads(NTH);
#pragma omp parallel shared(P)
		{
#pragma region mkl
			int size_r = (int)(N*M*Niter / NTH) + 1;
			float *r = new float[size_r];
			int index = 0;
			VSLStreamStatePtr stream;
			vslNewStream(&stream, BRNG, seed + omp_get_thread_num());
			vsRngGaussian(METHOD, stream, size_r, r, 0.0, 1.0);
#pragma endregion
#pragma region koef
			double knu = 1.0 / sqrt(1.0 - V*V);
			double t2 = 1.0 / (dt*dt);
			double x2 = 1.0 / (dx*dx);
			double k1 = 2.0*dt*dt / (2.0 + alpha*dt);
			double k2 = 0.5*alpha / dt;
			double ksi = sqrt(2.0*d*alpha / (dt*dx));
			double invNaver = 1.0 / NAver;
#pragma endregion
			double **f = NULL;
			New(f, N, 3);
#pragma omp for
			for (int s = 0; s < Niter; s++)
			{

				for (int i = 0; i < N; i++)
				{
					f[i][0] = 0.0;// 4.0*atan(exp((i*dx - x0)*knu));//NU

					if ((-PI < f[i][0]) && (f[i][0] < PI))
					{
						P[i][0] += invNaver;
					}
					f[i][1] = 0.0;// 4.0*atan(exp(((i*dx - x0) - V*dt)*knu));//NU
					if ((-PI < f[i][1]) && (f[i][1] < PI))
					{
						P[i][1] += invNaver;
					}
				}
				f[0][0] = f[1][0];
				f[0][1] = f[1][1];
				f[N - 1][0] = f[N - 2][0];
				f[N - 1][1] = f[N - 2][1];
				for (int j = 1; j < M - 1; j++)
				{
					f[1][2] = (ksi*(double)r[index] + I + x2*(f[2][1] - 2.0*f[1][1] + f[0][1]) - sin(f[1][1]) + k2*f[1][0] - t2*(-2.0*f[1][1] + f[1][0]))*k1;
					index++;
					f[0][2] = f[1][2];
					if ((-PI < f[0][2]) && (f[0][2] < PI))
					{
						P[0][j + 1] += invNaver;
						P[1][j + 1] += invNaver;
					}
					f[0][0] = f[0][1];
					f[0][1] = f[0][2];

					for (int i = 2; i < N - 1; i++)
					{
						f[i][2] = (ksi*(double)r[index] + I + x2*(f[i + 1][1] - 2.0*f[i][1] + f[i - 1][1]) - sin(f[i][1]) + k2*f[i][0] - t2*(-2.0*f[i][1] + f[i][0]))*k1;
						index++;
						if ((-PI < f[i][2]) && (f[i][2] < PI))
						{
							P[i][j + 1] += invNaver;
						}
						f[i - 1][0] = f[i - 1][1];
						f[i - 1][1] = f[i - 1][2];
					}
					f[N - 2][0] = f[N - 2][1];
					f[N - 2][1] = f[N - 2][2];
					f[N - 1][2] = f[N - 2][2];
					if ((-PI < f[N - 1][2]) && (f[N - 1][2] < PI))
					{
						P[N - 1][j + 1] += invNaver;
					}
					f[N - 1][0] = f[N - 1][1];
					f[N - 1][1] = f[N - 1][2];
				}
			}
			vslDeleteStream(&stream);
			Delete(f, N);
			delete[]r;
		}
	}
}

void CHM1_explicit_Prob(double *P, double I, double G1, double G2, double alpha, double V, double dt, double dx,
	int N, int M, double x0, double d, int NAver, int NTH)
{
	int Niter = 100,
		Nvect = (int)(NAver / Niter);
	Niter = (Niter < NAver) ? (Niter) : NAver;
	for (int l = 0; l < Nvect; l++)
	{
		int seed = time(NULL);
		omp_set_num_threads(NTH);
#pragma omp parallel shared(P)
		{
#pragma region mkl
			int size_r = (int)(N*M*Niter / NTH) + 1;
			cout << size_r << ' ';
			float *r = static_cast<float*>(_mm_malloc(size_r * sizeof(float), 32)); //new float[size_r];
			if (r == NULL)
			{
				cout << "fail" << ' ';
				system("pause");
			}
			int index = 0;
			VSLStreamStatePtr stream;
			vslNewStream(&stream, BRNG, seed + omp_get_thread_num());
			vsRngGaussian(METHOD, stream, size_r, r, 0.0, 1.0);
#pragma endregion
#pragma region koef
			double knu = 1.0 / sqrt(1.0 - V*V);
			double t2 = 1.0 / (dt*dt);
			double x2 = 1.0 / (dx*dx);
			double k1 = 2.0*dt*dt / (2.0 + alpha*dt);
			double k2 = 0.5*alpha / dt;
			double ksi = sqrt(2.0*d*alpha / (dt*dx));
			double invNaver = 1.0 / (NAver*N);
#pragma endregion
			double **f = NULL;
			New(f, N, 3);
#pragma omp for
			for (int s = 0; s < Niter; s++)
			{

				for (int i = 0; i < N; i++)
				{
					f[i][0] = 0.0;// 4.0*atan(exp((i*dx - x0)*knu));//NU

					if ((-PI < f[i][0]) && (f[i][0] < PI))
					{
						P[0] += invNaver;
					}
					f[i][1] = 0.0;// 4.0*atan(exp(((i*dx - x0) - V*dt)*knu));//NU
					if ((-PI < f[i][1]) && (f[i][1] < PI))
					{
						P[1] += invNaver;
					}
				}
				f[0][0] = f[1][0];
				f[0][1] = f[1][1];
				f[N - 1][0] = f[N - 2][0];
				f[N - 1][1] = f[N - 2][1];
				for (int j = 1; j < M - 1; j++)
				{
					f[1][2] = (ksi*(double)r[index] + I + x2*(f[2][1] - 2.0*f[1][1] + f[0][1]) - sin(f[1][1]) + k2*f[1][0] - t2*(-2.0*f[1][1] + f[1][0]))*k1;
					index++;
					f[0][2] = f[1][2];
					if ((-PI < f[0][2]) && (f[0][2] < PI))
					{
						P[j + 1] += invNaver;
						P[j + 1] += invNaver;
					}
					f[0][0] = f[0][1];
					f[0][1] = f[0][2];

					for (int i = 2; i < N - 1; i++)
					{
						f[i][2] = (ksi*(double)r[index] + I + x2*(f[i + 1][1] - 2.0*f[i][1] + f[i - 1][1]) - sin(f[i][1]) + k2*f[i][0] - t2*(-2.0*f[i][1] + f[i][0]))*k1;
						index++;
						if ((-PI < f[i][2]) && (f[i][2] < PI))
						{
							P[j + 1] += invNaver;
						}
						f[i - 1][0] = f[i - 1][1];
						f[i - 1][1] = f[i - 1][2];
					}
					f[N - 2][0] = f[N - 2][1];
					f[N - 2][1] = f[N - 2][2];
					f[N - 1][2] = f[N - 2][2];
					if ((-PI < f[N - 1][2]) && (f[N - 1][2] < PI))
					{
						P[j + 1] += invNaver;
					}
					f[N - 1][0] = f[N - 1][1];
					f[N - 1][1] = f[N - 1][2];
				}
			}
			vslDeleteStream(&stream);
			Delete(f, N);
			_mm_free(r);//delete[]r;
		}
	}
}
#pragma endregion

int main(int argc, char* argv[])
{

#pragma region parametrs
	//объявляются все необходимые параметры и значения поумолчанию
	double A = 0.0, B = 10.0, T = 30.0;
	double alpha = 0.0;
	double dt = 0.001, dx = 0.05;
	double t0 = 0.0, x0 = 0.0;
	double I = 0.0;
	double G1 = 0, G2 = 0;
	double V = 0.0;
	double d = 0.0;
	int Naver = 100, Niter = 100;
	string fname = "default.txt";
#pragma endregion

#pragma region input parametrs
	//получение значений параметров из аргументов
	try
	{
		if (argc > 1)
		{
			fname = argv[1];
			if (fname == "testmod")
			{
				cout << "Input FILENAME for save results: ";
				cin >> fname;
				fname += ".txt";
				cout << endl;
				cout << "If you want to change argument then input it number else input 0: ";
				int arg = 0;
				cin >> arg;
				cout << endl;
				while (arg != 0)
				{
					cout << "Current value: " << argv[arg] << endl << "New value: ";
					cin >> argv[arg];
					cout << endl << "Number any arg: ";
					cin >> arg;
					cout << endl;
				}
			}

			A = atof(argv[2]),
				B = atof(argv[3]),
				T = atof(argv[4]);
			//cout << T;
			x0 = atof(argv[5]);
			alpha = atof(argv[6]);
			dt = atof(argv[7]), dx = atof(argv[8]);
			I = atof(argv[9]);
			d = atof(argv[10]);
			Naver = atoi(argv[11]);
			Niter = atoi(argv[12]);
		}
	}
	catch (exception e)
	{
		cout << e.what() << endl;
	}
	if (Naver == 0)
	{
		system("Pause");
		return 0;
	}
#pragma endregion

#pragma region setting save
	string catalog = "result";
	mkdir(catalog.c_str());
	catalog += "/";
	int i_ = 0;
	while ((fname[i_] != '.') && (fname[i_] != '\0'))
	{
		catalog += fname[i_];
		i_++;
	}
	mkdir(catalog.c_str());
#pragma endregion

#pragma region calculations
	double**f = NULL, **Po = NULL, **P = NULL, *P1;
	int N, M = 0;
	int start, stop, worktime1, worktime2;
	M = (int)(T / dt) + 1;
	N = (int)((B - A) / dx) + 1;
	cout << M << ' ' << N << endl;

	//CHM1_explicit(f, I, G1, G2, alpha, V, dt, dx, N, M, x0, d);//f(x,t)
	//Print(catalog + "/exp_F_" + fname, f, N, M, dx, dt);
	/*New(f, N, M);
	start = time(NULL);
	CHM2_Crank_Nucolson(f, I, G1, G2, alpha, V, dt, dx, N, M, x0, d);
	stop = time(NULL);
	worktime1 = stop - start;
	Print(catalog + "/crn_F_nu0" + fname, f, N, M, dx, dt);
	CHM2_Crank_Nucolson(f, I, G1, G2, alpha, V, dt, dx, N, M, x0, d, false);
	Print(catalog + "/crn_F_" + fname, f, N, M, dx, dt);
	Delete(f, N);*/
	//d_dx(f, N, M, dx);//df(x,t)/dx
	//Print(catalog + "/crn_dFdx_" + fname, f, N, M, dx, dt);
	/*New(P, N, M);
	start = time(NULL);
	CHM2_Crank_Nucolson_Prob(P, I, G1, G2, alpha, V, dt, dx, N, M, x0, d, Naver, 2, Niter);
	stop = time(NULL);
	worktime2 = stop - start;
	Print(catalog + "/crn_p_nu0" + fname, P, N, M, dx, dt);
	New(P, N, M);
	CHM2_Crank_Nucolson_Prob(P, I, G1, G2, alpha, V, dt, dx, N, M, x0, d, Naver, 2, Niter, false);
	Print(catalog + "/crn_p_" + fname, P, N, M, dx, dt);
	Delete(P, N);*/
	//New(P, N, M);
	//CHM1_explicit_Prob(P, I, G1, G2, alpha, V, dt, dx, N, M, x0, d, Naver, 2, Niter);
	//Print(catalog + "/exp_p_" + fname, P, N, M, dx, dt);
	//Delete(P, N);

	P1 = new double[M];
	for (int i = 0; i < M; i++)
		P1[i] = 0.0;
	start = time(NULL);
	CHM2_Crank_Nucolson_Prob_DynamicI(P1, I, G1, G2, alpha, V, dt, dx, N, M, x0, d, Naver, 2, Niter);
	stop = time(NULL);
	Print(catalog + "/crn_p_" + fname, P1, M, dt);

	d_dt(P1, M, dt);
	Print(catalog + "/crn_dp_" + fname, P1, M, dt);

	worktime1 = stop - start;
	delete[]P1;
#pragma endregion

#pragma region info
	FILE* out;
	string ni = catalog + "/info.txt";
	out = fopen(ni.c_str(), "w");
	fprintf(out, "Имя файла: %s\n", fname.c_str());
	fprintf(out, "Границы по координате x : %f %f\nВерхняя граница по времени : %f\n", A, B, T);
	fprintf(out, "Начальное смещение х0 : %f\nКоэффициент альфа : %f\nШаг интегрирования по t, по x : %f, %f\n", x0, alpha, dt, dx);
	fprintf(out, "Скорость изменения токов : %f\nИнтенсивность флуктуаций : %f\nУсреднение : %d\nV : %f\n", I, d, Naver, V);
	fprintf(out, "Время работы вычисления вероятности: crn - %d", worktime1);
	fclose(out);
#pragma endregion
	return 0;
}

