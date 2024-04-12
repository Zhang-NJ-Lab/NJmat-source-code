/*getlatticeparm.h: Definition of
	(1) utility functions:	Max(); Min(); Rand();
	(2) mainly:	

	class GetLatticeParm(int spacegroup, double* latticeMins, double* latticeMaxes, int dimention, int choice, int celltype)
	
	bool GetLatticeParm::get(double* M)
	returns false if latticeMax < latticeMin or can't make specific latticeparm to 90/120 degree. 
	@M is the latticeparm matrix to return, but in arguements because one function only returns one value. 
	! IMPORTNT ! M=[ ax,bx,cx, 	0,by,cy,	0,0,cz ]  please note it is a transposed matrix compared to the commonly used latticeparm matrix such as in ASE.
	@latticeMaxes[6] and latticeMins[6] are cell constraints in format [a, b, c, alpha, beta, gamma].
	@spacegroup (number) of @dimention (2 or 3). If 2D, choice could be 0 (planegroup) or 1 (layergroup).
	@celltype (1-6) for primitive cell type if to get primitive cell.*/


#pragma once
#include <cstdlib> 
#include <cmath> 
#include <iostream> 
#define M_PI 3.14159265358979323846
using namespace std;

double Max(double a, double b) ;
double Max(double a, double b, double c) ;
double Max(double a, double b, double c, double d) ;

double Min(double a, double b) ;
double Min(double a, double b, double c) ;
double Min(double a, double b, double c, double d) ;

double Rand(void) //generate numbers between 0 and 1
{
	return 1.0*rand() / RAND_MAX;
};

/*Base class for primitive cell symmetry.*/
class BaseP
{
public:
	double *lcmax, *lcmin, *lpmax, *lpmin;				//latticemax/min for conventional cell/primitive cell.
	bool selfcheck;												//check nearly every thing. 

	BaseP() {selfcheck = true;};
	void Init(double * _lcmax, double* _lcmin, double* _lpmax, double* _lpmin) ;		//Init while makes sure the input cellmin < cellmax.
	virtual void get_range_lc() =0;						//get range for conventional cell.
	virtual void cal_lp(double *lc, double* lp) =0;				//calculate lattice_primitive from lattice_conventional. used in getlp.
	virtual ~BaseP() {};
	bool checklc() ;											//checks if latticemin > latticemax for conventional cell.

	bool checklp(double* lp) ;								//checks if the lp[6] satisfy the given condition.
	bool get_lp(double *lc, double* lp) ;				//cal_lp + checklp.
};
class GetLatticeParm
{
public:
	BaseP* basep;
	double *latticeMins, *latticeMaxes;
	double CcellMins[6], CcellMaxes[6];
	bool (*getlatticeparm) (double*, double*, double*) ;

	GetLatticeParm(int spg, double* Mins, double* Maxes, int _dimention, int _choice = 0, int _celltype =1) ;
	~GetLatticeParm() ;
	bool GetLattice(double* M);
};