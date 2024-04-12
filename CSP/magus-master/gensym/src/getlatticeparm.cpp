#include <cstdlib> 
#include "getlatticeparm.h"
using namespace std;

#define M_PI 3.14159265358979323846

double tol = 1e-2;

double Max(double a, double b) 
{
	if (a > b) return a;
	else return b;
};
double Max(double a, double b, double c)
{
	if (a > b)
	{
		if (a > c) return a;
		return c;
	}
	if (b > c) return b;
	return c;
};
double Max(double a, double b, double c, double d)
{
	return Max(Max(a,b,c), d);
};

double Min(double a, double b)
{
	if (a < b) return a;
	return b;
};
double Min(double a, double b, double c)
{
	if (a < b)
	{
		if (a < c) return a;
		return c;
	}
	if (b < c) return b;
	return c;
};
double Min(double a, double b, double c, double d)
{
	return Min(Min(a,b,c), d);
};
ostream & operator<<(ostream &out, double* &latticeparm)
{
	for (int i=0;i<3;i++)	out<<latticeparm[i]<<'\t';
	for (int i=3;i<6;i++)	out<<latticeparm[i]/M_PI *180<<'\t';
	out<<endl;
	return out;
};
bool ChooseTriclinic(double* latticeparm, double* maxes, double* mins)
{
	for (int i = 0; i < 6; i++) latticeparm[i] = Rand()*(maxes[i] - mins[i]) + mins[i];
	//a1!=a2!=a3;alpha!=beta!=gamma
	return true;
};
bool Monoclinic(double* latticeparm, double* maxes, double* mins, int uniqueangle)
{
	for (int i = 0; i < 6; i++) latticeparm[i] = Rand()*(maxes[i] - mins[i]) + mins[i];
	for (int i = 3; i < 6; i++)
	{
		if(i==uniqueangle) {}
		else
		{
			if(maxes[i]< M_PI / 2 - tol || mins[i]> M_PI / 2 + tol) return false;
			latticeparm[i] = M_PI / 2;
		}
	}
	//a1!=a2!=a3;alpha=gamma=pi/2,!=beta
	//for 3D monos, unique angle is beta[4]. for 2D monos with layergroup 3-7, gamma[5]; with layergroup 8-18, alpha[3].
	return true;
};
bool ChooseMonoclinic_alpha(double* latticeparm, double* maxes, double* mins)
{
	return Monoclinic(latticeparm, maxes, mins, 3);
};//ChooseMonoclinic with unique angle alpha
bool ChooseMonoclinic_beta(double* latticeparm, double* maxes, double* mins)
{
	return Monoclinic(latticeparm, maxes, mins, 4);
};//ChooseMonoclinic with unique angle beta
bool ChooseMonoclinic_gamma(double* latticeparm, double* maxes, double* mins)
{
	return Monoclinic(latticeparm, maxes, mins, 5);
};//ChooseMonoclinic with unique angle gamma
bool ChooseOrthorhombic(double* latticeparm, double* maxes, double* mins)
{
	for (int i = 0; i < 3; i++) latticeparm[i] = Rand()*(maxes[i] - mins[i]) + mins[i];
	for (int i = 3; i < 6; i++) if(maxes[i]< M_PI / 2 - tol || mins[i]> M_PI / 2 + tol ) return false;
	for (int i = 3; i < 6; i++) latticeparm[i] = M_PI / 2;
	//a1!=a2!=a3;alpha=beta=gamma=pi/2
	return true;
};
bool ChooseTetragonal(double* latticeparm, double* maxes, double* mins)
{
	double t1 = Max(mins[0], mins[1]);
	double t2 = Min(maxes[0], maxes[1]);
	if(t1 > t2 + tol ) return false;
	latticeparm[0] = Rand()*(t2 - t1) + t1;
	latticeparm[1] = latticeparm[0];
	latticeparm[2] = Rand()*(maxes[2] - mins[2]) + mins[2];
	for (int i = 3; i < 6; i++) if(maxes[i]< M_PI / 2 - tol || mins[i]> M_PI / 2 + tol ) return false;
	for (int i = 3; i < 6; i++) latticeparm[i] = M_PI / 2;
	//a1=a2!=a3;alpha=beta=gamma=pi/2
	return true;
};
bool ChooseTrigonal(double* latticeparm, double* maxes, double* mins)
{
	double t1 = Max(mins[0], mins[1]);
	double t2 = Min(maxes[0], maxes[1]);
	if(t1 > t2 + tol) return false;
	latticeparm[0] = Rand()*(t2 - t1) + t1;
	latticeparm[1] = latticeparm[0];
	latticeparm[2] = Rand()*(maxes[2] - mins[2]) + mins[2];

	if(maxes[3]< M_PI / 2 - tol || mins[3]> M_PI / 2 + tol ) return false;
	if(maxes[4]< M_PI / 2 - tol || mins[4]> M_PI / 2 + tol ) return false;
	if(maxes[5]< 2 * M_PI / 3 - tol || mins[5]> 2 * M_PI / 3 + tol ) return false;

	latticeparm[3] = M_PI / 2;
	latticeparm[4] = M_PI / 2;
	latticeparm[5] = 2 * M_PI / 3;
	//a1=a2!=a3;alpha=beta=pi/2,gamma=2pi/3	
	return true;
};
bool ChooseCubic(double* latticeparm, double* maxes, double* mins)
{
	double t1 = Max(mins[0], mins[1], mins[2]);
	double t2 = Min(maxes[0], maxes[1], maxes[2]);
	if(t1>t2) return false;
	latticeparm[0] = Rand()*(t2 - t1) + t1;
	latticeparm[1] = latticeparm[0];
	latticeparm[2] = latticeparm[0];

	for (int i = 3; i < 6; i++) if(maxes[i]< M_PI / 2 - tol || mins[i]> M_PI / 2 + tol ) return false;
	for (int i = 3; i < 6; i++) latticeparm[i] = M_PI / 2;
	//a1=a2=a3;alpha=beta=gamma=pi/2
	return true;
};
void latticetrans(double* M, double * latticeparm)
{
	for (int i = 0; i < 9; i++) M[i] = 0;
	M[0] = latticeparm[0]; //ax
	M[1] = latticeparm[1] * cos(latticeparm[5]);//bx
	M[4] = latticeparm[1] * sin(latticeparm[5]);//by
	M[2] = latticeparm[2] * cos(latticeparm[4]);//cx
	M[5] = (latticeparm[2] * latticeparm[1] * cos(latticeparm[3]) - M[2] * M[1]) / M[4];//cy
	M[8] = sqrt(pow(latticeparm[2], 2) - M[2] * M[2] - M[5] * M[5]);//cz
	//M=[[ax,bx,cx],[0,by,cy],[0,0,cz]]  
	for (int i = 0; i < 9; i++) if (fabs(M[i]) < 1e-6) M[i] = 0;
};
//notes here:
//alpha=b^c, beta=a^c, gamma=a^b

typedef bool (*chosecell) (double*, double*, double*);

chosecell GetLatticeParm_3D(int spg)
{
	if ((spg >= 1) & (spg <= 2)) //ChooseTriclinic	
		return ChooseTriclinic;

	else if ((spg >= 3) & (spg <= 15)) //ChooseMonoclinic
		return ChooseMonoclinic_beta;
	
	else if ((spg >= 16) & (spg <= 74)) //ChooseOrthorhombic
		return ChooseOrthorhombic;
	
	else if ((spg >= 75) & (spg <= 142)) //ChooseTetragonal
		return ChooseTetragonal;

	else if ((spg >= 143) & (spg <= 167)) //ChooseTrigonal
		return ChooseTrigonal;

	else if ((spg >= 168) & (spg <= 194)) //ChooseHexagonal
		return ChooseTrigonal;

	else if ((spg >= 195) & (spg <= 230)) //ChooseCubic
		return ChooseCubic;

	return 0;
};
chosecell GetLatticeParm_Lyg(int layergroup)						//Choose LayerGroup
{
	if ((layergroup >= 1) & (layergroup <= 2)) //ChooseTriclinic	
		return ChooseTriclinic;

	else if ((layergroup >= 3) & (layergroup <= 7)) //ChooseMonoclinic
		return ChooseMonoclinic_gamma;
	else if ((layergroup >= 8) & (layergroup <= 18)) //ChooseMonoclinic
		return ChooseMonoclinic_alpha;

	else if ((layergroup >= 19) & (layergroup <= 48)) //ChooseOrthorhombic
		return ChooseOrthorhombic;

	else if ((layergroup >= 49) & (layergroup <= 64)) //ChooseTetragonal
		return ChooseTetragonal;

	else if ((layergroup >= 65) & (layergroup <= 80)) //ChooseTrigonal/Hexagonal
		return ChooseTrigonal;

	return 0;
};
chosecell GetLatticeParm_RdG(int rodgroup)						//Choose RodGroup
{
	if ((rodgroup >= 1) & (rodgroup <= 2)) //ChooseTriclinic	
		return ChooseTriclinic;

	else if ((rodgroup >= 3) & (rodgroup <= 7)) //ChooseMonoclinic
		return ChooseMonoclinic_gamma;
	else if ((rodgroup >= 8) & (rodgroup <= 12)) //ChooseMonoclinic					#TODO: ???
		return ChooseMonoclinic_alpha;

	else if ((rodgroup >= 13) & (rodgroup <= 22)) //ChooseOrthorhombic
		return ChooseOrthorhombic;

	else if ((rodgroup >= 23) & (rodgroup <= 41)) //ChooseTetragonal
		return ChooseTetragonal;

	else if ((rodgroup >= 42) & (rodgroup <= 75)) //ChooseTrigonal/Hexagonal
		return ChooseTrigonal;

	return 0;
};

bool precheck(double* latticeparm, double* maxes, double* mins)
{
	//notes here:
	//alpha=b^c, beta=a^c, gamma=a^b
	if(maxes[3]< M_PI / 2 - tol || mins[3]> M_PI / 2 + tol ) return false;
	if(maxes[4]< M_PI / 2 - tol || mins[4]> M_PI / 2 + tol ) return false;
	latticeparm[3] = M_PI / 2;
	latticeparm[4] = M_PI / 2;
	//alpha = beta = pi/2

	latticeparm[2] = (maxes[2] + mins[2])/2;
	return true;
};

bool ChooseOblique(double* latticeparm, double* maxes, double* mins)
{
	if (! precheck(latticeparm, maxes, mins)) return false;
	latticeparm[0] = Rand()*(maxes[0] - mins[0]) + mins[0];
	latticeparm[1] = Rand()*(maxes[1] - mins[1]) + mins[1];
	latticeparm[5] = Rand()*(maxes[5] - mins[5]) + mins[5];
	return true;
};//a!=b
bool ChooseRectangular(double* latticeparm, double* maxes, double* mins)
{
	if (! precheck(latticeparm, maxes, mins)) return false; 

	latticeparm[0] = Rand()*(maxes[0] - mins[0]) + mins[0];
	latticeparm[1] = Rand()*(maxes[1] - mins[1]) + mins[1];

	if(maxes[5]< M_PI / 2 - tol || mins[5]> M_PI / 2 + tol ) return false;
	latticeparm[5] = M_PI / 2;
	return true;
};//a!=b, gamma=pi/2
bool ChooseRhombic(double* latticeparm, double* maxes, double* mins)
{
	if (! precheck(latticeparm, maxes, mins)) return false; 

	double t1 = Max(mins[0], mins[1]);
	double t2 = Min(maxes[0], maxes[1]);
	if(t1>t2) return false;
	latticeparm[0] = Rand()*(t2 - t1) + t1;
	latticeparm[1] = latticeparm[0];
	latticeparm[5] = Rand()*(maxes[5] - mins[5]) + mins[5];
	return true;
};//a=b
bool ChooseSquare(double* latticeparm, double* maxes, double* mins)
{
	if (! precheck(latticeparm, maxes, mins)) return false; 
	double t1 = Max(mins[0], mins[1]);
	double t2 = Min(maxes[0], maxes[1]);
	if(t1>t2) return false;
	latticeparm[0] = Rand()*(t2 - t1) + t1;
	latticeparm[1] = latticeparm[0];
	if(maxes[5]< M_PI / 2 - tol || mins[5]> M_PI / 2 + tol) return false;
	latticeparm[5] = M_PI / 2;
	return true;
};//a=b, gamma=pi/2
bool ChooseHexagonal(double* latticeparm, double* maxes, double* mins)
{
	if (! precheck(latticeparm, maxes, mins)) return false; 
	double t1 = Max(mins[0], mins[1]);
	double t2 = Min(maxes[0], maxes[1]);
	if(t1>t2) return false;
	latticeparm[0] = Rand()*(t2 - t1) + t1;
	latticeparm[1] = latticeparm[0];
	
	if(maxes[5]< 2 * M_PI / 3 - tol || mins[5]> 2 * M_PI / 3 + tol ) return false;
	latticeparm[5] = 2 * M_PI / 3;
			
	return true;
};//a1=a2!=a3;alpha=beta=pi/2,gamma=2pi/3	//a=b, gamma=2pi/3

chosecell GetLatticeParm_Plg(int planegroup)						//Choose PlaneGroup
{
	if ((planegroup >= 1) & (planegroup <= 2)) //ChooseOblique	
		return ChooseOblique;	
	else if ( (planegroup >= 3) & (planegroup <= 9)  ) //ChooseRectangular
		return ChooseRectangular;
	/*else if ((planegroup == 5) || (planegroup == 9)) //ChooseRhombic
		return ChooseRhombic;*/
	else if ((planegroup >= 10) & (planegroup <= 12)) //ChooseSquare
		return ChooseSquare;
	else if ((planegroup >= 13) & (planegroup <= 17)) //ChooseHexagonal
		return ChooseHexagonal;
	return 0; 
};


void BaseP::Init(double * _lcmax, double* _lcmin, double* _lpmax, double* _lpmin)
{
	lcmax=_lcmax; lcmin=_lcmin; lpmax=_lpmax; lpmin = _lpmin;
	for(int i =0;i<6;i++) 
		if ( lpmin[i] > lpmax[i] + tol )  { selfcheck=false; }
}
bool BaseP::checklc()
{
	for(int i =0;i<6;i++) 
		if ( lcmin[i] > lcmax[i] + tol )  { selfcheck=false ;return false;}
	return true;
}
bool BaseP::checklp(double* lp)
{
	for(int i =0;i<6;i++) 
		if ( lp[i] > lpmax[i] + tol ||  lp[i] < lpmin[i] - tol )  return false;
	return true;
}
bool BaseP::get_lp(double *lc, double* lp) 
{
	cal_lp(lc, lp);
	//cout<<lp;
	return checklp(lp);
};

class PN:virtual public BaseP
{
public:
	PN() {};
	~PN() {};
	void get_range_lc() 
	{
		for(int i = 0;i<6;i++)
		{
			lcmax[i] = lpmax[i];
			lcmin[i] = lpmin[i];
		}
	};
	void cal_lp(double *lc, double* lp) 
	{
		for(int i = 0;i<6;i++) lp[i] = lc[i];
	};
};
class PC:virtual public BaseP
{
public:
	int type;
	PC(int _type):BaseP() {type = _type;};			//type of unique angle. For PCs, only two possibilities: beta[4] for 3ds and alpha[3] for 2ds.
	~PC(){};
	/* a1 = 0.5 * sqrt(a*a + b*b )
		b1 = 0.5 * sqrt(a*a + b*b )
		c1 = c

		alpha1 = acos( (-a*cos(beta) +b*cos(alpha) )/ 2/b1)
		beta1 = acos( ( a * cos(beta) + b*cos(alpha) ) / 2/a1)
		gamma1 = acos((b*b - a*a) / 4/ a1 / b1 )
		--------------------------------------------------------
		a*a = 0.5 * (1-cos(gamma1)) * (a*a+b*b)
		b*b = 0.5 * (1+cos(gamma1)) * (a*a+b*b)
		
		[unique = beta4] cos(alpha1) + cos(beta1) = 0
								sqrt(a*a + b*b) *cos(alpha1) = -a*cos(beta)
								cos(beta) = -cos(alpha1) / sqrt(0.5*(1-cos(gamma1)))
		[unique = alpha3] cos(alpha1) = cos(beta1)
								 sqrt(a*a + b*b) *cos(alpha1) = b*cos(alpha)
								 cos(alpha) = cos(alpha1) / sqrt(0.5*(1+cos(gamma1)))		*/
	double eq1_3(double(*f)(double, double), double d1, double d2)
	{
		return pow(f(d1, d2),2)*4;
	}	
	void refine_angle_range(double &cosangle)
	{
		if (cosangle < -1) cosangle = -1;
		else if(cosangle > 1) cosangle = 1;
	};
	void get_range_lc()
	{
		//def A = a*a + b*b
		{
			double &a1 = lpmin[0], &b1 = lpmin[1], &c1 = lpmin[2]; 
			double A = eq1_3(Max, a1, b1);
			{
				double &gamma1 = lpmin[5];
				lcmin[0] = sqrt(0.5 * A * (1-cos(gamma1))); 
				lcmin[2] = c1;
			}
			{
				double &gamma1 = lpmax[5];
				lcmin[1] = sqrt(0.5 * A * (1+cos(gamma1))); 
			}
		}
		{
			double &a1 = lpmax[0], &b1 = lpmax[1], &c1 = lpmax[2]; 
			double A = eq1_3(Min, a1, b1);
			{
				double &gamma1 = lpmax[5];
				lcmax[0] = sqrt(0.5 * A * (1-cos(gamma1))); 
				lcmax[2] = c1;
			}
			{
				double &gamma1 = lpmin[5];
				lcmax[1] = sqrt(0.5 * A * (1+cos(gamma1))); 
			}
		}
		double lc[4];
		if(type == 4)		//unique = beta
		{
			double alpha1 = Max(lpmin[3], M_PI - lpmin[4]), alpha2 = Min(lpmax[3], M_PI - lpmax[4]);
			lc[0] = cos(alpha1); lc[1] = cos(alpha2);
			lc[3] = -sqrt(0.5*(1-cos(lcmin[5]))); lc[4] = -sqrt(0.5*(1-cos(lcmax[5])));
			lcmax[4] = Min(lc[0]/lc[3], lc[0]/lc[4], lc[1]/lc[3], lc[1]/lc[4]);
			lcmin[4] = Max(lc[0]/lc[3], lc[0]/lc[4], lc[1]/lc[3], lc[1]/lc[4]);
			
			refine_angle_range(lcmax[4]); refine_angle_range(lcmin[4]);
			lcmax[4] = acos(lcmax[4]); lcmin[4] = acos(lcmin[4]); 
			lcmin[3] = M_PI / 2; lcmax[3] = M_PI / 2;
			lcmin[5] = M_PI / 2; lcmax[5] = M_PI / 2;
		}
		else		//unique = alpha
		{
			double alpha1 = Max(lpmin[3], lpmin[4]), alpha2 = Min(lpmax[3], lpmax[4]);
			lc[0] = cos(alpha1); lc[1] = cos(alpha2);
			lc[3] = sqrt(0.5*(1+cos(lcmin[5]))); lc[4] = sqrt(0.5*(1+cos(lcmax[5])));
			lcmax[3] = Min(lc[0]/lc[3], lc[0]/lc[4], lc[1]/lc[3], lc[1]/lc[4]);
			lcmin[3] = Max(lc[0]/lc[3], lc[0]/lc[4], lc[1]/lc[3], lc[1]/lc[4]);
			refine_angle_range(lcmax[3]); refine_angle_range(lcmin[3]);
			lcmax[3] = acos(lcmax[3]); lcmin[3] = acos(lcmin[3]); 
			lcmin[4] = M_PI / 2; lcmax[4] = M_PI / 2;
			lcmin[5] = M_PI / 2; lcmax[5] = M_PI / 2;
		}
	}
	void cal_lp(double *lc, double* lp)
	{
		double &a = lc[0], &b = lc[1], &c = lc[2], &alpha = lc[3], &beta = lc[4], &gamma = lc[5];
		lp[0] = 0.5 * sqrt(a*a + b*b ) ;
		lp[1] = lp[0] ;
		lp[2] = c ;

		lp[3] = acos( (-a*cos(beta) +b*cos(alpha) )/ 2/lp[0]) ;
		lp[4] = acos( ( a * cos(beta) + b*cos(alpha) ) / 2/lp[0]) ;
		lp[5] = acos((b*b - a*a) / 4/ lp[0] / lp[0] );
		return;
	}
};

class PF:virtual public BaseP
{
public:
	PF():BaseP() {};
	
	double eq1_3(double d1, double d2, double d3)
	{
		return 2*(d1*d1 + d2*d2 - d3*d3);
	}
	double eq4_6(double a1, double d2, double d3)
	{
		return 4*cos(a1) *d2*d3;
	}
	/* a1 = 0.5 * sqrt(b*b + c*c)
		b1 = 0.5 * sqrt(a*a + c*c)
		c1 = 0.5 * sqrt(a*a + b*b)
		alpha1 = arccos( a*a / 4 / b1 / c1 )
		beta1 = arccos( b*b / 4 / a1 / c1 )
		gamma1 = arccos( c*c / 4 / a1 / b1 )
	--------------------------------------------
		a*a = 2*(b1*b1 + c1*c1 - a1*a1)
		b*b =  2*(a1*a1 + c1*c1 - b1*b1)
		c*c = 2*(a1*a1 + b1*b1 - c1*c1)
		4 * cos(alpha1) *b1*c1 = a*a
		4 * cos(beta1) *a1*c1 = b*b
		4 * cos(gamma1) *a1*b1 = c*c		*/
	void get_range_lc()
	{
		{
			double &a1 = lpmin[0], &b1 = lpmin[1], &c1 = lpmin[2], &alpha1 = lpmax[3], &beta1 = lpmax[4], &gamma1 = lpmax[5];
			lcmin[0] = sqrt(Max(eq4_6(alpha1, b1, c1), eq1_3(b1, c1, a1), 0)) ;
			lcmin[1] = sqrt(Max(eq4_6(beta1, a1, c1), eq1_3(a1, c1, b1), 0)) ;
			lcmin[2] = sqrt(Max(eq4_6(gamma1, a1, b1), eq1_3(a1, b1, c1), 0)) ;
		}
		{
			double &a1 = lpmax[0], &b1 = lpmax[1], &c1 = lpmax[2], &alpha1 = lpmin[3], &beta1 = lpmin[4], &gamma1 = lpmin[5];
			lcmax[0] = sqrt(Min(eq4_6(alpha1, b1, c1), eq1_3(b1, c1, a1), 4*b1*b1, 4*c1*c1)) ;
			lcmax[1] = sqrt(Min(eq4_6(beta1, a1, c1), eq1_3(a1, c1, b1), 4*a1*a1, 4*c1*c1)) ;
			lcmax[2] = sqrt(Min(eq4_6(gamma1, a1, b1), eq1_3(a1, b1, c1), 4*a1*a1, 4*b1*b1)) ;
		}
		for(int i = 3; i < 6 ; i++) 
		{
			lcmin[i] = M_PI / 2; lcmax[i] = M_PI / 2;
		}
		return;
	};
	void cal_lp(double *lc, double* lp)
	{
		double &a = lc[0], &b = lc[1], &c = lc[2], &alpha = lc[3], &beta = lc[4], &gamma = lc[5];
		lp[0] = 0.5 * sqrt(b*b + c*c);
		lp[1] = 0.5 * sqrt(a*a + c*c);
		lp[2] = 0.5 * sqrt(a*a + b*b);
		lp[3] = acos( a*a / 4 / lp[1] / lp[2] );
		lp[4] = acos( b*b / 4 / lp[0] / lp[2] );
		lp[5] = acos( c*c / 4 / lp[0] / lp[1] );
		return;
	};
};

class PI:virtual public BaseP
{
public:
	PI():BaseP() {};

	/* a1 = 0.5 * sqrt(a*a + b*b + c*c)
		b1 = 0.5 * sqrt(a*a + b*b + c*c)
		c1 = 0.5 * sqrt(a*a + b*b + c*c)
		alpha1 = acos( (a*a - b*b - c*c) / (a*a + b*b + c*c) )
		beta1 = acos( (b*b - a*a - c*c) / (a*a + b*b + c*c) )
		gamma1 = acos( (c*c - a*a - b*b) / (a*a + b*b + c*c) )
	---------------------------------------------------------------------
		(cos(alpha1) - 1 )*a*a + (cos(alpha1) + 1) (b*b + c*c) = 0
		(cos(beta1) - 1 )*b*b + (cos(beta1) + 1) (a*a + c*c) = 0
	---------------------------------------------------------------------	
		cos(alpha1) + cos(beta1) + cos(gamma1) +1 = 0
		a*a = -c*c *  (1+cos(alpha1)) / (cos(beta1) + cos(alpha1))  [aborted]
        b*b = -c*c * (1+cos(beta1)) / (cos(beta1) + cos(alpha1))    [aborted]
	---------------------------------------------------------------------	
		a*a = 0.5 * (1+cos(alpha1)) * (a*a + b*b + c*c)
		b*b = 0.5 * (1+cos(beta1)) * (a*a + b*b + c*c)
		c*c = 0.5 * (1+cos(gamma1))	* (a*a + b*b + c*c)				*/

	double eq1_3(double(*f)(double, double, double), double d1, double d2, double d3)
	{
		return pow(f(d1, d2, d3), 2)*4;
	}
	double eq4_6(double(*f)(double, double), double angle1, double angle2, double angle3)
	{
		return f(1+cos(angle1), -cos(angle2)-cos(angle3));
	}
	void get_range_lc()
	{
		//def A = (a*a + b*b + c*c), B = (1+cos(angle))
		{
			double &a1 = lpmin[0], &b1 = lpmin[1], &c1 = lpmin[2]; 
			double A = eq1_3(Max, a1, b1, c1);
			{
				double &alpha1 = lpmax[3], &beta1 = lpmin[4], &gamma1 = lpmin[5];
				lcmin[0] = sqrt(Max(A*0.5*eq4_6(Max, alpha1, beta1, gamma1), 0 ));
			}
			{
				double &alpha1 = lpmin[3], &beta1 = lpmax[4], &gamma1 = lpmin[5];
				lcmin[1] = sqrt(Max(A*0.5*eq4_6(Max, beta1, alpha1, gamma1), 0 ));
			}
			{
				double &alpha1 = lpmin[3], &beta1 = lpmin[4], &gamma1 = lpmax[5];
				lcmin[2] = sqrt(Max(A*0.5*eq4_6(Max, gamma1, alpha1, beta1), 0 ));
			}
		}
		{
			double &a1 = lpmax[0], &b1 = lpmax[1], &c1 = lpmax[2]; 
			double A = eq1_3(Min, a1, b1, c1);
			{
				double &alpha1 = lpmin[3], &beta1 = lpmax[4], &gamma1 = lpmax[5];
				lcmax[0] = sqrt(A*0.5*eq4_6(Max, alpha1, beta1, gamma1));
			}
			{
				double &alpha1 = lpmax[3], &beta1 = lpmin[4], &gamma1 = lpmax[5];
				lcmax[1] = sqrt(A*0.5*eq4_6(Max, beta1, alpha1, gamma1));
			}
			{
				double &alpha1 = lpmax[3], &beta1 = lpmax[4], &gamma1 = lpmin[5];
				lcmax[2] = sqrt(A*0.5*eq4_6(Max, gamma1, alpha1, beta1));
			}
		}
		for(int i = 3; i < 6 ; i++) 
		{
			lcmin[i] = M_PI / 2; lcmax[i] = M_PI / 2;
		}
		return;
	};

	void cal_lp(double *lc, double* lp)
	{
		double &a = lc[0], &b = lc[1], &c = lc[2], &alpha = lc[3], &beta = lc[4], &gamma = lc[5];
		double A = a*a + b*b + c*c;
		lp[0] = 0.5 * sqrt(A);
		lp[1] = lp[0];
		lp[2] = lp[0];
		lp[3] = acos( (a*a - b*b - c*c) / A );
		lp[4] = acos( (b*b - a*a - c*c) / A );
		lp[5] = acos( (c*c - a*a - b*b) / A );
		return ;
	};
};
class PA:virtual  public BaseP
{
public:
	PA():BaseP() {};

	/* a1 = a
		b1 = 0.5 * sqrt(b*b + c*c)
		c1 = 0.5 * sqrt(b*b + c*c)

		alpha1 = acos( (b*b - c*c) / (b*b + c*c) )
		beta1 = pi / 2
		gamma1 = pi / 2
	------------------------------------------------------------
		b*b = 0.5* (1+cos(alpha1)) * (b*b+c*c)
		c*c =  0.5* (1-cos(alpha1))	* (b*b+c*c)				*/

	double eq1_3(double(*f)(double, double), double d1, double d2)
	{
		return pow(f(d1, d2), 2)*4;
	};
	
	void get_range_lc()
	{
		//def A = (b*b + c*c)
		{
			double &a1 = lpmin[0], &b1 = lpmin[1], &c1 = lpmin[2]; 
			double A = eq1_3(Max, b1, c1);
			{
				double &alpha1 = lpmax[3], &beta1 = lpmin[4], &gamma1 = lpmin[5];
				lcmin[0] = a1;
				lcmin[1] = sqrt(Max(A*0.5*(1+cos(alpha1)), 0 ));
				lcmin[3] = M_PI / 2;
				lcmin[4] = beta1;
				lcmin[5] = gamma1;
			}
			{
				double &alpha1 = lpmin[3];
				lcmin[2] = sqrt(Max(A*0.5*(1-cos(alpha1)), 0 ));
			}
		}
		{
			double &a1 = lpmax[0], &b1 = lpmax[1], &c1 = lpmax[2]; 
			double A = eq1_3(Min, b1, c1);
			{
				double &alpha1 = lpmin[3], &beta1 = lpmax[4], &gamma1 = lpmax[5];
				lcmax[0] = a1;
				lcmax[1] = sqrt(A*0.5*(1+cos(alpha1)));
				lcmax[3] = M_PI / 2;
				lcmax[4] = beta1;
				lcmax[5] = gamma1;
			}
			{
				double &alpha1 = lpmax[3];
				lcmax[2] = sqrt(A*0.5*(1-cos(alpha1)));
			}
		}
		return;
	}
	void cal_lp(double *lc, double* lp)
	{
		double &a = lc[0], &b = lc[1], &c = lc[2], &alpha = lc[3], &beta = lc[4], &gamma = lc[5];
		double A = b*b + c*c;
		lp[0] = a;
		lp[1] = 0.5 * sqrt(A);
		lp[2] = lp[1];
		lp[3] = acos( (b*b - c*c) /A );
		lp[4] = M_PI / 2;
		lp[5] = M_PI / 2;
		return ;
	};
};
class PR:virtual public BaseP
{
public:
	PR():BaseP() {};
	/* a1 = 1.0/3 * sqrt(7*a*a + c*c ) 
		b1 = 1.0/3 * sqrt(a*a + 4*c*c )
		c1 = 1.0/3 * sqrt(a*a + c*c )

		alpha1 = acos( (a*a - 2*c*c ) / sqrt(a*a + 4*c*c ) / sqrt(a*a + c*c ) )
		beta1 = acos( (0.5*a*a - c*c ) / sqrt(7*a*a + c*c ) / sqrt(a*a + c*c ) )
		gamma1 = acos( (0.5*a*a + 2*c*c ) / sqrt(7*a*a + c*c ) / sqrt(a*a + 4*c*c )
	-------------------------------------------------------------------------------------------
		a1*a1 = 9*c1*c1 - 2*b1*b1
		a*a = 12*c1*c1 -3*b1*b1
		c*c = 3*b1*b1 - 3*c1*c1
		
		2*cos(beta1)*a1  = cos(alpha1)*b1
		a*a = 12*cos(beta1)*a1*c1 + 6*cos(gamma1)*a1*b1
		c*c = 3*cos(gamma1)*a1*b1 - 3*cos(beta1)*a1*c1						*/
	double refine_angle_range(double cosangle)
	{
		if (cosangle < -1) return M_PI;
		else if(cosangle > 1) return 0;
		else return acos(cosangle);
	};
	double refine_length_range(double l)
	{
		if (l < 0) return 0;
		else return sqrt(l);
	};

	void refinerangelp()
	{	
		{
			double &a1 = lpmin[0], &b1 = lpmax[1], &c1 = lpmin[2]; 
			a1 = Max(a1, refine_length_range(9*c1*c1 - 2*b1*b1));
		}
		{
			double &a1 = lpmax[0], &b1 = lpmin[1], &c1 = lpmax[2]; 
			a1 = Min(a1, refine_length_range(9*c1*c1 - 2*b1*b1));
		}
		{
			double &a1 = lpmax[0], &b1 = lpmin[1], &alpha1 = lpmax[3], &beta1 = lpmax[4];
			beta1 = Min(beta1, refine_angle_range(cos(alpha1) * b1/2/a1));
		}
		{
			double &a1 = lpmin[0], &b1 = lpmax[1], &alpha1 = lpmin[3], &beta1 = lpmin[4];
			beta1 = Max(beta1, refine_angle_range(cos(alpha1) * b1/2/a1));
		}
	};

	void get_range_lc()
	{
		refinerangelp();
		{
			double &b1 = lpmax[1], &c1 = lpmin[2];
			lcmin[0] = Max(12*c1*c1 -3*b1*b1, 0);	
			lcmax[2] = 3*b1*b1 - 3*c1*c1;
		}
		{
			double &b1 = lpmin[1], &c1 = lpmax[2];
			lcmax[0] = 12*c1*c1 -3*b1*b1;	
			lcmin[2] = Max(3*b1*b1 - 3*c1*c1, 0);
		}
		
		{
			double &a1 = lpmin[0], &b1 = lpmin[1], &c1 = lpmin[2], &beta1 = lpmax[4], &gamma1 = lpmax[5];
			lcmin[0] = Max(lcmin[0], 6*a1*(2*cos(beta1)*c1 + cos(gamma1)*b1));
		}
		{
			double &a1 = lpmax[0], &b1 = lpmax[1], &c1 = lpmax[2], &beta1 = lpmin[4], &gamma1 = lpmin[5];
			lcmax[0] = Min(lcmax[0], 6*a1*(2*cos(beta1)*c1 + cos(gamma1)*b1));
		}
		{
			double &a1 = lpmin[0], &b1 = lpmin[1], &c1 = lpmax[2], &beta1 = lpmin[4], &gamma1 = lpmax[5];
			lcmin[2] = Max(lcmin[2], 3*a1* (b1*cos(gamma1) - c1*cos(beta1)));
		}
		{
			double &a1 = lpmax[0], &b1 = lpmax[1], &c1 = lpmin[2], &beta1 = lpmax[4], &gamma1 = lpmin[5];
			lcmax[2] = Min(lcmax[2], 3*a1* (b1*cos(gamma1) - c1*cos(beta1)));
		}
		lcmin[0] = refine_length_range(lcmin[0]) ;lcmax[0] = refine_length_range(lcmax[0]) ;
		lcmin[2] = refine_length_range(lcmin[2]) ;lcmax[2] = refine_length_range(lcmax[2]) ;
		lcmin[1] =lcmin[0]; lcmax[1] = lcmax[0];
		lcmin[3] = M_PI / 2; lcmax[3] = M_PI / 2;
		lcmin[4] = M_PI / 2; lcmax[4] = M_PI / 2;
		lcmin[5] = 2 * M_PI / 3; lcmax[5] = 2 * M_PI / 3;
		return;
	}
	void cal_lp(double *lc, double* lp)
	{
		double &a = lc[0], &b = lc[1], &c = lc[2], &alpha = lc[3], &beta = lc[4], &gamma = lc[5];

		lp[0] = 1.0/3 * sqrt(7*a*a + c*c );
		lp[1] = 1.0/3 * sqrt(a*a + 4*c*c );
		lp[2] = 1.0/3 * sqrt(a*a + c*c );
		lp[3] = acos( (a*a - 2*c*c ) / 9 / lp[1] / lp[2] );
		lp[4] = acos( (0.5*a*a - c*c ) / 9/ lp[0] / lp[2] );
		lp[5] = acos( (0.5*a*a + 2*c*c ) / 9/ lp[0] / lp[1] );
		return ;
	};

};
BaseP* getbasep(int celltype)
{
	switch (celltype)
	{
	case 1:			//symtype = 'P'
	{
		PN* pn = new PN();
		return pn;
	};
		break;
	case 23:			//symtype = 'PC' for 3D with beta[4]
	{
		PC* pc = new PC(4);
		return pc;
	};
		break;
	case 22:			//symtype = 'PC' for 2D with alpha[3]
	{
		PC* pc = new PC(3);
		return pc;
	};
		break;
	case 3:			//symtype = 'PF'
	{
		PF* pf = new PF();
		return pf;
	};
		break;
	case 4:			//symtype = 'PI'
	{
		PI* pi = new PI();
		return pi;
	};
		break;
	case 5:			//symtype = 'PA'
	{
		PA* pa = new PA();
		return pa;
	};
		break;
	case 6:			//symtype = 'PR'
	{
		PR* pr = new PR();
		return pr;
	};
		break;
	default:
	    return 0;
		break;
	}
};

GetLatticeParm::GetLatticeParm(int spg, double* Mins, double* Maxes, int _dimention, int _choice, int _celltype)
{
	if (_dimention ==3)
		getlatticeparm = GetLatticeParm_3D(spg);
	else if(_dimention ==2)
	{
		if (_choice == 0) getlatticeparm = GetLatticeParm_Plg(spg);
		else if(_choice ==1) getlatticeparm = GetLatticeParm_Lyg(spg);
	}	
	else if (_dimention ==1)
		getlatticeparm = GetLatticeParm_RdG(spg);

	latticeMins = Mins; latticeMaxes = Maxes;
	//cout<<"celltype" <<_celltype<<endl;
	if (_celltype == 2) _celltype = _celltype*10 + _dimention;
	basep = getbasep(_celltype);
	basep->Init(CcellMaxes, CcellMins, latticeMaxes, latticeMins);
	basep->get_range_lc();
	//cout<<"range_lp\n"<<latticeMins<<latticeMaxes;
	//cout<<"range_lc\n"<<(basep->lcmin)<<(basep->lcmax);

	basep->checklc();
}
GetLatticeParm::~GetLatticeParm()
{
	if (basep) delete basep;
}

bool GetLatticeParm::GetLattice(double* M)
{
	double lp[6], lc[6];
	//cout<<"cp_1"<<endl;
	if (basep->selfcheck == false) return false;
	//cout<<"cp_2"<<endl;
	if (getlatticeparm(lc, CcellMaxes, CcellMins) == false) return false;
	else
	{
		//cout<<"cp_3"<<endl;
		int trynum=10;
		bool m = false;
		for (int i=0;i<trynum;i++)
		{
			if (basep->get_lp(lc, lp) == true) {m=true; break;}
			else 
				getlatticeparm(lc, CcellMaxes, CcellMins);
		}
		if(m == false) {return false;}
	}

	latticetrans(M, lc);
	return true;
};
