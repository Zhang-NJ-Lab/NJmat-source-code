/*position.h: Definition of 
	class position;
	function CheckDistance() between position(s)*/
	
#pragma once
#include <cstdlib>
#include <vector>
#include <fstream>
using namespace std;


class position
{
public:
	double x;
	double y;
	double z;
	position(double a = 0, double b = 0, double c = 0)
	{
		x = a; y = b; z = c;
	};
	position(const position &p)
	{
		x = p.x; y = p.y; z = p.z;
	}
	position(double* p)
	{
		x=p[0]; y=p[1]; z=p[2];
	}
	void operator =(const position&p)
	{
		x = p.x; y = p.y; z = p.z;
	}

	bool operator==(const position&p)
	{	
		double tolerance=1e-4;
		if (fabs(x - p.x) > tolerance) return false;
		if (fabs(y - p.y) > tolerance) return false;
		if (fabs(z - p.z) > tolerance) return false;
		return true;
	};
	
	position operator -(const position&p)
	{
		return position(x-p.x, y-p.y, z-p.z);
	}
	position operator +(const position&p)
	{
		return position(x+p.x, y+p.y, z+p.z);
	}
	void operator +=(const position& p)
	{
		x+=p.x; y+=p.y; z+=p.z;
		return;
	}
	void operator -=(const position& p)
	{
		x-=p.x; y-=p.y; z-=p.z;
		return;
	}
	position operator *(const double& d)
	{
		return position(x*d, y*d, z*d);
	}
	void operator *=(const double& d)
	{
		x*=d; y*=d; z*=d;
		return;
	}
	void clear(void)
	{
		x=0;y=0;z=0;
		return;
	}
	void renormalize(void)
	{
		double d=sqrt(x*x+y*y+z*z);
		x/=d; y/=d; z/=d;
	}
	bool sym(const position &p, double tolerance=-1);		//defined in clutersym.cpp

	position rotate(double* m)
	{
		return position(m[0]*x+m[1]*y+m[2]*z, m[4]*x+m[5]*y+m[6]*z, m[8]*x+m[9]*y+m[10]*z);
	};

};
ostream & operator<<(ostream &out, const position & p)
{
	out<<p.x<<'\t'<<p.y<<'\t'<<p.z;
	return out;
}
double Dotproduct(const position& p1, const position& p2)//dot product of p1 and p2
{
	return p1.x*p2.x + p1.y*p2.y + p1.z*p2.z;
};
double Dotproduct(position* p1, position* p2)//dot product of p1 and p2
{
	return p1->x*p2->x + p1->y*p2->y + p1->z*p2->z;
};
position Crossproduct(position* p1, position* p2) //cross product of p1 and p2
{
	return position(p1->y*p2->z-p1->z*p2->y, p1->z*p2->x-p1->x*p2->z, p1->x*p2->y-p1->y*p2->x);
}

void Padd(double a, double b, position* p1, position* p2, position* p)//p=a*p1+b*p2
{
	p->x = a * p1->x + b * p2->x;
	p->y = a * p1->y + b * p2->y;
	p->z = a * p1->z + b * p2->z;
	return;
};//Padd(1,-1,p1,p2,p):p=p1-p2

/*void solve(double M[][3], double b[], double x[])
{
	double Mstar[3][3];//Inverse of M
	Mstar[0][0] = M[1][1] * M[2][2] - M[1][2] * M[2][1]; Mstar[0][1] = M[1][0] * M[2][2] - M[1][2] * M[2][0]; Mstar[0][2] = M[1][0] * M[2][1] - M[2][0] * M[1][1];
	Mstar[1][0] = M[0][1] * M[2][2] - M[0][2] * M[2][1]; Mstar[1][1] = M[0][0] * M[2][2] - M[0][2] * M[2][0]; Mstar[1][2] = M[0][0] * M[2][1] - M[2][0] * M[0][1];
	Mstar[2][0] = M[0][1] * M[1][2] - M[0][2] * M[1][1]; Mstar[2][1] = M[0][0] * M[1][2] - M[0][2] * M[1][0]; Mstar[2][2] = M[0][0] * M[1][1] - M[1][0] * M[0][1];
	double detM = 0;
	for (int i = 0; i < 3; i++) detM += pow(-1, i)*M[0][i] * Mstar[0][i];
	for (int i = 0; i < 3; i++) x[i] = 0;
	for (int j = 0; j < 3; j++)
	{
		for (int i = 0; i < 3; i++) x[j] += Mstar[i][0] * b[i] * pow(-1, i + j);
		x[j] /= detM;
	}
	return;
};//MX=b;X=solve(M,b)=InverseM*b*/

//wrap a frac_pos to unit cell
position Standform(position* p, bool * _pbc=0)
{
	position temp = *p;

	bool pbc[3] = {true, true, true};
	if (_pbc) for(int _a = 0;_a<3;_a++) pbc[_a] = _pbc[_a];
	if (pbc[0])
	{
		temp.x = temp.x - ((int)temp.x);
		if (temp.x < 0) temp.x += 1;
		if(fabs(temp.x)<1e-4 || fabs(temp.x-1)<1e-4) temp.x=0;
	}
	if(pbc[1])
	{
		temp.y = temp.y - ((int)temp.y);
		if (temp.y < 0) temp.y += 1;
		if(fabs(temp.y)<1e-4 || fabs(temp.y-1)<1e-4) temp.y=0;
	}
	if(pbc[2])
	{
		temp.z = temp.z - ((int)temp.z);
		if (temp.z < 0) temp.z += 1;
		if(fabs(temp.z)<1e-4 || fabs(temp.z-1)<1e-4) temp.z=0;
	}
	return temp;
};

double CalDistance(position* p1, position* p2) //p1 and p2 must be in Orthorhombic 
{
	return  ( pow( p1->x - p2->x , 2) + pow( p1->y - p2->y , 2 )+ pow( p1->z - p2->z , 2) );
};
double CalDistance(position* p)
{
	return  ( pow( p->x , 2) + pow( p->y, 2 )+ pow( p->z , 2) );
}
void postrans(position* pr, position* po,double* lp) //pr=po.dot(latticeparm)
{
	pr->x= po->x*lp[0] + po->y*lp[1] + po->z*lp[2];
	pr->y= po->y*lp[4] + po->z*lp[5];
	pr->z= po->z*lp[8];
	return;
}
void inversepostrans(position* pr,position* po,double* lp) //pr=po.dot(latticeparm^-1)
{
	pr->z=  po->z/lp[8];
	pr->y= (po->y - pr->z*lp[5])/lp[4];
	pr->x= (po->x - pr->y*lp[1] - pr->z*lp[2])/lp[0];
	return;
}
bool parallel(position* p1, position* p2)
{
	double temp=pow(Dotproduct(p1, p2), 2) ;
	if(fabs(temp - CalDistance(p1)*CalDistance(p2))<1e-6) return true;
	else return false;
}
bool CheckDistance(position* p1, position* p2, double r1, double r2, double* lp, double threshold, bool cart_pos=false, bool* _pbc = 0)
{
	double mindis = pow((r1 + r2)*threshold, 2);
	position p2_prime((p2->x-p1->x),(p2->y-p1->y),(p2->z-p1->z));
	position p;
	if(!cart_pos) postrans(&p,&p2_prime,lp);
	else p=p2_prime; 

	vector<position> neiborsa;
	vector<position> neiborsb;
//	cout<<p.x<<'\t'<<p.y<<p.z<<endl;
	double maxdistance=CalDistance(&p);if(maxdistance< mindis) return false;
	bool pbc[3] = {true, true, true};
	if (_pbc) for(int _a = 0;_a<3;_a++) pbc[_a] = _pbc[_a];
	position temp;
	{
		temp=p;neiborsa.push_back(temp);
		if(pbc[2])
		{
			double tempz=temp.z-lp[8];
			while(tempz*tempz<=maxdistance)
			{
				temp.z=tempz;temp.y-=lp[5];temp.x-=lp[2];
				neiborsa.push_back(temp); tempz=temp.z-lp[8];
				if( fabs(temp.x)<fabs(p.x) || fabs(temp.y)<fabs(p.y) ||fabs(temp.z)<fabs(p.z)  )
				{
					maxdistance=min(CalDistance(&temp),maxdistance);if(maxdistance< mindis) return false;
				}
			}	
			temp=p;tempz=temp.z+lp[8];
			while(tempz*tempz<=maxdistance)
			{
				temp.z=tempz;temp.y+=lp[5];temp.x+=lp[2];
				neiborsa.push_back(temp);tempz=temp.z+lp[8];
				if( fabs(temp.x)<fabs(p.x) || fabs(temp.y)<fabs(p.y) ||fabs(temp.z)<fabs(p.z)  )
				{
					maxdistance=min(CalDistance(&temp),maxdistance);if(maxdistance< mindis) return false;
				}
			}
		}
			
	}
	
	for(int i=0;i<neiborsa.size();i++)
	{
		if(neiborsa[i].z*neiborsa[i].z>maxdistance) continue;
		
		if(neiborsa[i].y*neiborsa[i].y>maxdistance) 
		{
			int direction;
			if(neiborsa[i].y*lp[4]>0) direction=1;else direction=-1;
			double py2=neiborsa[i].y*neiborsa[i].y;
			double tempy=neiborsa[i].y;
			while(py2>maxdistance) 
			{
				neiborsa[i].y-=direction*lp[4];neiborsa[i].x-=direction*lp[1];
				py2=neiborsa[i].y*neiborsa[i].y;
				if(py2>maxdistance & tempy*neiborsa[i].y<0) break;
			}
			if(py2>maxdistance) continue;
		}
		
		temp=neiborsa[i];
		neiborsb.push_back(temp);maxdistance=min(CalDistance(&temp),maxdistance);if(maxdistance< mindis) return false;
		double tempy=temp.y-lp[4];
		while(tempy*tempy<=maxdistance)
		{	
			temp.y=tempy;temp.x-=lp[1];neiborsb.push_back(temp);tempy=temp.y-lp[4];
			if( fabs(temp.x)<fabs(neiborsa[i].x) || fabs(temp.y)<fabs(neiborsa[i].y) ) 
			{
				maxdistance=min(CalDistance(&temp),maxdistance);if(maxdistance< mindis) return false;
			}  
		}	
		temp=neiborsa[i];tempy=temp.y+lp[4];
		while(tempy*tempy<=maxdistance)
		{	
			temp.y=tempy;temp.x+=lp[1];neiborsb.push_back(temp);tempy=temp.y+lp[4];
			if( fabs(temp.x)<fabs(neiborsa[i].x) || fabs(temp.y)<fabs(neiborsa[i].y) ) 
			{
				maxdistance=min(CalDistance(&temp),maxdistance);if(maxdistance< mindis) return false;
			}  
		}	
		
	}
	
	for(int i=0;i<neiborsb.size();i++)
	{	
		if(neiborsb[i].z*neiborsb[i].z>maxdistance) continue;
		if(neiborsb[i].y*neiborsb[i].y>maxdistance) continue;
		if(neiborsb[i].x*neiborsb[i].x>maxdistance) 
		{
			int direction;
			if(neiborsb[i].x*lp[0]>0) direction=1;else direction=-1;
			double px2=neiborsb[i].x*neiborsb[i].x;
			double tempx=neiborsb[i].x;
			while(px2>maxdistance) 
			{
				neiborsb[i].x-=direction*lp[0];
				px2=neiborsb[i].x*neiborsb[i].x;
				if(px2>maxdistance & tempx*neiborsb[i].x<0) break;
			}
			if(px2>maxdistance) continue;
		}
		temp=neiborsb[i];maxdistance=min(CalDistance(&temp),maxdistance);if(maxdistance< mindis) return false;
		double tempx=temp.x-lp[0];
		while(tempx*tempx<=maxdistance)
		{	
			temp.x=tempx;tempx=temp.x-lp[0];
			if(fabs(temp.x)<fabs(neiborsb[i].x))
			{
				maxdistance=min(CalDistance(&temp),maxdistance); if(maxdistance< mindis) return false;
			}	
		}	
		temp=neiborsb[i];tempx=temp.x+lp[0];
		while(tempx*tempx<=maxdistance)
		{	
			temp.x+=lp[0];tempx=temp.x+lp[0];
			if(fabs(temp.x)<fabs(neiborsb[i].x))
			{
				maxdistance=min(CalDistance(&temp),maxdistance); if(maxdistance< mindis) return false;
			}	
		}	
	}
	
	/*for(int i=0;i<neiborsa.size();i++) 
	{
		cout<<neiborsa[i].x<<'\t'<<neiborsa[i].y<<'\t'<<neiborsa[i].z<<endl;
	}
	cout<<"b:\n";
	for(int i=0;i<neiborsb.size();i++) 
	{
		cout<<neiborsb[i].x<<'\t'<<neiborsb[i].y<<'\t'<<neiborsb[i].z<<endl;
	}
	*/

	return true;
}

bool CheckDistance(vector<position>* p1s, vector<position>* p2s, double r1, double r2, double* latticeparm, double threshold, bool cart_pos=false, bool* pbc=0)
{
	for (int i = 0; i < p1s->size(); i++)
		for (int j = 0; j < p2s->size(); j++)
		{
			if (CheckDistance(&(*p1s)[i], &(*p2s)[j], r1, r2, latticeparm, threshold, cart_pos, pbc) == false) return false;
		}
	return true;
};//for every atom in p1s and p2s,if one atom is too close to another, return false 
bool CheckDistance(vector<position>* p1s, vector<position>*p2s, double r, double*latticeparm, double threshold, bool cart_pos=false, bool* pbc=0)
{
	for (int i = 0; i < p1s->size(); i++)
	{
		for (int j = 0; j < i; j++)
		{
			if (CheckDistance(&(*p1s)[i], &(*p1s)[j], r, r, latticeparm, threshold, cart_pos, pbc) == false)  return false;
		}
	}

	return (CheckDistance(p1s, p2s, r, r, latticeparm, threshold, cart_pos, pbc));
}
bool CheckDistance_withinCluster(vector<position> &p1s, vector<position> &p2s, double r1, double r2, double threshold)
{
	double mindistance = pow((r1+r2)*threshold,2);
	for (int i=0;i<p1s.size();i++)
		for (int j=0;j<p2s.size();j++)
			if(CalDistance(&p1s[i], &p2s[j])<mindistance) return false;
	return true;
}
bool CheckDistance_withinCluster(vector<position> &p1s, vector<position> &p2s, double r, double threshold)
{
	double mindistance = pow(r*2*threshold,2);
	for (int i=0;i<p1s.size();i++)
		for (int j=0;j<i;j++)
			if(CalDistance(&p1s[i], &p1s[j])<mindistance) return false;
			
	return CheckDistance_withinCluster(p1s, p2s, r,r,threshold);
}




