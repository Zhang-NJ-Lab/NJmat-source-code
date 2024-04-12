/*wyckoffposition.h: Definition of
    class WyckPos;
    class WyckGroup*/

#pragma once
#include <cstdlib> 
#include <cmath> 
#include"position.h"
#include"debug.h"

using namespace std;

class WyckPos
{
public:
	char label;
	int multiplicity;
	double wyckmatrix[12];

	/*rotatematrix and transmatrix will be same for the same spacegroup. */
	vector<double> rotatematrix;
	vector<double> transmatrix;

	bool unique;
	int variables;		//variables in wyckmatrix. For example, variables=3 if wyckmatrix=[x,y,z], and 0 if unique=true.
	int symop;		//num of site symmetry operations
	
	/*symops is a 10*n 2D int matrix. The first dimension stands for 
	[ 'None', 'minus1' , '2' , 'mirror', '4', 'minus4', '3', 'minus3', '6', 'minus6' ]
	and the second dimension stands for the detailed symmetry type(which axis token) in sitesymData.h.
	For example, if symmops[2]=[0], then the '2' axis = [1,0,0]. 
	symmatrix is the angles between all axises. symmatrix[i][j][ii][jj] stands for angle between symops[i][j] and symops[ii][jj].*/

	vector< vector<int> > symops;		
	vector< vector <vector <vector <double> > > > symmatrix;


	WyckPos(char l, int m, double* w, vector<double>* r, vector<double>* t, bool u, vector<int>& s);
					/*label, multiplicity, wyckmatrix[12], rot_matrix, trans_matrix, unique, symmetry_ops*/ 
	WyckPos(const WyckPos& w);
	void operator =(const WyckPos& w);

	/*GetOnePosition returns one random position dot wyckmatrix. 
	[For non-0D cases, i.e., void arguement, it generates random position which x,y,z are in range(0,1).]
	Then it is passed to GetAllPosition as p and is rotated/translated to get the whole set of positions.
	For clusters, p stands for the center of the cluster. Given clusterpos, all positions of cluster atoms were calculated (positions). Positions of center_of_cluster is also calculated (com_poss).*/
	position GetOnePosition(std::function<double(double, double, double)> distribution);
	void GetAllPosition(vector<position>* positions, position p, bool* pbc = 0);
	void GetAllPosition(vector<position>* positions, position p, vector<position>& clusterpos, vector<position>& com_poss);
	
	/*GetOnePosition returns one random position dot wyckmatrix. 
	[For 0D cases, i.e. maxdistance given, it generates random position which distance to point(0,0,0) falls in range(mindistance,maxdistance).]
	GetAllPosition_PointGroup is slightly modified from GetAllPosition, which standform is not used for there is no periodic boundary conditions.*/
	position GetOnePosition(double mindistance, double maxdistance) ;
	void GetAllPosition_PointGroup(vector<position>* positions, position p);
};

class WyckGroup
{
public:
	int count;		
	vector<WyckPos>* SimilarWyck;

	WyckGroup(const WyckGroup& w)
	{
		count = w.count;
		SimilarWyck=w.SimilarWyck;
	}
	void operator =(const WyckGroup& w)
	{
		count = w.count;
		SimilarWyck = w.SimilarWyck;
	}
	WyckGroup(vector<WyckPos>* w)
	{
		count = 0;
		SimilarWyck = w;
	}
};





/****************Here begins the detailed function. ****/

WyckPos::WyckPos(const WyckPos& w)
{
	label = w.label;
	multiplicity = w.multiplicity;
	for (int i = 0; i < 12; i++) wyckmatrix[i] = w.wyckmatrix[i];
	for (int i = 0; i < w.rotatematrix.size(); i++) rotatematrix.push_back(w.rotatematrix[i]);
	for (int i = 0; i < w.transmatrix.size(); i++) transmatrix.push_back(w.transmatrix[i]);
	unique = w.unique;
	variables = w.variables;
	symop=w.symop;
	for(int i=0;i<w.symops.size();i++) symops.push_back(vector<int>(w.symops[i]));

	symmatrix.resize(w.symmatrix.size());
	for(int i=0;i<w.symmatrix.size();i++) 
	{
		symmatrix[i].resize(w.symmatrix[i].size());
		for(int j=0;j<w.symmatrix[i].size();j++)
			for(int k=0;k<w.symmatrix[i][j].size();k++) symmatrix[i][j].push_back(vector<double>(w.symmatrix[i][j][k]));
	}
}

void WyckPos::operator =(const WyckPos& w)
{
	label = w.label;
	multiplicity = w.multiplicity;
	for (int i = 0; i < 12; i++) wyckmatrix[i] = w.wyckmatrix[i];
	rotatematrix.clear(); transmatrix.clear();
	for (int i = 0; i < w.rotatematrix.size(); i++) rotatematrix.push_back(w.rotatematrix[i]);
	for (int i = 0; i < w.transmatrix.size(); i++) transmatrix.push_back(w.transmatrix[i]);
	unique = w.unique;
	variables = w.variables;
	symops.clear(); symmatrix.clear();
	symop=w.symop;
	for(int i=0;i<w.symops.size();i++) symops.push_back(vector<int>(w.symops[i]));

	symmatrix.resize(w.symmatrix.size());
	for(int i=0;i<w.symmatrix.size();i++) 
	{
		symmatrix[i].resize(w.symmatrix[i].size());
		for(int j=0;j<w.symmatrix[i].size();j++)
			for(int k=0;k<w.symmatrix[i][j].size();k++) symmatrix[i][j].push_back(vector<double>(w.symmatrix[i][j][k]));
	}
}

WyckPos::WyckPos(char l, int m, double* w, vector<double>* r, vector<double>* t, bool u, vector<int>& s)
{
	label = l;
	for (int i = 0; i < 12; i++) wyckmatrix[i] = w[i];
	for (int i = 0; i < r->size(); i++) rotatematrix.push_back((*r)[i]);
	for (int i = 0; i < t->size(); i++) transmatrix.push_back((*t)[i]);
	unique = u;
	multiplicity = m;
	variables = fabs(wyckmatrix[0]) + fabs(wyckmatrix[5]) + fabs(wyckmatrix[10]);

	symop=s.size();

	if(symop>0) 
	{
		symops.resize(10);
		for(int i=0;i<10;i++) symops[i].resize(0);
		for(int i=0;i<s.size();i++) 
		{
			int t=s[i]/10;
			int tt=s[i]-t*10;
			if(t>10) {t/=10;tt=s[i]-t*100;}
			symops[t].push_back(tt);
		}

		symmatrix.resize(10);
		for(int i=2;i<symmatrix.size();i++)
		{
			symmatrix[i].resize(symops[i].size());
			for(int j=0;j<symmatrix[i].size();j++)				
			{
				symmatrix[i][j].resize(10);
				for(int k=2;k<10;k++)
				{
					symmatrix[i][j][k].resize(symops[k].size());
				}
			}
		}
		for(int i=2;i<symmatrix.size();i++)
		{
			for(int j=0;j<symmatrix[i].size();j++)
			{
				for(int ii=2;ii<symmatrix[i][j].size();ii++)
				{
					for(int jj=0;jj<symmatrix[i][j][ii].size();jj++)
					{
						symmatrix[i][j][ii][jj]=fabs(Dotproduct(symaxis[i][symops[i][j]], symaxis[ii][symops[ii][jj]]));
					}
				}
			}
		}
	}
	return;
};



void GetPosition(double* m, position* p, position* presult)
{
	position temp1(m[0], m[1], m[2]);
	position temp2(m[4], m[5], m[6]);
	position temp3(m[8], m[9], m[10]);
	position temp4(m[3], m[7], m[11]);

	position temp(Dotproduct(&temp1, p), Dotproduct(&temp2, p), Dotproduct(&temp3, p));
	Padd(1, 1, &temp, &temp4, presult);
	return;
};//m[:,:-1].dot(p)+m[:,-1]

/*202203: acceptance-rejection method of generation random xr of d(x) distribution:
	(1) d(x) known, guess a g(x) >= d(x) [g(x)=1 used]
	(2) u = uniform(0,1), if u <= d(xr)/cg(xr), accept xr
	**Note: d(x)/g(x)[g=1] <= 1, c = 1
*/

position WyckPos::GetOnePosition(std::function<double(double, double, double)> distribution = 0)
{
	position randp(Rand(), Rand(), Rand());
	if (distribution == 0) {}
	else
		for (int i=0; i<500; i++)
		//while(true)
		{
			randp.x = Rand(); randp.y = Rand(); randp.z = Rand();
			double u = Rand();
			//DEBUG_INFO("%f, %f, %f, %f, %f\n", u, randp.x, randp.y, randp.z, distribution(randp.x, randp.y, randp.z));
			if (u > distribution(randp.x, randp.y, randp.z)) break;
		}
	
	position temp;
	GetPosition(wyckmatrix, &randp, &temp);
	return Standform(&temp);
}

position WyckPos::GetOnePosition(double mindistance, double maxdistance)
{
	double theta = acos(2.0*Rand() -1) , phi=2*M_PI*Rand();
	double dis = Rand()*(maxdistance-mindistance)+mindistance;
	position randp = position (dis*sin(theta)*cos(phi), dis*sin(theta)*sin(phi), dis*cos(theta)) ;	
	position temp;
	GetPosition(wyckmatrix, &randp, &temp);
	if (temp.sym(position(0,0,0), 0.01)) {}
	else
		temp *= dis / sqrt(CalDistance(&temp));
	return temp;
}
void WyckPos::GetAllPosition(vector<position>* positions, position p, bool* pbc)
{
	vector<position> tempositions;
	position temp;
	positions->clear();
	double matrix[12];
	for (int i = 0; i < rotatematrix.size() / 12; i++)
	{
		for (int j = 0; j < 12; j++) matrix[j] = rotatematrix[12 * i + j];
		GetPosition(matrix, &p, &temp);
		tempositions.push_back(temp);
	}

	for (int j = 0; j < tempositions.size(); j++)
	{
		for (int i = 0; i < transmatrix.size() / 3; i++)
		{
			position tempp(transmatrix[3 * i], transmatrix[3 * i + 1], transmatrix[3 * i + 2]);
			Padd(1, 1, &tempp, &tempositions[j], &temp);
			tempp = Standform(&temp, pbc);
			if (positions->size() == 0) positions->push_back(tempp);
			else
				for (int i = 0; i < positions->size(); i++)
				{
					if (tempp.sym( (*positions)[i], 1e-2))  break;
					if (i == positions->size() - 1) { positions->push_back(tempp); break; }
				}

			if (positions->size() == multiplicity) return;
		}
	}
	return;
}
void WyckPos::GetAllPosition(vector<position>* positions, position p, vector<position>& clusterpos, vector<position>& com_poss)
{
	positions->clear();
	com_poss.clear();
	vector<int> rotatetype;
	vector<position> tempositions;
	position temp;
	//for(int i=0;i<clusterpos.size();i++) cout<< clusterpos[i].x<<'\t'<<clusterpos[i].y<<'\t'<<clusterpos[i].z<<endl;
	double matrix[12];
	for (int i = 0; i < rotatematrix.size() / 12; i++)
	{
		for (int j = 0; j < 12; j++) matrix[j] = rotatematrix[12 * i + j];
		GetPosition(matrix, &p, &temp);
		tempositions.push_back(temp);
	}

	for (int j = 0; j < tempositions.size(); j++)
	{
		for (int i = 0; i < transmatrix.size() / 3; i++)
		{
			position tempp(transmatrix[3 * i], transmatrix[3 * i + 1], transmatrix[3 * i + 2]);
			Padd(1, 1, &tempp, &tempositions[j], &temp);
			tempp = Standform(&temp);
			if (com_poss.size() == 0) {com_poss.push_back(tempp); rotatetype.push_back(j);}
			else
				for (int i = 0; i < com_poss.size(); i++)
				{
					if (tempp.sym( com_poss[i]))  break;
					if (i == com_poss.size() - 1) { com_poss.push_back(tempp); rotatetype.push_back(j); break; }
				}
				
			if (com_poss.size() == multiplicity) break;
		}
		if (com_poss.size() == multiplicity) break;
	}

	for(int i=0;i<com_poss.size();i++)
	{
		
		double matrix[12];
		for (int j = 0; j < 12; j++) matrix[j] = rotatematrix[12 *rotatetype[i] + j];
		
		for(int a=0;a<clusterpos.size();a++)
		{
			position tp=clusterpos[a].rotate(matrix);
			positions->push_back(com_poss[i]+tp);
		}
	}
	
	//for(int i=0;i<positions->size();i++) cout<<(*positions)[i].x<<'\t'<<(*positions)[i].y<<'\t'<<(*positions)[i].z<<'\t'<<endl;

	return;
};
void WyckPos::GetAllPosition_PointGroup(vector<position>* positions, position p)
{
	vector<position> tempositions;
	position temp;
	positions->clear();
	double matrix[12];
	for (int i = 0; i < rotatematrix.size() / 12; i++)
	{
		for (int j = 0; j < 12; j++) matrix[j] = rotatematrix[12 * i + j];
		GetPosition(matrix, &p, &temp);
		tempositions.push_back(temp);
	}

	for (int j = 0; j < tempositions.size(); j++)
	{
		if (positions->size() == 0) positions->push_back(tempositions[j]);
		else
			for (int i = 0; i < positions->size(); i++)
			{
				if (tempositions[j].sym( (*positions)[i], 1e-2))  break;
				if (i == positions->size() - 1) { positions->push_back(tempositions[j]); break; }
			}

		if (positions->size() == multiplicity) return;
	}
	return;
}
