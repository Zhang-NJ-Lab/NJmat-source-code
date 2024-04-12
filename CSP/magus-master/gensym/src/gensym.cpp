/*gensym.cpp: Definition of
	class Atoms;
	class Structure;
	function
	@[Two methods to get combinations]
		[method=1, randomly get some (not all) combinations to use] 
			void GetAllCombination(Structure structure, vector<WyckGroup> wycks, vector<Structure> &combinations,bool forceMostGeneralWyckPos,vector<int>* biasedw,int wsum)
		[method=2, get all possible combinations] 
			void GetAllCombinations(Structure structure, vector<WyckGroup> wycks, vector<Structure> &combinations, bool forceMostGeneralWyckPos, int attemptstoGetCombs)
	@[cluster wyckoff positions according to multiplicity, unique, (and symmetry in molucule structure). ]
		void GetWyckPosGrouped(vector<WyckGroup>& wycks,int i, bool dividesym=false)

	are used in main.cpp*/

#include <cstdlib> 
#include <cmath> 
#include <ctime> 
#include <vector>
#include <iostream> 
#include <fstream>
#include <string>
#include <map>
#include <functional>

using namespace std;

#include"DataBase/spacegroupData.h"
#include"DataBase/sitesymData.h"
#include"DataBase/layergroupData.h"
#include"DataBase/planegroupData.h"
#include"DataBase/pointgroupData.h"
#include"DataBase/rodgroupData.h"
#include"getlatticeparm.h"
#include"position.h"
#include"cluster.h"
#include"wyckoffposition.h"
#include"atom.h"

#include"debug.h"

#define M_PI 3.14159265358979323846

class Atoms
{
public:
	Atom atom;
	int number;
	int left;
	vector<WyckGroup> wyckGroups;
	
	vector<int> chosenWycks;
	vector<position> positions;
	vector<position> positions_wyck;
	vector<position> pos_molcenter;

	bool UsedMostGeneral; //UsedMostGeneral is for the second method.
	Atoms(int n, const char* name, double r, bool m, const char* filename =0, double _symprec = 1e-2)  ;
	Atoms(int n, const char* name, cluster* clus) ;
	Atoms(const Atoms& a) ;
	void operator =(const Atoms& a) ;
};

class Structure
{
public:
	vector<Atoms> atoms;
	double latticeparm[9];
	double volume;
	double atomvolume;
	int spg;
	bool legal;
	double maxr;
	double mu_gussian;
	bool UsedMostGeneral;

	int dimension;
	bool pbc[3];
	int choice;
	double vacuum;

	Structure() { legal=false; for(int i =0;i<3;i++) pbc[i]=true; mu_gussian = 1;};
	Structure(const Structure& s) ;
	void operator =(const Structure& s) ;
	
	
	bool AllAtomUsed(void);
	/*check if each atom in structure has a position*/
	double GetVolume(void) ;
	/*calculate structure volume*/
	int ChooseLattice(double* latticeMins, double* latticeMaxes, double volumeMin, double volumeMax, int celltype = 1) ;
	/*get lattice parameters with constrains @lattice Min and Max, @volume Min and Max, and spacegroup. 
	    @celltype [If set to generate primitive cell parms, 1-6 to represent 'P/C/F/I/A/R' cell.]*/
	void ChooseWyck(void) ;
	/*fill chosenWycks in class Atoms*/

	double _atom_distribution_(double x, double y, double z);
	/*get the distribution of atom positions in this structure. Normally One atom has a guassian distribution of 
		g = [1/sqrt(2*pi)^d] * exp[-(x-x0)^2/2/mu_x^2 -(y-y0)^2/2/mu_y^2 -(z-z0)^2/2/mu_z^2]
		And this function gives the sum of all existed atoms in this structure 
		to make random positions of incoming new atoms distributes better.*/
	
	void MakeCrystal(std::function<double(const char*, const char*)> threshold, double threshold_mol, int maxAttemps, double* latticeMins, double* latticeMaxes) ;
	/*get a position for each atom. Arguements: @threshold [ distance between two atoms must larger than sum(radius)*threshold ]
		@threshold_mol [distance between two center_of_molecules must larger than sum(radius_BoundingSphere)*threshold_mol ]
		@maxAttemps
		if fails, self.legal will be set to false. */
	void MakeCluster(std::function<double(const char*, const char*)> threshold, int maxAttemps) ;
	/*make a cluster according to point group, and get a position for each atom. 
		Arguements: @threshold [ distance between two atoms must larger than sum(radius)*threshold ]
		@maxAttemps
		if fails, self.legal will be set to false. */
	/*update210928: changed threshold from double to a pointer to a function defines threshold(double). Aiming get different thds for different atom types.*/
	void WritePoscar(string* filename,char c);
	/*write a file contains all information to 'filename'. Default setting is vasp-poscar format. Other formats such as gulp-gin need to be slightly modified. */
	void AddWyckGroup(vector<WyckGroup>* wycks);
	/*a function only to pass data. */
};

/**********************************Here begins the detailed function. ***/
Atoms::Atoms(int n, const char* name, double r, bool m, const char* filename, double _symprec) :atom (name, r, m,filename, _symprec)
{
	number = n;
	left = n;
	UsedMostGeneral = false;
};
Atoms::Atoms(int n, const char* name, cluster* clus): atom(name,clus)
{
	number=n;
	left = n;
	UsedMostGeneral = false;
};
Atoms::Atoms(const Atoms& a)
{
	atom = a.atom;
	number = a.number;
	left = a.left;
	for (int i = 0; i < a.wyckGroups.size(); i++) wyckGroups.push_back(a.wyckGroups[i]);
	for (int i = 0; i < a.chosenWycks.size(); i++) chosenWycks.push_back(a.chosenWycks[i]);
	for (int i = 0; i < a.positions.size(); i++) positions.push_back(a.positions[i]);
	for (int i = 0; i < a.positions_wyck.size(); i++) positions_wyck.push_back(a.positions_wyck[i]);
	for (int i = 0; i < a.pos_molcenter.size(); i++) pos_molcenter.push_back(a.pos_molcenter[i]);
	UsedMostGeneral = a.UsedMostGeneral;
};
void Atoms::operator =(const Atoms& a)
{
	atom = a.atom;
	number = a.number;
	left = a.left;
	wyckGroups.clear(); chosenWycks. clear(); positions.clear(); positions_wyck.clear();
	pos_molcenter.clear();
	for (int i = 0; i < a.wyckGroups.size(); i++) wyckGroups.push_back(a.wyckGroups[i]);
	for (int i = 0; i < a.chosenWycks.size(); i++) chosenWycks.push_back(a.chosenWycks[i]);
	for (int i = 0; i < a.positions.size(); i++) positions.push_back(a.positions[i]);
	for (int i = 0; i < a.positions_wyck.size();i++) positions_wyck.push_back(a.positions_wyck[i]);
	for (int i = 0; i < a.pos_molcenter.size(); i++) pos_molcenter.push_back(a.pos_molcenter[i]);
	UsedMostGeneral = a.UsedMostGeneral;
};

vector<WyckPos> wyckpositions;

void GetAllPosition(WyckPos *w, vector<position>* positions, position p, Atom *atom, vector<position>& pos_com, int dimention = 3, bool* pbc = 0)
{
	if ( dimention != 0)
	{
		if(atom->num==1)
		{
			w->GetAllPosition(positions,p, pbc);
		}
		else
		{
			w->GetAllPosition(positions, p, (atom->c).cart_positions_frac, pos_com);
			//for(int i=0;i<(atom->c).cart_positions_frac.size();i++) 
				//DEBUG_INFO("%d \t %d \t %d \n", (atom->c).cart_positions_frac[i].x, (atom->c).cart_positions_frac[i].y,(atom->c).cart_positions_frac[i].z);
		}
	}
	else 
	{
		w->GetAllPosition_PointGroup(positions,p);
	}
	
	return;
};

void matchmatrix(const vector< vector<position> >& clusym,const vector< vector <vector <vector <double> > > >& wycksym,  int ith, vector< vector<position> >& chosenaxis, bool& matrixmatch)
{
	
	if(matrixmatch==true) return;

	if(ith==9) 
	{
		bool match=true;
		for(int i=0;i<wycksym[ith].size();i++)
		{
			bool mark=false;
			for(int a=0;a<clusym[ith].size();a++)
			{
				position p(clusym[ith][a]);
				bool chosep=true;

				for(int x=2;x<10;x++)
				{
					for(int xx=0;xx<chosenaxis[x].size();xx++)
					{
						if(fabs(fabs(Dotproduct(&p, &chosenaxis[x][xx]))-wycksym[ith][i][x][xx])>1e-2)
						{
							chosep=false;break;
						}  
					}
					if(chosep==false) break;
				}	
				if(chosep) 
				{
					chosenaxis[ith].push_back(p);mark=true;break;
				}
			}
			if(mark==false) {match=false; break;}
		}
		if(match) {matrixmatch=true;return;}
	}

	else
	{
		if(wycksym[ith].size()==0)  return matchmatrix(clusym, wycksym, ith+1, chosenaxis, matrixmatch);
		else
		{
			for(int i=0;i<wycksym[ith].size();i++)
			{
				for(int a=0;a<clusym[ith].size();a++)
				{
					position p(clusym[ith][a]);
					bool chosep=true;

					for(int x=2;x<10;x++)
					{
						for(int xx=0;xx<chosenaxis[x].size();xx++)
						{
							if(fabs(fabs(Dotproduct(&p, &chosenaxis[x][xx]))-wycksym[ith][i][x][xx])>1e-2)
							{
								chosep=false;break;
							}  
						}
						if(chosep==false) break;
					}	
					if(chosep) 
					{
						vector< vector<position> > axis;
						axis.resize(chosenaxis.size());
						for(int x=0;x<chosenaxis.size();x++)
							for(int xx=0;xx<chosenaxis[x].size();xx++)
								axis[x].push_back(chosenaxis[x][xx]);

						axis[ith].push_back(p);
						return matchmatrix(clusym, wycksym, ith+1, axis,matrixmatch);
					}
				}
				
			}
		}
	}
};


bool matchsym(const vector< vector<int> >& wycksymops, int symop, const vector< vector <vector <vector <double> > > >& wycksymmatrix ,const vector< vector<position> >& clusym, int atomdim, const position& atomvec)
{
	
	if(atomdim>1)
	{
		for(int i=0;i<wycksymops.size();i++)
		{
			if(clusym[i].size()<wycksymops[i].size())  return false;
		}
	}
	else 
	{	
		for(int i=0;i<wycksymops.size();i++)
		{
			if(i==3) continue;
			if(i==2 & clusym[1].size()>0) continue;
			if(clusym[i].size()<wycksymops[i].size())  return false;
		}
		if(symop>1)
		{
			int i; position tp; 
			if(clusym[1].size()==0) i=2; else i=4;
			for(;i<wycksymops.size();i++)
			{
				if(i==3) continue;
				if(wycksymops[i].size()>0) 
				{
					tp=symaxis[i][wycksymops[i][0]];
					break;
				}
			}
			if(tp==position(0,0,0)) return true;

			i++;
			for(;i<wycksymops.size();i++)
			{
				if(i==3) continue;
				if(wycksymops[i].size()>0) 
				{
					if(fabs(fabs(Dotproduct(symaxis[i][wycksymops[i][0]], tp))-1)>1e-2) return false;
				}
			}
			for(int j=0;j<wycksymops[3].size();j++)
			{
				double d=fabs(Dotproduct(symaxis[3][wycksymops[3][j]], tp));
				if(d<1e-2) continue;
				if(fabs(d-1)<1e-2) 
				{
					if(clusym[3].size()>0) continue;
				}
				return false;
			}
			if(clusym[1].size()>0)
			{
				for(int j=0;j<wycksymops[2].size();j++)
				{
					double d=fabs(Dotproduct(symaxis[2][wycksymops[2][j]], tp));
					if(d<1e-2) continue;
					if(fabs(d-1)<1e-2) continue;
					return false;
				}
			}
		}
		return true;
	}

	if(symop==1) return true;

	else
	{
		vector< vector<position> >chosenaxis;
		chosenaxis.resize(10);
		bool match=false; 
		matchmatrix(clusym, wycksymmatrix, 2, chosenaxis,match);
		if(match==true) return true;
		else return false;
	}

}

bool IsUsable(WyckGroup* w, Atoms* atoms)
{
	if ((*w->SimilarWyck)[0].multiplicity > atoms->left)  return false;
	if (((*w->SimilarWyck)[0].unique == true)&(w->count >= (w->SimilarWyck)->size())) return false;

	if (atoms->atom.num>1) 
	{
		if((*w->SimilarWyck)[0].symop>0)
			if ( matchsym((*w->SimilarWyck)[0].symops,(*w->SimilarWyck)[0].symop, (*w->SimilarWyck)[0].symmatrix, atoms->atom.c.clus->symmetry, atoms->atom.c.clus->dimention, atoms->atom.c.clus->vec)==false) return false;
	}
	return true;
};
void AddVacuum(vector<position> & p, double* la, double vacuum, int spg)
{
	int axis;
	/*non-periodic axis c */
	
	double ratio;

	axis=2; //axis c
	double c=sqrt(la[2]*la[2]+ la[5]*la[5] + la[8]*la[8]);
	ratio= c/(c+vacuum);

	/*axis=0; //axis a
	ratio= la[0]/(la[0]+vacuum);*/
	

	for(int i=0;i<p.size();i++)
	{
		p[i].z*=ratio;
		//else if(axis==0) p[i].x*=ratio;
	}

	return;
}

Structure::Structure(const Structure& s)
{
	for (int i = 0; i < s.atoms.size(); i++) atoms.push_back(Atoms(s.atoms[i]));
	for (int i = 0; i < 9; i++) latticeparm[i] = s.latticeparm[i];
	volume = s.volume;
	atomvolume = s.atomvolume;
	spg = s.spg;
	legal = s.legal;
	maxr = s.maxr;
	mu_gussian = s.mu_gussian;
	UsedMostGeneral = s.UsedMostGeneral;
	for(int i = 0;i<3;i++) pbc[i] = s.pbc[i];
};
void Structure::operator =(const Structure& s)
{
	atoms.clear();
	for (int i = 0; i < s.atoms.size(); i++) atoms.push_back(Atoms(s.atoms[i]));
	for (int i = 0; i < 9; i++) latticeparm[i] = s.latticeparm[i];
	for(int i = 0;i<3;i++) pbc[i] = s.pbc[i];
	volume = s.volume;
	atomvolume = s.atomvolume;
	spg = s.spg;
	legal = s.legal;
	maxr = s.maxr;
	mu_gussian = s.mu_gussian;
	UsedMostGeneral = s.UsedMostGeneral;
};

bool Structure::AllAtomUsed(void)
{
	for (int i = 0; i < atoms.size(); i++)
		if (atoms[i].left > 0) return false;
	return true;
};



double Structure::GetVolume(void)
{
	double*t = latticeparm;
	return t[0] * (t[4] * t[8] - t[5] * t[7]) - t[1] * (t[3] * t[8] - t[5] * t[6]) + t[2] * (t[3] * t[7] - t[4] * t[6]);
	//determinant of 3*3 matrix "latticeparm"
};

int Structure::ChooseLattice(double* latticeMins, double* latticeMaxes, double volumeMin, double volumeMax, int celltype)
{
	//For 0D clusters, skip choose lattice, copy latticemin to latticeparm[0] and latticemax to lattceparm[1].
	if (dimension ==0) 
	{
		latticeparm[0] = Min(latticeMins[0], latticeMaxes[0]); 
		latticeparm[1]= Max(latticeMins[0], latticeMaxes[0]);  
		return 1;
	}

	int attempt = 0;
	double v = 0;
	GetLatticeParm g = GetLatticeParm(spg, latticeMins, latticeMaxes, dimension, choice, celltype);
	if(g.GetLattice(latticeparm) == false)
	{
		DEBUG_INFO("error: spacegroup does not match anglemins and anglemaxes\n");
		return -2;
	};
	v = GetVolume();
	while (v >= volumeMax+0.1 || v <= volumeMin-0.1 || std::isnan(v))
	{
		attempt++;
		if (attempt > 10000) 
		{ 
			DEBUG_INFO("error: failed ChooseLattice(), reconsider volumeMax and volumeMin\n"); 
			return -1; 
		}
		g.GetLattice(latticeparm);
		v = GetVolume();
	}
	volume = v;
	return 1;
};
void Structure::ChooseWyck(void)
{
	for (int i = 0; i < atoms[0].wyckGroups.size(); i++)
	{
		if ((*atoms[0].wyckGroups[i].SimilarWyck)[0].unique == true)
		{
			int count = 0;
			for (int j = 0; j < atoms.size(); j++)
				count += atoms[j].wyckGroups[i].count;
			vector<int> chosen;
			while (chosen.size() < count)
			{
				bool l = true;
				int temp = rand() % (atoms[0].wyckGroups[i].SimilarWyck->size());
				for (int k = 0; k < chosen.size(); k++)
				{
					if (temp == chosen[k]) {l = false; break;}	
				}
				if (l == true) chosen.push_back(temp);
			}
			count = 0;
			for (int j = 0; j < atoms.size(); j++)
			{
				for (int k = count; k < count+atoms[j].wyckGroups[i].count; k++) 
					atoms[j].chosenWycks.push_back((*atoms[0].wyckGroups[i].SimilarWyck)[chosen[k]].label-'a');
				count += atoms[j].wyckGroups[i].count;
			}
		}
		else
		{
			for (int j = 0; j < atoms.size(); j++) 
			{
				int chosen = 0;
				while (chosen < atoms[j].wyckGroups[i].count)
				{
					int k = rand() % (atoms[j].wyckGroups[i].SimilarWyck->size());
					atoms[j].chosenWycks.push_back((*atoms[0].wyckGroups[i].SimilarWyck)[k].label - 'a');
					chosen++;
				}
			}				
		}			
	}
	return;
}

double Structure::_atom_distribution_(double x, double y, double z)
{
	double g = 0;
	int num = 0; 
	for (int i =0; i<atoms.size();i++)
		for (int j=0; j<atoms[i].positions.size();j++)
		{
			g += exp( (-pow(x-atoms[i].positions[j].x, 2) -pow(y-atoms[i].positions[j].y, 2) -pow(z-atoms[i].positions[j].z, 2) )
									/2/pow(mu_gussian*atoms[i].atom.radius, 2) );
			num++;
		}
	if (num > 0) g = g/num;  

	return g;
};

void Structure::MakeCrystal(std::function<double(const char*, const char*)> threshold,double threshold_mol, int maxAttemps, double* latticeMins, double* latticeMaxes)
{
	int num = 0;
	int attempt = 0;
	bool shouldadd;
	position cluster_center;
	double cluster_radius;
	
	for (int i = 0; i < atoms.size(); i++)
	{
		for (int j = 0; j < atoms[i].chosenWycks.size(); j++)
		{
			attempt = 0;
			WyckPos* w_atomi_chosenj=&wyckpositions[atoms[i].chosenWycks[j]];

			while (attempt < abs(w_atomi_chosenj->variables)*maxAttemps / 2 + 1)
			{
				if(atoms[i].atom.num!=1) 
				{
					if(w_atomi_chosenj->symop>0)
					{	
						if(atoms[i].atom.c.RePlace(latticeparm,w_atomi_chosenj->symops, w_atomi_chosenj->symop)) {}
						else {legal=false; return;}
					}
					else
						atoms[i].atom.c.RePlace(latticeparm);
				}
				//DEBUG_INFO("makecrystal-A %u \n", attempt);
				shouldadd = true;
				vector<position> p;
				vector<position> p_com;
				position pos_wyck=w_atomi_chosenj->GetOnePosition(std::bind(&Structure::_atom_distribution_, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3) );
				GetAllPosition(w_atomi_chosenj, &p, pos_wyck, &(atoms[i].atom), p_com, 3 ,pbc);

				if (p.size() != w_atomi_chosenj->multiplicity*atoms[i].atom.num)
					{ shouldadd = false; attempt++; continue; }
				//DEBUG_INFO("makecrystal-B\n");
				
				if (CheckDistance(&p, &p_com, &(atoms[i].positions), &(atoms[i].pos_molcenter), &(atoms[i].atom), latticeparm, threshold, threshold_mol, pbc) == false)
				{
					shouldadd = false;
					attempt++;
					continue;
				}
				//DEBUG_INFO("makecrystal-C\n");
				for (int k = 0; k < i; k++)
				{
					if (CheckDistance(&p, &p_com, &(atoms[k].positions), &(atoms[k].pos_molcenter), &(atoms[i].atom), &(atoms[k].atom), latticeparm, threshold, threshold_mol, pbc) == false)
					{
						shouldadd = false;
						break;
					}
				}

				if (pbc[2] == false)
				{
					if ( i+j == 0)
						{cluster_center.z = p[0].z; cluster_radius=0;}
				
					double new_center = cluster_center.z, r = cluster_radius;
					//cout<<"a\t"<<"cluster-center = "<<cluster_center.z<<"\tradius= "<<cluster_radius<<endl;
					for(int k=0; k<p.size(); k++)
					{
						if (p[k].z > new_center + r) 
						{
							new_center = (p[k].z + new_center - r) /2 ;
							r = p[k].z - new_center;
						}	
						else if ( p[k].z < new_center - r )
						{
							new_center = (new_center + r + p[k].z ) /2 ;
							r = new_center - p[k].z;
						}
						//cout<<"position = "<<p[k]<<endl;
						//cout<<"b\t"<<"new-cluster-center = "<<new_center<<"\tradius= "<<r<<endl;
						if (2 * r*latticeparm[8] > latticeMaxes[2])
							{shouldadd = false; break;}

					}
					if (shouldadd)
						{cluster_center.z = new_center; cluster_radius = r;}
				}

				//DEBUG_INFO("makecrystal-D\n");
				if (shouldadd)
				{
					for (int ii = 0; ii < p.size(); ii++) atoms[i].positions.push_back(p[ii]);
					atoms[i].positions_wyck.push_back(pos_wyck);
					for(int ii=0;ii<p_com.size();ii++) atoms[i].pos_molcenter.push_back(p_com[ii]);
					break;
				}
				attempt++;
			}
			if (shouldadd == false)
			{
				legal = false;
				return;
			}
		}
	}
	for (int i = 0; i < atoms.size(); i++)
		if (atoms[i].positions.size() != atoms[i].number*atoms[i].atom.num)
		{
			legal = false;
			DEBUG_INFO("%s \t %d \t %d \n", atoms[i].atom.name, atoms[i].positions.size(), atoms[i].number*atoms[i].atom.num) ;
			for (int i = 0; i < atoms.size(); i++)
			{
				DEBUG_INFO("errorlog for %s :\n", atoms[i].atom.name);
				for (int j = 0; j < atoms[i].chosenWycks.size(); j++)
					DEBUG_INFO("%u \t %s,\t", wyckpositions[atoms[i].chosenWycks[j]].multiplicity , wyckpositions[atoms[i].chosenWycks[j]].label );
				DEBUG_INFO("\n");
			}
			return;
		}

	if(dimension==2) 
	{
		//cout<<"thickness = "<<2*cluster_radius*sqrt(latticeparm[2]*latticeparm[2] + latticeparm[5]*latticeparm[5]+ latticeparm[8]*latticeparm[8])<<endl;
		if(2*cluster_radius*latticeparm[8] > latticeMins[2]) {legal = false;return;}
		
		double c=sqrt(latticeparm[2]*latticeparm[2]+ latticeparm[5]*latticeparm[5] + latticeparm[8]*latticeparm[8]);
		double ratio=(c+vacuum)/c;
		latticeparm[2]*=ratio;
		latticeparm[5]*=ratio;
		latticeparm[8]*=ratio;
		

		double bottom = 0 ; 
		for (int i = 0; i < atoms.size(); i++)
		{		
			for(int j=0;j<atoms[i].positions.size();j++)
			{
				atoms[i].positions[j].z/=ratio;
			}
			for (int j = 0; j<atoms[i].positions.size(); j++) bottom = Min(bottom, atoms[i].positions[j].z);
		}
		for (int i = 0; i < atoms.size(); i++)
			for (int j = 0; j<atoms[i].positions.size(); j++) 
				atoms[i].positions[j].z -= bottom;
	}
	else if (dimension == 1)
	{
		double a = latticeparm[0], b = sqrt(latticeparm[1]*latticeparm[1] + latticeparm[4]*latticeparm[4]);
		double ra=(a+vacuum)/a, rb=(b+vacuum)/b;
		latticeparm[0]*=ra;
		latticeparm[1]*=rb;
		latticeparm[4]*=rb;
		

		double bottom_a = 0, bottom_b = 0 ; 
		for (int i = 0; i < atoms.size(); i++)
		{		
			for(int j=0;j<atoms[i].positions.size();j++)
			{
				atoms[i].positions[j].x/=ra;
				atoms[i].positions[j].y/=rb;
			}
			for (int j = 0; j<atoms[i].positions.size(); j++) 
			{
				bottom_a = Min(bottom_a, atoms[i].positions[j].x);
				bottom_b = Min(bottom_b, atoms[i].positions[j].y);
			}
		}
		for (int i = 0; i < atoms.size(); i++)
			for (int j = 0; j<atoms[i].positions.size(); j++) 
			{	
				atoms[i].positions[j].x -= bottom_a;
				atoms[i].positions[j].y -= bottom_b;
			}
	}
	legal = true;
	return;
};
void Structure::MakeCluster(std::function<double(const char*, const char*)> threshold, int maxAttemps)
{
	int num = 0;
	int attempt = 0;
	bool shouldadd;

	for (int i = 0; i < atoms.size(); i++)
	{
		for (int j = 0; j < atoms[i].chosenWycks.size(); j++)
		{
			attempt = 0;
			WyckPos* w_atomi_chosenj=&wyckpositions[atoms[i].chosenWycks[j]];

			while (attempt < abs(w_atomi_chosenj->variables)*maxAttemps / 2 + 1)
			{
				shouldadd = true;
				vector<position> p;
				position pos_wyck=w_atomi_chosenj->GetOnePosition(latticeparm[0], latticeparm[1]);
				vector<position> pos_;	//to match number of arguements, not used.
				GetAllPosition(w_atomi_chosenj, &p, pos_wyck, &(atoms[i].atom), pos_, 0);

				if (p.size() != w_atomi_chosenj->multiplicity*atoms[i].atom.num)
					{ shouldadd = false; attempt++; continue; }
				//DEBUG_INFO("makecrystal-A %u \n", attempt);
				
				//Checks if has atom too far from origin
				for (int _a = 0;_a<p.size();_a++)
				{
					if (CalDistance(&p[_a]) > latticeparm[1]*latticeparm[1]) 
						{ shouldadd = false; attempt++; continue; }
				}
				//DEBUG_INFO("makecrystal-B\n");
				
				if (CheckDistance_withinCluster(p, atoms[i].positions, atoms[i].atom.radius, threshold(atoms[i].atom.name, atoms[i].atom.name) ) == false)
				{
					shouldadd = false;
					attempt++;
					continue;
				}
				//DEBUG_INFO("makecrystal-C\n");
				for (int k = 0; k < i; k++)
				{
					if (CheckDistance_withinCluster(p, atoms[k].positions, atoms[i].atom.radius, atoms[k].atom.radius,threshold(atoms[i].atom.name, atoms[k].atom.name)) == false)
					{
						shouldadd = false;
						break;
					}
				}
				//DEBUG_INFO("makecrystal-D\n");
				if (shouldadd)
				{
					for (int ii = 0; ii < p.size(); ii++) atoms[i].positions.push_back(p[ii]);
					atoms[i].positions_wyck.push_back(pos_wyck);
					break;
				}
				attempt++;
			}
			if (shouldadd == false)
			{
				legal = false;
				return;
			}
		}
	}
	for (int i = 0; i < atoms.size(); i++)
		if (atoms[i].positions.size() != atoms[i].number*atoms[i].atom.num)
		{
			legal = false;
			DEBUG_INFO("%s \t %d \t %d \n", atoms[i].atom.name, atoms[i].positions.size(), atoms[i].number*atoms[i].atom.num) ;
			for (int i = 0; i < atoms.size(); i++)
			{
				DEBUG_INFO("errorlog for %s :\n", atoms[i].atom.name);
				for (int j = 0; j < atoms[i].chosenWycks.size(); j++)
					DEBUG_INFO("%u \t %s,\t", wyckpositions[atoms[i].chosenWycks[j]].multiplicity , wyckpositions[atoms[i].chosenWycks[j]].label );
				DEBUG_INFO("\n");
			}
			return;
		}

	//get a set of latticeparm for compatibility here.

	double x1 =0, x2 =0, y1=0, y2=0, z1=0, z2=0;
	for (int i=0;i<atoms.size();i++)
		for(int j=0;j<atoms[i].positions.size();j++)
		{
			x1 = Min(x1, atoms[i].positions[j].x); x2 = Max(x1, atoms[i].positions[j].x);
			y1 = Min(y1, atoms[i].positions[j].y); y2 = Max(y1, atoms[i].positions[j].y);
			z1 = Min(z1, atoms[i].positions[j].z); z2 = Max(z1, atoms[i].positions[j].z);
		}
	double lattice = (Max(x2-x1, y2-y1, z2-z1) + vacuum) / 2;

	for(int i=0;i<9;i++) latticeparm[i]=0;
	latticeparm[0] = 2*lattice; latticeparm[4] = latticeparm[0]; latticeparm[8] = latticeparm[0];
	for (int i = 0; i < atoms.size(); i++)
	{
		for (int j=0;j<atoms[i].positions.size();j++)
		{
			position _temp(atoms[i].positions[j]);
			_temp += position(lattice, lattice, lattice);
			position _pr;
			inversepostrans(&_pr, &_temp, latticeparm);
			atoms[i].positions[j] = _pr; 
		}
	}

	legal = true;
	return;
};
/*void Structure::newposition(position* presult, position* p0, position* p1, position* center, double theta, double phi)
{
	position vec;
	Padd(1, -1, p1, center, &vec);
	double newvec[3];
	newvec[0] = vec.x*cos(theta) - vec.y*sin(theta)*cos(phi) + vec.z*sin(theta)*sin(phi);
	newvec[1] = vec.x*sin(theta) + vec.y*cos(theta)*cos(phi) - vec.z*cos(theta)*sin(phi);
	newvec[2] = vec.y*sin(phi) + vec.z*cos(phi);
	double temp[3][3];
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)	temp[i][j] = latticeparm[3 * i + j];
	double fvec[3];
	solve(temp, newvec, fvec);
	presult->x = p0->x + fvec[0]; presult->y = p0->y + fvec[1]; presult->z = p0->z + fvec[2];
	return;
	//vec=p1-center 
	//M1=[[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]]
	//M2=[[1,0,0],[0,np.cos(phi),-np.sin(phi)],[0,np.sin(phi),np.cos(phi)]]
	//newvec=M1.dot(M2).dot(vec)
	//fvec=LA.solve(self.latticeparm.T,newvec) 
};*/

void Structure::WritePoscar(string* filename,char c)
{
	ofstream out((*filename).c_str());
	switch (c)
	{
	case 'v':
		out << (*filename).c_str() << '\n';
		if (legal)
		{
			out << "1.0\n";
			for (int i = 0; i < 3; i++)
				out << latticeparm[i] << '\t' << latticeparm[i + 3] << '\t' << latticeparm[i + 6] << '\n';
			
			for(int i=0;i<atoms.size();i++)
			{
				if(atoms[i].atom.num==1)  out<<atoms[i].atom.name<<'\t';
				else
					for(int j=0;j<atoms[i].atom.c.clus->Name.size();j++)
						out<<atoms[i].atom.c.clus->Name[j]<<'\t';
			}
			out<<'\n';
			for(int i=0;i<atoms.size();i++)
			{
				if(atoms[i].atom.num==1)  out<<atoms[i].number<<'\t';
				else
				{
					cluster* c=atoms[i].atom.c.clus;
					int num=1;
					const char* ch=c->name[0];
					for(int j=1;j<c->name.size();j++)
					{
						if(c->name[j]==ch) num++;
						else
						{
							out<<num*atoms[i].number<<'\t';
							ch=c->name[j];
							num=1;
						}
					}
					out<<num*atoms[i].number<<'\t';
				}
			}

			out << "\nDirect\n";
			for (int i = 0; i < atoms.size(); i++)
			{
				if(atoms[i].atom.num==1)
					for (int j = 0; j < atoms[i].positions.size(); j++)
						out << atoms[i].positions[j].x << '\t' << atoms[i].positions[j].y << '\t' << atoms[i].positions[j].z << '\n';
				else
				{
					for(int column=0;column<atoms[i].atom.num;column++)
						for(int row=0;row<atoms[i].number;row++)
							out<<atoms[i].positions[row*atoms[i].atom.num+column].x<<'\t'<<atoms[i].positions[row*atoms[i].atom.num+column].y<<'\t'<<atoms[i].positions[row*atoms[i].atom.num+column].z<<'\n';
				}
						
			}
				
		}
		out.close();
		break;

	case 'g':
		out << "opti conj conp\nswitch_minimiser bfgs gnorm 0.5\nvectors\n";
		if (legal)
		{
			for (int i = 0; i < 3; i++)
				out << latticeparm[i] << '\t' << latticeparm[i + 3] << '\t' << latticeparm[i + 6] << '\n';
			out << "fractional\n";
			for (int i = 0; i < atoms.size(); i++)
			{
				
				for (int j = 0; j < atoms[i].positions.size(); j++)
				{
					if(atoms[i].atom.num==1) 
						out << atoms[i].atom.name << " core " << atoms[i].positions[j].x << '\t' << atoms[i].positions[j].y << '\t' << atoms[i].positions[j].z << '\n';
					else 
						out << atoms[i].atom.c.clus->name[j%atoms[i].atom.num] << " core " << atoms[i].positions[j].x << '\t' << atoms[i].positions[j].y << '\t' << atoms[i].positions[j].z << '\n';
				}
			}
					
			out << "species\nTi 2.196\nO -1.098\nbuck\n";
			out << "Ti Ti 31120.1 0.1540 5.25 15\nO  O  11782.7 0.2340 30.22 15\nTi O  16957.5 0.1940 12.59 15\n";
			out << "lennard 12 6\nTi Ti 1 0 15\nO  O  1 0 15\nTi O  1 0 15\n";
		}
		out.close();
		break;
		
	case 't':
		if(legal)
		{
			out<<"import spglib\nlattice=[";
			for (int i = 0; i < 3; i++)
				out << '['<<latticeparm[i] << ',' << latticeparm[i + 3] << ',' << latticeparm[i + 6] << "],\n";
			out<<"]\n";
			out<<"positions=[";
			for (int i = 0; i < atoms.size(); i++)
			{
				if(atoms[i].atom.num==1)
					for (int j = 0; j < atoms[i].positions.size(); j++)
						out << '['<<atoms[i].positions[j].x << ',' << atoms[i].positions[j].y << ',' << atoms[i].positions[j].z << "],\n";
				else
				{
					for(int column=0;column<atoms[i].atom.num;column++)
						for(int row=0;row<atoms[i].number;row++)
							out<<'['<<atoms[i].positions[row*atoms[i].atom.num+column].x<<','<<atoms[i].positions[row*atoms[i].atom.num+column].y<<','<<atoms[i].positions[row*atoms[i].atom.num+column].z<<"],\n";
				}
						
			}
			out<<"]\n";out<<"numbers=[]";
			int count=1;
			for(int i=0;i<atoms.size();i++)
			{
				if(atoms[i].atom.num==1)  {out<<"["<<count<<",]*"<<atoms[i].number;count++;}
				else
				{
					cluster* c=atoms[i].atom.c.clus;
					int num=1;
					const char* ch=c->name[0];
					for(int j=1;j<c->name.size();j++)
					{
						if(c->name[j]==ch) num++;
						else
						{
							out<<"+["<<count<<",]*"<<num*atoms[i].number;count++;
							ch=c->name[j];
							num=1;
						}
					}
					out<<"+["<<count<<",]*"<<num*atoms[i].number;count++;
				}
			}
			
			out<<"\ncell=(lattice,positions,numbers)\n";
			//out<<"lattice, scaled_positions, numbers = spglib.find_primitive(cell, symprec=1e-5)\nprint(lattice)\nprint(scaled_positions)\n";
			out<<"spacegroup=spglib.get_spacegroup(cell, symprec=1e-2)\nprint(spacegroup)\n";
			//out<<"def caldis(p1, p2):\n\treturn (p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2\n\n";
			//out<<"for i in range(289):\n\tfor j in range(i):\n\t\tif(caldis(positions[i], positions[j])<0.76*0.76):\n\t\t\tprint(\"error\")";

			/*out<<"!useKeyWords\n!title\n"<< (*filename).c_str() << "\n!latticeTolerance\n0.0001\n";
			out << "!latticeBasisVectors" << '\n';
			for (int i = 0; i < 3; i++)
				out << latticeparm[i] << '\t' << latticeparm[i + 3] << '\t' << latticeparm[i + 6] << '\n';
			int count = 0;
			for (int i = 0; i < atoms.size(); i++) count += atoms[i].number;
			out << "!atomCount"<<'\n'<<count<<'\n'<<"!atomType"<<'\n';
			for (int i = 0; i < atoms.size(); i++) out << atoms[i].number << "*" << atoms[i].atom.name << ' ';
			out << '\n' << "!atomPosition" << '\n';
			for (int i = 0; i < atoms.size(); i++)
					for (int j = 0; j < atoms[i].positions.size(); j++)
						out << atoms[i].positions[j].x << '\t' << atoms[i].positions[j].y << '\t' << atoms[i].positions[j].z << '\n';*/
		}
		out.close();
		break;
	}
	return;
};

void Structure::AddWyckGroup(vector<WyckGroup>* wycks)
{
	for(int i=0;i<atoms.size();i++)
		for (int j = 0; j < wycks->size(); j++)
		{
			atoms[i].wyckGroups.push_back((*wycks)[j]);
		}
	return;
};

//Here begins the first solution!
void AddWyck(Structure &s, int i, vector<WyckGroup> &wycks, int j)
{
	s.atoms[i].left -= (*wycks[j].SimilarWyck)[0].multiplicity;
	wycks[j].count++;
	s.atoms[i].wyckGroups[j].count++;
	return;
};

bool GetAtomsCanUse(int &i, int &j, Structure* s, vector<WyckGroup>* w,vector<int>* biasedw,int wsum)
{
	for (int attempt = 0; attempt < s->atoms.size() * 5; attempt++)
	{
		i = rand() % s->atoms.size();
		if ((s->atoms)[i].left > 0)
		{
			for (int a = 0; a < w->size() *5; a++)
			{
				int r=rand()%wsum;
				int t=0;
				for(j=0;j<biasedw->size();j++)
				{
					t+=(*biasedw)[j];
					if(t>r) break;
				}

				if (IsUsable(&(*w)[j], &(s->atoms)[i])) return true;
			}
		}
			
	}
	for (int k = 0; k < s->atoms.size(); k++)
		if ((s->atoms)[k].left > 0)
			for (j = w->size()-1; j >=0; j--) if (IsUsable(&(*w)[j], &(s->atoms)[k]))
			{
				i = k;
				return true;
			}
	i = 0; j = 0;
	return false;
};
void GetAllCombination(Structure structure, vector<WyckGroup> wycks, vector<Structure> &combinations,bool forceMostGeneralWyckPos,vector<int>* biasedw,int wsum)
{
	if (structure.AllAtomUsed())
	{
		if (forceMostGeneralWyckPos == true)
		{
			if (wycks[wycks.size() - 1].count > 0)
				{structure.UsedMostGeneral = true;combinations.push_back(structure); }
			return;
		}			
		combinations.push_back(structure);
		return;
	}
	int i = 0, j = 0;
	if (GetAtomsCanUse(i, j, &structure, &wycks,biasedw,wsum))
	{
		AddWyck(structure, i, wycks, j);
		GetAllCombination(structure, wycks, combinations, forceMostGeneralWyckPos,biasedw,wsum);
	}
	return;
};
//The first solution ends here.
//And here begins the second solution, just for test!


void AddWyck(Atoms* atoms, WyckGroup* wyck)
{
	atoms->left -= (*wyck->SimilarWyck)[0].multiplicity;	
	wyck->count++;
	return;
}
void GetCombinationforAtoms(Atoms atoms, vector<WyckGroup> wycks, vector<Atoms> &combinations,int i)
{
	for (int j = i; j < wycks.size(); j++)
	{
		if (IsUsable(&(wycks[j]), &atoms) == true)
		{
			if(j!=wycks.size()-1) GetCombinationforAtoms(atoms, wycks, combinations, j + 1);
			AddWyck(&atoms, &(wycks[j]));
			if (atoms.left == 0)
			{
				for (int i = 0; i < wycks.size(); i++) atoms.wyckGroups[i].count=wycks[i].count;
				if (wycks[wycks.size() - 1].count > 0) atoms.UsedMostGeneral = true;
				combinations.push_back(atoms);
				return;
			}
			else GetCombinationforAtoms(atoms, wycks, combinations, j);
			break;
		}
	}
	return;
}
bool CheckUnique(Structure* s, Atoms* ats,vector<int>* Uniquewycks)
{
	for(int i=0;i<Uniquewycks->size();i++)
	{
		int count = ats->wyckGroups[(*Uniquewycks)[i]].count;
		for (int k = 0; k < s->atoms.size(); k++)
				count += s->atoms[k].wyckGroups[(*Uniquewycks)[i]].count;
		if (count > (ats->wyckGroups[(*Uniquewycks)[i]].SimilarWyck)->size())
			return false;
	}
	return true;
}

bool AddAtomstoStructure(Structure* structure, Atoms* atoms, int i, vector<Structure> &combinations, vector<int>* Uniquewycks)
{
	if (CheckUnique(structure, atoms, Uniquewycks) == true)
	{
		Structure s(*structure);
		for (int l = 0; l < s.atoms[i].wyckGroups.size(); l++)
			s.atoms[i].wyckGroups[l].count = atoms->wyckGroups[l].count;
		s.atoms[i].UsedMostGeneral = atoms->UsedMostGeneral;
		s.UsedMostGeneral = (s.UsedMostGeneral || s.atoms[i].UsedMostGeneral);
		combinations.push_back(s);
		return true;
	}
	return false;
};

void GetAllCombinations(Structure structure, vector<WyckGroup> wycks, vector<Structure> &combinations, bool forceMostGeneralWyckPos, int attemptstoGetCombs)
{
	vector<Structure> tempcombs;

	vector<int> Uniquewycks;
	for (int i = 0; i < wycks.size(); i++)
		if ((*wycks[i].SimilarWyck)[0].unique == true) Uniquewycks.push_back(i);

	vector< vector<Atoms> > combinationsofAtoms;
	for (int i = 0; i < structure.atoms.size(); i++)
	{
		vector<Atoms> comb;
		GetCombinationforAtoms(structure.atoms[i], wycks, comb, 0);
		combinationsofAtoms.push_back(comb);
		//Here begins the logfile.
		
		/*for (int j = 0; j < comb.size(); j++)
		{
			cout << structure.atoms[i].atom.name <<" comb "<<j<<'\n';
			for (int k = 0; k < comb[j].wyckGroups.size(); k++)
			{
				cout << comb[j].wyckGroups[k].count << "(";

				for (int l = 0; l < (comb[j].wyckGroups[k].SimilarWyck)->size(); l++)
					cout << (*comb[j].wyckGroups[k].SimilarWyck)[l].multiplicity << (*comb[j].wyckGroups[k].SimilarWyck)[l].label << ',';

				cout<<") , ";
			}
			cout << endl;
		}*/
		
		//And it ends here!
	}

	long int combnum = 1;
	for (int i = 0; i < structure.atoms.size(); i++) combnum *= combinationsofAtoms[i].size();
	if(combnum<=1500)
	{
		tempcombs.push_back(structure);
		for (int i = 0; i < structure.atoms.size(); i++)
		{
			int n = tempcombs.size();
			for (int j = 0; j < n; j++)
			{
				for (int k = 0; k < combinationsofAtoms[i].size(); k++)
					AddAtomstoStructure(&tempcombs[j], &combinationsofAtoms[i][k], i, tempcombs, &Uniquewycks);			
			}
			tempcombs.erase(tempcombs.begin(), tempcombs.begin() + n);
		}
	}
	else	
	{
		DEBUG_INFO("Notice: The number of all combinations can be up to about %d , so we just ignored some of them randomly.\n", combnum);
		int attempt = 0;
		vector<Structure> temps;		

		while (attempt < attemptstoGetCombs)
		{
			temps.push_back(structure);
			bool l = true;
			for (int i = 0; i < structure.atoms.size(); i++)
			{
				int k = rand() % combinationsofAtoms[i].size();
				l=AddAtomstoStructure(&temps[temps.size() - 1], &combinationsofAtoms[i][k], i, temps, &Uniquewycks);
				if (l == false) break;
			}
			if(l==true) tempcombs.push_back(temps[temps.size() - 1]);
			temps.clear();
			attempt++;
		}		
	}
	
	if (forceMostGeneralWyckPos == true)
	{
		for (int i = 0; i < tempcombs.size(); i++)
		{
			if(tempcombs[i].UsedMostGeneral==true) combinations.push_back(tempcombs[i]);
		}
	}
	else
		for (int i = 0; i < tempcombs.size(); i++)
			combinations.push_back(tempcombs[i]);

	return;
}

//The second solution ends here.



void Initialize(Structure &structure,int dimension, int spg, vector<Atoms> atomlist, vector<WyckPos> &wycks, int choice=0)
{
	wycks.clear();
	vector<double> symmetry;
	int temps = spg - 1;
	vector<double> translate;

	const vector< vector<double> >* tr;
	const vector< vector<double> >* wy;
	const vector< vector<sitepos> >* sp;

	if(dimension==3) 
	{
		if( temps < 0 || temps >= 230 ) DEBUG_INFO("error: spacegroup must be in range of 1-230. \n");
		tr = &trans; wy = &wyck; sp = &SitePosition; 
	}
	else if(dimension ==2)
	{
		structure.pbc[2] = false;
		switch (choice)
		{
		case 1:
			if( temps < 0 || temps >= 80 ) DEBUG_INFO("error: layergroup must be in range of 1-80. \n");
			tr = &trans_2D; wy = &wyck_2D; sp = &SitePosition_2D; 
			break;
		
		default:
			if( temps < 0 || temps >= 17 ) DEBUG_INFO("error: planegroup must be in range of 1-17. \n");
			tr = &trans_PL; wy = &wyck_PL; sp = &SitePosition_PL; 
			break;
		}
	}
	else if(dimension == 1)
	{
		structure.pbc[0] = false; structure.pbc[1] = false; 
		if( temps < 0 || temps >= 75 ) DEBUG_INFO("error: rodgroup must be in range of 1-75. \n");
		tr = &trans_1D; wy = &wyck_1D; sp = &SitePosition_1D; 
	}
	else if(dimension ==0)
	{
		structure.pbc[0] = false; structure.pbc[1] = false; structure.pbc[2] = false;
		if( temps < 0 || temps >= 56 ) DEBUG_INFO("error: pointgroup must be in range of 1-56. \n");
		tr = &trans_0D; wy = &wyck_0D; sp = 0; 
	}
	else DEBUG_INFO("error: Initialize error, dimension cannot be %d \n", dimension);


	int symmetrynum = (int)(*tr)[temps][0];
	int translatenum = (int)(*tr)[temps][1];
	for (int i = 0; i < 12 * symmetrynum; i++)
		symmetry.push_back((*tr)[temps][i + 2]);
	for (int i = 0; i < 3 * translatenum; i++)
		translate.push_back((*tr)[temps][12 * symmetrynum + i + 2]);
	int wycksnum = (int)(*wy)[temps][0];
	for (int i = 0; i < wycksnum; i++)
	{
		char label = (char)(*wy)[temps][15 * i + 1];
		double matrix[12];
		for (int j = 0; j < 12; j++)
			matrix[j] = (*wy)[temps][15 * i + j + 2];
		int multiplicity = (int)(*wy)[temps][15 * i + 14];
		bool unique = (int)(*wy)[temps][15 * i + 15];
		if(dimension!=0)
		{
			vector<int> symop(get<3>((*sp)[temps][wycksnum-i-1]));
			if(get<1>((*sp)[temps][wycksnum-i-1])!=label) DEBUG_INFO("error: database error, in group %d (dim %d ),  order of label dismatch. \n", spg, dimension);
			wycks.push_back(WyckPos(label, multiplicity, matrix, &symmetry, &translate, unique, symop));
		}
		else
		{
			vector<int> symop;
			wycks.push_back(WyckPos(label, multiplicity, matrix, &symmetry, &translate, unique, symop));
		}
	}

	double atomvolume = 0;
	double maxr = 0;

	for (int i = 0; i < atomlist.size(); i++)
	{
		structure.atoms.push_back(atomlist[i]);
		if(atomlist[i].atom.num==1)
		{
			atomvolume += 4 * M_PI / 3 * atomlist[i].number*pow((atomlist[i].atom.radius), 3);
			maxr = Max(maxr, atomlist[i].atom.radius);
		}
		else
		{
			for(int a=0;a<atomlist[i].atom.c.clus->radius.size();a++)
			{
				double r=atomlist[i].atom.c.clus->radius[a];
				atomvolume += 4 * M_PI / 3 * atomlist[i].number*pow((r), 3);
				maxr = Max(maxr, r);
			}
		}
		
			
	}

	structure.atomvolume = atomvolume;
	structure.spg = spg;
	structure.maxr = maxr;
	structure.UsedMostGeneral = false;
	
	structure.dimension=dimension;
	structure.choice=choice;

	return;
}
bool issametype(WyckPos& a, WyckPos& b, bool dividesym)
{
	if(a.multiplicity!=b.multiplicity) return false;
	if(a.unique!=b.unique) return false;
	if(dividesym)
	{
		if(a.symop!=b.symop) return false;
		if(a.symop==0)	{}
		else if(a.symop==1)
		{
			for(int i=0;i<a.symops.size();i++)
				if(a.symops[i].size()!=b.symops[i].size()) return false;
		}
		else
		{
			for(int i=0;i<a.symops.size();i++)
				if(a.symops[i].size()!=b.symops[i].size()) return false;

			for(int i=2;i<a.symmatrix.size();i++)
				for(int j=0;j<a.symmatrix[i].size();j++)
					for(int ii=2;ii<a.symmatrix[i][j].size();ii++)
						for(int jj=0;jj<a.symmatrix[i][j][ii].size();jj++)
						{
							if(fabs(a.symmatrix[i][j][ii][jj]-b.symmatrix[i][j][ii][jj])>1e-4) return false;
						}
				
		}
		
	}
	return true;
}
void GetWyckPosGrouped(vector<WyckGroup>& wycks,int i, bool dividesym=false)
{
	vector<WyckPos>* SimilarWyck = new vector<WyckPos>;
	SimilarWyck->push_back(wyckpositions[i]);
	wycks.push_back(WyckGroup(SimilarWyck));
	if (i == wyckpositions.size() - 1) return;
	
	int j;
	for (j = i + 1; j < wyckpositions.size(); j++)
	{
		if (issametype(wyckpositions[j], wyckpositions[i], dividesym))
			SimilarWyck->push_back(wyckpositions[j]);
		else break;
	}
	GetWyckPosGrouped(wycks, j, dividesym);
};
void GetWycksDeleted(vector<WyckGroup>& wycks)
{
	for (int i = 0; i < wycks.size(); i++)
	{
		wycks[i].SimilarWyck->clear();
		delete wycks[i].SimilarWyck;
	}
	return;
};
