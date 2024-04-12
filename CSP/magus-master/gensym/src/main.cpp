#include"getlatticeparm.cpp"
#include"gensym.cpp"
#include"cluster.cpp"
#include"clustersym.cpp"
#include"DataBase/celltransData.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

class Info
{
public:

	int dimension;	
	double vacuum;		//add vacuum if dimention=2
	int choice;			//choice = 0 [plane group] and choice = 1 [layergroup]
	double minVolume;
	double maxVolume;
	double threshold;
	map<string, double> sp_threshold;		//considering different thresholds. threshold["ab"] = threshold of type a and type b and eqs threshold["ba"]
	double threshold_mol;
	double latticeMins[6];
	double latticeMaxes[6];
	const char* outputdir;

	bool forceMostGeneralWyckPos;
	double biasedrand;
	vector<int> biasedwycks;

	int spg;
	int spgnumber;		//number of structures to be generated
	
	/*if you want to get a primitive cell, program generates a atom_num*primitiveCellnum conventional cell first and then transforms it to primitive.*/
	double primitivector[9];
	double inversePrimitivector[9];
	int primitiveCellnum;
	int celltype;	    //1 'P'; 2 'C'; 3 'F'; 4 'I'; 5 'A'; 6 'R'; =1 if GetConventional = true
	char UselocalCellTrans;
	bool GetConventional;

	int maxAttempts;
	int maxattempts;
	int attemptstoGetCombs;

	int method;
	char fileformat;
	vector<Atoms> atomlist;
	vector<Structure> ans;
	vector<Structure> primitiveans;

	Info(double min = 0, double max = 0) ;
	~Info() ;

	/*Checks if there are user-defined LatticeMins/Maxes, volumeMin/Max, complete them if not.*/
	bool Check(double *m) ;
	void CompleteParm(Structure* s) ;

	/*Transform generated conventional cell to primitive cell.*/
	void CellTrans(vector<Structure>* ans, vector<Structure>* primitiveans) ;

	/*a call of Generate() automaticly calls for _generate_().
		In Generate, CellTrans() is called for cell transformation.
		In _generate_, wyckoff positions of user-defined spacegroup is clustered into wycks.
		Then search for combinations of atoms and wyckoff positions.
		Lastly get a set of lattice parameters and get positions for each atom. */
	double _get_threshold_(const char* a, const char* b) ;
	bool _generate_(vector<Structure> &ans) ;
	int Generate(int seed) ;

	/*interface to set atoms/molecules.*/
	void AppendAtoms(int num,const char* name,double radius,bool m, const char* filename=0, double _symprec = 1e-2) ;
	void AppendMoles(int number, const char* clustername, vector<double> radius, vector<double> positioninfo, vector<int> numinfo, vector<string> namesinfo, double _symprec = 1e-2) ;

	/*python interface. returns numpy-list. Arguement 'n' stands for the 'n'_th structure.*/
	py::list GetLattice(int n) ;
	void GetAtom(int n) ;
	py::list GetPosition(int n) ;
	py::list GetWyckPos(int n) ;
	py::list GetWyckLabel(int n) ;

	void SetLatticeMins(double a, double b, double c, double d, double e, double f) ;
	void SetLatticeMaxes(double a, double b, double c, double d, double e, double f) ;
	void SpThreshold(const char* a, const char* b, double thre) ;
};


/*******************************Here begins the detailed function.*/

int GetCellNum(double* pv) //double* inverse primitive vector
{
    return abs( (int) ( pv[0]*(pv[4]*pv[8]-pv[5]*pv[7]) - pv[1]*(pv[3]*pv[8]-pv[5]*pv[6]) + pv[2]*(pv[3]*pv[7]-pv[4]*pv[6]) ));
}
void GetPrimPos(vector<position>& pr, vector<position>& po, double* inversePrimitivector, int atomnum)
{
	for(int i=0;i<po.size();i++)
	{
		position tp;
		tp.x=po[i].x*inversePrimitivector[0]+po[i].y*inversePrimitivector[1]+po[i].z*inversePrimitivector[2];
		tp.y=po[i].x*inversePrimitivector[3]+po[i].y*inversePrimitivector[4]+po[i].z*inversePrimitivector[5];
		tp.z=po[i].x*inversePrimitivector[6]+po[i].y*inversePrimitivector[7]+po[i].z*inversePrimitivector[8];
		tp=Standform(&tp);

		if(pr.size()==0) pr.push_back(tp);
		else
			for(int a=0;a<pr.size();a++)
			{
				if(tp==pr[a]) break;
				if(a==(pr.size()-1))
				{
					pr.push_back(tp);
					break;
				}
			}
		if(pr.size()==atomnum) return;
	} 
};

Info::Info(double min , double max )
{
	minVolume = min;
	maxVolume = max;
	for (int i = 0; i < 6; i++) latticeMins[i] = latticeMaxes[i] = 0;

	dimension=3;
	vacuum=10;			//for 2D structure and 0D structure
	choice=0;				//for 2D, choose planegroup by default 

	threshold = 0;
	threshold_mol=1.0;
	forceMostGeneralWyckPos = true;
	biasedrand=1;
	maxAttempts = 1000;
	maxattempts = 500;
	attemptstoGetCombs = 0;

	method = 2;
	fileformat = 'v';
	UselocalCellTrans='y';
	GetConventional=false;
	celltype = 1;
	outputdir = 0;

};
Info::~Info()
{
	for(int i=0;i<atomlist.size();i++) 
	{
		if(atomlist[i].atom.num!=1) 
		{
			for(int j=0;j<atomlist[i].atom.c.clus->Name.size();j++)
				delete[] atomlist[i].atom.c.clus->Name[j];
			delete atomlist[i].atom.c.clus;
		}
		delete[] atomlist[i].atom.name;
	}
};


bool Info::Check(double *m)
{
	for (int i = 0; i < 6; i++) if (m[i] != 0) return false;
	return true;
}

void Info::CompleteParm(Structure* s)
{
	int temps=spg-1;
	if(dimension==3) vacuum=0;
	else if(dimension==0) GetConventional=true;
	
	s->vacuum=vacuum;

	if(!GetConventional)
	{	
		const vector<int>* vc_type_choice;
		if (dimension == 3) vc_type_choice = &vector_type_choice;
		else if (dimension == 2)
		{
			if (choice == 1)
				vc_type_choice = &vector_type_choice_layergroup; 
			else 
				vc_type_choice = &vector_type_choice_planegroup;
		}
		celltype = (*vc_type_choice)[temps] ; 
		for(int i=0;i<9;i++) primitivector[i]=primitive_vector_type[celltype-1][i];
		for(int i=0;i<9;i++) inversePrimitivector[i]=inverse_primitive_vector[celltype-1][i];
		primitiveCellnum=GetCellNum(inversePrimitivector);
	}
	else UselocalCellTrans='n';

	double maxr;
	if (minVolume == 0)
		minVolume = s->atomvolume * 1;
	if (maxVolume == 0)
		maxVolume = s->atomvolume * 3;
	if (threshold == 0)
		threshold = 1;
	if (Check(latticeMins))
	{
		if (s->maxr < 1.5)
		{
			for (int i = 0; i < 3; i++) { latticeMins[i] = 3.0; latticeMins[i + 3] = 60.0; }
			maxr = 3;
		}
		else
		{
			for (int i = 0; i < 3; i++) { latticeMins[i] = 2 * s->maxr; latticeMins[i + 3] = 60.0; }
			maxr = 2 * s->maxr;
		}
	}
	else maxr = 3;
	double maxlen = maxVolume / (maxr*maxr);
	if (Check(latticeMaxes))
	{
		if (maxlen > 3) for (int i = 0; i < 3; i++) { latticeMaxes[i] = maxlen; latticeMaxes[i + 3] = 120.0; }
		else for (int i = 0; i < 3; i++) { latticeMaxes[i] = 4.0; latticeMaxes[i + 3] = 120.0; }
	}
	if(!GetConventional)
	{
		//Here begins Get primitive cell transformed to conventional cell.
		maxVolume*=primitiveCellnum;
		minVolume*=primitiveCellnum;
		s->atomvolume*=primitiveCellnum;
		//the part of change latticeMin and latticeMax is moved to ChooseLattice().
		for(int i=0;i<s->atoms.size();i++)
		{
			s->atoms[i].number*=primitiveCellnum;
			s->atoms[i].left*=primitiveCellnum;
			atomlist[i].number*=primitiveCellnum;
		}

		//Here ends Get primitive cell transformed to conventional cell.
	}
	if (attemptstoGetCombs == 0)
	{
		if (method == 1) attemptstoGetCombs = sqrt(spgnumber) * 40;
		else if (method == 2) attemptstoGetCombs = 500;
	}
	return;
};

void Info::CellTrans(vector<Structure>* ans, vector<Structure>* primitiveans)
{
	for(int i=0;i<ans->size();i++)
	{
		Structure s;
		Structure* temps=&(*ans)[i]; double* lp=temps->latticeparm;
		s.volume=temps->volume/primitiveCellnum;
		s.spg = temps->spg;
		s.legal = temps->legal;
		s.UsedMostGeneral = temps->UsedMostGeneral;

		for(int j=0;j<3;j++)
		{
			s.latticeparm[j]=primitivector[j]*lp[0]+primitivector[j+3]*lp[1]+primitivector[j+6]*lp[2];
			s.latticeparm[j+3]=primitivector[j+3]*lp[4]+primitivector[j+6]*lp[5];
			s.latticeparm[j+6]=primitivector[j+6]*lp[8];
		}

		for(int j=0;j<temps->atoms.size();j++)
		{
			if(temps->atoms[j].atom.num==1) 
				s.atoms.push_back(Atoms(temps->atoms[j].number/primitiveCellnum,temps->atoms[j].atom.name,temps->atoms[j].atom.radius,false));
			else
				s.atoms.push_back(Atoms(temps->atoms[j].number/primitiveCellnum,temps->atoms[j].atom.name,temps->atoms[j].atom.c.clus));
			
			for(int k=0;k<temps->atoms[j].positions_wyck.size();k++) s.atoms[j].positions_wyck.push_back(temps->atoms[j].positions_wyck[k]);
			
			GetPrimPos(s.atoms[j].positions, temps->atoms[j].positions, inversePrimitivector, s.atoms[j].number*s.atoms[j].atom.num);
		}

		primitiveans->push_back(s);

	}
	return;
};

bool Info::_generate_(vector<Structure> &ans)
{
	Structure structure;
	Initialize(structure, dimension, spg, atomlist, wyckpositions, choice);
	CompleteParm(&structure);
	for (int i = 3; i < 6; i++)
	{
		latticeMins[i] = latticeMins[i] * M_PI / 180;
		latticeMaxes[i] = latticeMaxes[i] * M_PI / 180;
	}
	if (forceMostGeneralWyckPos == true)
	{
		bool l=false;
		for (int i = 0; i < atomlist.size(); i++)
			if (atomlist[i].number >= wyckpositions[wyckpositions.size() - 1].multiplicity)  {l = true; break;}
		if (l == false)
		{
			DEBUG_INFO("error: cannot generate a structure with most general wyckpos, turnning out the option may solve this problem \n");
			return false;
		}
	}
	vector<WyckGroup> wycks;
	bool dividesym=false;
	for(int i=0;i<atomlist.size();i++)
	{
		if(atomlist[i].atom.num!=1) {dividesym=true;break;}
	}
	GetWyckPosGrouped(wycks,0,dividesym);

	structure.AddWyckGroup(&wycks);
	DEBUG_INFO("Initialize success with spg %d (%dD).\n", spg, dimension);

	vector<Structure> combinations;

	switch(method)
	{
	case 1:
	{
		int wsum=0;
		for(int i=0;i<wycks.size();i++)
		{
			biasedwycks.push_back(pow((*wycks[i].SimilarWyck)[0].multiplicity,biasedrand));
			wsum+=biasedwycks[i];
		}

		for (int attempt = 0; attempt < attemptstoGetCombs; attempt++)
		{
			GetAllCombination(structure, wycks, combinations, forceMostGeneralWyckPos,&biasedwycks,wsum);
			if (combinations.size() >= sqrt(spgnumber) * 15) break;
		}
	}
	break;
	case 2:
		GetAllCombinations(structure, wycks, combinations, forceMostGeneralWyckPos, attemptstoGetCombs);
	break;
	}

	//Here begins the logfile for combinations!
	/*for (int i = 0; i < combinations.size(); i++)
	{
		cout << "structure combination " << i << endl;
		for (int j = 0; j < combinations[i].atoms.size(); j++)
		{
			cout << combinations[i].atoms[j].atom.name << " : ";
			for (int k = 0; k < combinations[i].atoms[j].wyckGroups.size(); k++)
			{
				cout << combinations[i].atoms[j].wyckGroups[k].count << "(";
				for (int l = 0; l < (combinations[i].atoms[j].wyckGroups[k].SimilarWyck)->size(); l++)
					cout << (*combinations[i].atoms[j].wyckGroups[k].SimilarWyck)[l].multiplicity << (*combinations[i].atoms[j].wyckGroups[k].SimilarWyck)[l].label << ',';
				cout << ") , ";
			}
			cout << endl;
		}
	}*/
	//The logfile ends here.

	if (combinations.size() == 0)
	{
		ans.push_back(Structure(structure));
		DEBUG_INFO("error: Combination does not exist.\n");
		GetWycksDeleted(wycks);
		return false;
	}
	else DEBUG_INFO("GetAllCombination success: got %d combination(s).\n" , combinations.size() );


	int attemps = 0;
	int ans_size = 0;
	int maxfailures = Min( 5, spgnumber );
	while (ans_size < spgnumber)
	{
		for (int j = 0; j < maxAttempts; j++)
		{
			//DEBUG_INFO("attempt %d of %d attempts.\n", j, maxAttempts);
			structure = combinations[ rand() % combinations.size()];
			int statuscode = structure.ChooseLattice(latticeMins, latticeMaxes, minVolume, maxVolume, celltype);
			if ( statuscode == -2 )
			{
				GetWycksDeleted(wycks);
				return false;
			} 
			else if ( statuscode == 1 ) 
			{
				structure.ChooseWyck();
				if (dimension!=0)
					structure.MakeCrystal(std::bind(&Info::_get_threshold_, this, std::placeholders::_1, std::placeholders::_2), threshold_mol, maxattempts, latticeMins, latticeMaxes);
				else
					structure.MakeCluster(std::bind(&Info::_get_threshold_, this, std::placeholders::_1, std::placeholders::_2), maxattempts);
					
				//DEBUG_INFO("is attempt %d a successful try ? %d (1 for y /0 for n); stored structures: %d\n", j, structure.legal, ans_size);
			}
			if (structure.legal == true)  break;
			if (j == maxAttempts - 1)
			{
				attemps++;
				ans_size++;
				DEBUG_INFO("error: failed ChooseLattice()/MakeCrystal(), already made %d crystal(s).\n", ans.size());
			}
		}
		if (structure.legal == true)
		{
			ans.push_back(Structure(structure));
			ans_size++;
			//Here begins the logfile for crystal.
			/*ofstream out("log_structurecombs.txt",ios::app);
			out << "spg= "<<spg<<", structure= " << ans.size()  << '\n';
			for (int i = 0; i < structure.atoms.size(); i++)
			{
				out << structure.atoms[i].atom.name << " : ";
				for (int j = 0; j < structure.atoms[i].chosenWycks.size(); j++)
					out << wyckpositions[structure.atoms[i].chosenWycks[j]].multiplicity << wyckpositions[structure.atoms[i].chosenWycks[j]].label<<",";
				out << '\n';
			}
			out.close();*/
			//And here it ends.
		}
		if (attemps >= maxfailures)
		{
			GetWycksDeleted(wycks);
			if (ans.size() > 0)
			{
				DEBUG_INFO("Notice: exit for too many MakeCrystal() failures; %d crystal(s) were generated in total.\n", ans.size());
				return true;
			}
			else return false;
		}
	}
	GetWycksDeleted(wycks);
	return true;
};

void Info::AppendAtoms(int num,const char* name,double radius,bool m, const char* filename, double _symprec)
{
	char* atomname=new char[strlen(name)+1];
	strcpy(atomname, name);
	const char * _name = atomname;
	atomlist.push_back(Atoms(num, _name, radius,m,filename, _symprec));
	return;
}
void Info::AppendMoles(int number, const char* clustername, vector<double> radius, vector<double> positioninfo, vector<int> numinfo, vector<string> namesinfo, double _symprec)
{   
	vector<string> names = namesinfo;
	vector<double> r = radius;
	vector<int> num = numinfo;
	vector<double> pos = positioninfo;

	cluster* clus=new cluster(num, r,  pos, names, _symprec);

	char* clusname=new char[strlen(clustername)+1];
	strcpy(clusname, clustername);
	const char * _name = clusname;

	atomlist.push_back(Atoms(number, _name, clus));

	return;
};

double Info::_get_threshold_(const char* a, const char* b) 
{
	if (!sp_threshold.empty())
	{
		string s = string(a).append(b);
		map<string, double>::iterator t = sp_threshold.find(s);
		if (t != sp_threshold.end()) return t->second;
	} 
	return threshold;
}

int Info::Generate(int seed)
{
	srand((unsigned)time(NULL)+(unsigned)seed);
	bool legel = _generate_(ans);

	if (legel)
	{
		switch(UselocalCellTrans)
		{
		case 'y':
			CellTrans(&ans,&primitiveans);
			break;
		case 'n':
		{
			for(int i=0;i<ans.size();i++)
				primitiveans.push_back(ans[i]);
		}
			break;
		}
		DEBUG_INFO("Generate success!\n");

		if(outputdir)
		{
			string output(outputdir);
			for (int i = 0; i < primitiveans[0].atoms.size(); i++)
			{
				output.append(primitiveans[0].atoms[i].atom.name);
				output.append(to_string(primitiveans[0].atoms[i].number));
			}
			output.append("_"); output.append(to_string(spg)); output.append("-");
			for (int i = 0; i < primitiveans.size(); i++)
			{
				string filename(output);
				filename.append(to_string(i + 1));
				if(fileformat=='g') filename.append(".gin");
				if(fileformat=='t') filename.append(".py");
				primitiveans[i].WritePoscar(&filename,fileformat);
				if(UselocalCellTrans)
				{
					filename.append("-Cell.py");
					ans[i].WritePoscar(&filename,fileformat);
				}
			}
		}
		
		return (int)primitiveans.size();
	}
	else DEBUG_INFO("error: Generate error\n");
	return 0;
};

py::list Info::GetLattice(int n)
{
	py::list l;
	if (n >= primitiveans.size()) 
	{ 
		DEBUG_INFO("Please input a smaller number than %d \n", primitiveans.size()); 
		return l; 
	}
	for (int i = 0; i < 3; i++)
	{
		l.append(primitiveans[n].latticeparm[i]);
		l.append(primitiveans[n].latticeparm[i+3]);
		l.append(primitiveans[n].latticeparm[i+6]);
	}
	return l;
};
void Info::GetAtom(int n)
{
	if (n >= primitiveans.size()) { DEBUG_INFO("Please input a smaller number than %d \n", primitiveans.size());}
	DEBUG_INFO("There's %d type(s) of atoms in this structure.\n", primitiveans[n].atoms.size()+1);
	for (int i = 0; i < primitiveans[n].atoms.size(); i++)
	{
		DEBUG_INFO("%s, %d\n", primitiveans[n].atoms[i].atom.name, primitiveans[n].atoms[i].positions.size()+1);
	}

	return ;
};
py::list Info::GetPosition(int n)
{
	py::list l;
	if (n >= primitiveans.size()) { DEBUG_INFO("Please input a smaller number than %d \n", primitiveans.size()); return l; }
	for (int i = 0; i < primitiveans[n].atoms.size(); i++)
	{
		if(primitiveans[n].atoms[i].atom.num==1)
			for (int j = 0; j < primitiveans[n].atoms[i].positions.size(); j++)
			{
				l.append(primitiveans[n].atoms[i].positions[j].x);
				l.append(primitiveans[n].atoms[i].positions[j].y);
				l.append(primitiveans[n].atoms[i].positions[j].z);
			}
		else
		{
			for(int column=0;column<primitiveans[n].atoms[i].atom.num;column++)
				for(int row=0;row<primitiveans[n].atoms[i].number;row++)
				{
					l.append(primitiveans[n].atoms[i].positions[row*primitiveans[n].atoms[i].atom.num+column].x);
					l.append(primitiveans[n].atoms[i].positions[row*primitiveans[n].atoms[i].atom.num+column].y);
					l.append(primitiveans[n].atoms[i].positions[row*primitiveans[n].atoms[i].atom.num+column].z);
				}
		}
						
	}

	return l;
};
py::list Info::GetWyckPos(int n)
{
	py::list l;
	if (n >= primitiveans.size()) { DEBUG_INFO("Please input a smaller number than %d \n", primitiveans.size()); return l; }
	for (int i = 0; i < primitiveans[n].atoms.size(); i++)
	{
		for (int j = 0; j < primitiveans[n].atoms[i].positions_wyck.size(); j++)
		{
			l.append(primitiveans[n].atoms[i].positions_wyck[j].x);
			l.append(primitiveans[n].atoms[i].positions_wyck[j].y);
			l.append(primitiveans[n].atoms[i].positions_wyck[j].z);
		}
	}
	return l;
};
py::list Info::GetWyckLabel(int n)
{
	py::list l;
	if (n >= primitiveans.size()) { DEBUG_INFO("Please input a smaller number than %d \n", primitiveans.size()); return l; }
	for (int i = 0; i < primitiveans[n].atoms.size(); i++)
	{
		for (int j = 0; j < primitiveans[n].atoms[i].positions_wyck.size(); j++)
		{
			l.append(primitiveans[n].atoms[i].atom.name);
		}
	}
	return l;
};

void Info::SetLatticeMins(double a, double b, double c, double d, double e, double f)
{
	latticeMins[0] = a; latticeMins[1] = b; latticeMins[2] = c;
	latticeMins[3] = d; latticeMins[4] = e; latticeMins[5] = f;
	return;
};
void Info::SetLatticeMaxes(double a, double b, double c, double d, double e, double f)
{
	latticeMaxes[0] = a; latticeMaxes[1] = b; latticeMaxes[2] = c;
	latticeMaxes[3] = d; latticeMaxes[4] = e; latticeMaxes[5] = f;
	return;
};
void Info::SpThreshold(const char* a, const char* b, double thre)
{
	string s = string(a).append(b);
	sp_threshold[s] = thre;
	s = string(b).append(a);
	sp_threshold[s] = thre;
	/*for(auto i = sp_threshold.begin(); i!=sp_threshold.end(); i++)
		cout<<i->first<<'\t'<<i->second<<endl;
	cout<<endl;*/
	return;
};

/*int main()
{
	for (int i =177; i <= 177; i++)
	{
		Info info;
		info.spg=i;
		info.dimension=2;
		cout << "spg=" <<info.spg<< ": start at "<<1.0*clock()/CLOCKS_PER_SEC<<"s"<<endl;
		//info.AppendAtoms(18, "Ti", 1.6,false);
		info.AppendAtoms(20, "C6H6", 0,true,"CH4.txt");
		//info.AppendAtoms(15, "Mg", 1.41,false);
		//info.AppendAtoms(15, "Si", 1.11,false);
		info.minVolume = 590;
		info.maxVolume = 910;
		info.maxAttempts = 100;
		info.spgnumber = 5;
		info.threshold = 0.4;
		info.forceMostGeneralWyckPos = true;
		info.method =2;
		info.outputdir = "outputtest/";
		info.fileformat='t';
		//info.GetConventional=true;
		//info.biasedrand=3;
		double mins[6] = { 3,3,3,60,60,60 };
		for (int j = 0; j < 6; j++) info.latticeMins[j] = mins[j];
		double maxs[6] = { 10,10,10,120,120,120 };
		for (int j = 0; j < 6; j++) info.latticeMaxes[j] = maxs[j];
		info.Generate(i);
		//info.GetWyckPos(0);
	}
	return 0;
}*/

PYBIND11_MODULE(gensym, m){
    m.doc() = "gensym";
    py::class_<Info>(m, "Info")
		.def(py::init())

		.def_readwrite("minVolume", &Info::minVolume)
		.def_readwrite("maxVolume", &Info::maxVolume)
		.def_readwrite("threshold",&Info::threshold)
		.def_readwrite("threshold_mol",&Info::threshold_mol)
        .def_readwrite("outputdir",&Info::outputdir)
		.def_readwrite("spg", &Info::spg)
		.def_readwrite("dimension", &Info::dimension)
		.def_readwrite("vacuum", &Info::vacuum)
		.def_readwrite("choice", &Info::choice)
		.def_readwrite("spgnumber", &Info::spgnumber)
        .def_readwrite("maxAttempts", &Info::maxAttempts)
		.def_readwrite("forceMostGeneralWyckPos",&Info::forceMostGeneralWyckPos)
        .def_readwrite("biasedrand",&Info::biasedrand)
		.def_readwrite("method",&Info::method)
        .def_readwrite("fileformat",&Info::fileformat)
		.def_readwrite("UselocalCellTrans",&Info::UselocalCellTrans)
		.def_readwrite("GetConventional",&Info::GetConventional)

		.def("AppendAtoms", &Info::AppendAtoms, py::arg("num"), py::arg("name"), py::arg("radius"), py::arg("m"), py::arg("filename")="0", py::arg("_symprec")=1e-2)
		.def("AppendMoles", &Info::AppendMoles, py::arg("number"), py::arg("clustername"), py::arg("radius"), py::arg("positioninfo"), py::arg("numinfo"), py::arg("namesinfo"), py::arg("_symprec")=1e-2)
		.def("Generate", &Info::Generate)
		.def("GetLattice", &Info::GetLattice)
		.def("GetAtom", &Info::GetAtom)
		.def("GetPosition", &Info::GetPosition)
		.def("GetWyckPos", &Info:: GetWyckPos)
		.def("GetWyckLabel", &Info:: GetWyckLabel)
		.def("SetLatticeMins", &Info::SetLatticeMins)
		.def("SetLatticeMaxes", &Info::SetLatticeMaxes)
		.def("SpThreshold", &Info::SpThreshold)
		;
			
    	py::object wyckoff_positions_3d = py::cast(wyck);
    	m.attr("wyckoff_positions_3d") = wyckoff_positions_3d;

		py::object symbols_0d = py::cast(symbols_0D);
    	m.attr("symbols_0d") = symbols_0d;
	
}
