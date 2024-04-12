/*cluster.h: Definition of
    class Cluster in class Atom*/

#pragma once
#include <cstdlib>
#include <vector>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstring>
#include"position.h"

//#include"sitesymData.h"

using namespace std;

#define M_PI 3.14159265358979323846

extern double symprec ;

class sph_position
{
public:
    double r;
    double theta;
    double phi;

    sph_position(double a,double b, double c)
    {
        r=a;theta=b/180*M_PI;phi=c/180*M_PI;
    }

    position cart_position(void)
    {
        return position(r*sin(theta)*cos(phi), r*sin(theta)*sin(phi),r*cos(theta));
    }
};

class cluster
{
public:
    vector<const char*> name;
    vector<const char*> Name;
    vector<double> radius;
    vector<position> cart_positions;
    vector<position> cart_originreset;
    int dimention;
    position vec;
    vector< vector<position> > symmetry;
    /*molecule symmetry:
        Sym_minus1 =symmetry[1]****Sym_2 =symmetry[2]****Sym_m =symmetry[3]
        Sym_4 =symmetry[4]****Sym_minus4 =symmetry[5]****Sym_3 =symmetry[6]
        Sym_minus3 =symmetry[7]****Sym_6 =symmetry[8]****Sym_minus6 =symmetry[9]    */

	double radius_BoundingSphere;

    cluster(){};
    cluster(vector<int>& atomnum, vector<double>& r, vector<double>& pos, vector<string>&names, double _symprec = 1e-2) ;
    cluster(const char* filename, double _symprec = 1e-2) ;
};

class Cluster
{
public:
    cluster* clus;
    vector<position> cart_positions_frac;
    
    Cluster(){clus=0;} ;
    Cluster(cluster* c)     {  clus=c; } ;
    Cluster(const Cluster &c) ;
    void Input(const char* filename, double _symprec = 1e-2) ;
    void operator =(const Cluster &c) ;

    /*function RePlace rotates a molecule. 
    If it were to ocuupy a wyckoff position with specific site symmetry[given in symops], it rotates with constrains and returns false if fails.*/
    void RePlace(double *latticeparm) ;
    bool RePlace(double *latticeparm, const vector<vector<int> >& symops, int symop) ;
};


/*calculate dimention for a molecule.*/
int CalDimention(vector<position> &pos, position& vec)  ;
/*get symmetry of a molecule.*/
void GetSym(vector<const char*> name, vector<position> pos, vector< vector<position> > &symmetry, int dimention, double & radius_BoundingSphere, position* selfvec=0) ;