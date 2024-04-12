/*atom.h: Definition of 
    class Atom; 
    function CheckDistance(vector<position>*, vector<position>*, Atom*, (Atom*), double* latticeparm, double threshold)*/
	
#pragma once
#include<cstdlib>
#include<fstream>
#include<iostream>
#include<vector>
using namespace std;

#include"position.h"
#include"cluster.h"

class Atom
{
public:
	const char* name;
	double radius;
	int num;		//num of total atoms in Atom
	Cluster c;

	Atom() {};
	Atom(const char* n, double r , bool m, const char* filename=0, double _symprec=1e-2)                          
	{
		name = n;
		radius = r;
		if(m) 
		{
			c.Input(filename, _symprec);
			num=c.clus->cart_positions.size();
			radius=c.clus->radius_BoundingSphere;
		}
		else num=1;
	};
	Atom(const char* n, cluster* clus)                          
	{
		name = n;

		num=clus->cart_positions.size();
		c.clus=clus;
		radius = c.clus->radius_BoundingSphere;
	};
	Atom(const Atom &a)
	{
		name = a.name;
		radius = a.radius;
		num=a.num;
		if(num!=1) c=a.c;
	}
	void operator =(const Atom &a)
	{
		name = a.name;
		radius = a.radius;
		num=a.num;
		if(num!=1) c=a.c;
	}
};

bool CheckDistance(vector<position>* p1s, vector<position>* p1s_coms, vector<position>* p2s,vector<position>* p2s_coms, Atom* atom1, Atom* atom2, double* latticeparm, std::function<double(const char*, const char*)> threshold, double threshold_r, bool* pbc=0)
{
	if(atom1->num==1 & atom2->num==1) return(CheckDistance(p1s,p2s,atom1->radius,atom2->radius,latticeparm,threshold(atom1->name, atom2->name), false, pbc));
	
	vector<double>* ratom1=&(atom1->c.clus->radius);
	vector<double>* ratom2=&(atom2->c.clus->radius);
	vector<const char*>* name1=&(atom1->c.clus->name);
	vector<const char*>* name2=&(atom2->c.clus->name);
	double r1,r2;
	const char* c1; const char* c2;

	if(atom1->num==1) p1s_coms=p1s;
	if(atom2->num==1) p2s_coms=p2s;

	for (int i = 0; i < p1s_coms->size(); i++)
		for (int j = 0; j < p2s_coms->size(); j++)
		{	
			if(CheckDistance(&(*p1s_coms)[i], &(*p2s_coms)[j], atom1->radius, atom2->radius, latticeparm, threshold_r) == false) return false;
		}

	for (int i = 0; i < p1s->size(); i++)
		for (int j = 0; j < p2s->size(); j++)
		{	
			if(atom1->num==1) {r1=atom1->radius; c1=atom1->name;} else {r1=(*ratom1)[i%atom1->num];c1=(*name1)[i%atom1->num];}
			if(atom2->num==1) {r2=atom2->radius; c2=atom2->name; } else {r2=(*ratom2)[j%atom2->num]; c2=(*name2)[j%atom2->num];}
			if(CheckDistance(&(*p1s)[i], &(*p2s)[j], r1, r2, latticeparm, threshold(c1, c2)) == false) return false;
		}
	return true;
};//for every atom in p1s and p2s,if one atom is too close to another, return false 
#include <string>
bool CheckDistance(vector<position>* p1s,vector<position>* p1s_coms, vector<position>*p2s,vector<position>* p2s_coms, Atom* atom, double*latticeparm, std::function<double(const char*, const char*)> threshold, double threshold_r, bool* pbc=0)
{	
	if(atom->num==1) return (CheckDistance(p1s,p2s,atom->radius,latticeparm,threshold(atom->name, atom->name), false, pbc));

	for (int i = 0; i < p1s_coms->size(); i++)
	{
		for (int j = 0; j < i; j++)
		{
			if (CheckDistance(&(*p1s_coms)[i], &(*p1s_coms)[j], atom->radius, atom->radius, latticeparm, threshold_r) == false)  return false;
		}
	}

	vector<double>* ratom=&(atom->c.clus->radius);
	vector<const char*>* name=&(atom->c.clus->name);
	for(int i=0;i<p1s->size();i++) 
		for(int j=0;j<(i/atom->num)*atom->num;j++)
		{
			if(CheckDistance(&(*p1s)[i], &(*p1s)[j], (*ratom)[i%atom->num],  (*ratom)[j%atom->num], latticeparm, threshold( (*name)[i%atom->num], (*name)[j%atom->num] )) == false)  return false;
		}

	return (CheckDistance(p1s, p1s_coms, p2s, p2s_coms, atom, atom, latticeparm, threshold, threshold_r));
};