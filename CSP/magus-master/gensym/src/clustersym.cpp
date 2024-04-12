/*clustersym.cpp: Definition of
    functions of checking symmetry in class cluster

    void GetSym(vector<const char*> name, vector<position> pos, vector< vector<position> > &symmetry, int dimention, double & radius_BoundingSphere, position* selfvec=0)
        arguements: @vector of atomnames, @vector of atompositions, @molecule_dimention, @selfvec[None if molecule_dimention=3; normal vector if molecule_dimention=2; self if molecule_dimention=1]
        returns: @symmetry  [ 10*n 2D array, first dimension stands for " 'None', 'minus1' , '2' , 'mirror', '4', 'minus4', '3', 'minus3', '6', 'minus6' ", second dimension stores symm_axis of corresponding symmetry.]
                    @radius_BoundingSphere [ distance of the atom furthest from center] 
                    
    In this file, check_'symm' functions check whether arguement 'vec' is the symm_axis. 
                        Sym_'symm' functions look for trial symm_axises and call check_'symm' functions to check them.*/


#include <cstdlib>
#include <cmath>
#include <vector>
#include"position.h"
#include"getlatticeparm.h"
using namespace std;


#include "cluster.h"

using namespace std;

extern double symprec;

bool position::sym(const position &p,  double tolerance)
{
    if (tolerance ==-1) tolerance = symprec;

    if (fabs(x - p.x) > tolerance) return false;
    if (fabs(y - p.y) > tolerance) return false;
    if (fabs(z - p.z) > tolerance) return false;
    return true;
}

class sym       //atoms of same distance and species
{
public:
    double distance;
    const char* name;
    vector<position> pos;

    sym(double d, const char* n, position p){distance=d; name=n; pos.push_back(p);};
    sym(const sym &s)
    {
        distance=s.distance;
        name=s.name;
        for(int i=0;i<s.pos.size();i++) pos.push_back(s.pos[i]);
    }
};

bool check_minus1(vector<position> pos)           //"-1"                  
{
    for(int i=0;i<pos.size();i++)
    {
        bool mark=false;
        position p(-pos[i].x, -pos[i].y, -pos[i].z);
        for(int j=i+1;j<pos.size();j++)
        {
            if( p.sym(pos[j]) )  { pos.erase(pos.begin()+j); mark=true; break;}
        }
        if(mark==false) return false;
    }
    return true;
}
bool Sym_minus1(vector<sym>& vectorsym)
{
    for(int i=0;i<vectorsym.size();i++)
    {
        if(vectorsym[i].distance<symprec) continue;
        if(check_minus1(vectorsym[i].pos)==false) return false;
    }
    return true;
}

//to check if a plane whose normal vector equals vec is the mirror
bool check_m(vector<position> pos, position* vec) 
{
    for(int i=0;i<pos.size();i++)
    {
        double posi_vec=Dotproduct(&pos[i],vec);
        if(fabs(posi_vec)<symprec) {continue;}
        double co=-2*posi_vec;
        position p(pos[i].x+co*vec->x, pos[i].y+co*vec->y, pos[i].z+co*vec->z);
        /*cout<<"originpos\t"<<pos[i].x<<'\t'<<pos[i].y<<'\t'<<pos[i].z<<'\t'<<endl;
            cout<<"mirror\t"<<vec->x<<'\t'<<vec->y<<'\t'<<vec->z<<'\t'<<endl;
            cout<<"mpos\t"<<p.x<<'\t'<<p.y<<'\t'<<p.z<<'\t'<<endl;*/
        if(p.sym(pos[i])) {continue;}
        else
        {
            bool mark=false;
            for(int j=i+1;j<pos.size();j++)
            {    
                if(p.sym(pos[j])) { pos.erase(pos.begin()+j);mark=true; break;}
            }
            if(mark==false) return false;
        }
    }
    return true;
};

bool check_2(vector<position> pos, position* vec) 
{
    for(int i=0;i<pos.size();i++)
    {
        double co=2*Dotproduct(&pos[i],vec);
        position p(-pos[i].x+co*vec->x, -pos[i].y+co*vec->y, -pos[i].z+co*vec->z);
        if(p.sym(pos[i])) {continue;}
        else 
        {
            bool mark=false;  
            for(int j=i+1;j<pos.size();j++)
            {    
                if(p.sym(pos[j])) { pos.erase(pos.begin()+j);mark=true; break;}
            }
            if(mark==false) return false;
        }
    }
    return true;
};
bool check_4(vector<position> pos, position* vec) 
{
    for(int i=0;i<pos.size();i++)
    {
        double co=2*Dotproduct(&pos[i],vec);
        position z=Crossproduct(&pos[i],vec);
        vector<position> ps;
        ps.push_back(position(-pos[i].x+co*vec->x, -pos[i].y+co*vec->y, -pos[i].z+co*vec->z));

        if(ps[0].sym(pos[i])) {continue;}

        co/=2;
        ps.push_back(position(co*vec->x+z.x, co*vec->y+z.y, co*vec->z+z.z));
        ps.push_back(position(co*vec->x-z.x, co*vec->y-z.y, co*vec->z-z.z));
        
        for(int a=0;a<3;a++) 
        {
            bool mark=false;
            for(int j=i+1;j<pos.size();j++)
            {    
                if(ps[a].sym(pos[j])) {pos.erase(pos.begin()+j);mark=true; break;}
            }
            if(mark==false) return false;
        }        
    }
    return true;
}
bool check_minus4(vector<position> pos, position* vec) 
{
    for(int i=0;i<pos.size();i++)
    {
        double co=2*Dotproduct(&pos[i],vec);
        position z=Crossproduct(&pos[i],vec);
        vector<position> ps;
        ps.push_back(position(-pos[i].x+co*vec->x, -pos[i].y+co*vec->y, -pos[i].z+co*vec->z));

        if(ps[0].sym(pos[i])) {continue;}

        co/=(-2); 
        ps.push_back(position(co*vec->x+z.x, co*vec->y+z.y, co*vec->z+z.z));
        ps.push_back(position(co*vec->x-z.x, co*vec->y-z.y, co*vec->z-z.z));
        
        for(int a=0;a<3;a++) 
        {
            bool mark=false;
            for(int j=i+1;j<pos.size();j++)
            {    
                if(ps[a].sym(pos[j])) {pos.erase(pos.begin()+j);mark=true; break;}
            }
            if(mark==false) return false;
        }        
    }
    return true;
}
bool check_3(vector<position> pos, position* vec) 
{
    for(int i=0;i<pos.size();i++)
    {
        double co=1.5*Dotproduct(&pos[i],vec);
        position z=Crossproduct(&pos[i],vec)*0.86603;
        vector<position> ps;
        ps.push_back(position(co*vec->x+z.x-0.5*pos[i].x, co*vec->y+z.y-0.5*pos[i].y, co*vec->z+z.z-0.5*pos[i].z));

        if(ps[0].sym(pos[i])) {continue;}

        z*=(-1);
        ps.push_back(position(co*vec->x+z.x-0.5*pos[i].x, co*vec->y+z.y-0.5*pos[i].y, co*vec->z+z.z-0.5*pos[i].z));
        
        for(int a=0;a<2;a++) 
        {
            bool mark=false;
            for(int j=i+1;j<pos.size();j++)
            {    
                if(ps[a].sym(pos[j])) {pos.erase(pos.begin()+j);mark=true; break;}
            }
            if(mark==false) return false;
        }        
    }
    return true;
}
bool check_minus3(vector<position> pos, position* vec) 
{
    for(int i=0;i<pos.size();i++)
    {
        double co=-1.5*Dotproduct(&pos[i],vec);
        position z=Crossproduct(&pos[i],vec)*(-0.86603);
        vector<position> ps;
        ps.push_back(position(co*vec->x+z.x+0.5*pos[i].x, co*vec->y+z.y+0.5*pos[i].y, co*vec->z+z.z+0.5*pos[i].z));

        if(ps[0].sym(pos[i])) {continue;}

        ps.push_back(ps[0]*(-1));
        z*=(-1);
        ps.push_back(position(co*vec->x+z.x+0.5*pos[i].x, co*vec->y+z.y+0.5*pos[i].y, co*vec->z+z.z+0.5*pos[i].z));
        ps.push_back(ps[2]*(-1));
        ps.push_back(position (-pos[i].x, -pos[i].y, -pos[i].z));
        
        for(int a=0;a<5;a++) 
        {
            bool mark=false;
            for(int j=i+1;j<pos.size();j++)
            {    
                if(ps[a].sym(pos[j])) {pos.erase(pos.begin()+j);mark=true; break;}
            }
            if(mark==false) return false;
        }        
    }
    return true;
}


void Sym_pos(position& v, char type, vector<sym>* vectorsym, vector<position>* vecs)
{
    v.renormalize();
    bool isnewvec=true;
    for(int t=0;t<vecs->size();t++) if(v.sym((*vecs)[t], 1e-2)||(v+(*vecs)[t]).sym(position(0,0,0), 1e-2)) {isnewvec=false;break;}
    if(isnewvec)
    {
        bool add=true;
        switch (type)
        {
            case 'm':
            {
                for(int j=0;j<vectorsym->size();j++)
                    if(check_m((*vectorsym)[j].pos, &v)==false) {add=false;break;}
            }
            break;
        
            case '2':
            {
                for(int j=0;j<vectorsym->size();j++)
                    if(check_2((*vectorsym)[j].pos, &v)==false) {add=false;break;}
            }
            break;

            case '4':
            {
                for(int j=0;j<vectorsym->size();j++)
                    if(check_4((*vectorsym)[j].pos, &v)==false) {add=false;break;}
            }
            break;

            case 'F':
            {
                for(int j=0;j<vectorsym->size();j++)
                    if(check_minus4((*vectorsym)[j].pos, &v)==false) {add=false;break;}
            }
            break;

            case '3':
            {
                for(int j=0;j<vectorsym->size();j++)
                    if(check_3((*vectorsym)[j].pos, &v)==false) {add=false;break;}
            }
            break;

            case 'T':
            {
                for(int j=0;j<vectorsym->size();j++)
                    if(check_minus3((*vectorsym)[j].pos, &v)==false) {add=false;break;}
            }
            break;


            default: {}
            break;
        }
        
        if(add) vecs->push_back(v);
    }
    return;
}


//find mirror plane in pos
bool Sym_m(vector<sym>& vectorsym, vector<position>& vecs)              //"m"
{
    for(int i=0;i<vectorsym.size();i++)
    {
        if(vectorsym[i].distance<symprec) continue;

        for(int a=0;a<vectorsym[i].pos.size();a++)
            for(int b=a+1;b<vectorsym[i].pos.size();b++)
            {
                position v=vectorsym[i].pos[a]-vectorsym[i].pos[b];
                Sym_pos(v, 'm', &vectorsym, &vecs);
            }
    }

    if(vecs.size()>0) return true;
    return false;
}


bool Sym_2(vector<sym>& vectorsym, vector<position>& vecs)              //"2"
{
    bool stepmark=false;

    for(int i=0;i<vectorsym.size();i++)
    {
        if(vectorsym[i].distance<symprec) continue;
        if(vectorsym[i].pos.size()==1)
        {
            position v=vectorsym[i].pos[0];
            v.renormalize();
            for(int j=0;j<vectorsym.size();j++)
                if(check_2(vectorsym[j].pos, &v)==false) {return false;}
            vecs.push_back(v);
            return true;
        }

        for(int a=0;a<vectorsym[i].pos.size();a++)
            for(int b=a+1;b<vectorsym[i].pos.size();b++)
            {
                position v=vectorsym[i].pos[a]+vectorsym[i].pos[b];
                if(v.sym(position(0,0,0))) {stepmark=true;continue;}               
                Sym_pos(v, '2', &vectorsym, &vecs);
            }
    }

    if(stepmark==true)
    {
        for(int i=0;i<vectorsym.size();i++)
            for(int a=0;a<vectorsym[i].pos.size();a++)
            {
                position v=vectorsym[i].pos[a];
                Sym_pos(v, '2', &vectorsym, &vecs);
            }
    }
        

    if(vecs.size()>0) return true;
    return false;
};


/*
void mids(int n, vector<position>* originpos, int i, vector<position>& resultpos, vector<position>& rfinal)
{
    if(n==1) 
    {
        for(int j=i+1;j<originpos->size();j++) 
            for(int a=0;a<resultpos.size();a++)
                rfinal.push_back(position(resultpos[a]+(*originpos)[j]));

        return;
    }
    else
    {
        for(int j=i+1;j<originpos->size();j++) 
        {
            vector<position> tempr;
            for(int a=0;a<resultpos.size();a++)
                tempr.push_back(position(resultpos[a]+(*originpos)[j]));
            return mids(n-1, originpos, j, tempr, rfinal);
        }
        return;
    }
}
*/
bool Sym_4(vector<sym>& vectorsym, vector<position>& vecs, vector<position>* vec_2)              //"4"
{
    for(int i=0;i<vectorsym.size();i++)
    {
        if(vectorsym[i].distance< symprec ) continue;
        if(vectorsym[i].pos.size()==3) return false;
        else if(vectorsym[i].pos.size()<4)
        {
            position v=vectorsym[i].pos[0];
            v.renormalize();
            for(int j=0;j<vectorsym.size();j++)
                if(check_4(vectorsym[j].pos, &v)==false) {return false;}
            vecs.push_back(v);
            return true;
        }
        else break;
    }


    for(int i=0;i<vec_2->size();i++)
    {
        position v=(*vec_2)[i];
        bool add=true;
        for(int j=0;j<vectorsym.size();j++)
            if(check_4(vectorsym[j].pos, &v)==false) {add=false;break;}
        if(add)  vecs.push_back(v);
    }



        /*bool stepmark=false;

        vector<position> midpos;
        vector<position> temp; temp.push_back(position(0,0,0));

        if(vectorsym[i].pos.size()<6) mids(4,&vectorsym[i].pos,0,temp,midpos);
        else mids(2,&vectorsym[i].pos,0,temp,midpos);

        for(int i=0;i<midpos.size();i++)
        {
            position v=midpos[i];
            if(v.sym(position(0,0,0))) {stepmark=true;continue;}
            Sym_pos(v, '4', &vectorsym, &vecs);
        }*/

        /*if(stepmark==true)
        {
            for(int a=0;a<vectorsym[i].pos.size();a++)
            {
                position v=vectorsym[i].pos[a];
                Sym_pos(v, '4', &vectorsym, &vecs);
            }
        }*/
    

    if(vecs.size()>0) return true;
    return false;
};


bool Sym_minus4(vector<sym>& vectorsym, vector<position>& vecs, vector<position>* vec_2)              //"-4"
{
    for(int i=0;i<vectorsym.size();i++)
    {
        if(vectorsym[i].distance<symprec) continue;
        if(vectorsym[i].pos.size()==3) return false;
        else if(vectorsym[i].pos.size()<4)
        {
            position v=vectorsym[i].pos[0];
            v.renormalize();
            for(int j=0;j<vectorsym.size();j++)
                if(check_minus4(vectorsym[j].pos, &v)==false) {return false;}
            vecs.push_back(v);
            return true;
        }
        else break;
    }

    for(int i=0;i<vec_2->size();i++)
    {
        position v=(*vec_2)[i];
        bool add=true;
        for(int j=0;j<vectorsym.size();j++)
            if(check_minus4(vectorsym[j].pos, &v)==false) {add=false;break;}
        if(add)  vecs.push_back(v);
    }



        /*bool stepmark=false;

        vector<position> midpos;
        vector<position> temp; temp.push_back(position(0,0,0));
        
        mids(2,&vectorsym[i].pos,0,temp,midpos);

        for(int i=0;i<midpos.size();i++)
        {
            position v=midpos[i];
            if(v.sym(position(0,0,0))) {stepmark=true;continue;}
            Sym_pos(v, 'F', &vectorsym, &vecs);
        }*/

        /*if(stepmark==true)
        {
            for(int a=0;a<vectorsym[i].pos.size();a++)
            {
                position v=vectorsym[i].pos[a];
                Sym_pos(v, 'F', &vectorsym, &vecs);
            }
        }*/
    

    if(vecs.size()>0) return true;
    return false;
};




bool Sym_3(vector<sym>& vectorsym, vector<position>& vecs)              //"3"
{
    for(int i=0;i<vectorsym.size();i++)
    {
        if(vectorsym[i].distance<symprec) continue;
        if(vectorsym[i].pos.size()<3)
        {
            position v=vectorsym[i].pos[0];
            v.renormalize();
            for(int j=0;j<vectorsym.size();j++)
                if(check_3(vectorsym[j].pos, &v)==false) {return false;}
            vecs.push_back(v);
            return true;
        }

        for(int a=0;a<vectorsym[i].pos.size();a++)
        {
            for(int b=a+1;b<vectorsym[i].pos.size();b++)
            {
                for(int c=b+1;c<vectorsym[i].pos.size();c++)
                {
                    position v=vectorsym[i].pos[a]+vectorsym[i].pos[b]+vectorsym[i].pos[c];
                    if(v.sym(position(0,0,0)))  v=Crossproduct(&vectorsym[i].pos[a], &vectorsym[i].pos[b]);
                    Sym_pos(v, '3', &vectorsym, &vecs);
                }
            }
        }
        
    
        /*bool stepmark=false;

        //vector<position> midpos;
        //vector<position> temp; temp.push_back(position(0,0,0));
        
        //mids(3,&vectorsym[i].pos,0,temp,midpos);

        for(int i=0;i<midpos.size();i++)
        {
            position v=midpos[i];
            if(v.sym(position(0,0,0))) {stepmark=true;continue;}
            Sym_pos(v, '3', &vectorsym, &vecs);
        }*/

        /*if(stepmark==true)
        {
            for(int a=0;a<vectorsym[i].pos.size();a++)
            {
                position v=vectorsym[i].pos[a];
                Sym_pos(v, 'F', &vectorsym, &vecs);
            }
        }*/
    
    }
    if(vecs.size()>0) return true;
    return false;
};

bool Sym_minus3(vector<sym>& vectorsym, vector<position>& vecs, bool sym_minus1,vector<position>* vec_3)              //"-3"
{
    if(sym_minus1==false) return false;
    else for(int a=0;a<vec_3->size();a++)
        vecs.push_back((*vec_3)[a]);
    if(vecs.size()>0) return true;
    return false;
};

/*bool check_6(vector<position> pos, position* vec) 
{
    for(int i=0;i<pos.size();i++)
    {
        double co=2*Dotproduct(&pos[i],vec);
        vector<position> ps;
        ps.push_back(position(-pos[i].x+co*vec->x, -pos[i].y+co*vec->y, -pos[i].z+co*vec->z));          //180
        
        if(ps[0].sym(pos[i])) {continue;}

        co/=4;
        position z=0.86603*Crossproduct(&pos[i],vec);
        ps.push_back(position(co*vec->x+z.x+0.5*pos[i].x, co*vec->y+z.y+0.5*pos[i].y, co*vec->z+z.z+0.5*pos[i].z));     //60

        z*=(-1);
        ps.push_back(position(co*vec->x+z.x+0.5*pos[i].x, co*vec->y+z.y+0.5*pos[i].y, co*vec->z+z.z+0.5*pos[i].z));     //300

        co*=3;
        ps.push_back(position(co*vec->x+z.x-0.5*pos[i].x, co*vec->y+z.y-0.5*pos[i].y, co*vec->z+z.z-0.5*pos[i].z));     //120

        z*=(-1);
        ps.push_back(position(co*vec->x+z.x-0.5*pos[i].x, co*vec->y+z.y-0.5*pos[i].y, co*vec->z+z.z-0.5*pos[i].z));     //240

        for(int a=0;a<5;a++) 
        {
            bool mark=false;
            for(int j=i+1;j<pos.size();j++)
            {    
                if(ps[a].sym(pos[j])) {pos.erase(pos.begin()+j);mark=true; break;}
            }
            if(mark==false) return false;
        }        
    }
    return true;
}*/

bool Sym_6(vector<sym>& vectorsym, vector<position>& vecs, vector<position>* vec_2, vector<position>* vec_3)              //"6"
{
    for(int i=0;i<vec_2->size();i++)
        for(int j=0;j<vec_3->size();j++)
        {
            if((*vec_2)[i].sym((*vec_3)[j])) vecs.push_back((*vec_2)[i]);
            else if(((*vec_2)[i]+(*vec_3)[j]).sym(position(0,0,0))) vecs.push_back((*vec_2)[i]);
        }
    
    if(vecs.size()>0) return true;
    return false;
};

bool Sym_minus6(vector<sym>& vectorsym, vector<position>& vecs, vector<position>* vec_3, vector<position>* vec_m)              //"-6"
{
    for(int i=0;i<vec_3->size();i++)
        for(int j=0;j<vec_m->size();j++)
        {
            if((*vec_3)[i].sym((*vec_m)[j])) vecs.push_back((*vec_3)[i]);
            else if(((*vec_3)[i]+(*vec_m)[j]).sym(position(0,0,0))) vecs.push_back((*vec_3)[i]);
        }
    
    if(vecs.size()>0) return true;
    return false;
};


int CalDimention(vector<position> &pos, position& vec)  
{   
    position p; double v=0; 
    for(int i=0;i<pos.size();i++)
    {
        if(pos[i].sym(position(0,0,0))) continue;
        for(int j=i+1; j<pos.size();j++)
        {
            p=Crossproduct(&pos[i],&pos[j]);
            if(p.sym(position(0,0,0))) continue;
            else for(int k=j+1;k<pos.size();k++)
            {
                v=Dotproduct(&p, &pos[k]);
                if(fabs(v)>1e-2) return 3;
            }
        }
    }
    if(p.sym(position(0,0,0))) 
    {
        vec=pos[1]-pos[0];
        vec.renormalize();
        return 1;
    }
    vec=p;
    vec.renormalize();
    return 2;
}

void GetSym(vector<const char*> name, vector<position> pos, vector< vector<position> > &symmetry, int dimention, double & radius_BoundingSphere, position* selfvec)
{
    //cout<<"symprec = "<<symprec<<endl;
    radius_BoundingSphere=0;
    //cout<<dimention<<endl;
    vector<sym> sortsym;
    for(int i=0;i<pos.size();i++)
    {
        double d=sqrt(CalDistance(&pos[i]));
        radius_BoundingSphere = Max(d, radius_BoundingSphere);
        bool mark=false;
        for(int j=0;j<sortsym.size();j++)
        {
            if(fabs(sortsym[j].distance-d)< symprec) 
                if(sortsym[j].name==name[i])
                {
                    mark=true;
                    sortsym[j].pos.push_back(pos[i]);
                    break;
                }
        }
        if(mark==false) sortsym.push_back(sym(d, name[i], pos[i]));
    }

    vector<sym> sorted;
    for(int i=0;i<sortsym.size(); i++)
    {
        int j;
        for(j=0;j<sorted.size();j++)
        {
            if(sortsym[i].pos.size()>sorted[j].pos.size()) {}
            else break;
        }
        sorted.insert(sorted.begin()+j, sortsym[i]);
    };

    /********************************************************
    Sym_minus1 =sym[1]****Sym_2 =sym[2]****Sym_m =sym[3]
    Sym_4 =sym[4]****Sym_minus4 =sym[5]****Sym_3 =sym[6]
    Sym_minus3 =sym[7]****Sym_6 =sym[8]****Sym_minus6 =sym[9]
    ********************************************************/
    symmetry.resize(10);
    if(Sym_minus1(sorted)) symmetry[1].push_back(position(0,0,0));
    Sym_2(sorted, symmetry[2]);
    Sym_m(sorted, symmetry[3]);
    Sym_4(sorted, symmetry[4], &symmetry[2]);
    Sym_minus4(sorted, symmetry[5], &symmetry[2]);
    Sym_3(sorted, symmetry[6]);
    Sym_minus3(sorted, symmetry[7], (symmetry[1].size()>0), &symmetry[6]);
    Sym_6(sorted, symmetry[8], &symmetry[2], &symmetry[6]);
    Sym_minus6(sorted, symmetry[9], &symmetry[6], &symmetry[3]);

    if(dimention==2)
    {
        symmetry[3].push_back(*selfvec);
        bool sym2=false;
        {
            bool add=true;
            for(int j=0;j<sorted.size();j++)
                if(check_2(sorted[j].pos, selfvec)==false) {add=false; break;}
            if(add) 
            {
                symmetry[2].push_back(*selfvec);
                sym2=true;
            }
        }
        if(sym2)
        {
            bool add=true;
            for(int j=0;j<sorted.size();j++)
                if(check_4(sorted[j].pos, selfvec)==false) {add=false; break;}
            if(add) 
            {
                symmetry[4].push_back(*selfvec);
                symmetry[5].push_back(*selfvec);
            }
        }
        bool sym3=false;
        {
            for(int a=0;a<symmetry[6].size();a++)  
                if(symmetry[6][a].sym(*selfvec)||(symmetry[6][a]+*selfvec).sym(position(0,0,0))) sym3=true;
        }
        if(sym3)
        {
            symmetry[9].push_back(*selfvec);
        }
        if(sym3&sym2)
        {
            symmetry[8].push_back(*selfvec);
        }
        
    }
    
    
    /*Plus:
    For 1D clusters, CrossProduct of randp and selfvec is also mirror vec
    if its selfvec is mirror vec,  CrossProduct of randp and selfvec is also 2-axis*/

};


