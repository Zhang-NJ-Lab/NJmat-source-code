#include"cluster.h"
#include"DataBase/sitesymData.h"

using namespace std;

/****************Here begins the detailed function. ****/
double symprec =1e-1;


cluster::cluster(vector<int>& atomnum, vector<double>& r, vector<double>& pos, vector<string>&names, double _symprec) 
{
    int mark=0;
    for(int i=0;i<atomnum.size();i++)
    {
        char* atomname=new char[names[i].size()+1];
        for(int j=0;j<names[i].size();j++) atomname[j]=names[i][j];atomname[names[i].size()]='\0';
        const char* tname=atomname;
        Name.push_back(tname);

        for(int j=0;j<atomnum[i];j++)
        {
            name.push_back(tname);
            radius.push_back(r[i]);
            cart_positions.push_back(position(pos[mark*3], pos[mark*3+1], pos[mark*3+2]));
            mark++;
        }
    }

    position center;
    for(int i=0;i<cart_positions.size();i++)
    {
        center+=cart_positions[i];
    }
    center.x/=cart_positions.size(); center.y/=cart_positions.size(); center.z/=cart_positions.size(); 

    for(int i=0;i<cart_positions.size();i++)
    {
        cart_originreset.push_back(position(cart_positions[i]-center));
    }
    for(int i=0;i<cart_originreset.size();i++)
    {
        if(fabs(cart_originreset[i].x)<1e-6) cart_originreset[i].x=0;
        if(fabs(cart_originreset[i].y)<1e-6) cart_originreset[i].y=0;
        if(fabs(cart_originreset[i].z)<1e-6) cart_originreset[i].z=0;

        //cout<<cart_originreset[i].x<<'\t'<<cart_originreset[i].y<<'\t'<<cart_originreset[i].z<<'\t'<<endl;
    }
    
    dimention=CalDimention(cart_originreset, vec);
    symprec = _symprec;
    GetSym(name, cart_originreset, symmetry, dimention, radius_BoundingSphere, &vec);
    DEBUG_INFO("radius_BoundingSphere= %f \n", radius_BoundingSphere);
    DEBUG_INFO("found symmetry operation of cluster:\n");
    string sym[] = {"", "minus1 ", "axis 2 ", "mirror plane ", "axis 4 ", "minus 4 ", "axis 3 ", "minus 3 ", "axis 6 ", "minus 6 "};
    for(int i = 1; i<symmetry.size(); i++)
    {
        if(symmetry[i].size()==0)   {DEBUG_INFO("%s: None", sym[i]);}
        else for(int j=0;j<symmetry[i].size();j++)  DEBUG_INFO("%s: %f\t%f\t%f\t", sym[i], symmetry[i][j].x, symmetry[i][j].y, symmetry[i][j].z);
        DEBUG_INFO("\n");
    }
};

cluster::cluster(const char* filename, double _symprec) 
{   
    ifstream in(filename);
    int n; in>>n; //n is types for atoms 
    for(int i=0;i<n;i++)
    {
        string AtomInfo;
        getline(in,AtomInfo);
        getline(in,AtomInfo);
        string names(AtomInfo.begin(), AtomInfo.begin()+(int)AtomInfo.find(":"));
        char* atomname=new char[names.length()+1];
        for(int i=0;i<names.length();i++) atomname[i]=names[i];atomname[names.length()]='\0';
        const char* tname=atomname;
        Name.push_back(tname);
        string rs(AtomInfo.begin()+(int)AtomInfo.find(":")+1, AtomInfo.begin()+(int)AtomInfo.find(" "));
        double r= atof(rs.c_str());
        string ns(AtomInfo.begin()+(int)AtomInfo.find(" ")+1, AtomInfo.end());
        int atomnum=(int) (atof(ns.c_str()));
        
        for(int j=0;j<atomnum;j++)
        {
            name.push_back(tname);
            radius.push_back(r);
            double x,y,z;
            in>>x>>y>>z;
            cart_positions.push_back(position(x,y,z));
            
        }
    }
    in.close();

    dimention=CalDimention(cart_positions, vec);
    
    position center;
    for(int i=0;i<cart_positions.size();i++)
    {
        center+=cart_positions[i];
    }
    center.x/=cart_positions.size(); center.y/=cart_positions.size(); center.z/=cart_positions.size(); 

    for(int i=0;i<cart_positions.size();i++)
    {
        cart_originreset.push_back(position(cart_positions[i]-center));
    }
    /*for(int i=0;i<cart_originreset.size();i++)
    {
        cout<<cart_originreset[i].x<<'\t'<<cart_originreset[i].y<<'\t'<<cart_originreset[i].z<<'\t'<<endl;
    }
    cout<<endl;*/
    symprec = _symprec;
    GetSym(name, cart_originreset, symmetry, dimention, radius_BoundingSphere, &vec);
    DEBUG_INFO("radius_BoundingSphere= %f \n", radius_BoundingSphere);
    DEBUG_INFO("found symmetry operation of cluster:\n");
    string sym[] = {"", "minus1 ", "axis 2 ", "mirror plane ", "axis 4 ", "minus 4 ", "axis 3 ", "minus 3 ", "axis 6 ", "minus 6 "};
    for(int i = 1; i<symmetry.size(); i++)
    {
        if(symmetry[i].size()==0)   {DEBUG_INFO("%s: None", sym[i]);}
        else for(int j=0;j<symmetry[i].size();j++)  DEBUG_INFO("%s: %f\t%f\t%f\t", sym[i], symmetry[i][j].x, symmetry[i][j].y, symmetry[i][j].z);
        DEBUG_INFO("\n");
    }
};


position PosRotate_frac(position* cart_p, position* axis, double Omega, double* latticeparm, bool get_frac_pos=true)
{
    double x=cart_p->x, y=cart_p->y, z=cart_p->z;
    double r=sqrt(CalDistance(axis));
    double r1=axis->x/r; double r2=axis->y/r; double r3= axis->z/r;
    double cosOmega=cos(Omega), sinOmega=sin(Omega);
    position cart_position;

    cart_position.x=x*(r1*r1*(1-cosOmega)+cosOmega)+y*(r1*r2*(1-cosOmega)-r3*sinOmega)+z*(r1*r3*(1-cosOmega)+r2*sinOmega);
    cart_position.y=x*(r1*r2*(1-cosOmega)+r3*sinOmega)+y*(r2*r2*(1-cosOmega)+cosOmega)+z*(r2*r3*(1-cosOmega)-r1*sinOmega);
    cart_position.z=x*(r1*r3*(1-cosOmega)-r2*sinOmega)+y*(r2*r3*(1-cosOmega)+r1*sinOmega)+z*(r3*r3*(1-cosOmega)+cosOmega);

    if(get_frac_pos)
    {
        position position_frac;
        inversepostrans(&position_frac, &cart_position, latticeparm);
        return position_frac;
    }
    else return cart_position;
}
position PosRotate_frac(position* cart_p, double Theta, double Phi, double Omega, double* latticeparm)
{
    double x=cart_p->x, y=cart_p->y, z=cart_p->z;
    double r1=sin(Theta)*cos(Phi), r2=sin(Theta)*sin(Phi), r3=cos(Theta);
    double cosOmega=cos(Omega), sinOmega=sin(Omega);
    position cart_position;

    cart_position.x=x*(r1*r1*(1-cosOmega)+cosOmega)+y*(r1*r2*(1-cosOmega)-r3*sinOmega)+z*(r1*r3*(1-cosOmega)+r2*sinOmega);
    cart_position.y=x*(r1*r2*(1-cosOmega)+r3*sinOmega)+y*(r2*r2*(1-cosOmega)+cosOmega)+z*(r2*r3*(1-cosOmega)-r1*sinOmega);
    cart_position.z=x*(r1*r3*(1-cosOmega)-r2*sinOmega)+y*(r2*r3*(1-cosOmega)+r1*sinOmega)+z*(r3*r3*(1-cosOmega)+cosOmega);

    position position_frac;
    inversepostrans(&position_frac, &cart_position, latticeparm);
   // cout<<cart_p->x<<'\t'<<cart_p->y<<'\t'<<cart_p->z<<"\tto\t"<<position_frac.x<<'\t'<<position_frac.y<<'\t'<<position_frac.z<<endl;
    return position_frac;
}




void clusterotate(position& tempaxis, double angle, vector<position>& cart_positions, vector< vector<position> >& clussym)
{
    //cout<<"tempaxis= "<<tempaxis.x<<'\t'<<tempaxis.y<<'\t'<<tempaxis.z<<'\t'<<"angle= "<<angle<<endl;
    for(int i=0;i<cart_positions.size();i++)
    {
        position tp=cart_positions[i];
        //cout<<"before "<<cart_positions[i].x<<'\t'<<cart_positions[i].y<<'\t'<<cart_positions[i].z<<endl;
        cart_positions[i]=PosRotate_frac(&tp, &tempaxis, angle, 0, false) ;
        //cout<<"after "<<cart_positions[i].x<<'\t'<<cart_positions[i].y<<'\t'<<cart_positions[i].z<<endl;
    }
    //cout<<"end"<<endl;
    for(int i=2;i<clussym.size();i++)
        for(int j=0;j<clussym[i].size();j++)
        {
            position tp=clussym[i][j];
            clussym[i][j]=PosRotate_frac(&tp, &tempaxis, angle, 0,false);
        }
    return;
}
void clusterotate(position& tempaxis, double angle,  vector<position>& cart_positions, position &selfvec)
{
    for(int i=0;i<cart_positions.size();i++)
    {
        position tp=cart_positions[i];
        cart_positions[i]=PosRotate_frac(&tp, &tempaxis, angle, 0, false) ;
    }
    position tp=selfvec;
    selfvec=PosRotate_frac(&tp, &tempaxis, angle, 0,false);
    return;
}


bool replace(vector<position>& cart_positions, cluster* clus, double *latticeparm, const vector<vector<int> >& symops)
{
    cart_positions.clear();
    //cout<<"replacestart"<<endl;
    for(int i=0;i<clus->cart_originreset.size();i++)
    {
        cart_positions.push_back(clus->cart_originreset[i]) ;
        //cout<< cart_positions[i].x<<'\t'<<cart_positions[i].y<<'\t'<<cart_positions[i].z<<endl;
    }

    vector<position> usedaxis;
    vector< vector<position> > clussym;
    clussym.resize(10);
    //position selfvec=clus->vec;
    for(int i=2;i<clus->symmetry.size();i++)
        for(int j=0;j<clus->symmetry[i].size();j++)
            clussym[i].push_back(clus->symmetry[i][j]);
    for(int i=2;i<symops.size();i++)
    {
        for(int j=0;j<symops[i].size();j++)
        {
            position symops_ij=symaxis[i][symops[i][j]];
            if(usedaxis.size()==0)
            {
                int ra=rand() % (clussym[i].size());
                position p=clussym[i][ra];
                if(p.sym(symops_ij)||(p+symops_ij).sym(position(0,0,0))) {}
                else 
                {
                    /*cout<<"symbefore"<<endl;
                    for(int j=0;j<clussym[i].size();j++) cout<<clussym[i][j].x<<'\t'<<clussym[i][j].y<<'\t'<<clussym[i][j].z<<'\t'<<endl;*/
                    //cout<<"p= "<<ra<<"_th"<<endl;
                    position tempaxis=Crossproduct(&symops_ij, &p);
                    double angle=-acos(Dotproduct(&symops_ij, &p));
                    //cout<<"from "<<p.x<<'\t'<<p.y<<'\t'<<p.z<<'\t'<<'\t'<<symops_ij.x<<'\t'<<symops_ij.y<<'\t'<<symops_ij.z<<'\t'<<endl;
                    clusterotate(tempaxis, angle, cart_positions, clussym);
                    /*cout<<"symafter"<<endl;
                    for(int j=0;j<clussym[i].size();j++) cout<<clussym[i][j].x<<'\t'<<clussym[i][j].y<<'\t'<<clussym[i][j].z<<'\t'<<endl;*/
                }
                usedaxis.push_back(symops_ij);
            }
            else if(usedaxis.size()==1)
            {
                if(usedaxis[0].sym(symops_ij)||(usedaxis[0]+symops_ij).sym(position(0,0,0))) 
                {
                    bool m=false;
                    for(int a=0;a<clussym[i].size();a++) 
                    {
                       if(clussym[i][a].sym(usedaxis[0])||(clussym[i][a]+usedaxis[0]).sym(position(0,0,0))) {m=true;break;}
                    }
                    if(m==false) {return false;}
                    continue;
                }

                position p=clussym[i][rand() % (clussym[i].size())];
                bool m=true; int att=0; double targetangle=fabs(Dotproduct(&usedaxis[0], &symops_ij));
                while (m&(att<20))
                {
                    if(p.sym(usedaxis[0])||(p+usedaxis[0]).sym(position(0,0,0))) 
                    {
                        att++;
                        p=clussym[i][rand() % (clussym[i].size())];
                    }
                    else 
                    {
                        if(fabs(fabs(Dotproduct(&usedaxis[0], &p))-targetangle)<1e-2) {m=false;break;}
                        else
                        {
                            att++;
                            p=clussym[i][rand() % (clussym[i].size())];
                        }
                    }
                }
                if(m) return false;
                
                if(p.sym(symops_ij)||(p+symops_ij).sym(position(0,0,0))) {}
                else 
                {
                    position tempaxis=usedaxis[0];
                    double angle=-acos(Dotproduct(&symops_ij, &p));
                    //cout<<"symbefore"<<endl;
                    //for(int j=0;j<clussym[i].size();j++) cout<<clussym[i][j].x<<'\t'<<clussym[i][j].y<<'\t'<<clussym[i][j].z<<'\t'<<endl;
                    //cout<<"from "<<p.x<<'\t'<<p.y<<'\t'<<p.z<<'\t'<<'\t'<<symops_ij.x<<'\t'<<symops_ij.y<<'\t'<<symops_ij.z<<'\t'<<endl;
                    clusterotate(tempaxis, angle, cart_positions, clussym);
                    //cout<<"symafter"<<endl;
                    //for(int j=0;j<clussym[i].size();j++) cout<<clussym[i][j].x<<'\t'<<clussym[i][j].y<<'\t'<<clussym[i][j].z<<'\t'<<endl;
                }
                usedaxis.push_back(symops_ij);
            }
            else 
            {
                bool m=false;
                
                for(int a=0;a<clussym[i].size();a++)
                {
                    if(clussym[i][a].sym(symops_ij)||(clussym[i][a]+symops_ij).sym(position(0,0,0))) {m=true;break;}
                }
                if(m==false) {return false;}
            }
        }
    }

    /*cout<<"usedaxis: ";
    for(int i=0;i<usedaxis.size();i++) cout<<usedaxis[i]<<'\t';
    cout<<endl;*/

    if(usedaxis.size()==0)
    {
        double Theta=acos(2.0*rand()/RAND_MAX-1);
        double Phi=2*M_PI*rand()/RAND_MAX;
        double Omega=2*M_PI*rand()/RAND_MAX;

        cart_positions.clear();
        for(int i=0;i<clus->cart_originreset.size();i++)
        {
            cart_positions.push_back(PosRotate_frac(&clus->cart_originreset[i],Theta, Phi, Omega, latticeparm)) ;
            //cout<< cart_positions[i].x<<'\t'<<cart_positions[i].y<<'\t'<<cart_positions[i].z<<endl;
        }
        return true;
    }
    else if(usedaxis.size()==1)
    {
        double Omega=2*M_PI*rand()/RAND_MAX;
        for(int i=0;i<cart_positions.size();i++)
        {
            position tp=cart_positions[i];
            cart_positions[i]=PosRotate_frac(&tp, &usedaxis[0], Omega, latticeparm) ;
            //cout<< cart_positions[i].x<<'\t'<<cart_positions[i].y<<'\t'<<cart_positions[i].z<<endl;
        }
        return true;
    }
    else
    {
        for(int i=0;i<cart_positions.size();i++)
        {   
            position tp=cart_positions[i];
            inversepostrans(&cart_positions[i], &tp, latticeparm);
            //cout<< cart_positions[i].x<<'\t'<<cart_positions[i].y<<'\t'<<cart_positions[i].z<<endl;
        }
        return true;
    }
    
};
bool replace_1d(vector<position>& cart_positions, cluster* clus, double *latticeparm, const vector<vector<int> >& symops)
{
    cart_positions.clear();

    for(int i=0;i<clus->cart_originreset.size();i++)
        cart_positions.push_back(clus->cart_originreset[i]) ;
    
    position selfvec=clus->vec;

    bool m=false; position wycksym; 
    for(int i=4 ; i<symops.size();i++)
    {   
        if(symops[i].size()>0) {m=true; wycksym=symaxis[i][symops[i][0]];break;} 
    }
    if(m==false)
        if(clus->symmetry[1].size()==0) 
            if(symops[2].size()>0) {m=true;wycksym=symaxis[2][symops[2][0]];} 

    if(m) 
    {
        position p=selfvec;
        if(p.sym(wycksym)||(p+wycksym).sym(position(0,0,0))) {}
        else 
        {
           
            position tempaxis=Crossproduct(&wycksym, &p);
            double angle=-acos(Dotproduct(&wycksym, &p));
            clusterotate(tempaxis, angle, cart_positions, selfvec); 
        }
        //cout<<"caseA\tselfvec= "<<selfvec<<endl;
    }
    else
    {
        vector<position> usedaxis;

        int i;
        if(clus->symmetry[1].size()==0) i=3;else i=2;
        for(;i<4;i++)
        {
            for(int j=0;j<symops[i].size();j++)
            {
                position symops_ij=symaxis[i][symops[i][j]];
                char choice;
                if(usedaxis.size()==0)
                {
                    position p;
                    if(rand()%2==0)
                    {
                        p=selfvec;
                        choice='a';
                    }
                    else
                    {   
                        double randtheta=acos(2.0*rand()/RAND_MAX-1);
                        double sintheta=sin(randtheta);
                        double randphi=2*M_PI*rand()/RAND_MAX;
       
                        position randp(sintheta*cos(randphi), sintheta*sin(randphi), cos(randtheta));
                        
                        p=Crossproduct(&randp, &selfvec);
                        p.renormalize();
                        choice='b';
                    }

                    if(p.sym(symops_ij)||(p+symops_ij).sym(position(0,0,0))) {}
                    else 
                    {  
                        position tempaxis=Crossproduct(&symops_ij, &p);
                        double angle=-acos(Dotproduct(&symops_ij, &p));
                        clusterotate(tempaxis, angle, cart_positions, selfvec); 
                        
                    }
                    usedaxis.push_back(symops_ij);
                }
                else if(usedaxis.size()==1)
                {
                    position p;
                    if(usedaxis[0].sym(symops_ij)||(usedaxis[0]+symops_ij).sym(position(0,0,0))) continue;
                    else if(choice=='a')
                    {
                        if(fabs(Dotproduct(&usedaxis[0], &symops_ij))>1e-2) return false;
                    }
                    else
                    {
                        if(rand()%4==0 & (fabs(Dotproduct(&usedaxis[0],&selfvec))<1e-2))
                        {
                            p=selfvec;
                            choice='a';
                            usedaxis.clear();

                            if(p.sym(symops_ij)||(p+symops_ij).sym(position(0,0,0))) {}
                            else 
                            {  
                                position tempaxis=Crossproduct(&symops_ij, &p);
                                double angle=-acos(Dotproduct(&symops_ij, &p));
                                clusterotate(tempaxis, angle, cart_positions, selfvec); 
                               
                            }
                            usedaxis.push_back(symops_ij);
                        }
                        else
                        {
                            position newvecpos=Crossproduct(&usedaxis[0],&symops_ij);
                            if(newvecpos.sym(selfvec)||(newvecpos+selfvec).sym(position(0,0,0))) {}
                            else
                            {  
                                position tempaxis=Crossproduct(&newvecpos, &selfvec);
                                double angle=-acos(Dotproduct(&newvecpos, &selfvec));
                                clusterotate(tempaxis, angle, cart_positions, selfvec);
                               
                            }
                            usedaxis.push_back(symops_ij);
                        }
                    }
                }
                else 
                {
                    double d=fabs(Dotproduct(&selfvec, &symops_ij));
                    if(d<1e-2) continue;
                    if(fabs(d-1)<1e-2) continue;
                    return false;
                }
            }
        }
        //cout<<"caseB\tselfvec= "<<selfvec<<endl;
        
    }

    for(int i=0;i<cart_positions.size();i++)
    {   
        position tp=cart_positions[i];
        inversepostrans(&cart_positions[i], &tp, latticeparm);
    }
    return true;
}

Cluster::Cluster(const Cluster &c)  
{
    clus=c.clus;

    for(int i=0;i<c.cart_positions_frac.size();i++)
        cart_positions_frac.push_back(c.cart_positions_frac[i]);
};

void Cluster::Input(const char* filename, double _symprec) 
{
    if(filename)
        clus=new cluster(filename, _symprec);
}
void Cluster::operator =(const Cluster &c)   
{
    clus=c.clus;

    cart_positions_frac.clear();
    for(int i=0;i<c.cart_positions_frac.size();i++)
        cart_positions_frac.push_back(c.cart_positions_frac[i]);
};

void Cluster::RePlace(double *latticeparm)   
{ 
    double Theta=acos(2.0*rand()/RAND_MAX-1);
    double Phi=2*M_PI*rand()/RAND_MAX;
    double Omega=2*M_PI*rand()/RAND_MAX;

    cart_positions_frac.clear();
    
    for(int i=0;i<clus->cart_originreset.size();i++)
    {
        cart_positions_frac.push_back(PosRotate_frac(&clus->cart_originreset[i],Theta, Phi, Omega, latticeparm)) ;
        
    }
    return;
};
bool Cluster::RePlace(double *latticeparm, const vector<vector<int> >& symops, int symop)
{
    cart_positions_frac.clear();
    if((symop==1)&(symops[1].size()>0))
    {
        
        double Theta=acos(2.0*rand()/RAND_MAX-1);
        double Phi=2*M_PI*rand()/RAND_MAX;
        double Omega=2*M_PI*rand()/RAND_MAX;

        for(int i=0;i<clus->cart_originreset.size();i++)
        {
            cart_positions_frac.push_back(PosRotate_frac(&clus->cart_originreset[i],Theta, Phi, Omega, latticeparm)) ;
        }
        return true;
    }

    if(clus->dimention==1)
    {
        
        int attempt=0;

        bool m=false;
        while (attempt<100)
        {
            if(replace_1d(cart_positions_frac, clus, latticeparm, symops)==true) {m=true;break;}
            else attempt++;
        }        
        if(m==false) return false;
        return true;
    }
    else 
    {
        
        int attempt=0;
        bool m=false;
        while (attempt<100)
        {
            if(replace(cart_positions_frac, clus, latticeparm, symops)==true) 
            {       
                m=true;break;
            }
            else attempt++;
        }        
        if(m==false) return false;
        
        return true;
    }
}; 