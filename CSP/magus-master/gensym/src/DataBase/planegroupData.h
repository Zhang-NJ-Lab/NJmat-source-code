#pragma once
#include <cstdlib>
#include <vector>
using namespace std;
/*transform matrix [trans_PL] and wyckoff position matrix [wyck_PL] of planegroup 1-17.
    export from Bilbao Crystallographic Server, 20201013
	DataBase: vector< vector<double> > wyck_PL;
					vector< vector<double> > trans_PL;
					*/
static const vector< vector<double> > wyck_PL =
{
    //Wyckoff Positions of Plane Group p 1 (No. 1)
    {/*num of wyckpos =*/1, /*label=*/97, 1,0,0,0,0,1,0,0,0,0,0,0, /*mul=*/1, /*uni=*/0	},
    //Wyckoff Positions of Plane Group p 2 (No. 2)
    {/*num of wyckpos =*/5, /*label=*/97, 0,0,0,0,0,0,0,0,0,0,0,0, /*mul=*/1, /*uni=*/1,	/*label=*/98, 0,0,0,0,0,0,0,0.5,0,0,0,0, /*mul=*/1, /*uni=*/1,	/*label=*/99, 0,0,0,0.5,0,0,0,0,0,0,0,0, /*mul=*/1, /*uni=*/1,	/*label=*/100, 0,0,0,0.5,0,0,0,0.5,0,0,0,0, /*mul=*/1, /*uni=*/1,	/*label=*/101, 1,0,0,0,0,1,0,0,0,0,0,0, /*mul=*/2, /*uni=*/0	},
    //Wyckoff Positions of Plane Group p 1 m 1 (No. 3)
    {/*num of wyckpos =*/3, /*label=*/97, 0,0,0,0,0,1,0,0,0,0,0,0, /*mul=*/1, /*uni=*/0,	/*label=*/98, 0,0,0,0.5,0,1,0,0,0,0,0,0, /*mul=*/1, /*uni=*/0,	/*label=*/99, 1,0,0,0,0,1,0,0,0,0,0,0, /*mul=*/2, /*uni=*/0	},
    //Wyckoff Positions of Plane Group p 1 g 1 (No. 4)
    {/*num of wyckpos =*/1, /*label=*/97, 1,0,0,0,0,1,0,0,0,0,0,0, /*mul=*/2, /*uni=*/0	},
    //Wyckoff Positions of Plane Group c 1 m 1 (No. 5)
    {/*num of wyckpos =*/2, /*label=*/97, 0,0,0,0,0,1,0,0,0,0,0,0, /*mul=*/2, /*uni=*/0,	/*label=*/98, 1,0,0,0,0,1,0,0,0,0,0,0, /*mul=*/4, /*uni=*/0	},
    //Wyckoff Positions of Plane Group p 2 m m (No. 6)
    {/*num of wyckpos =*/9, /*label=*/97, 0,0,0,0,0,0,0,0,0,0,0,0, /*mul=*/1, /*uni=*/1,	/*label=*/98, 0,0,0,0,0,0,0,0.5,0,0,0,0, /*mul=*/1, /*uni=*/1,	/*label=*/99, 0,0,0,0.5,0,0,0,0,0,0,0,0, /*mul=*/1, /*uni=*/1,	/*label=*/100, 0,0,0,0.5,0,0,0,0.5,0,0,0,0, /*mul=*/1, /*uni=*/1,	/*label=*/101, 1,0,0,0,0,0,0,0,0,0,0,0, /*mul=*/2, /*uni=*/0,	/*label=*/102, 1,0,0,0,0,0,0,0.5,0,0,0,0, /*mul=*/2, /*uni=*/0,	/*label=*/103, 0,0,0,0,0,1,0,0,0,0,0,0, /*mul=*/2, /*uni=*/0,	/*label=*/104, 0,0,0,0.5,0,1,0,0,0,0,0,0, /*mul=*/2, /*uni=*/0,	/*label=*/105, 1,0,0,0,0,1,0,0,0,0,0,0, /*mul=*/4, /*uni=*/0	},
    //Wyckoff Positions of Plane Group p 2 m g (No. 7)
    {/*num of wyckpos =*/4, /*label=*/97, 0,0,0,0,0,0,0,0,0,0,0,0, /*mul=*/2, /*uni=*/1,	/*label=*/98, 0,0,0,0,0,0,0,0.5,0,0,0,0, /*mul=*/2, /*uni=*/1,	/*label=*/99, 0,0,0,0.25,0,1,0,0,0,0,0,0, /*mul=*/2, /*uni=*/0,	/*label=*/100, 1,0,0,0,0,1,0,0,0,0,0,0, /*mul=*/4, /*uni=*/0	},
    //Wyckoff Positions of Plane Group p 2 g g (No. 8)
    {/*num of wyckpos =*/3, /*label=*/97, 0,0,0,0,0,0,0,0,0,0,0,0, /*mul=*/2, /*uni=*/1,	/*label=*/98, 0,0,0,0.5,0,0,0,0,0,0,0,0, /*mul=*/2, /*uni=*/1,	/*label=*/99, 1,0,0,0,0,1,0,0,0,0,0,0, /*mul=*/4, /*uni=*/0	},
    //Wyckoff Positions of Plane Group c 2 m m (No. 9)
    {/*num of wyckpos =*/6, /*label=*/97, 0,0,0,0,0,0,0,0,0,0,0,0, /*mul=*/2, /*uni=*/1,	/*label=*/98, 0,0,0,0,0,0,0,0.5,0,0,0,0, /*mul=*/2, /*uni=*/1,	/*label=*/99, 0,0,0,0.25,0,0,0,0.25,0,0,0,0, /*mul=*/4, /*uni=*/1,	/*label=*/100, 1,0,0,0,0,0,0,0,0,0,0,0, /*mul=*/4, /*uni=*/0,	/*label=*/101, 0,0,0,0,0,1,0,0,0,0,0,0, /*mul=*/4, /*uni=*/0,	/*label=*/102, 1,0,0,0,0,1,0,0,0,0,0,0, /*mul=*/8, /*uni=*/0	},
    //Wyckoff Positions of Plane Group p 4 (No. 10)
    {/*num of wyckpos =*/4, /*label=*/97, 0,0,0,0,0,0,0,0,0,0,0,0, /*mul=*/1, /*uni=*/1,	/*label=*/98, 0,0,0,0.5,0,0,0,0.5,0,0,0,0, /*mul=*/1, /*uni=*/1,	/*label=*/99, 0,0,0,0.5,0,0,0,0,0,0,0,0, /*mul=*/2, /*uni=*/1,	/*label=*/100, 1,0,0,0,0,1,0,0,0,0,0,0, /*mul=*/4, /*uni=*/0	},
    //Wyckoff Positions of Plane Group p 4 m m (No. 11)
    {/*num of wyckpos =*/7, /*label=*/97, 0,0,0,0,0,0,0,0,0,0,0,0, /*mul=*/1, /*uni=*/1,	/*label=*/98, 0,0,0,0.5,0,0,0,0.5,0,0,0,0, /*mul=*/1, /*uni=*/1,	/*label=*/99, 0,0,0,0.5,0,0,0,0,0,0,0,0, /*mul=*/2, /*uni=*/1,	/*label=*/100, 1,0,0,0,0,0,0,0,0,0,0,0, /*mul=*/4, /*uni=*/0,	/*label=*/101, 1,0,0,0,0,0,0,0.5,0,0,0,0, /*mul=*/4, /*uni=*/0,	/*label=*/102, 1,0,0,0,1,0,0,0,0,0,0,0, /*mul=*/4, /*uni=*/0,	/*label=*/103, 1,0,0,0,0,1,0,0,0,0,0,0, /*mul=*/8, /*uni=*/0	},
    //Wyckoff Positions of Plane Group p 4 g m (No. 12)
    {/*num of wyckpos =*/4, /*label=*/97, 0,0,0,0,0,0,0,0,0,0,0,0, /*mul=*/2, /*uni=*/1,	/*label=*/98, 0,0,0,0.5,0,0,0,0,0,0,0,0, /*mul=*/2, /*uni=*/1,	/*label=*/99, 1,0,0,0,1,0,0,0.5,0,0,0,0, /*mul=*/4, /*uni=*/0,	/*label=*/100, 1,0,0,0,0,1,0,0,0,0,0,0, /*mul=*/8, /*uni=*/0	},
    //Wyckoff Positions of Plane Group p 3 (No. 13)
    {/*num of wyckpos =*/4, /*label=*/97, 0,0,0,0,0,0,0,0,0,0,0,0, /*mul=*/1, /*uni=*/1,	/*label=*/98, 0,0,0,0.333333,0,0,0,0.666667,0,0,0,0, /*mul=*/1, /*uni=*/1,	/*label=*/99, 0,0,0,0.666667,0,0,0,0.333333,0,0,0,0, /*mul=*/1, /*uni=*/1,	/*label=*/100, 1,0,0,0,0,1,0,0,0,0,0,0, /*mul=*/3, /*uni=*/0	},
    //Wyckoff Positions of Plane Group p 3 m 1 (No. 14)
    {/*num of wyckpos =*/5, /*label=*/97, 0,0,0,0,0,0,0,0,0,0,0,0, /*mul=*/1, /*uni=*/1,	/*label=*/98, 0,0,0,0.333333,0,0,0,0.666667,0,0,0,0, /*mul=*/1, /*uni=*/1,	/*label=*/99, 0,0,0,0.666667,0,0,0,0.333333,0,0,0,0, /*mul=*/1, /*uni=*/1,	/*label=*/100, 1,0,0,0,-1,0,0,0,0,0,0,0, /*mul=*/3, /*uni=*/0,	/*label=*/101, 1,0,0,0,0,1,0,0,0,0,0,0, /*mul=*/6, /*uni=*/0	},
    //Wyckoff Positions of Plane Group p 3 1 m (No. 15)
    {/*num of wyckpos =*/4, /*label=*/97, 0,0,0,0,0,0,0,0,0,0,0,0, /*mul=*/1, /*uni=*/1,	/*label=*/98, 0,0,0,0.333333,0,0,0,0.666667,0,0,0,0, /*mul=*/2, /*uni=*/1,	/*label=*/99, 1,0,0,0,0,0,0,0,0,0,0,0, /*mul=*/3, /*uni=*/0,	/*label=*/100, 1,0,0,0,0,1,0,0,0,0,0,0, /*mul=*/6, /*uni=*/0	},
    //Wyckoff Positions of Plane Group p 6 (No. 16)
    {/*num of wyckpos =*/4, /*label=*/97, 0,0,0,0,0,0,0,0,0,0,0,0, /*mul=*/1, /*uni=*/1,	/*label=*/98, 0,0,0,0.333333,0,0,0,0.666667,0,0,0,0, /*mul=*/2, /*uni=*/1,	/*label=*/99, 0,0,0,0.5,0,0,0,0,0,0,0,0, /*mul=*/3, /*uni=*/1,	/*label=*/100, 1,0,0,0,0,1,0,0,0,0,0,0, /*mul=*/6, /*uni=*/0	},
    //Wyckoff Positions of Plane Group p 6 m m (No. 17)
    {/*num of wyckpos =*/6, /*label=*/97, 0,0,0,0,0,0,0,0,0,0,0,0, /*mul=*/1, /*uni=*/1,	/*label=*/98, 0,0,0,0.333333,0,0,0,0.666667,0,0,0,0, /*mul=*/2, /*uni=*/1,	/*label=*/99, 0,0,0,0.5,0,0,0,0,0,0,0,0, /*mul=*/3, /*uni=*/1,	/*label=*/100, 1,0,0,0,0,0,0,0,0,0,0,0, /*mul=*/6, /*uni=*/0,	/*label=*/101, 1,0,0,0,-1,0,0,0,0,0,0,0, /*mul=*/6, /*uni=*/0,	/*label=*/102, 1,0,0,0,0,1,0,0,0,0,0,0, /*mul=*/12, /*uni=*/0	},
};
static const vector< vector<double> > trans_PL =
{
    //Wyckoff Positions of Plane Group p 1 (No. 1)
    {/*rotmatrix num=*/1, /*transmatrix num=*/1, /*rotmatrix =*/1,0,0,0,0,1,0,0,0,0,0,0,	/*transmatrix =*/0,0,0,   },
    //Wyckoff Positions of Plane Group p 2 (No. 2)
    {/*rotmatrix num=*/2, /*transmatrix num=*/1, /*rotmatrix =*/1,0,0,0,0,1,0,0,0,0,0,0,	-1,0,0,0,0,-1,0,0,0,0,0,0,	/*transmatrix =*/0,0,0,   },
    //Wyckoff Positions of Plane Group p 1 m 1 (No. 3)
    {/*rotmatrix num=*/2, /*transmatrix num=*/1, /*rotmatrix =*/1,0,0,0,0,1,0,0,0,0,0,0,	-1,0,0,0,0,1,0,0,0,0,0,0,	/*transmatrix =*/0,0,0,   },
    //Wyckoff Positions of Plane Group p 1 g 1 (No. 4)
    {/*rotmatrix num=*/2, /*transmatrix num=*/1, /*rotmatrix =*/1,0,0,0,0,1,0,0,0,0,0,0,	-1,0,0,0,0,1,0,0.5,0,0,0,0,	/*transmatrix =*/0,0,0,   },
    //Wyckoff Positions of Plane Group c 1 m 1 (No. 5)
    {/*rotmatrix num=*/2, /*transmatrix num=*/2, /*rotmatrix =*/1,0,0,0,0,1,0,0,0,0,0,0,	-1,0,0,0,0,1,0,0,0,0,0,0,	/*transmatrix =*/0,0,0,   0.5,0.5,0,    },
    //Wyckoff Positions of Plane Group p 2 m m (No. 6)
    {/*rotmatrix num=*/4, /*transmatrix num=*/1, /*rotmatrix =*/1,0,0,0,0,1,0,0,0,0,0,0,	-1,0,0,0,0,-1,0,0,0,0,0,0,	-1,0,0,0,0,1,0,0,0,0,0,0,	1,0,0,0,0,-1,0,0,0,0,0,0,	/*transmatrix =*/0,0,0,   },
    //Wyckoff Positions of Plane Group p 2 m g (No. 7)
    {/*rotmatrix num=*/4, /*transmatrix num=*/1, /*rotmatrix =*/1,0,0,0,0,1,0,0,0,0,0,0,	-1,0,0,0,0,-1,0,0,0,0,0,0,	-1,0,0,0.5,0,1,0,0,0,0,0,0,	1,0,0,0.5,0,-1,0,0,0,0,0,0,	/*transmatrix =*/0,0,0,   },
    //Wyckoff Positions of Plane Group p 2 g g (No. 8)
    {/*rotmatrix num=*/4, /*transmatrix num=*/1, /*rotmatrix =*/1,0,0,0,0,1,0,0,0,0,0,0,	-1,0,0,0,0,-1,0,0,0,0,0,0,	-1,0,0,0.5,0,1,0,0.5,0,0,0,0,	1,0,0,0.5,0,-1,0,0.5,0,0,0,0,	/*transmatrix =*/0,0,0,   },
    //Wyckoff Positions of Plane Group c 2 m m (No. 9)
    {/*rotmatrix num=*/4, /*transmatrix num=*/2, /*rotmatrix =*/1,0,0,0,0,1,0,0,0,0,0,0,	-1,0,0,0,0,-1,0,0,0,0,0,0,	-1,0,0,0,0,1,0,0,0,0,0,0,	1,0,0,0,0,-1,0,0,0,0,0,0,	/*transmatrix =*/0,0,0,   0.5,0.5,0,    },
    //Wyckoff Positions of Plane Group p 4 (No. 10)
    {/*rotmatrix num=*/4, /*transmatrix num=*/1, /*rotmatrix =*/1,0,0,0,0,1,0,0,0,0,0,0,	-1,0,0,0,0,-1,0,0,0,0,0,0,	0,-1,0,0,1,0,0,0,0,0,0,0,	0,1,0,0,-1,0,0,0,0,0,0,0,	/*transmatrix =*/0,0,0,   },
    //Wyckoff Positions of Plane Group p 4 m m (No. 11)
    {/*rotmatrix num=*/8, /*transmatrix num=*/1, /*rotmatrix =*/1,0,0,0,0,1,0,0,0,0,0,0,	-1,0,0,0,0,-1,0,0,0,0,0,0,	0,-1,0,0,1,0,0,0,0,0,0,0,	0,1,0,0,-1,0,0,0,0,0,0,0,	-1,0,0,0,0,1,0,0,0,0,0,0,	1,0,0,0,0,-1,0,0,0,0,0,0,	0,1,0,0,1,0,0,0,0,0,0,0,	0,-1,0,0,-1,0,0,0,0,0,0,0,	/*transmatrix =*/0,0,0,   },
    //Wyckoff Positions of Plane Group p 4 g m (No. 12)
    {/*rotmatrix num=*/8, /*transmatrix num=*/1, /*rotmatrix =*/1,0,0,0,0,1,0,0,0,0,0,0,	-1,0,0,0,0,-1,0,0,0,0,0,0,	0,-1,0,0,1,0,0,0,0,0,0,0,	0,1,0,0,-1,0,0,0,0,0,0,0,	-1,0,0,0.5,0,1,0,0.5,0,0,0,0,	1,0,0,0.5,0,-1,0,0.5,0,0,0,0,	0,1,0,0.5,1,0,0,0.5,0,0,0,0,	0,-1,0,0.5,-1,0,0,0.5,0,0,0,0,	/*transmatrix =*/0,0,0,   },
    //Wyckoff Positions of Plane Group p 3 (No. 13)
    {/*rotmatrix num=*/3, /*transmatrix num=*/1, /*rotmatrix =*/1,0,0,0,0,1,0,0,0,0,0,0,	0,-1,0,0,1,-1,0,0,0,0,0,0,	-1,1,0,0,-1,0,0,0,0,0,0,0,	/*transmatrix =*/0,0,0,   },
    //Wyckoff Positions of Plane Group p 3 m 1 (No. 14)
    {/*rotmatrix num=*/6, /*transmatrix num=*/1, /*rotmatrix =*/1,0,0,0,0,1,0,0,0,0,0,0,	0,-1,0,0,1,-1,0,0,0,0,0,0,	-1,1,0,0,-1,0,0,0,0,0,0,0,	0,-1,0,0,-1,0,0,0,0,0,0,0,	-1,1,0,0,0,1,0,0,0,0,0,0,	1,0,0,0,1,-1,0,0,0,0,0,0,	/*transmatrix =*/0,0,0,   },
    //Wyckoff Positions of Plane Group p 3 1 m (No. 15)
    {/*rotmatrix num=*/6, /*transmatrix num=*/1, /*rotmatrix =*/1,0,0,0,0,1,0,0,0,0,0,0,	0,-1,0,0,1,-1,0,0,0,0,0,0,	-1,1,0,0,-1,0,0,0,0,0,0,0,	0,1,0,0,1,0,0,0,0,0,0,0,	1,-1,0,0,0,-1,0,0,0,0,0,0,	-1,0,0,0,-1,1,0,0,0,0,0,0,	/*transmatrix =*/0,0,0,   },
    //Wyckoff Positions of Plane Group p 6 (No. 16)
    {/*rotmatrix num=*/6, /*transmatrix num=*/1, /*rotmatrix =*/1,0,0,0,0,1,0,0,0,0,0,0,	0,-1,0,0,1,-1,0,0,0,0,0,0,	-1,1,0,0,-1,0,0,0,0,0,0,0,	-1,0,0,0,0,-1,0,0,0,0,0,0,	0,1,0,0,-1,1,0,0,0,0,0,0,	1,-1,0,0,1,0,0,0,0,0,0,0,	/*transmatrix =*/0,0,0,   },
    //Wyckoff Positions of Plane Group p 6 m m (No. 17)
    {/*rotmatrix num=*/12, /*transmatrix num=*/1, /*rotmatrix =*/1,0,0,0,0,1,0,0,0,0,0,0,	0,-1,0,0,1,-1,0,0,0,0,0,0,	-1,1,0,0,-1,0,0,0,0,0,0,0,	-1,0,0,0,0,-1,0,0,0,0,0,0,	0,1,0,0,-1,1,0,0,0,0,0,0,	1,-1,0,0,1,0,0,0,0,0,0,0,	0,-1,0,0,-1,0,0,0,0,0,0,0,	-1,1,0,0,0,1,0,0,0,0,0,0,	1,0,0,0,1,-1,0,0,0,0,0,0,	0,1,0,0,1,0,0,0,0,0,0,0,	1,-1,0,0,0,-1,0,0,0,0,0,0,	-1,0,0,0,-1,1,0,0,0,0,0,0,	/*transmatrix =*/0,0,0,   },
};
static const vector< vector<sitepos> > SitePosition_PL=
{
    //Wyckoff Positions of Plane Group 1
    {
        sitepos(1, 'a', "1", vector<int>( {} ) ),
    },
    //Wyckoff Positions of Plane Group 2
    {
        sitepos(2, 'e', "1", vector<int>( {} ) ),
        sitepos(1, 'd', "2", vector<int>( {/*Sym_2*/22, } ) ),
        sitepos(1, 'c', "2", vector<int>( {/*Sym_2*/22, } ) ),
        sitepos(1, 'b', "2", vector<int>( {/*Sym_2*/22, } ) ),
        sitepos(1, 'a', "2", vector<int>( {/*Sym_2*/22, } ) ),
    },
    //Wyckoff Positions of Plane Group 3
    {
        sitepos(2, 'c', "1", vector<int>( {} ) ),
        sitepos(1, 'b', ".m.", vector<int>( {/*Sym_m*/30, } ) ),
        sitepos(1, 'a', ".m.", vector<int>( {/*Sym_m*/30, } ) ),
    },
    //Wyckoff Positions of Plane Group 4
    {
        sitepos(2, 'a', "1", vector<int>( {} ) ),
    },
    //Wyckoff Positions of Plane Group 5
    {
        sitepos(4, 'b', "1", vector<int>( {} ) ),
        sitepos(2, 'a', ".m.", vector<int>( {/*Sym_m*/30, } ) ),
    },
    //Wyckoff Positions of Plane Group 6
    {
        sitepos(4, 'i', "1", vector<int>( {} ) ),
        sitepos(2, 'h', ".m.", vector<int>( {/*Sym_m*/30, } ) ),
        sitepos(2, 'g', ".m.", vector<int>( {/*Sym_m*/30, } ) ),
        sitepos(2, 'f', "..m", vector<int>( {/*Sym_m*/31, } ) ),
        sitepos(2, 'e', "..m", vector<int>( {/*Sym_m*/31, } ) ),
        sitepos(1, 'd', "2mm", vector<int>( {/*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, } ) ),
        sitepos(1, 'c', "2mm", vector<int>( {/*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, } ) ),
        sitepos(1, 'b', "2mm", vector<int>( {/*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, } ) ),
        sitepos(1, 'a', "2mm", vector<int>( {/*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, } ) ),
    },
    //Wyckoff Positions of Plane Group 7
    {
        sitepos(4, 'd', "1", vector<int>( {} ) ),
        sitepos(2, 'c', ".m.", vector<int>( {/*Sym_m*/30, } ) ),
        sitepos(2, 'b', "2..", vector<int>( {/*Sym_2*/22, } ) ),
        sitepos(2, 'a', "2..", vector<int>( {/*Sym_2*/22, } ) ),
    },
    //Wyckoff Positions of Plane Group 8
    {
        sitepos(4, 'c', "1", vector<int>( {} ) ),
        sitepos(2, 'b', "2..", vector<int>( {/*Sym_2*/22, } ) ),
        sitepos(2, 'a', "2..", vector<int>( {/*Sym_2*/22, } ) ),
    },
    //Wyckoff Positions of Plane Group 9
    {
        sitepos(8, 'f', "1", vector<int>( {} ) ),
        sitepos(4, 'e', ".m.", vector<int>( {/*Sym_m*/30, } ) ),
        sitepos(4, 'd', "..m", vector<int>( {/*Sym_m*/31, } ) ),
        sitepos(4, 'c', "2..", vector<int>( {/*Sym_2*/22, } ) ),
        sitepos(2, 'b', "2mm", vector<int>( {/*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, } ) ),
        sitepos(2, 'a', "2mm", vector<int>( {/*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, } ) ),
    },
    //Wyckoff Positions of Plane Group 10
    {
        sitepos(4, 'd', "1", vector<int>( {} ) ),
        sitepos(2, 'c', "2..", vector<int>( {/*Sym_2*/22, } ) ),
        sitepos(1, 'b', "4..", vector<int>( {/*Sym_4*/41, } ) ),
        sitepos(1, 'a', "4..", vector<int>( {/*Sym_4*/41, } ) ),
    },
    //Wyckoff Positions of Plane Group 11
    {
        sitepos(8, 'g', "1", vector<int>( {} ) ),
        sitepos(4, 'f', "..m", vector<int>( {/*Sym_m*/33, } ) ),
        sitepos(4, 'e', ".m.", vector<int>( {/*Sym_m*/31, } ) ),
        sitepos(4, 'd', ".m.", vector<int>( {/*Sym_m*/31, } ) ),
        sitepos(2, 'c', "2mm.", vector<int>( {/*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, } ) ),
        sitepos(1, 'b', "4mm", vector<int>( {/*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/33, /*Sym_m*/36, /*Sym_4*/41, } ) ),
        sitepos(1, 'a', "4mm", vector<int>( {/*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/33, /*Sym_m*/36, /*Sym_4*/41, } ) ),
    },
    //Wyckoff Positions of Plane Group 12
    {
        sitepos(8, 'd', "1", vector<int>( {} ) ),
        sitepos(4, 'c', "..m", vector<int>( {/*Sym_m*/33, } ) ),
        sitepos(2, 'b', "2.mm", vector<int>( {/*Sym_2*/22, /*Sym_m*/33, /*Sym_m*/36, } ) ),
        sitepos(2, 'a', "4..", vector<int>( {/*Sym_4*/41, } ) ),
    },
    //Wyckoff Positions of Plane Group 13
    {
        sitepos(3, 'd', "1", vector<int>( {} ) ),
        sitepos(1, 'c', "3..", vector<int>( {/*Sym_3*/61, } ) ),
        sitepos(1, 'b', "3..", vector<int>( {/*Sym_3*/61, } ) ),
        sitepos(1, 'a', "3..", vector<int>( {/*Sym_3*/61, } ) ),
    },
    //Wyckoff Positions of Plane Group 14
    {
        sitepos(6, 'e', "1", vector<int>( {} ) ),
        sitepos(3, 'd', ".m.", vector<int>( {/*Sym_m*/315, } ) ),
        sitepos(1, 'c', "3m.", vector<int>( {/*Sym_m*/310, /*Sym_m*/311, /*Sym_m*/315, /*Sym_3*/61, } ) ),
        sitepos(1, 'b', "3m.", vector<int>( {/*Sym_m*/310, /*Sym_m*/311, /*Sym_m*/315, /*Sym_3*/61, } ) ),
        sitepos(1, 'a', "3m.", vector<int>( {/*Sym_m*/310, /*Sym_m*/311, /*Sym_m*/315, /*Sym_3*/61, } ) ),
    },
    //Wyckoff Positions of Plane Group 15
    {
        sitepos(6, 'd', "1", vector<int>( {} ) ),
        sitepos(3, 'c', "..m", vector<int>( {/*Sym_m*/39, } ) ),
        sitepos(2, 'b', "3..", vector<int>( {/*Sym_3*/61, } ) ),
        sitepos(1, 'a', "3.m", vector<int>( {/*Sym_m*/39, /*Sym_m*/312, /*Sym_m*/314, /*Sym_3*/61, } ) ),
    },
    //Wyckoff Positions of Plane Group 16
    {
        sitepos(6, 'd', "1", vector<int>( {} ) ),
        sitepos(3, 'c', "2..", vector<int>( {/*Sym_2*/213, } ) ),
        sitepos(2, 'b', "3..", vector<int>( {/*Sym_3*/61, } ) ),
        sitepos(1, 'a', "6..", vector<int>( {/*Sym_6*/81, } ) ),
    },
    //Wyckoff Positions of Plane Group 17
    {
        sitepos(12, 'f', "1", vector<int>( {} ) ),
        sitepos(6, 'e', ".m.", vector<int>( {/*Sym_m*/315, } ) ),
        sitepos(6, 'd', "..m", vector<int>( {/*Sym_m*/39, } ) ),
        sitepos(3, 'c', "2mm", vector<int>( {/*Sym_2*/213, /*Sym_m*/39, /*Sym_m*/310, } ) ),
        sitepos(2, 'b', "3m.", vector<int>( {/*Sym_m*/310, /*Sym_m*/311, /*Sym_m*/315, /*Sym_3*/61, } ) ),
        sitepos(1, 'a', "6mm", vector<int>( {/*Sym_m*/39, /*Sym_m*/310, /*Sym_m*/311, /*Sym_m*/312, /*Sym_m*/314, /*Sym_m*/315, /*Sym_6*/81, } ) ),
    },
};