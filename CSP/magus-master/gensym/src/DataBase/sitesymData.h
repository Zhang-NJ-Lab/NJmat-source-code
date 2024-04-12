/*sitesymData.h: Database of 
   symmetry matrix of site positions in a spacegroup*/
/*note: spg=178,179,180,181,182,194; 191, 192, 193 might be incorrect*/
/*note: label of 8A of spg47 was changed to char('z'+1) for convenience*/

/*modified: 178 b, 179 b, 180 j, 180 i, 181 j, 181 i, 182 h, 191 o, 191 m, 191 l, 192 k, 193 i, 194 k, 194 h*/

#pragma once
#include <tuple>
#include <cstdlib>
#include <vector>
#include "../position.h"
using namespace std;

typedef std::tuple<int, char, const char*, vector<int> > sitepos;
/*{ int multiplicity; char label; const char* symmetry; vector<int> Sym_matrix;},*/

static const vector< vector < position > > symaxis=
{
	{
	}, //sym[0]
	//vector< vector<double> > Sym_minus1 =
	{
		/*{-1, 0, 0,	0, -1, 0,	0, 0, -1} //Sym_minus1[0], */ 			position(),
	},//sym[1]
	//vector< vector<double> > Sym_2 =
	{
		/*{1, 0, 0,	0, -1, 0,	0, 0, -1}, //Sym_2[0], */ 						position(1.0, 0.0, 0.0), 
		/*{-1, 0, 0,	0, 1, 0,	0, 0, -1}, //Sym_2[1], */ 					position(0.0, 1.0, 0.0), 
		/*{-1, 0, 0,	0, -1, 0,	0, 0, 1}, //Sym_2[2], */ 					position(0.0, 0.0, 1.0), 

		/*{0, 1, 0,	1, 0, 0,	0, 0, -1}, //Sym_2[3], */ 						position(0.7071067811865476, 0.7071067811865476, 0.0), 
		/*{-1, 0, 0,	0, 0, 1,	0, 1, 0}, //Sym_2[4], */ 					position(0.0, 0.7071067811865475, 0.7071067811865475), 
		/*{0, 0, 1,	0, -1, 0,	1, 0, 0}, //Sym_2[5], */ 						position(0.7071067811865475, 0.0, 0.7071067811865475), 

		/*{0, -1, 0,	-1, 0, 0,	0, 0, -1}, //Sym_2[6], */ 					position(-0.7071067811865475, 0.7071067811865475, 0.0), 
		/*{-1, 0, 0,	0, 0, -1,	0, -1, 0}, //Sym_2[7], */ 					position(0.0, -0.7071067811865475, 0.7071067811865475), 
		/*{0, 0, -1,	0, -1, 0,	-1, 0, 0}, //Sym_2[8], */ 					position(-0.7071067811865475, 0.0, 0.7071067811865475), 
			
		/*{-1, 1, 0,	0, 1, 0,	0, 0, -1}, //Sym_2[9], */ 					position(0.0, 1.0, 0.0), 
		/*{1, -1, 0,	0, -1, 0,	0, 0, -1}, //Sym_2[10], */ 					position(1.0, 0.0, 0.0), 
		/*{1, 0, 0,	1, -1, 0,	0, 0, -1}, //Sym_2[11], */ 						position(0.8660254037844387, 0.5, 0.0), 
		/*{-1, 0, 0,	-1, 1, 0,	0, 0, -1}, //Sym_2[12], */ 					position(-0.5, 0.8660254037844387, 0.0), 

		/*{-1, 0, 0,	0, -1, 0,	0, 0, 1}, //Sym_2[13 |2 ], */ 					position(0.0, 0.0, 1.0), 
		/*{0, 1, 0,	1, 0, 0,	0, 0, -1}, //Sym_2[14 |3 ], */ 						position(0.5, 0.8660265528314685, 0.0), 
		/*{0, -1, 0,	-1, 0, 0,	0, 0, -1}, //Sym_2[15 |6 ], */ 					position(0.866024254729786, -0.5, 0.0), 

	},//sym[2]
		
	//vector< vector<double> > Sym_m =
	{
		/*{-1, 0, 0,	0, 1, 0,	0, 0, 1}, //Sym_m[0], */ 					position(1.0, 0.0, 0.0), 
		/*{1, 0, 0,	0, -1, 0,	0, 0, 1}, //Sym_m[1], */ 						position(0.0, 1.0, 0.0), 
		/*{1, 0, 0,	0, 1, 0,	0, 0, -1}, //Sym_m[2], */ 						position(0.0, 0.0, 1.0), 

		/*{0, 1, 0,	1, 0, 0,	0, 0, 1}, //Sym_m[3], */ 						position(-0.7071067811865475, 0.7071067811865475, 0.0), 
		/*{0, 0, 1,	0, 1, 0,	1, 0, 0}, //Sym_m[4], */ 						position(-0.7071067811865475, 0.0, 0.7071067811865475), 
		/*{1, 0, 0,	0, 0, 1,	0, 1, 0}, //Sym_m[5], */ 						position(0.0, -0.7071067811865475, 0.7071067811865475), 

		/*{0, -1, 0,	-1, 0, 0,	0, 0, 1}, //Sym_m[6], */ 					position(0.7071067811865476, 0.7071067811865476, 0.0), 
		/*{0, 0, -1,	0, 1, 0,	-1, 0, 0}, //Sym_m[7], */ 					position(0.7071067811865475, 0.0, 0.7071067811865475), 
		/*{1, 0, 0,	0, 0, -1,	0, -1, 0}, //Sym_m[8], */ 						position(0.0, 0.7071067811865475, 0.7071067811865475), 

		/*{1, -1, 0,	0, -1, 0,	0, 0, 1}, //Sym_m[9], */ 					position(0.0, 1.0, 0.0), 
		/*{-1, 1, 0,	0, 1, 0,	0, 0, 1}, //Sym_m[10], */ 					position(1.0, 0.0, 0.0), 
		/*{1, 0, 0,	1, -1, 0,	0, 0, 1}, //Sym_m[11], */ 						position(-0.5, 0.8660254037844387, 0.0), 
		/*{-1, 0, 0,	-1, 1, 0,	0, 0, 1} //Sym_m[12], */ 					position(0.8660254037844387, 0.5, 0.0), 
		
		/*{1, 0, 0,	0, 1, 0,	0, 0, -1}, //Sym_m[13 |2 ], */ 						position(0.0, 0.0, 1.0), 
		/*{0, 1, 0,	1, 0, 0,	0, 0, 1}, //Sym_m[14 |3 ], */ 						position(0.866024254729786, -0.5, 0.0), 
		/*{0, -1, 0,	-1, 0, 0,	0, 0, 1}, //Sym_m[15 |6 ], */ 					position(0.5, 0.8660265528314685, 0.0), 


	},//sym[3]
	//vector< vector<double> > Sym_4 =
	{
		/*{0, -1, 0,	1, 0, 0,	0, 0, 1},  //Sym_4[0], */ 					position(),
		/*{0, 1, 0,	-1, 0, 0,	0, 0, 1}, //Sym_4[1], */ 						position(0.0, 0.0, 1.0), 

		/*{0, 0, 1,	0, 1, 0,	-1, 0, 0}, //Sym_4[2], */ 						position(),
		/*{0, 0, -1,	0, 1, 0,	1, 0, 0}, //Sym_4[3], */ 					position(0.0, 1.0, 0.0), 

		/*{1, 0, 0,	0, 0, 1,	0, -1, 0}, //Sym_4[4], */ 						position(),
		/*{1, 0, 0,	0, 0, -1,	0, 1, 0} //Sym_4[5], */ 						position(1.0, 0.0, 0.0), 
	},//sym[4]
	//vector< vector<double> > Sym_minus4 =
	{
		/*{0, 1, 0,	-1, 0, 0,	0, 0, -1}, //Sym_minus4[0], */ 				position(),
		/*{0, -1, 0,	1, 0, 0,	0, 0, -1}, //Sym_minus4[1], */ 			position(0.0, 0.0, 1.0), 

		/*{0, 0, -1,	0, -1, 0,	1, 0, 0}, //Sym_minus4[2], */ 			position(),
		/*{0, 0, 1,	0, -1, 0,	-1, 0, 0}, //Sym_minus4[3], */ 				position(0.0, 1.0, 0.0), 

		/*{-1, 0, 0,	0, 0, -1,	0, 1, 0}, //Sym_minus4[4], */ 			position(),
		/*{-1, 0, 0,	0, 0, 1,	0, -1, 0}//Sym_minus4[5], */ 			position(1.0, 0.0, 0.0), 
	},//sym[5]
	/****************************************************x=i, y=-0.5i+0.866025j*/
	//vector< vector<double> > Sym_3 =
	{
		/*{0, -1, 0,	1, -1, 0,	0, 0, 1}, //Sym_3[0], */ 					position(),
		/*{-1, 1, 0,	-1, 0, 0,	0, 0, 1}, //Sym_3[1], */ 					position(0.0, 0.0, 1.0), 
	
		/*{0, 0, 1,	1, 0, 0,	0, 1, 0}, //Sym_3[2], */ 						position(),
		/*{0, 1, 0,	0, 0, 1,	1, 0, 0}, //Sym_3[3], */ 						position(0.5773502691896257, 0.5773502691896257, 0.5773502691896257), 

		/*{0, 0, 1,	-1, 0, 0,	0, -1, 0}, //Sym_3[4], */ 						position(),
		/*{0, -1, 0,	0, 0, -1,	1, 0, 0}, //Sym_3[5], */ 					position(0.5773502691896257, -0.5773502691896257, 0.5773502691896257), 

		/*{0, 0, -1,	-1, 0, 0,	0, 1, 0}, //Sym_3[6], */ 					position(),
		/*{0, -1, 0,	0, 0, 1,	-1, 0, 0}, //Sym_3[7], */ 					position(-0.5773502691896258, 0.5773502691896258, 0.5773502691896258), 

		/*{0, 0, -1,	1, 0, 0,	0, -1, 0}, //Sym_3[8], */ 					position(),
		/*{0, 1, 0,	0, 0, -1,	-1, 0, 0}, //Sym_3[9], */ 						position(0.5773502691896257, 0.5773502691896257, -0.5773502691896257), 
	},//sym[6]
	//vector< vector<double> > Sym_minus3 =
	{
		/*{0, 1, 0,	-1, 1, 0,	0, 0, -1}, //Sym_minus3[0], */ 				position(),
		/*{1, -1, 0,	1, 0, 0,	0, 0, -1}, //Sym_minus3[1], */ 			position(0.0, 0.0, 1.0), 

		/*{0, 0, -1,	-1, 0, 0,	0, -1, 0}, //Sym_minus3[2], */ 			position(),
		/*{0, -1, 0,	0, 0, -1,	-1, 0, 0}, //Sym_minus3[3], */ 			position(0.5773502691896257, 0.5773502691896257, 0.5773502691896257), 

		/*{0, 0, -1,	1, 0, 0,	0, 1, 0}, //Sym_minus3[4], */ 			position(),
		/*{0, 1, 0,	0, 0, 1,	-1, 0, 0}, //Sym_minus3[5], */ 				position(0.5773502691896257, -0.5773502691896257, 0.5773502691896257), 

		/*{0, 0, 1,	1, 0, 0,	0, -1, 0}, //Sym_minus3[6], */ 				position(),
		/*{0, 1, 0,	0, 0, -1,	1, 0, 0}, //Sym_minus3[7], */ 				position(-0.5773502691896258, 0.5773502691896258, 0.5773502691896258), 

		/*{0, 0, 1,	-1, 0, 0,	0, 1, 0}, //Sym_minus3[8], */ 				position(),
		/*{0, -1, 0,	0, 0, 1,	1, 0, 0}//Sym_minus3[9], */ 			position(0.5773502691896257, 0.5773502691896257, -0.5773502691896257), 
	},//sym[7]
	//vector< vector<double> > Sym_6 =
	{
		/*{0, 1, 0,	-1, 1, 0,	0, 0, 1}, //Sym_6[0], */ 						position(),
		/*{1, -1, 0,	1, 0, 0,	0, 0, 1}//Sym_6[1], */ 						position(0.0, 0.0, 1.0), 
	},//sym[8]
	//vector< vector<double> > Sym_minus6 =
	{
		/*{0, -1, 0,	1, -1, 0,	0, 0, -1}, //Sym_minus6[0], */ 			position(),
		/*{-1, 1, 0,	-1, 0, 0,	0, 0, -1}//Sym_minus6[1], */ 			position(0.0, 0.0, 1.0), 
	},//sym[9]
};

static const vector< vector<sitepos> > SitePosition=
{	
	//Wyckoff Positions of Group 1
	{
		sitepos(1, 'a', "1", vector<int>( {} ) ),
	},
	//Wyckoff Positions of Group 2
	{
		sitepos(2, 'i', "1", vector<int>( {} ) ),
		sitepos(1, 'h', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(1, 'g', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(1, 'f', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(1, 'e', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(1, 'd', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(1, 'c', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(1, 'b', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(1, 'a', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
	},
	//Wyckoff Positions of Group 3
	{
		sitepos(2, 'e', "1", vector<int>( {} ) ),
		sitepos(1, 'd', "2", vector<int>( {/*Sym_2*/21, } ) ),
		sitepos(1, 'c', "2", vector<int>( {/*Sym_2*/21, } ) ),
		sitepos(1, 'b', "2", vector<int>( {/*Sym_2*/21, } ) ),
		sitepos(1, 'a', "2", vector<int>( {/*Sym_2*/21, } ) ),
	},
	//Wyckoff Positions of Group 4
	{
		sitepos(2, 'a', "1", vector<int>( {} ) ),
	},
	//Wyckoff Positions of Group 5
	{
		sitepos(4, 'c', "1", vector<int>( {} ) ),
		sitepos(2, 'b', "2", vector<int>( {/*Sym_2*/21, } ) ),
		sitepos(2, 'a', "2", vector<int>( {/*Sym_2*/21, } ) ),
	},
	//Wyckoff Positions of Group 6
	{
		sitepos(2, 'c', "1", vector<int>( {} ) ),
		sitepos(1, 'b', "m", vector<int>( {/*Sym_m*/31, } ) ),
		sitepos(1, 'a', "m", vector<int>( {/*Sym_m*/31, } ) ),
	},
	//Wyckoff Positions of Group 7
	{
		sitepos(2, 'a', "1", vector<int>( {} ) ),
	},
	//Wyckoff Positions of Group 8
	{
		sitepos(4, 'b', "1", vector<int>( {} ) ),
		sitepos(2, 'a', "m", vector<int>( {/*Sym_m*/31, } ) ),
	},
	//Wyckoff Positions of Group 9
	{
		sitepos(4, 'a', "1", vector<int>( {} ) ),
	},
	//Wyckoff Positions of Group 10
	{
		sitepos(4, 'o', "1", vector<int>( {} ) ),
		sitepos(2, 'n', "m", vector<int>( {/*Sym_m*/31, } ) ),
		sitepos(2, 'm', "m", vector<int>( {/*Sym_m*/31, } ) ),
		sitepos(2, 'l', "2", vector<int>( {/*Sym_2*/21, } ) ),
		sitepos(2, 'k', "2", vector<int>( {/*Sym_2*/21, } ) ),
		sitepos(2, 'j', "2", vector<int>( {/*Sym_2*/21, } ) ),
		sitepos(2, 'i', "2", vector<int>( {/*Sym_2*/21, } ) ),
		sitepos(1, 'h', "2/m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/21, /*Sym_m*/31, } ) ),
		sitepos(1, 'g', "2/m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/21, /*Sym_m*/31, } ) ),
		sitepos(1, 'f', "2/m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/21, /*Sym_m*/31, } ) ),
		sitepos(1, 'e', "2/m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/21, /*Sym_m*/31, } ) ),
		sitepos(1, 'd', "2/m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/21, /*Sym_m*/31, } ) ),
		sitepos(1, 'c', "2/m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/21, /*Sym_m*/31, } ) ),
		sitepos(1, 'b', "2/m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/21, /*Sym_m*/31, } ) ),
		sitepos(1, 'a', "2/m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/21, /*Sym_m*/31, } ) ),
	},
	//Wyckoff Positions of Group 11
	{
		sitepos(4, 'f', "1", vector<int>( {} ) ),
		sitepos(2, 'e', "m", vector<int>( {/*Sym_m*/31, } ) ),
		sitepos(2, 'd', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(2, 'c', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(2, 'b', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(2, 'a', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
	},
	//Wyckoff Positions of Group 12
	{
		sitepos(8, 'j', "1", vector<int>( {} ) ),
		sitepos(4, 'i', "m", vector<int>( {/*Sym_m*/31, } ) ),
		sitepos(4, 'h', "2", vector<int>( {/*Sym_2*/21, } ) ),
		sitepos(4, 'g', "2", vector<int>( {/*Sym_2*/21, } ) ),
		sitepos(4, 'f', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(4, 'e', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(2, 'd', "2/m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/21, /*Sym_m*/31, } ) ),
		sitepos(2, 'c', "2/m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/21, /*Sym_m*/31, } ) ),
		sitepos(2, 'b', "2/m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/21, /*Sym_m*/31, } ) ),
		sitepos(2, 'a', "2/m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/21, /*Sym_m*/31, } ) ),
	},
	//Wyckoff Positions of Group 13
	{
		sitepos(4, 'g', "1", vector<int>( {} ) ),
		sitepos(2, 'f', "2", vector<int>( {/*Sym_2*/21, } ) ),
		sitepos(2, 'e', "2", vector<int>( {/*Sym_2*/21, } ) ),
		sitepos(2, 'd', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(2, 'c', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(2, 'b', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(2, 'a', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
	},
	//Wyckoff Positions of Group 14
	{
		sitepos(4, 'e', "1", vector<int>( {} ) ),
		sitepos(2, 'd', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(2, 'c', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(2, 'b', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(2, 'a', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
	},
	//Wyckoff Positions of Group 15
	{
		sitepos(8, 'f', "1", vector<int>( {} ) ),
		sitepos(4, 'e', "2", vector<int>( {/*Sym_2*/21, } ) ),
		sitepos(4, 'd', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(4, 'c', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(4, 'b', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(4, 'a', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
	},
	//Wyckoff Positions of Group 16
	{
		sitepos(4, 'u', "1", vector<int>( {} ) ),
		sitepos(2, 't', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(2, 's', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(2, 'r', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(2, 'q', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(2, 'p', ".2.", vector<int>( {/*Sym_2*/21, } ) ),
		sitepos(2, 'o', ".2.", vector<int>( {/*Sym_2*/21, } ) ),
		sitepos(2, 'n', ".2.", vector<int>( {/*Sym_2*/21, } ) ),
		sitepos(2, 'm', ".2.", vector<int>( {/*Sym_2*/21, } ) ),
		sitepos(2, 'l', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(2, 'k', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(2, 'j', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(2, 'i', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(1, 'h', "222", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(1, 'g', "222", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(1, 'f', "222", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(1, 'e', "222", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(1, 'd', "222", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(1, 'c', "222", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(1, 'b', "222", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(1, 'a', "222", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
	},
	//Wyckoff Positions of Group 17
	{
		sitepos(4, 'e', "1", vector<int>( {} ) ),
		sitepos(2, 'd', ".2.", vector<int>( {/*Sym_2*/21, } ) ),
		sitepos(2, 'c', ".2.", vector<int>( {/*Sym_2*/21, } ) ),
		sitepos(2, 'b', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(2, 'a', "2..", vector<int>( {/*Sym_2*/20, } ) ),
	},
	//Wyckoff Positions of Group 18
	{
		sitepos(4, 'c', "1", vector<int>( {} ) ),
		sitepos(2, 'b', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(2, 'a', "..2", vector<int>( {/*Sym_2*/22, } ) ),
	},
	//Wyckoff Positions of Group 19
	{
		sitepos(4, 'a', "1", vector<int>( {} ) ),
	},
	//Wyckoff Positions of Group 20
	{
		sitepos(8, 'c', "1", vector<int>( {} ) ),
		sitepos(4, 'b', ".2.", vector<int>( {/*Sym_2*/21, } ) ),
		sitepos(4, 'a', "2..", vector<int>( {/*Sym_2*/20, } ) ),
	},
	//Wyckoff Positions of Group 21
	{
		sitepos(8, 'l', "1", vector<int>( {} ) ),
		sitepos(4, 'k', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'j', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'i', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'h', ".2.", vector<int>( {/*Sym_2*/21, } ) ),
		sitepos(4, 'g', ".2.", vector<int>( {/*Sym_2*/21, } ) ),
		sitepos(4, 'f', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(4, 'e', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(2, 'd', "222", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(2, 'c', "222", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(2, 'b', "222", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(2, 'a', "222", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
	},
	//Wyckoff Positions of Group 22
	{
		sitepos(16, 'k', "1", vector<int>( {} ) ),
		sitepos(8, 'j', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(8, 'i', ".2.", vector<int>( {/*Sym_2*/21, } ) ),
		sitepos(8, 'h', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(8, 'g', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(8, 'f', ".2.", vector<int>( {/*Sym_2*/21, } ) ),
		sitepos(8, 'e', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(4, 'd', "222", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(4, 'c', "222", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(4, 'b', "222", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(4, 'a', "222", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
	},
	//Wyckoff Positions of Group 23
	{
		sitepos(8, 'k', "1", vector<int>( {} ) ),
		sitepos(4, 'j', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'i', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'h', ".2.", vector<int>( {/*Sym_2*/21, } ) ),
		sitepos(4, 'g', ".2.", vector<int>( {/*Sym_2*/21, } ) ),
		sitepos(4, 'f', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(4, 'e', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(2, 'd', "222", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(2, 'c', "222", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(2, 'b', "222", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(2, 'a', "222", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
	},
	//Wyckoff Positions of Group 24
	{
		sitepos(8, 'd', "1", vector<int>( {} ) ),
		sitepos(4, 'c', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'b', ".2.", vector<int>( {/*Sym_2*/21, } ) ),
		sitepos(4, 'a', "2..", vector<int>( {/*Sym_2*/20, } ) ),
	},
	//Wyckoff Positions of Group 25
	{
		sitepos(4, 'i', "1", vector<int>( {} ) ),
		sitepos(2, 'h', "m..", vector<int>( {/*Sym_m*/30, } ) ),
		sitepos(2, 'g', "m..", vector<int>( {/*Sym_m*/30, } ) ),
		sitepos(2, 'f', ".m.", vector<int>( {/*Sym_m*/31, } ) ),
		sitepos(2, 'e', ".m.", vector<int>( {/*Sym_m*/31, } ) ),
		sitepos(1, 'd', "mm2", vector<int>( {/*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, } ) ),
		sitepos(1, 'c', "mm2", vector<int>( {/*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, } ) ),
		sitepos(1, 'b', "mm2", vector<int>( {/*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, } ) ),
		sitepos(1, 'a', "mm2", vector<int>( {/*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, } ) ),
	},
	//Wyckoff Positions of Group 26
	{
		sitepos(4, 'c', "1", vector<int>( {} ) ),
		sitepos(2, 'b', "m..", vector<int>( {/*Sym_m*/30, } ) ),
		sitepos(2, 'a', "m..", vector<int>( {/*Sym_m*/30, } ) ),
	},
	//Wyckoff Positions of Group 27
	{
		sitepos(4, 'e', "1", vector<int>( {} ) ),
		sitepos(2, 'd', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(2, 'c', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(2, 'b', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(2, 'a', "..2", vector<int>( {/*Sym_2*/22, } ) ),
	},
	//Wyckoff Positions of Group 28
	{
		sitepos(4, 'd', "1", vector<int>( {} ) ),
		sitepos(2, 'c', "m..", vector<int>( {/*Sym_m*/30, } ) ),
		sitepos(2, 'b', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(2, 'a', "..2", vector<int>( {/*Sym_2*/22, } ) ),
	},
	//Wyckoff Positions of Group 29
	{
		sitepos(4, 'a', "1", vector<int>( {} ) ),
	},
	//Wyckoff Positions of Group 30
	{
		sitepos(4, 'c', "1", vector<int>( {} ) ),
		sitepos(2, 'b', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(2, 'a', "..2", vector<int>( {/*Sym_2*/22, } ) ),
	},
	//Wyckoff Positions of Group 31
	{
		sitepos(4, 'b', "1", vector<int>( {} ) ),
		sitepos(2, 'a', "m..", vector<int>( {/*Sym_m*/30, } ) ),
	},
	//Wyckoff Positions of Group 32
	{
		sitepos(4, 'c', "1", vector<int>( {} ) ),
		sitepos(2, 'b', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(2, 'a', "..2", vector<int>( {/*Sym_2*/22, } ) ),
	},
	//Wyckoff Positions of Group 33
	{
		sitepos(4, 'a', "1", vector<int>( {} ) ),
	},
	//Wyckoff Positions of Group 34
	{
		sitepos(4, 'c', "1", vector<int>( {} ) ),
		sitepos(2, 'b', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(2, 'a', "..2", vector<int>( {/*Sym_2*/22, } ) ),
	},
	//Wyckoff Positions of Group 35
	{
		sitepos(8, 'f', "1", vector<int>( {} ) ),
		sitepos(4, 'e', "m..", vector<int>( {/*Sym_m*/30, } ) ),
		sitepos(4, 'd', ".m.", vector<int>( {/*Sym_m*/31, } ) ),
		sitepos(4, 'c', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(2, 'b', "mm2", vector<int>( {/*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, } ) ),
		sitepos(2, 'a', "mm2", vector<int>( {/*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, } ) ),
	},
	//Wyckoff Positions of Group 36
	{
		sitepos(8, 'b', "1", vector<int>( {} ) ),
		sitepos(4, 'a', "m..", vector<int>( {/*Sym_m*/30, } ) ),
	},
	//Wyckoff Positions of Group 37
	{
		sitepos(8, 'd', "1", vector<int>( {} ) ),
		sitepos(4, 'c', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'b', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'a', "..2", vector<int>( {/*Sym_2*/22, } ) ),
	},
	//Wyckoff Positions of Group 38
	{
		sitepos(8, 'f', "1", vector<int>( {} ) ),
		sitepos(4, 'e', "m..", vector<int>( {/*Sym_m*/30, } ) ),
		sitepos(4, 'd', "m..", vector<int>( {/*Sym_m*/30, } ) ),
		sitepos(4, 'c', ".m.", vector<int>( {/*Sym_m*/31, } ) ),
		sitepos(2, 'b', "mm2", vector<int>( {/*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, } ) ),
		sitepos(2, 'a', "mm2", vector<int>( {/*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, } ) ),
	},
	//Wyckoff Positions of Group 39
	{
		sitepos(8, 'd', "1", vector<int>( {} ) ),
		sitepos(4, 'c', ".m.", vector<int>( {/*Sym_m*/31, } ) ),
		sitepos(4, 'b', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'a', "..2", vector<int>( {/*Sym_2*/22, } ) ),
	},
	//Wyckoff Positions of Group 40
	{
		sitepos(8, 'c', "1", vector<int>( {} ) ),
		sitepos(4, 'b', "m..", vector<int>( {/*Sym_m*/30, } ) ),
		sitepos(4, 'a', "..2", vector<int>( {/*Sym_2*/22, } ) ),
	},
	//Wyckoff Positions of Group 41
	{
		sitepos(8, 'b', "1", vector<int>( {} ) ),
		sitepos(4, 'a', "..2", vector<int>( {/*Sym_2*/22, } ) ),
	},
	//Wyckoff Positions of Group 42
	{
		sitepos(16, 'e', "1", vector<int>( {} ) ),
		sitepos(8, 'd', ".m.", vector<int>( {/*Sym_m*/31, } ) ),
		sitepos(8, 'c', "m..", vector<int>( {/*Sym_m*/30, } ) ),
		sitepos(8, 'b', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'a', "mm2", vector<int>( {/*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, } ) ),
	},
	//Wyckoff Positions of Group 43
	{
		sitepos(16, 'b', "1", vector<int>( {} ) ),
		sitepos(8, 'a', "..2", vector<int>( {/*Sym_2*/22, } ) ),
	},
	//Wyckoff Positions of Group 44
	{
		sitepos(8, 'e', "1", vector<int>( {} ) ),
		sitepos(4, 'd', "m..", vector<int>( {/*Sym_m*/30, } ) ),
		sitepos(4, 'c', ".m.", vector<int>( {/*Sym_m*/31, } ) ),
		sitepos(2, 'b', "mm2", vector<int>( {/*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, } ) ),
		sitepos(2, 'a', "mm2", vector<int>( {/*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, } ) ),
	},
	//Wyckoff Positions of Group 45
	{
		sitepos(8, 'c', "1", vector<int>( {} ) ),
		sitepos(4, 'b', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'a', "..2", vector<int>( {/*Sym_2*/22, } ) ),
	},
	//Wyckoff Positions of Group 46
	{
		sitepos(8, 'c', "1", vector<int>( {} ) ),
		sitepos(4, 'b', "m..", vector<int>( {/*Sym_m*/30, } ) ),
		sitepos(4, 'a', "..2", vector<int>( {/*Sym_2*/22, } ) ),
	},
	//Wyckoff Positions of Group 47
	{
		sitepos(8, '{', "1", vector<int>( {} ) ),
		sitepos(4, 'z', "..m", vector<int>( {/*Sym_m*/32, } ) ),
		sitepos(4, 'y', "..m", vector<int>( {/*Sym_m*/32, } ) ),
		sitepos(4, 'x', ".m.", vector<int>( {/*Sym_m*/31, } ) ),
		sitepos(4, 'w', ".m.", vector<int>( {/*Sym_m*/31, } ) ),
		sitepos(4, 'v', "m..", vector<int>( {/*Sym_m*/30, } ) ),
		sitepos(4, 'u', "m..", vector<int>( {/*Sym_m*/30, } ) ),
		sitepos(2, 't', "mm2", vector<int>( {/*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, } ) ),
		sitepos(2, 's', "mm2", vector<int>( {/*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, } ) ),
		sitepos(2, 'r', "mm2", vector<int>( {/*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, } ) ),
		sitepos(2, 'q', "mm2", vector<int>( {/*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, } ) ),
		sitepos(2, 'p', "m2m", vector<int>( {/*Sym_2*/21, /*Sym_m*/30, /*Sym_m*/32, } ) ),
		sitepos(2, 'o', "m2m", vector<int>( {/*Sym_2*/21, /*Sym_m*/30, /*Sym_m*/32, } ) ),
		sitepos(2, 'n', "m2m", vector<int>( {/*Sym_2*/21, /*Sym_m*/30, /*Sym_m*/32, } ) ),
		sitepos(2, 'm', "m2m", vector<int>( {/*Sym_2*/21, /*Sym_m*/30, /*Sym_m*/32, } ) ),
		sitepos(2, 'l', "2mm", vector<int>( {/*Sym_2*/20, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(2, 'k', "2mm", vector<int>( {/*Sym_2*/20, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(2, 'j', "2mm", vector<int>( {/*Sym_2*/20, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(2, 'i', "2mm", vector<int>( {/*Sym_2*/20, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(1, 'h', "mmm", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(1, 'g', "mmm", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(1, 'f', "mmm", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(1, 'e', "mmm", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(1, 'd', "mmm", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(1, 'c', "mmm", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(1, 'b', "mmm", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(1, 'a', "mmm", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/32, } ) ),
	},
	//Wyckoff Positions of Group 48
	{
		sitepos(8, 'm', "1", vector<int>( {} ) ),
		sitepos(4, 'l', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'k', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'j', ".2.", vector<int>( {/*Sym_2*/21, } ) ),
		sitepos(4, 'i', ".2.", vector<int>( {/*Sym_2*/21, } ) ),
		sitepos(4, 'h', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(4, 'g', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(4, 'f', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(4, 'e', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(2, 'd', "222", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(2, 'c', "222", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(2, 'b', "222", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(2, 'a', "222", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
	},
	//Wyckoff Positions of Group 49
	{
		sitepos(8, 'r', "1", vector<int>( {} ) ),
		sitepos(4, 'q', "..m", vector<int>( {/*Sym_m*/32, } ) ),
		sitepos(4, 'p', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'o', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'n', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'm', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'l', ".2.", vector<int>( {/*Sym_2*/21, } ) ),
		sitepos(4, 'k', ".2.", vector<int>( {/*Sym_2*/21, } ) ),
		sitepos(4, 'j', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(4, 'i', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(2, 'h', "222", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(2, 'g', "222", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(2, 'f', "222", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(2, 'e', "222", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(2, 'd', "..2/m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/22, /*Sym_m*/32, } ) ),
		sitepos(2, 'c', "..2/m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/22, /*Sym_m*/32, } ) ),
		sitepos(2, 'b', "..2/m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/22, /*Sym_m*/32, } ) ),
		sitepos(2, 'a', "..2/m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/22, /*Sym_m*/32, } ) ),
	},
	//Wyckoff Positions of Group 50
	{
		sitepos(8, 'm', "1", vector<int>( {} ) ),
		sitepos(4, 'l', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'k', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'j', ".2.", vector<int>( {/*Sym_2*/21, } ) ),
		sitepos(4, 'i', ".2.", vector<int>( {/*Sym_2*/21, } ) ),
		sitepos(4, 'h', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(4, 'g', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(4, 'f', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(4, 'e', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(2, 'd', "222", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(2, 'c', "222", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(2, 'b', "222", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(2, 'a', "222", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
	},
	//Wyckoff Positions of Group 51
	{
		sitepos(8, 'l', "1", vector<int>( {} ) ),
		sitepos(4, 'k', "m..", vector<int>( {/*Sym_m*/30, } ) ),
		sitepos(4, 'j', ".m.", vector<int>( {/*Sym_m*/31, } ) ),
		sitepos(4, 'i', ".m.", vector<int>( {/*Sym_m*/31, } ) ),
		sitepos(4, 'h', ".2.", vector<int>( {/*Sym_2*/21, } ) ),
		sitepos(4, 'g', ".2.", vector<int>( {/*Sym_2*/21, } ) ),
		sitepos(2, 'f', "mm2", vector<int>( {/*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, } ) ),
		sitepos(2, 'e', "mm2", vector<int>( {/*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, } ) ),
		sitepos(2, 'd', ".2/m.", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/21, /*Sym_m*/31, } ) ),
		sitepos(2, 'c', ".2/m.", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/21, /*Sym_m*/31, } ) ),
		sitepos(2, 'b', ".2/m.", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/21, /*Sym_m*/31, } ) ),
		sitepos(2, 'a', ".2/m.", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/21, /*Sym_m*/31, } ) ),
	},
	//Wyckoff Positions of Group 52
	{
		sitepos(8, 'e', "1", vector<int>( {} ) ),
		sitepos(4, 'd', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(4, 'c', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'b', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(4, 'a', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
	},
	//Wyckoff Positions of Group 53
	{
		sitepos(8, 'i', "1", vector<int>( {} ) ),
		sitepos(4, 'h', "m..", vector<int>( {/*Sym_m*/30, } ) ),
		sitepos(4, 'g', ".2.", vector<int>( {/*Sym_2*/21, } ) ),
		sitepos(4, 'f', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(4, 'e', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(2, 'd', "2/m..", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_m*/30, } ) ),
		sitepos(2, 'c', "2/m..", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_m*/30, } ) ),
		sitepos(2, 'b', "2/m..", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_m*/30, } ) ),
		sitepos(2, 'a', "2/m..", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_m*/30, } ) ),
	},
	//Wyckoff Positions of Group 54
	{
		sitepos(8, 'f', "1", vector<int>( {} ) ),
		sitepos(4, 'e', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'd', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'c', ".2.", vector<int>( {/*Sym_2*/21, } ) ),
		sitepos(4, 'b', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(4, 'a', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
	},
	//Wyckoff Positions of Group 55
	{
		sitepos(8, 'i', "1", vector<int>( {} ) ),
		sitepos(4, 'h', "..m", vector<int>( {/*Sym_m*/32, } ) ),
		sitepos(4, 'g', "..m", vector<int>( {/*Sym_m*/32, } ) ),
		sitepos(4, 'f', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'e', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(2, 'd', "..2/m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/22, /*Sym_m*/32, } ) ),
		sitepos(2, 'c', "..2/m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/22, /*Sym_m*/32, } ) ),
		sitepos(2, 'b', "..2/m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/22, /*Sym_m*/32, } ) ),
		sitepos(2, 'a', "..2/m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/22, /*Sym_m*/32, } ) ),
	},
	//Wyckoff Positions of Group 56
	{
		sitepos(8, 'e', "1", vector<int>( {} ) ),
		sitepos(4, 'd', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'c', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'b', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(4, 'a', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
	},
	//Wyckoff Positions of Group 57
	{
		sitepos(8, 'e', "1", vector<int>( {} ) ),
		sitepos(4, 'd', "..m", vector<int>( {/*Sym_m*/32, } ) ),
		sitepos(4, 'c', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(4, 'b', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(4, 'a', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
	},
	//Wyckoff Positions of Group 58
	{
		sitepos(8, 'h', "1", vector<int>( {} ) ),
		sitepos(4, 'g', "..m", vector<int>( {/*Sym_m*/32, } ) ),
		sitepos(4, 'f', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'e', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(2, 'd', "..2/m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/22, /*Sym_m*/32, } ) ),
		sitepos(2, 'c', "..2/m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/22, /*Sym_m*/32, } ) ),
		sitepos(2, 'b', "..2/m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/22, /*Sym_m*/32, } ) ),
		sitepos(2, 'a', "..2/m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/22, /*Sym_m*/32, } ) ),
	},
	//Wyckoff Positions of Group 59
	{
		sitepos(8, 'g', "1", vector<int>( {} ) ),
		sitepos(4, 'f', ".m.", vector<int>( {/*Sym_m*/31, } ) ),
		sitepos(4, 'e', "m..", vector<int>( {/*Sym_m*/30, } ) ),
		sitepos(4, 'd', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(4, 'c', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(2, 'b', "mm2", vector<int>( {/*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, } ) ),
		sitepos(2, 'a', "mm2", vector<int>( {/*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, } ) ),
	},
	//Wyckoff Positions of Group 60
	{
		sitepos(8, 'd', "1", vector<int>( {} ) ),
		sitepos(4, 'c', ".2.", vector<int>( {/*Sym_2*/21, } ) ),
		sitepos(4, 'b', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(4, 'a', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
	},
	//Wyckoff Positions of Group 61
	{
		sitepos(8, 'c', "1", vector<int>( {} ) ),
		sitepos(4, 'b', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(4, 'a', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
	},
	//Wyckoff Positions of Group 62
	{
		sitepos(8, 'd', "1", vector<int>( {} ) ),
		sitepos(4, 'c', ".m.", vector<int>( {/*Sym_m*/31, } ) ),
		sitepos(4, 'b', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(4, 'a', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
	},
	//Wyckoff Positions of Group 63
	{
		sitepos(16, 'h', "1", vector<int>( {} ) ),
		sitepos(8, 'g', "..m", vector<int>( {/*Sym_m*/32, } ) ),
		sitepos(8, 'f', "m..", vector<int>( {/*Sym_m*/30, } ) ),
		sitepos(8, 'e', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(8, 'd', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(4, 'c', "m2m", vector<int>( {/*Sym_2*/21, /*Sym_m*/30, /*Sym_m*/32, } ) ),
		sitepos(4, 'b', "2/m..", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_m*/30, } ) ),
		sitepos(4, 'a', "2/m..", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_m*/30, } ) ),
	},
	//Wyckoff Positions of Group 64
	{
		sitepos(16, 'g', "1", vector<int>( {} ) ),
		sitepos(8, 'f', "m..", vector<int>( {/*Sym_m*/30, } ) ),
		sitepos(8, 'e', ".2.", vector<int>( {/*Sym_2*/21, } ) ),
		sitepos(8, 'd', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(8, 'c', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(4, 'b', "2/m..", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_m*/30, } ) ),
		sitepos(4, 'a', "2/m..", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_m*/30, } ) ),
	},
	//Wyckoff Positions of Group 65
	{
		sitepos(16, 'r', "1", vector<int>( {} ) ),
		sitepos(8, 'q', "..m", vector<int>( {/*Sym_m*/32, } ) ),
		sitepos(8, 'p', "..m", vector<int>( {/*Sym_m*/32, } ) ),
		sitepos(8, 'o', ".m.", vector<int>( {/*Sym_m*/31, } ) ),
		sitepos(8, 'n', "m..", vector<int>( {/*Sym_m*/30, } ) ),
		sitepos(8, 'm', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'l', "mm2", vector<int>( {/*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, } ) ),
		sitepos(4, 'k', "mm2", vector<int>( {/*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, } ) ),
		sitepos(4, 'j', "m2m", vector<int>( {/*Sym_2*/21, /*Sym_m*/30, /*Sym_m*/32, } ) ),
		sitepos(4, 'i', "m2m", vector<int>( {/*Sym_2*/21, /*Sym_m*/30, /*Sym_m*/32, } ) ),
		sitepos(4, 'h', "2mm", vector<int>( {/*Sym_2*/20, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(4, 'g', "2mm", vector<int>( {/*Sym_2*/20, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(4, 'f', "..2/m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/22, /*Sym_m*/32, } ) ),
		sitepos(4, 'e', "..2/m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/22, /*Sym_m*/32, } ) ),
		sitepos(2, 'd', "mmm", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(2, 'c', "mmm", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(2, 'b', "mmm", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(2, 'a', "mmm", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/32, } ) ),
	},
	//Wyckoff Positions of Group 66
	{
		sitepos(16, 'm', "1", vector<int>( {} ) ),
		sitepos(8, 'l', "..m", vector<int>( {/*Sym_m*/32, } ) ),
		sitepos(8, 'k', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(8, 'j', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(8, 'i', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(8, 'h', ".2.", vector<int>( {/*Sym_2*/21, } ) ),
		sitepos(8, 'g', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(4, 'f', "..2/m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/22, /*Sym_m*/32, } ) ),
		sitepos(4, 'e', "..2/m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/22, /*Sym_m*/32, } ) ),
		sitepos(4, 'd', "..2/m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/22, /*Sym_m*/32, } ) ),
		sitepos(4, 'c', "..2/m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/22, /*Sym_m*/32, } ) ),
		sitepos(4, 'b', "222", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(4, 'a', "222", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
	},
	//Wyckoff Positions of Group 67
	{
		sitepos(16, 'o', "1", vector<int>( {} ) ),
		sitepos(8, 'n', ".m.", vector<int>( {/*Sym_m*/31, } ) ),
		sitepos(8, 'm', "m..", vector<int>( {/*Sym_m*/30, } ) ),
		sitepos(8, 'l', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(8, 'k', ".2.", vector<int>( {/*Sym_2*/21, } ) ),
		sitepos(8, 'j', ".2.", vector<int>( {/*Sym_2*/21, } ) ),
		sitepos(8, 'i', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(8, 'h', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(4, 'g', "mm2", vector<int>( {/*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, } ) ),
		sitepos(4, 'f', ".2/m.", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/21, /*Sym_m*/31, } ) ),
		sitepos(4, 'e', ".2/m.", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/21, /*Sym_m*/31, } ) ),
		sitepos(4, 'd', "2/m..", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_m*/30, } ) ),
		sitepos(4, 'c', "2/m..", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_m*/30, } ) ),
		sitepos(4, 'b', "222", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(4, 'a', "222", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
	},
	//Wyckoff Positions of Group 68
	{
		sitepos(16, 'i', "1", vector<int>( {} ) ),
		sitepos(8, 'h', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(8, 'g', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(8, 'f', ".2.", vector<int>( {/*Sym_2*/21, } ) ),
		sitepos(8, 'e', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(8, 'd', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(8, 'c', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(4, 'b', "222", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(4, 'a', "222", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
	},
	//Wyckoff Positions of Group 69
	{
		sitepos(32, 'p', "1", vector<int>( {} ) ),
		sitepos(16, 'o', "..m", vector<int>( {/*Sym_m*/32, } ) ),
		sitepos(16, 'n', ".m.", vector<int>( {/*Sym_m*/31, } ) ),
		sitepos(16, 'm', "m..", vector<int>( {/*Sym_m*/30, } ) ),
		sitepos(16, 'l', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(16, 'k', ".2.", vector<int>( {/*Sym_2*/21, } ) ),
		sitepos(16, 'j', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(8, 'i', "mm2", vector<int>( {/*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, } ) ),
		sitepos(8, 'h', "m2m", vector<int>( {/*Sym_2*/21, /*Sym_m*/30, /*Sym_m*/32, } ) ),
		sitepos(8, 'g', "2mm", vector<int>( {/*Sym_2*/20, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(8, 'f', "222", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(8, 'e', "..2/m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/22, /*Sym_m*/32, } ) ),
		sitepos(8, 'd', ".2/m.", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/21, /*Sym_m*/31, } ) ),
		sitepos(8, 'c', "2/m..", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_m*/30, } ) ),
		sitepos(4, 'b', "mmm", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(4, 'a', "mmm", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/32, } ) ),
	},
	//Wyckoff Positions of Group 70
	{
		sitepos(32, 'h', "1", vector<int>( {} ) ),
		sitepos(16, 'g', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(16, 'f', ".2.", vector<int>( {/*Sym_2*/21, } ) ),
		sitepos(16, 'e', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(16, 'd', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(16, 'c', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(8, 'b', "222", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(8, 'a', "222", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
	},
	//Wyckoff Positions of Group 71
	{
		sitepos(16, 'o', "1", vector<int>( {} ) ),
		sitepos(8, 'n', "..m", vector<int>( {/*Sym_m*/32, } ) ),
		sitepos(8, 'm', ".m.", vector<int>( {/*Sym_m*/31, } ) ),
		sitepos(8, 'l', "m..", vector<int>( {/*Sym_m*/30, } ) ),
		sitepos(8, 'k', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(4, 'j', "mm2", vector<int>( {/*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, } ) ),
		sitepos(4, 'i', "mm2", vector<int>( {/*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, } ) ),
		sitepos(4, 'h', "m2m", vector<int>( {/*Sym_2*/21, /*Sym_m*/30, /*Sym_m*/32, } ) ),
		sitepos(4, 'g', "m2m", vector<int>( {/*Sym_2*/21, /*Sym_m*/30, /*Sym_m*/32, } ) ),
		sitepos(4, 'f', "2mm", vector<int>( {/*Sym_2*/20, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(4, 'e', "2mm", vector<int>( {/*Sym_2*/20, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(2, 'd', "mmm", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(2, 'c', "mmm", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(2, 'b', "mmm", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(2, 'a', "mmm", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/32, } ) ),
	},
	//Wyckoff Positions of Group 72
	{
		sitepos(16, 'k', "1", vector<int>( {} ) ),
		sitepos(8, 'j', "..m", vector<int>( {/*Sym_m*/32, } ) ),
		sitepos(8, 'i', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(8, 'h', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(8, 'g', ".2.", vector<int>( {/*Sym_2*/21, } ) ),
		sitepos(8, 'f', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(8, 'e', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(4, 'd', "..2/m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/22, /*Sym_m*/32, } ) ),
		sitepos(4, 'c', "..2/m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/22, /*Sym_m*/32, } ) ),
		sitepos(4, 'b', "222", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(4, 'a', "222", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
	},
	//Wyckoff Positions of Group 73
	{
		sitepos(16, 'f', "1", vector<int>( {} ) ),
		sitepos(8, 'e', "..2", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(8, 'd', ".2.", vector<int>( {/*Sym_2*/21, } ) ),
		sitepos(8, 'c', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(8, 'b', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(8, 'a', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
	},
	//Wyckoff Positions of Group 74
	{
		sitepos(16, 'j', "1", vector<int>( {} ) ),
		sitepos(8, 'i', ".m.", vector<int>( {/*Sym_m*/31, } ) ),
		sitepos(8, 'h', "m..", vector<int>( {/*Sym_m*/30, } ) ),
		sitepos(8, 'g', ".2.", vector<int>( {/*Sym_2*/21, } ) ),
		sitepos(8, 'f', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(4, 'e', "mm2", vector<int>( {/*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, } ) ),
		sitepos(4, 'd', ".2/m.", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/21, /*Sym_m*/31, } ) ),
		sitepos(4, 'c', ".2/m.", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/21, /*Sym_m*/31, } ) ),
		sitepos(4, 'b', "2/m..", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_m*/30, } ) ),
		sitepos(4, 'a', "2/m..", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_m*/30, } ) ),
	},
	//Wyckoff Positions of Group 75
	{
		sitepos(4, 'd', "1", vector<int>( {} ) ),
		sitepos(2, 'c', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(1, 'b', "4..", vector<int>( {/*Sym_4*/41, } ) ),
		sitepos(1, 'a', "4..", vector<int>( {/*Sym_4*/41, } ) ),
	},
	//Wyckoff Positions of Group 76
	{
		sitepos(4, 'a', "1", vector<int>( {} ) ),
	},
	//Wyckoff Positions of Group 77
	{
		sitepos(4, 'd', "1", vector<int>( {} ) ),
		sitepos(2, 'c', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(2, 'b', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(2, 'a', "2..", vector<int>( {/*Sym_2*/22, } ) ),
	},
	//Wyckoff Positions of Group 78
	{
		sitepos(4, 'a', "1", vector<int>( {} ) ),
	},
	//Wyckoff Positions of Group 79
	{
		sitepos(8, 'c', "1", vector<int>( {} ) ),
		sitepos(4, 'b', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(2, 'a', "4..", vector<int>( {/*Sym_4*/41, } ) ),
	},
	//Wyckoff Positions of Group 80
	{
		sitepos(8, 'b', "1", vector<int>( {} ) ),
		sitepos(4, 'a', "2..", vector<int>( {/*Sym_2*/22, } ) ),
	},
	//Wyckoff Positions of Group 81
	{
		sitepos(4, 'h', "1", vector<int>( {} ) ),
		sitepos(2, 'g', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(2, 'f', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(2, 'e', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(1, 'd', "-4..", vector<int>( {/*Sym_minus4*/51, } ) ),
		sitepos(1, 'c', "-4..", vector<int>( {/*Sym_minus4*/51, } ) ),
		sitepos(1, 'b', "-4..", vector<int>( {/*Sym_minus4*/51, } ) ),
		sitepos(1, 'a', "-4..", vector<int>( {/*Sym_minus4*/51, } ) ),
	},
	//Wyckoff Positions of Group 82
	{
		sitepos(8, 'g', "1", vector<int>( {} ) ),
		sitepos(4, 'f', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'e', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(2, 'd', "-4..", vector<int>( {/*Sym_minus4*/51, } ) ),
		sitepos(2, 'c', "-4..", vector<int>( {/*Sym_minus4*/51, } ) ),
		sitepos(2, 'b', "-4..", vector<int>( {/*Sym_minus4*/51, } ) ),
		sitepos(2, 'a', "-4..", vector<int>( {/*Sym_minus4*/51, } ) ),
	},
	//Wyckoff Positions of Group 83
	{
		sitepos(8, 'l', "1", vector<int>( {} ) ),
		sitepos(4, 'k', "m..", vector<int>( {/*Sym_m*/32, } ) ),
		sitepos(4, 'j', "m..", vector<int>( {/*Sym_m*/32, } ) ),
		sitepos(4, 'i', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(2, 'h', "4..", vector<int>( {/*Sym_4*/41, } ) ),
		sitepos(2, 'g', "4..", vector<int>( {/*Sym_4*/41, } ) ),
		sitepos(2, 'f', "2/m..", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/22, /*Sym_m*/32, } ) ),
		sitepos(2, 'e', "2/m..", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/22, /*Sym_m*/32, } ) ),
		sitepos(1, 'd', "4/m..", vector<int>( {/*Sym_minus1*/10, /*Sym_m*/32, /*Sym_4*/41, /*Sym_minus4*/51, } ) ),
		sitepos(1, 'c', "4/m..", vector<int>( {/*Sym_minus1*/10, /*Sym_m*/32, /*Sym_4*/41, /*Sym_minus4*/51, } ) ),
		sitepos(1, 'b', "4/m..", vector<int>( {/*Sym_minus1*/10, /*Sym_m*/32, /*Sym_4*/41, /*Sym_minus4*/51, } ) ),
		sitepos(1, 'a', "4/m..", vector<int>( {/*Sym_minus1*/10, /*Sym_m*/32, /*Sym_4*/41, /*Sym_minus4*/51, } ) ),
	},
	//Wyckoff Positions of Group 84
	{
		sitepos(8, 'k', "1", vector<int>( {} ) ),
		sitepos(4, 'j', "m..", vector<int>( {/*Sym_m*/32, } ) ),
		sitepos(4, 'i', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'h', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'g', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(2, 'f', "-4..", vector<int>( {/*Sym_minus4*/51, } ) ),
		sitepos(2, 'e', "-4..", vector<int>( {/*Sym_minus4*/51, } ) ),
		sitepos(2, 'd', "2/m..", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/22, /*Sym_m*/32, } ) ),
		sitepos(2, 'c', "2/m..", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/22, /*Sym_m*/32, } ) ),
		sitepos(2, 'b', "2/m..", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/22, /*Sym_m*/32, } ) ),
		sitepos(2, 'a', "2/m..", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/22, /*Sym_m*/32, } ) ),
	},
	//Wyckoff Positions of Group 85
	{
		sitepos(8, 'g', "1", vector<int>( {} ) ),
		sitepos(4, 'f', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'e', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(4, 'd', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(2, 'c', "4..", vector<int>( {/*Sym_4*/41, } ) ),
		sitepos(2, 'b', "-4..", vector<int>( {/*Sym_minus4*/51, } ) ),
		sitepos(2, 'a', "-4..", vector<int>( {/*Sym_minus4*/51, } ) ),
	},
	//Wyckoff Positions of Group 86
	{
		sitepos(8, 'g', "1", vector<int>( {} ) ),
		sitepos(4, 'f', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'e', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'd', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(4, 'c', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(2, 'b', "-4..", vector<int>( {/*Sym_minus4*/51, } ) ),
		sitepos(2, 'a', "-4..", vector<int>( {/*Sym_minus4*/51, } ) ),
	},
	//Wyckoff Positions of Group 87
	{
		sitepos(16, 'i', "1", vector<int>( {} ) ),
		sitepos(8, 'h', "m..", vector<int>( {/*Sym_m*/32, } ) ),
		sitepos(8, 'g', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(8, 'f', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(4, 'e', "4..", vector<int>( {/*Sym_4*/41, } ) ),
		sitepos(4, 'd', "-4..", vector<int>( {/*Sym_minus4*/51, } ) ),
		sitepos(4, 'c', "2/m..", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/22, /*Sym_m*/32, } ) ),
		sitepos(2, 'b', "4/m..", vector<int>( {/*Sym_minus1*/10, /*Sym_m*/32, /*Sym_4*/41, /*Sym_minus4*/51, } ) ),
		sitepos(2, 'a', "4/m..", vector<int>( {/*Sym_minus1*/10, /*Sym_m*/32, /*Sym_4*/41, /*Sym_minus4*/51, } ) ),
	},
	//Wyckoff Positions of Group 88
	{
		sitepos(16, 'f', "1", vector<int>( {} ) ),
		sitepos(8, 'e', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(8, 'd', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(8, 'c', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(4, 'b', "-4..", vector<int>( {/*Sym_minus4*/51, } ) ),
		sitepos(4, 'a', "-4..", vector<int>( {/*Sym_minus4*/51, } ) ),
	},
	//Wyckoff Positions of Group 89
	{
		sitepos(8, 'p', "1", vector<int>( {} ) ),
		sitepos(4, 'o', ".2.", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(4, 'n', ".2.", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(4, 'm', ".2.", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(4, 'l', ".2.", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(4, 'k', "..2", vector<int>( {/*Sym_2*/23, } ) ),
		sitepos(4, 'j', "..2", vector<int>( {/*Sym_2*/23, } ) ),
		sitepos(4, 'i', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(2, 'h', "4..", vector<int>( {/*Sym_4*/41, } ) ),
		sitepos(2, 'g', "4..", vector<int>( {/*Sym_4*/41, } ) ),
		sitepos(2, 'f', "222 .", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(2, 'e', "222 .", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(1, 'd', "422", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/23, /*Sym_2*/26, /*Sym_4*/41, } ) ),
		sitepos(1, 'c', "422", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/23, /*Sym_2*/26, /*Sym_4*/41, } ) ),
		sitepos(1, 'b', "422", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/23, /*Sym_2*/26, /*Sym_4*/41, } ) ),
		sitepos(1, 'a', "422", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/23, /*Sym_2*/26, /*Sym_4*/41, } ) ),
	},
	//Wyckoff Positions of Group 90
	{
		sitepos(8, 'g', "1", vector<int>( {} ) ),
		sitepos(4, 'f', "..2", vector<int>( {/*Sym_2*/23, } ) ),
		sitepos(4, 'e', "..2", vector<int>( {/*Sym_2*/23, } ) ),
		sitepos(4, 'd', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(2, 'c', "4..", vector<int>( {/*Sym_4*/41, } ) ),
		sitepos(2, 'b', "2.2 2", vector<int>( {/*Sym_2*/22, /*Sym_2*/23, /*Sym_2*/26, } ) ),
		sitepos(2, 'a', "2.2 2", vector<int>( {/*Sym_2*/22, /*Sym_2*/23, /*Sym_2*/26, } ) ),
	},
	//Wyckoff Positions of Group 91
	{
		sitepos(8, 'd', "1", vector<int>( {} ) ),
		sitepos(4, 'c', "..2", vector<int>( {/*Sym_2*/23, } ) ),
		sitepos(4, 'b', ".2.", vector<int>( {/*Sym_2*/21, } ) ),
		sitepos(4, 'a', ".2.", vector<int>( {/*Sym_2*/21, } ) ),
	},
	//Wyckoff Positions of Group 92
	{
		sitepos(8, 'b', "1", vector<int>( {} ) ),
		sitepos(4, 'a', "..2", vector<int>( {/*Sym_2*/23, } ) ),
	},
	//Wyckoff Positions of Group 93
	{
		sitepos(8, 'p', "1", vector<int>( {} ) ),
		sitepos(4, 'o', "..2", vector<int>( {/*Sym_2*/23, } ) ),
		sitepos(4, 'n', "..2", vector<int>( {/*Sym_2*/23, } ) ),
		sitepos(4, 'm', ".2.", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(4, 'l', ".2.", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(4, 'k', ".2.", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(4, 'j', ".2.", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(4, 'i', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'h', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'g', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(2, 'f', "2.2 2", vector<int>( {/*Sym_2*/22, /*Sym_2*/23, /*Sym_2*/26, } ) ),
		sitepos(2, 'e', "2.2 2", vector<int>( {/*Sym_2*/22, /*Sym_2*/23, /*Sym_2*/26, } ) ),
		sitepos(2, 'd', "222 .", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(2, 'c', "222 .", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(2, 'b', "222 .", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(2, 'a', "222 .", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
	},
	//Wyckoff Positions of Group 94
	{
		sitepos(8, 'g', "1", vector<int>( {} ) ),
		sitepos(4, 'f', "..2", vector<int>( {/*Sym_2*/23, } ) ),
		sitepos(4, 'e', "..2", vector<int>( {/*Sym_2*/23, } ) ),
		sitepos(4, 'd', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'c', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(2, 'b', "2.2 2", vector<int>( {/*Sym_2*/22, /*Sym_2*/23, /*Sym_2*/26, } ) ),
		sitepos(2, 'a', "2.2 2", vector<int>( {/*Sym_2*/22, /*Sym_2*/23, /*Sym_2*/26, } ) ),
	},
	//Wyckoff Positions of Group 95
	{
		sitepos(8, 'd', "1", vector<int>( {} ) ),
		sitepos(4, 'c', "..2", vector<int>( {/*Sym_2*/23, } ) ),
		sitepos(4, 'b', ".2.", vector<int>( {/*Sym_2*/21, } ) ),
		sitepos(4, 'a', ".2.", vector<int>( {/*Sym_2*/21, } ) ),
	},
	//Wyckoff Positions of Group 96
	{
		sitepos(8, 'b', "1", vector<int>( {} ) ),
		sitepos(4, 'a', "..2", vector<int>( {/*Sym_2*/23, } ) ),
	},
	//Wyckoff Positions of Group 97
	{
		sitepos(16, 'k', "1", vector<int>( {} ) ),
		sitepos(8, 'j', "..2", vector<int>( {/*Sym_2*/23, } ) ),
		sitepos(8, 'i', ".2.", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(8, 'h', ".2.", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(8, 'g', "..2", vector<int>( {/*Sym_2*/23, } ) ),
		sitepos(8, 'f', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'e', "4..", vector<int>( {/*Sym_4*/41, } ) ),
		sitepos(4, 'd', "2.2 2", vector<int>( {/*Sym_2*/22, /*Sym_2*/23, /*Sym_2*/26, } ) ),
		sitepos(4, 'c', "222 .", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(2, 'b', "422", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/23, /*Sym_2*/26, /*Sym_4*/41, } ) ),
		sitepos(2, 'a', "422", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/23, /*Sym_2*/26, /*Sym_4*/41, } ) ),
	},
	//Wyckoff Positions of Group 98
	{
		sitepos(16, 'g', "1", vector<int>( {} ) ),
		sitepos(8, 'f', ".2.", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(8, 'e', "..2", vector<int>( {/*Sym_2*/26, } ) ),
		sitepos(8, 'd', "..2", vector<int>( {/*Sym_2*/23, } ) ),
		sitepos(8, 'c', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'b', "2.2 2", vector<int>( {/*Sym_2*/22, /*Sym_2*/23, /*Sym_2*/26, } ) ),
		sitepos(4, 'a', "2.2 2", vector<int>( {/*Sym_2*/22, /*Sym_2*/23, /*Sym_2*/26, } ) ),
	},
	//Wyckoff Positions of Group 99
	{
		sitepos(8, 'g', "1", vector<int>( {} ) ),
		sitepos(4, 'f', ".m.", vector<int>( {/*Sym_m*/31, } ) ),
		sitepos(4, 'e', ".m.", vector<int>( {/*Sym_m*/31, } ) ),
		sitepos(4, 'd', "..m", vector<int>( {/*Sym_m*/33, } ) ),
		sitepos(2, 'c', "2mm .", vector<int>( {/*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, } ) ),
		sitepos(1, 'b', "4mm", vector<int>( {/*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/33, /*Sym_m*/36, /*Sym_4*/41, } ) ),
		sitepos(1, 'a', "4mm", vector<int>( {/*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/33, /*Sym_m*/36, /*Sym_4*/41, } ) ),
	},
	//Wyckoff Positions of Group 100
	{
		sitepos(8, 'd', "1", vector<int>( {} ) ),
		sitepos(4, 'c', "..m", vector<int>( {/*Sym_m*/33, } ) ),
		sitepos(2, 'b', "2.m m", vector<int>( {/*Sym_2*/22, /*Sym_m*/33, /*Sym_m*/36, } ) ),
		sitepos(2, 'a', "4..", vector<int>( {/*Sym_4*/41, } ) ),
	},
	//Wyckoff Positions of Group 101
	{
		sitepos(8, 'e', "1", vector<int>( {} ) ),
		sitepos(4, 'd', "..m", vector<int>( {/*Sym_m*/33, } ) ),
		sitepos(4, 'c', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(2, 'b', "2.m m", vector<int>( {/*Sym_2*/22, /*Sym_m*/33, /*Sym_m*/36, } ) ),
		sitepos(2, 'a', "2.m m", vector<int>( {/*Sym_2*/22, /*Sym_m*/33, /*Sym_m*/36, } ) ),
	},
	//Wyckoff Positions of Group 102
	{
		sitepos(8, 'd', "1", vector<int>( {} ) ),
		sitepos(4, 'c', "..m", vector<int>( {/*Sym_m*/33, } ) ),
		sitepos(4, 'b', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(2, 'a', "2.m m", vector<int>( {/*Sym_2*/22, /*Sym_m*/33, /*Sym_m*/36, } ) ),
	},
	//Wyckoff Positions of Group 103
	{
		sitepos(8, 'd', "1", vector<int>( {} ) ),
		sitepos(4, 'c', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(2, 'b', "4..", vector<int>( {/*Sym_4*/41, } ) ),
		sitepos(2, 'a', "4..", vector<int>( {/*Sym_4*/41, } ) ),
	},
	//Wyckoff Positions of Group 104
	{
		sitepos(8, 'c', "1", vector<int>( {} ) ),
		sitepos(4, 'b', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(2, 'a', "4..", vector<int>( {/*Sym_4*/41, } ) ),
	},
	//Wyckoff Positions of Group 105
	{
		sitepos(8, 'f', "1", vector<int>( {} ) ),
		sitepos(4, 'e', ".m.", vector<int>( {/*Sym_m*/31, } ) ),
		sitepos(4, 'd', ".m.", vector<int>( {/*Sym_m*/31, } ) ),
		sitepos(2, 'c', "2mm .", vector<int>( {/*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, } ) ),
		sitepos(2, 'b', "2mm .", vector<int>( {/*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, } ) ),
		sitepos(2, 'a', "2mm .", vector<int>( {/*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, } ) ),
	},
	//Wyckoff Positions of Group 106
	{
		sitepos(8, 'c', "1", vector<int>( {} ) ),
		sitepos(4, 'b', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'a', "2..", vector<int>( {/*Sym_2*/22, } ) ),
	},
	//Wyckoff Positions of Group 107
	{
		sitepos(16, 'e', "1", vector<int>( {} ) ),
		sitepos(8, 'd', ".m.", vector<int>( {/*Sym_m*/31, } ) ),
		sitepos(8, 'c', "..m", vector<int>( {/*Sym_m*/33, } ) ),
		sitepos(4, 'b', "2mm .", vector<int>( {/*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, } ) ),
		sitepos(2, 'a', "4mm", vector<int>( {/*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/33, /*Sym_m*/36, /*Sym_4*/41, } ) ),
	},
	//Wyckoff Positions of Group 108
	{
		sitepos(16, 'd', "1", vector<int>( {} ) ),
		sitepos(8, 'c', "..m", vector<int>( {/*Sym_m*/33, } ) ),
		sitepos(4, 'b', "2.m m", vector<int>( {/*Sym_2*/22, /*Sym_m*/33, /*Sym_m*/36, } ) ),
		sitepos(4, 'a', "4..", vector<int>( {/*Sym_4*/41, } ) ),
	},
	//Wyckoff Positions of Group 109
	{
		sitepos(16, 'c', "1", vector<int>( {} ) ),
		sitepos(8, 'b', ".m.", vector<int>( {/*Sym_m*/30, } ) ),
		sitepos(4, 'a', "2mm .", vector<int>( {/*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, } ) ),
	},
	//Wyckoff Positions of Group 110
	{
		sitepos(16, 'b', "1", vector<int>( {} ) ),
		sitepos(8, 'a', "2..", vector<int>( {/*Sym_2*/22, } ) ),
	},
	//Wyckoff Positions of Group 111
	{
		sitepos(8, 'o', "1", vector<int>( {} ) ),
		sitepos(4, 'n', "..m", vector<int>( {/*Sym_m*/33, } ) ),
		sitepos(4, 'm', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'l', ".2.", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(4, 'k', ".2.", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(4, 'j', ".2.", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(4, 'i', ".2.", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(2, 'h', "2.m m", vector<int>( {/*Sym_2*/22, /*Sym_m*/33, /*Sym_m*/36, } ) ),
		sitepos(2, 'g', "2.m m", vector<int>( {/*Sym_2*/22, /*Sym_m*/33, /*Sym_m*/36, } ) ),
		sitepos(2, 'f', "222 .", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(2, 'e', "222 .", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(1, 'd', "-42m", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_m*/33, /*Sym_m*/36, /*Sym_minus4*/51, } ) ),
		sitepos(1, 'c', "-42m", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_m*/33, /*Sym_m*/36, /*Sym_minus4*/51, } ) ),
		sitepos(1, 'b', "-42m", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_m*/33, /*Sym_m*/36, /*Sym_minus4*/51, } ) ),
		sitepos(1, 'a', "-42m", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_m*/33, /*Sym_m*/36, /*Sym_minus4*/51, } ) ),
	},
	//Wyckoff Positions of Group 112
	{
		sitepos(8, 'n', "1", vector<int>( {} ) ),
		sitepos(4, 'm', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'l', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'k', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'j', ".2.", vector<int>( {/*Sym_2*/21, } ) ),
		sitepos(4, 'i', ".2.", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(4, 'h', ".2.", vector<int>( {/*Sym_2*/21, } ) ),
		sitepos(4, 'g', ".2.", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(2, 'f', "-4..", vector<int>( {/*Sym_minus4*/51, } ) ),
		sitepos(2, 'e', "-4..", vector<int>( {/*Sym_minus4*/51, } ) ),
		sitepos(2, 'd', "222 .", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(2, 'c', "222 .", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(2, 'b', "222 .", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(2, 'a', "222 .", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
	},
	//Wyckoff Positions of Group 113
	{
		sitepos(8, 'f', "1", vector<int>( {} ) ),
		sitepos(4, 'e', "..m", vector<int>( {/*Sym_m*/33, } ) ),
		sitepos(4, 'd', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(2, 'c', "2.m m", vector<int>( {/*Sym_2*/22, /*Sym_m*/33, /*Sym_m*/36, } ) ),
		sitepos(2, 'b', "-4..", vector<int>( {/*Sym_minus4*/51, } ) ),
		sitepos(2, 'a', "-4..", vector<int>( {/*Sym_minus4*/51, } ) ),
	},
	//Wyckoff Positions of Group 114
	{
		sitepos(8, 'e', "1", vector<int>( {} ) ),
		sitepos(4, 'd', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'c', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(2, 'b', "-4..", vector<int>( {/*Sym_minus4*/51, } ) ),
		sitepos(2, 'a', "-4..", vector<int>( {/*Sym_minus4*/51, } ) ),
	},
	//Wyckoff Positions of Group 115
	{
		sitepos(8, 'l', "1", vector<int>( {} ) ),
		sitepos(4, 'k', ".m.", vector<int>( {/*Sym_m*/31, } ) ),
		sitepos(4, 'j', ".m.", vector<int>( {/*Sym_m*/31, } ) ),
		sitepos(4, 'i', "..2", vector<int>( {/*Sym_2*/23, } ) ),
		sitepos(4, 'h', "..2", vector<int>( {/*Sym_2*/23, } ) ),
		sitepos(2, 'g', "2mm .", vector<int>( {/*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, } ) ),
		sitepos(2, 'f', "2mm .", vector<int>( {/*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, } ) ),
		sitepos(2, 'e', "2mm .", vector<int>( {/*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, } ) ),
		sitepos(1, 'd', "-4m2", vector<int>( {/*Sym_2*/23, /*Sym_2*/26, /*Sym_m*/30, /*Sym_m*/31, /*Sym_minus4*/51, } ) ),
		sitepos(1, 'c', "-4m2", vector<int>( {/*Sym_2*/23, /*Sym_2*/26, /*Sym_m*/30, /*Sym_m*/31, /*Sym_minus4*/51, } ) ),
		sitepos(1, 'b', "-4m2", vector<int>( {/*Sym_2*/23, /*Sym_2*/26, /*Sym_m*/30, /*Sym_m*/31, /*Sym_minus4*/51, } ) ),
		sitepos(1, 'a', "-4m2", vector<int>( {/*Sym_2*/23, /*Sym_2*/26, /*Sym_m*/30, /*Sym_m*/31, /*Sym_minus4*/51, } ) ),
	},
	//Wyckoff Positions of Group 116
	{
		sitepos(8, 'j', "1", vector<int>( {} ) ),
		sitepos(4, 'i', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'h', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'g', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'f', "..2", vector<int>( {/*Sym_2*/23, } ) ),
		sitepos(4, 'e', "..2", vector<int>( {/*Sym_2*/23, } ) ),
		sitepos(2, 'd', "-4..", vector<int>( {/*Sym_minus4*/51, } ) ),
		sitepos(2, 'c', "-4..", vector<int>( {/*Sym_minus4*/51, } ) ),
		sitepos(2, 'b', "2.2 2", vector<int>( {/*Sym_2*/22, /*Sym_2*/23, /*Sym_2*/26, } ) ),
		sitepos(2, 'a', "2.2 2", vector<int>( {/*Sym_2*/22, /*Sym_2*/23, /*Sym_2*/26, } ) ),
	},
	//Wyckoff Positions of Group 117
	{
		sitepos(8, 'i', "1", vector<int>( {} ) ),
		sitepos(4, 'h', "..2", vector<int>( {/*Sym_2*/23, } ) ),
		sitepos(4, 'g', "..2", vector<int>( {/*Sym_2*/23, } ) ),
		sitepos(4, 'f', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'e', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(2, 'd', "2.2 2", vector<int>( {/*Sym_2*/22, /*Sym_2*/23, /*Sym_2*/26, } ) ),
		sitepos(2, 'c', "2.2 2", vector<int>( {/*Sym_2*/22, /*Sym_2*/23, /*Sym_2*/26, } ) ),
		sitepos(2, 'b', "-4..", vector<int>( {/*Sym_minus4*/51, } ) ),
		sitepos(2, 'a', "-4..", vector<int>( {/*Sym_minus4*/51, } ) ),
	},
	//Wyckoff Positions of Group 118
	{
		sitepos(8, 'i', "1", vector<int>( {} ) ),
		sitepos(4, 'h', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'g', "..2", vector<int>( {/*Sym_2*/23, } ) ),
		sitepos(4, 'f', "..2", vector<int>( {/*Sym_2*/26, } ) ),
		sitepos(4, 'e', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(2, 'd', "2.2 2", vector<int>( {/*Sym_2*/22, /*Sym_2*/23, /*Sym_2*/26, } ) ),
		sitepos(2, 'c', "2.2 2", vector<int>( {/*Sym_2*/22, /*Sym_2*/23, /*Sym_2*/26, } ) ),
		sitepos(2, 'b', "-4..", vector<int>( {/*Sym_minus4*/51, } ) ),
		sitepos(2, 'a', "-4..", vector<int>( {/*Sym_minus4*/51, } ) ),
	},
	//Wyckoff Positions of Group 119
	{
		sitepos(16, 'j', "1", vector<int>( {} ) ),
		sitepos(8, 'i', ".m.", vector<int>( {/*Sym_m*/31, } ) ),
		sitepos(8, 'h', "..2", vector<int>( {/*Sym_2*/23, } ) ),
		sitepos(8, 'g', "..2", vector<int>( {/*Sym_2*/23, } ) ),
		sitepos(4, 'f', "2mm .", vector<int>( {/*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, } ) ),
		sitepos(4, 'e', "2mm .", vector<int>( {/*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, } ) ),
		sitepos(2, 'd', "-4m2", vector<int>( {/*Sym_2*/23, /*Sym_2*/26, /*Sym_m*/30, /*Sym_m*/31, /*Sym_minus4*/51, } ) ),
		sitepos(2, 'c', "-4m2", vector<int>( {/*Sym_2*/23, /*Sym_2*/26, /*Sym_m*/30, /*Sym_m*/31, /*Sym_minus4*/51, } ) ),
		sitepos(2, 'b', "-4m2", vector<int>( {/*Sym_2*/23, /*Sym_2*/26, /*Sym_m*/30, /*Sym_m*/31, /*Sym_minus4*/51, } ) ),
		sitepos(2, 'a', "-4m2", vector<int>( {/*Sym_2*/23, /*Sym_2*/26, /*Sym_m*/30, /*Sym_m*/31, /*Sym_minus4*/51, } ) ),
	},
	//Wyckoff Positions of Group 120
	{
		sitepos(16, 'i', "1", vector<int>( {} ) ),
		sitepos(8, 'h', "..2", vector<int>( {/*Sym_2*/23, } ) ),
		sitepos(8, 'g', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(8, 'f', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(8, 'e', "..2", vector<int>( {/*Sym_2*/23, } ) ),
		sitepos(4, 'd', "2.2 2", vector<int>( {/*Sym_2*/22, /*Sym_2*/23, /*Sym_2*/26, } ) ),
		sitepos(4, 'c', "-4..", vector<int>( {/*Sym_minus4*/51, } ) ),
		sitepos(4, 'b', "-4..", vector<int>( {/*Sym_minus4*/51, } ) ),
		sitepos(4, 'a', "2.2 2", vector<int>( {/*Sym_2*/22, /*Sym_2*/23, /*Sym_2*/26, } ) ),
	},
	//Wyckoff Positions of Group 121
	{
		sitepos(16, 'j', "1", vector<int>( {} ) ),
		sitepos(8, 'i', "..m", vector<int>( {/*Sym_m*/33, } ) ),
		sitepos(8, 'h', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(8, 'g', ".2.", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(8, 'f', ".2.", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(4, 'e', "2.m m", vector<int>( {/*Sym_2*/22, /*Sym_m*/33, /*Sym_m*/36, } ) ),
		sitepos(4, 'd', "-4..", vector<int>( {/*Sym_minus4*/51, } ) ),
		sitepos(4, 'c', "222 .", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(2, 'b', "-42m", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_m*/33, /*Sym_m*/36, /*Sym_minus4*/51, } ) ),
		sitepos(2, 'a', "-42m", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_m*/33, /*Sym_m*/36, /*Sym_minus4*/51, } ) ),
	},
	//Wyckoff Positions of Group 122
	{
		sitepos(16, 'e', "1", vector<int>( {} ) ),
		sitepos(8, 'd', ".2.", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(8, 'c', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'b', "-4..", vector<int>( {/*Sym_minus4*/51, } ) ),
		sitepos(4, 'a', "-4..", vector<int>( {/*Sym_minus4*/51, } ) ),
	},
	//Wyckoff Positions of Group 123
	{
		sitepos(16, 'u', "1", vector<int>( {} ) ),
		sitepos(8, 't', ".m.", vector<int>( {/*Sym_m*/31, } ) ),
		sitepos(8, 's', ".m.", vector<int>( {/*Sym_m*/31, } ) ),
		sitepos(8, 'r', "..m", vector<int>( {/*Sym_m*/33, } ) ),
		sitepos(8, 'q', "m..", vector<int>( {/*Sym_m*/32, } ) ),
		sitepos(8, 'p', "m..", vector<int>( {/*Sym_m*/32, } ) ),
		sitepos(4, 'o', "m2m .", vector<int>( {/*Sym_2*/20, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(4, 'n', "m2m .", vector<int>( {/*Sym_2*/20, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(4, 'm', "m2m .", vector<int>( {/*Sym_2*/20, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(4, 'l', "m2m .", vector<int>( {/*Sym_2*/20, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(4, 'k', "m.2 m", vector<int>( {/*Sym_2*/23, /*Sym_m*/32, /*Sym_m*/33, } ) ),
		sitepos(4, 'j', "m.2 m", vector<int>( {/*Sym_2*/23, /*Sym_m*/32, /*Sym_m*/33, } ) ),
		sitepos(4, 'i', "2mm .", vector<int>( {/*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, } ) ),
		sitepos(2, 'h', "4mm", vector<int>( {/*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/33, /*Sym_m*/36, /*Sym_4*/41, } ) ),
		sitepos(2, 'g', "4mm", vector<int>( {/*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/33, /*Sym_m*/36, /*Sym_4*/41, } ) ),
		sitepos(2, 'f', "mmm .", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(2, 'e', "mmm .", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(1, 'd', "4/mmm", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/23, /*Sym_2*/26, /*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/32, /*Sym_m*/33, /*Sym_m*/36, /*Sym_4*/41, /*Sym_minus4*/51, } ) ),
		sitepos(1, 'c', "4/mmm", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/23, /*Sym_2*/26, /*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/32, /*Sym_m*/33, /*Sym_m*/36, /*Sym_4*/41, /*Sym_minus4*/51, } ) ),
		sitepos(1, 'b', "4/mmm", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/23, /*Sym_2*/26, /*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/32, /*Sym_m*/33, /*Sym_m*/36, /*Sym_4*/41, /*Sym_minus4*/51, } ) ),
		sitepos(1, 'a', "4/mmm", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/23, /*Sym_2*/26, /*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/32, /*Sym_m*/33, /*Sym_m*/36, /*Sym_4*/41, /*Sym_minus4*/51, } ) ),
	},
	//Wyckoff Positions of Group 124
	{
		sitepos(16, 'n', "1", vector<int>( {} ) ),
		sitepos(8, 'm', "m..", vector<int>( {/*Sym_m*/32, } ) ),
		sitepos(8, 'l', ".2.", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(8, 'k', ".2.", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(8, 'j', "..2", vector<int>( {/*Sym_2*/23, } ) ),
		sitepos(8, 'i', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'h', "4..", vector<int>( {/*Sym_4*/41, } ) ),
		sitepos(4, 'g', "4..", vector<int>( {/*Sym_4*/41, } ) ),
		sitepos(4, 'f', "222 .", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(4, 'e', "2/m..", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/22, /*Sym_m*/32, } ) ),
		sitepos(2, 'd', "4/m..", vector<int>( {/*Sym_minus1*/10, /*Sym_m*/32, /*Sym_4*/41, /*Sym_minus4*/51, } ) ),
		sitepos(2, 'c', "422", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/23, /*Sym_2*/26, /*Sym_4*/41, } ) ),
		sitepos(2, 'b', "4/m..", vector<int>( {/*Sym_minus1*/10, /*Sym_m*/32, /*Sym_4*/41, /*Sym_minus4*/51, } ) ),
		sitepos(2, 'a', "422", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/23, /*Sym_2*/26, /*Sym_4*/41, } ) ),
	},
	//Wyckoff Positions of Group 125
	{
		sitepos(16, 'n', "1", vector<int>( {} ) ),
		sitepos(8, 'm', "..m", vector<int>( {/*Sym_m*/36, } ) ),
		sitepos(8, 'l', ".2.", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(8, 'k', ".2.", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(8, 'j', "..2", vector<int>( {/*Sym_2*/23, } ) ),
		sitepos(8, 'i', "..2", vector<int>( {/*Sym_2*/23, } ) ),
		sitepos(4, 'h', "2.m m", vector<int>( {/*Sym_2*/22, /*Sym_m*/33, /*Sym_m*/36, } ) ),
		sitepos(4, 'g', "4..", vector<int>( {/*Sym_4*/41, } ) ),
		sitepos(4, 'f', "..2/m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/23, /*Sym_m*/36, } ) ),
		sitepos(4, 'e', "..2/m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/23, /*Sym_m*/36, } ) ),
		sitepos(2, 'd', "-42m", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_m*/33, /*Sym_m*/36, /*Sym_minus4*/51, } ) ),
		sitepos(2, 'c', "-42m", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_m*/33, /*Sym_m*/36, /*Sym_minus4*/51, } ) ),
		sitepos(2, 'b', "422", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/23, /*Sym_2*/26, /*Sym_4*/41, } ) ),
		sitepos(2, 'a', "422", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/23, /*Sym_2*/26, /*Sym_4*/41, } ) ),
	},
	//Wyckoff Positions of Group 126
	{
		sitepos(16, 'k', "1", vector<int>( {} ) ),
		sitepos(8, 'j', ".2.", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(8, 'i', ".2.", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(8, 'h', "..2", vector<int>( {/*Sym_2*/23, } ) ),
		sitepos(8, 'g', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(8, 'f', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(4, 'e', "4..", vector<int>( {/*Sym_4*/41, } ) ),
		sitepos(4, 'd', "-4..", vector<int>( {/*Sym_minus4*/51, } ) ),
		sitepos(4, 'c', "222 .", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(2, 'b', "422", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/23, /*Sym_2*/26, /*Sym_4*/41, } ) ),
		sitepos(2, 'a', "422", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/23, /*Sym_2*/26, /*Sym_4*/41, } ) ),
	},
	//Wyckoff Positions of Group 127
	{
		sitepos(16, 'l', "1", vector<int>( {} ) ),
		sitepos(8, 'k', "..m", vector<int>( {/*Sym_m*/33, } ) ),
		sitepos(8, 'j', "m..", vector<int>( {/*Sym_m*/32, } ) ),
		sitepos(8, 'i', "m..", vector<int>( {/*Sym_m*/32, } ) ),
		sitepos(4, 'h', "m.2 m", vector<int>( {/*Sym_2*/23, /*Sym_m*/32, /*Sym_m*/33, } ) ),
		sitepos(4, 'g', "m.2 m", vector<int>( {/*Sym_2*/23, /*Sym_m*/32, /*Sym_m*/33, } ) ),
		sitepos(4, 'f', "2.m m", vector<int>( {/*Sym_2*/22, /*Sym_m*/33, /*Sym_m*/36, } ) ),
		sitepos(4, 'e', "4..", vector<int>( {/*Sym_4*/41, } ) ),
		sitepos(2, 'd', "m.m m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/22, /*Sym_2*/23, /*Sym_2*/26, /*Sym_m*/32, /*Sym_m*/33, /*Sym_m*/36, } ) ),
		sitepos(2, 'c', "m.m m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/22, /*Sym_2*/23, /*Sym_2*/26, /*Sym_m*/32, /*Sym_m*/33, /*Sym_m*/36, } ) ),
		sitepos(2, 'b', "4/m..", vector<int>( {/*Sym_minus1*/10, /*Sym_m*/32, /*Sym_4*/41, /*Sym_minus4*/51, } ) ),
		sitepos(2, 'a', "4/m..", vector<int>( {/*Sym_minus1*/10, /*Sym_m*/32, /*Sym_4*/41, /*Sym_minus4*/51, } ) ),
	},
	//Wyckoff Positions of Group 128
	{
		sitepos(16, 'i', "1", vector<int>( {} ) ),
		sitepos(8, 'h', "m..", vector<int>( {/*Sym_m*/32, } ) ),
		sitepos(8, 'g', "..2", vector<int>( {/*Sym_2*/23, } ) ),
		sitepos(8, 'f', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'e', "4..", vector<int>( {/*Sym_4*/41, } ) ),
		sitepos(4, 'd', "2.2 2", vector<int>( {/*Sym_2*/22, /*Sym_2*/23, /*Sym_2*/26, } ) ),
		sitepos(4, 'c', "2/m..", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/22, /*Sym_m*/32, } ) ),
		sitepos(2, 'b', "4/m..", vector<int>( {/*Sym_minus1*/10, /*Sym_m*/32, /*Sym_4*/41, /*Sym_minus4*/51, } ) ),
		sitepos(2, 'a', "4/m..", vector<int>( {/*Sym_minus1*/10, /*Sym_m*/32, /*Sym_4*/41, /*Sym_minus4*/51, } ) ),
	},
	//Wyckoff Positions of Group 129
	{
		sitepos(16, 'k', "1", vector<int>( {} ) ),
		sitepos(8, 'j', "..m", vector<int>( {/*Sym_m*/33, } ) ),
		sitepos(8, 'i', ".m.", vector<int>( {/*Sym_m*/30, } ) ),
		sitepos(8, 'h', "..2", vector<int>( {/*Sym_2*/26, } ) ),
		sitepos(8, 'g', "..2", vector<int>( {/*Sym_2*/26, } ) ),
		sitepos(4, 'f', "2mm .", vector<int>( {/*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, } ) ),
		sitepos(4, 'e', "..2/m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/26, /*Sym_m*/33, } ) ),
		sitepos(4, 'd', "..2/m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/26, /*Sym_m*/33, } ) ),
		sitepos(2, 'c', "4mm", vector<int>( {/*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/33, /*Sym_m*/36, /*Sym_4*/41, } ) ),
		sitepos(2, 'b', "-4m2", vector<int>( {/*Sym_2*/23, /*Sym_2*/26, /*Sym_m*/30, /*Sym_m*/31, /*Sym_minus4*/51, } ) ),
		sitepos(2, 'a', "-4m2", vector<int>( {/*Sym_2*/23, /*Sym_2*/26, /*Sym_m*/30, /*Sym_m*/31, /*Sym_minus4*/51, } ) ),
	},
	//Wyckoff Positions of Group 130
	{
		sitepos(16, 'g', "1", vector<int>( {} ) ),
		sitepos(8, 'f', "..2", vector<int>( {/*Sym_2*/26, } ) ),
		sitepos(8, 'e', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(8, 'd', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(4, 'c', "4..", vector<int>( {/*Sym_4*/41, } ) ),
		sitepos(4, 'b', "-4..", vector<int>( {/*Sym_minus4*/51, } ) ),
		sitepos(4, 'a', "2.2 2", vector<int>( {/*Sym_2*/22, /*Sym_2*/23, /*Sym_2*/26, } ) ),
	},
	//Wyckoff Positions of Group 131
	{
		sitepos(16, 'r', "1", vector<int>( {} ) ),
		sitepos(8, 'q', "m..", vector<int>( {/*Sym_m*/32, } ) ),
		sitepos(8, 'p', ".m.", vector<int>( {/*Sym_m*/30, } ) ),
		sitepos(8, 'o', ".m.", vector<int>( {/*Sym_m*/30, } ) ),
		sitepos(8, 'n', "..2", vector<int>( {/*Sym_2*/23, } ) ),
		sitepos(4, 'm', "m2m .", vector<int>( {/*Sym_2*/20, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(4, 'l', "m2m .", vector<int>( {/*Sym_2*/20, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(4, 'k', "m2m .", vector<int>( {/*Sym_2*/20, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(4, 'j', "m2m .", vector<int>( {/*Sym_2*/20, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(4, 'i', "2mm .", vector<int>( {/*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, } ) ),
		sitepos(4, 'h', "2mm .", vector<int>( {/*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, } ) ),
		sitepos(4, 'g', "2mm .", vector<int>( {/*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, } ) ),
		sitepos(2, 'f', "-4m2", vector<int>( {/*Sym_2*/23, /*Sym_2*/26, /*Sym_m*/30, /*Sym_m*/31, /*Sym_minus4*/51, } ) ),
		sitepos(2, 'e', "-4m2", vector<int>( {/*Sym_2*/23, /*Sym_2*/26, /*Sym_m*/30, /*Sym_m*/31, /*Sym_minus4*/51, } ) ),
		sitepos(2, 'd', "mmm .", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(2, 'c', "mmm .", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(2, 'b', "mmm .", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(2, 'a', "mmm .", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/32, } ) ),
	},
	//Wyckoff Positions of Group 132
	{
		sitepos(16, 'p', "1", vector<int>( {} ) ),
		sitepos(8, 'o', "..m", vector<int>( {/*Sym_m*/33, } ) ),
		sitepos(8, 'n', "m..", vector<int>( {/*Sym_m*/32, } ) ),
		sitepos(8, 'm', ".2.", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(8, 'l', ".2.", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(8, 'k', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'j', "m.2 m", vector<int>( {/*Sym_2*/23, /*Sym_m*/32, /*Sym_m*/33, } ) ),
		sitepos(4, 'i', "m.2 m", vector<int>( {/*Sym_2*/23, /*Sym_m*/32, /*Sym_m*/33, } ) ),
		sitepos(4, 'h', "2.m m", vector<int>( {/*Sym_2*/22, /*Sym_m*/33, /*Sym_m*/36, } ) ),
		sitepos(4, 'g', "2.m m", vector<int>( {/*Sym_2*/22, /*Sym_m*/33, /*Sym_m*/36, } ) ),
		sitepos(4, 'f', "2/m..", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/22, /*Sym_m*/32, } ) ),
		sitepos(4, 'e', "222 .", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(2, 'd', "-42m", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_m*/33, /*Sym_m*/36, /*Sym_minus4*/51, } ) ),
		sitepos(2, 'c', "m.m m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/22, /*Sym_2*/23, /*Sym_2*/26, /*Sym_m*/32, /*Sym_m*/33, /*Sym_m*/36, } ) ),
		sitepos(2, 'b', "-42m", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_m*/33, /*Sym_m*/36, /*Sym_minus4*/51, } ) ),
		sitepos(2, 'a', "m.m m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/22, /*Sym_2*/23, /*Sym_2*/26, /*Sym_m*/32, /*Sym_m*/33, /*Sym_m*/36, } ) ),
	},
	//Wyckoff Positions of Group 133
	{
		sitepos(16, 'k', "1", vector<int>( {} ) ),
		sitepos(8, 'j', "..2", vector<int>( {/*Sym_2*/23, } ) ),
		sitepos(8, 'i', ".2.", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(8, 'h', ".2.", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(8, 'g', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(8, 'f', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(8, 'e', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(4, 'd', "-4..", vector<int>( {/*Sym_minus4*/51, } ) ),
		sitepos(4, 'c', "2.2 2", vector<int>( {/*Sym_2*/22, /*Sym_2*/23, /*Sym_2*/26, } ) ),
		sitepos(4, 'b', "222 .", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(4, 'a', "222 .", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
	},
	//Wyckoff Positions of Group 134
	{
		sitepos(16, 'n', "1", vector<int>( {} ) ),
		sitepos(8, 'm', "..m", vector<int>( {/*Sym_m*/36, } ) ),
		sitepos(8, 'l', "..2", vector<int>( {/*Sym_2*/23, } ) ),
		sitepos(8, 'k', "..2", vector<int>( {/*Sym_2*/23, } ) ),
		sitepos(8, 'j', ".2.", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(8, 'i', ".2.", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(8, 'h', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'g', "2.m m", vector<int>( {/*Sym_2*/22, /*Sym_m*/33, /*Sym_m*/36, } ) ),
		sitepos(4, 'f', "..2/m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/23, /*Sym_m*/36, } ) ),
		sitepos(4, 'e', "..2/m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/23, /*Sym_m*/36, } ) ),
		sitepos(4, 'd', "2.2 2", vector<int>( {/*Sym_2*/22, /*Sym_2*/23, /*Sym_2*/26, } ) ),
		sitepos(4, 'c', "222 .", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(2, 'b', "-42m", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_m*/33, /*Sym_m*/36, /*Sym_minus4*/51, } ) ),
		sitepos(2, 'a', "-42m", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_m*/33, /*Sym_m*/36, /*Sym_minus4*/51, } ) ),
	},
	//Wyckoff Positions of Group 135
	{
		sitepos(16, 'i', "1", vector<int>( {} ) ),
		sitepos(8, 'h', "m..", vector<int>( {/*Sym_m*/32, } ) ),
		sitepos(8, 'g', "..2", vector<int>( {/*Sym_2*/23, } ) ),
		sitepos(8, 'f', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(8, 'e', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'd', "2.2 2", vector<int>( {/*Sym_2*/22, /*Sym_2*/23, /*Sym_2*/26, } ) ),
		sitepos(4, 'c', "2/m..", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/22, /*Sym_m*/32, } ) ),
		sitepos(4, 'b', "-4..", vector<int>( {/*Sym_minus4*/51, } ) ),
		sitepos(4, 'a', "2/m..", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/22, /*Sym_m*/32, } ) ),
	},
	//Wyckoff Positions of Group 136
	{
		sitepos(16, 'k', "1", vector<int>( {} ) ),
		sitepos(8, 'j', "..m", vector<int>( {/*Sym_m*/33, } ) ),
		sitepos(8, 'i', "m..", vector<int>( {/*Sym_m*/32, } ) ),
		sitepos(8, 'h', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'g', "m.2 m", vector<int>( {/*Sym_2*/26, /*Sym_m*/32, /*Sym_m*/36, } ) ),
		sitepos(4, 'f', "m.2 m", vector<int>( {/*Sym_2*/23, /*Sym_m*/32, /*Sym_m*/33, } ) ),
		sitepos(4, 'e', "2.m m", vector<int>( {/*Sym_2*/22, /*Sym_m*/33, /*Sym_m*/36, } ) ),
		sitepos(4, 'd', "-4..", vector<int>( {/*Sym_minus4*/51, } ) ),
		sitepos(4, 'c', "2/m..", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/22, /*Sym_m*/32, } ) ),
		sitepos(2, 'b', "m.m m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/22, /*Sym_2*/23, /*Sym_2*/26, /*Sym_m*/32, /*Sym_m*/33, /*Sym_m*/36, } ) ),
		sitepos(2, 'a', "m.m m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/22, /*Sym_2*/23, /*Sym_2*/26, /*Sym_m*/32, /*Sym_m*/33, /*Sym_m*/36, } ) ),
	},
	//Wyckoff Positions of Group 137
	{
		sitepos(16, 'h', "1", vector<int>( {} ) ),
		sitepos(8, 'g', ".m.", vector<int>( {/*Sym_m*/30, } ) ),
		sitepos(8, 'f', "..2", vector<int>( {/*Sym_2*/26, } ) ),
		sitepos(8, 'e', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(4, 'd', "2mm .", vector<int>( {/*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, } ) ),
		sitepos(4, 'c', "2mm .", vector<int>( {/*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, } ) ),
		sitepos(2, 'b', "-4m2", vector<int>( {/*Sym_2*/23, /*Sym_2*/26, /*Sym_m*/30, /*Sym_m*/31, /*Sym_minus4*/51, } ) ),
		sitepos(2, 'a', "-4m2", vector<int>( {/*Sym_2*/23, /*Sym_2*/26, /*Sym_m*/30, /*Sym_m*/31, /*Sym_minus4*/51, } ) ),
	},
	//Wyckoff Positions of Group 138
	{
		sitepos(16, 'j', "1", vector<int>( {} ) ),
		sitepos(8, 'i', "..m", vector<int>( {/*Sym_m*/33, } ) ),
		sitepos(8, 'h', "..2", vector<int>( {/*Sym_2*/26, } ) ),
		sitepos(8, 'g', "..2", vector<int>( {/*Sym_2*/26, } ) ),
		sitepos(8, 'f', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(4, 'e', "2.m m", vector<int>( {/*Sym_2*/22, /*Sym_m*/33, /*Sym_m*/36, } ) ),
		sitepos(4, 'd', "..2/m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/26, /*Sym_m*/33, } ) ),
		sitepos(4, 'c', "..2/m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/26, /*Sym_m*/33, } ) ),
		sitepos(4, 'b', "-4..", vector<int>( {/*Sym_minus4*/51, } ) ),
		sitepos(4, 'a', "2.2 2", vector<int>( {/*Sym_2*/22, /*Sym_2*/23, /*Sym_2*/26, } ) ),
	},
	//Wyckoff Positions of Group 139
	{
		sitepos(32, 'o', "1", vector<int>( {} ) ),
		sitepos(16, 'n', ".m.", vector<int>( {/*Sym_m*/30, } ) ),
		sitepos(16, 'm', "..m", vector<int>( {/*Sym_m*/33, } ) ),
		sitepos(16, 'l', "m..", vector<int>( {/*Sym_m*/32, } ) ),
		sitepos(16, 'k', "..2", vector<int>( {/*Sym_2*/23, } ) ),
		sitepos(8, 'j', "m2m .", vector<int>( {/*Sym_2*/20, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(8, 'i', "m2m .", vector<int>( {/*Sym_2*/20, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(8, 'h', "m.2 m", vector<int>( {/*Sym_2*/23, /*Sym_m*/32, /*Sym_m*/33, } ) ),
		sitepos(8, 'g', "2mm .", vector<int>( {/*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, } ) ),
		sitepos(8, 'f', "..2/m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/26, /*Sym_m*/33, } ) ),
		sitepos(4, 'e', "4mm", vector<int>( {/*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/33, /*Sym_m*/36, /*Sym_4*/41, } ) ),
		sitepos(4, 'd', "-4m2", vector<int>( {/*Sym_2*/23, /*Sym_2*/26, /*Sym_m*/30, /*Sym_m*/31, /*Sym_minus4*/51, } ) ),
		sitepos(4, 'c', "mmm .", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(2, 'b', "4/mmm", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/23, /*Sym_2*/26, /*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/32, /*Sym_m*/33, /*Sym_m*/36, /*Sym_4*/41, /*Sym_minus4*/51, } ) ),
		sitepos(2, 'a', "4/mmm", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/23, /*Sym_2*/26, /*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/32, /*Sym_m*/33, /*Sym_m*/36, /*Sym_4*/41, /*Sym_minus4*/51, } ) ),
	},
	//Wyckoff Positions of Group 140
	{
		sitepos(32, 'm', "1", vector<int>( {} ) ),
		sitepos(16, 'l', "..m", vector<int>( {/*Sym_m*/33, } ) ),
		sitepos(16, 'k', "m..", vector<int>( {/*Sym_m*/32, } ) ),
		sitepos(16, 'j', ".2.", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(16, 'i', "..2", vector<int>( {/*Sym_2*/23, } ) ),
		sitepos(8, 'h', "m.2 m", vector<int>( {/*Sym_2*/23, /*Sym_m*/32, /*Sym_m*/33, } ) ),
		sitepos(8, 'g', "2.m m", vector<int>( {/*Sym_2*/22, /*Sym_m*/33, /*Sym_m*/36, } ) ),
		sitepos(8, 'f', "4..", vector<int>( {/*Sym_4*/41, } ) ),
		sitepos(8, 'e', "..2/m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/23, /*Sym_m*/36, } ) ),
		sitepos(4, 'd', "m.m m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/22, /*Sym_2*/23, /*Sym_2*/26, /*Sym_m*/32, /*Sym_m*/33, /*Sym_m*/36, } ) ),
		sitepos(4, 'c', "4/m..", vector<int>( {/*Sym_minus1*/10, /*Sym_m*/32, /*Sym_4*/41, /*Sym_minus4*/51, } ) ),
		sitepos(4, 'b', "-42m", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_m*/33, /*Sym_m*/36, /*Sym_minus4*/51, } ) ),
		sitepos(4, 'a', "422", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/23, /*Sym_2*/26, /*Sym_4*/41, } ) ),
	},
	//Wyckoff Positions of Group 141
	{
		sitepos(32, 'i', "1", vector<int>( {} ) ),
		sitepos(16, 'h', ".m.", vector<int>( {/*Sym_m*/30, } ) ),
		sitepos(16, 'g', "..2", vector<int>( {/*Sym_2*/23, } ) ),
		sitepos(16, 'f', ".2.", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(8, 'e', "2mm .", vector<int>( {/*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, } ) ),
		sitepos(8, 'd', ".2/m.", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_m*/30, } ) ),
		sitepos(8, 'c', ".2/m.", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_m*/30, } ) ),
		sitepos(4, 'b', "-4m2", vector<int>( {/*Sym_2*/23, /*Sym_2*/26, /*Sym_m*/30, /*Sym_m*/31, /*Sym_minus4*/51, } ) ),
		sitepos(4, 'a', "-4m2", vector<int>( {/*Sym_2*/23, /*Sym_2*/26, /*Sym_m*/30, /*Sym_m*/31, /*Sym_minus4*/51, } ) ),
	},
	//Wyckoff Positions of Group 142
	{
		sitepos(32, 'g', "1", vector<int>( {} ) ),
		sitepos(16, 'f', "..2", vector<int>( {/*Sym_2*/23, } ) ),
		sitepos(16, 'e', ".2.", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(16, 'd', "2..", vector<int>( {/*Sym_2*/22, } ) ),
		sitepos(16, 'c', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(8, 'b', "2.2 2", vector<int>( {/*Sym_2*/22, /*Sym_2*/23, /*Sym_2*/26, } ) ),
		sitepos(8, 'a', "-4..", vector<int>( {/*Sym_minus4*/51, } ) ),
	},
	//Wyckoff Positions of Group 143
	{
		sitepos(3, 'd', "1", vector<int>( {} ) ),
		sitepos(1, 'c', "3..", vector<int>( {/*Sym_3*/61, } ) ),
		sitepos(1, 'b', "3..", vector<int>( {/*Sym_3*/61, } ) ),
		sitepos(1, 'a', "3..", vector<int>( {/*Sym_3*/61, } ) ),
	},
	//Wyckoff Positions of Group 144
	{
		sitepos(3, 'a', "1", vector<int>( {} ) ),
	},
	//Wyckoff Positions of Group 145
	{
		sitepos(3, 'a', "1", vector<int>( {} ) ),
	},
	//Wyckoff Positions of Group 146
	{
		sitepos(9, 'b', "1", vector<int>( {} ) ),
		sitepos(3, 'a', "3.", vector<int>( {/*Sym_3*/61, } ) ),
	},
	//Wyckoff Positions of Group 147
	{
		sitepos(6, 'g', "1", vector<int>( {} ) ),
		sitepos(3, 'f', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(3, 'e', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(2, 'd', "3..", vector<int>( {/*Sym_3*/61, } ) ),
		sitepos(2, 'c', "3..", vector<int>( {/*Sym_3*/61, } ) ),
		sitepos(1, 'b', "-3..", vector<int>( {/*Sym_minus3*/71, } ) ),
		sitepos(1, 'a', "-3..", vector<int>( {/*Sym_minus3*/71, } ) ),
	},
	//Wyckoff Positions of Group 148
	{
		sitepos(18, 'f', "1", vector<int>( {} ) ),
		sitepos(9, 'e', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(9, 'd', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(6, 'c', "3.", vector<int>( {/*Sym_3*/61, } ) ),
		sitepos(3, 'b', "-3.", vector<int>( {/*Sym_minus3*/71, } ) ),
		sitepos(3, 'a', "-3.", vector<int>( {/*Sym_minus3*/71, } ) ),
	},
	//Wyckoff Positions of Group 149
	{
		sitepos(6, 'l', "1", vector<int>( {} ) ),
		sitepos(3, 'k', "..2", vector<int>( {/*Sym_2*/215, } ) ),
		sitepos(3, 'j', "..2", vector<int>( {/*Sym_2*/215, } ) ),
		sitepos(2, 'i', "3..", vector<int>( {/*Sym_3*/61, } ) ),
		sitepos(2, 'h', "3..", vector<int>( {/*Sym_3*/61, } ) ),
		sitepos(2, 'g', "3..", vector<int>( {/*Sym_3*/61, } ) ),
		sitepos(1, 'f', "3.2", vector<int>( {/*Sym_2*/29, /*Sym_2*/211, /*Sym_2*/215, /*Sym_3*/61, } ) ),
		sitepos(1, 'e', "3.2", vector<int>( {/*Sym_2*/29, /*Sym_2*/211, /*Sym_2*/215, /*Sym_3*/61, } ) ),
		sitepos(1, 'd', "3.2", vector<int>( {/*Sym_2*/29, /*Sym_2*/211, /*Sym_2*/215, /*Sym_3*/61, } ) ),
		sitepos(1, 'c', "3.2", vector<int>( {/*Sym_2*/29, /*Sym_2*/211, /*Sym_2*/215, /*Sym_3*/61, } ) ),
		sitepos(1, 'b', "3.2", vector<int>( {/*Sym_2*/29, /*Sym_2*/211, /*Sym_2*/215, /*Sym_3*/61, } ) ),
		sitepos(1, 'a', "3.2", vector<int>( {/*Sym_2*/29, /*Sym_2*/211, /*Sym_2*/215, /*Sym_3*/61, } ) ),
	},
	//Wyckoff Positions of Group 150
	{
		sitepos(6, 'g', "1", vector<int>( {} ) ),
		sitepos(3, 'f', ".2.", vector<int>( {/*Sym_2*/210, } ) ),
		sitepos(3, 'e', ".2.", vector<int>( {/*Sym_2*/210, } ) ),
		sitepos(2, 'd', "3..", vector<int>( {/*Sym_3*/61, } ) ),
		sitepos(2, 'c', "3..", vector<int>( {/*Sym_3*/61, } ) ),
		sitepos(1, 'b', "32.", vector<int>( {/*Sym_2*/210, /*Sym_2*/212, /*Sym_2*/214, /*Sym_3*/61, } ) ),
		sitepos(1, 'a', "32.", vector<int>( {/*Sym_2*/210, /*Sym_2*/212, /*Sym_2*/214, /*Sym_3*/61, } ) ),
	},
	//Wyckoff Positions of Group 151
	{
		sitepos(6, 'c', "1", vector<int>( {} ) ),
		sitepos(3, 'b', "..2", vector<int>( {/*Sym_2*/215, } ) ),
		sitepos(3, 'a', "..2", vector<int>( {/*Sym_2*/215, } ) ),
	},
	//Wyckoff Positions of Group 152
	{
		sitepos(6, 'c', "1", vector<int>( {} ) ),
		sitepos(3, 'b', ".2.", vector<int>( {/*Sym_2*/210, } ) ),
		sitepos(3, 'a', ".2.", vector<int>( {/*Sym_2*/210, } ) ),
	},
	//Wyckoff Positions of Group 153
	{
		sitepos(6, 'c', "1", vector<int>( {} ) ),
		sitepos(3, 'b', "..2", vector<int>( {/*Sym_2*/215, } ) ),
		sitepos(3, 'a', "..2", vector<int>( {/*Sym_2*/215, } ) ),
	},
	//Wyckoff Positions of Group 154
	{
		sitepos(6, 'c', "1", vector<int>( {} ) ),
		sitepos(3, 'b', ".2.", vector<int>( {/*Sym_2*/210, } ) ),
		sitepos(3, 'a', ".2.", vector<int>( {/*Sym_2*/210, } ) ),
	},
	//Wyckoff Positions of Group 155
	{
		sitepos(18, 'f', "1", vector<int>( {} ) ),
		sitepos(9, 'e', ".2", vector<int>( {/*Sym_2*/210, } ) ),
		sitepos(9, 'd', ".2", vector<int>( {/*Sym_2*/210, } ) ),
		sitepos(6, 'c', "3.", vector<int>( {/*Sym_3*/61, } ) ),
		sitepos(3, 'b', "32", vector<int>( {/*Sym_2*/210, /*Sym_2*/212, /*Sym_2*/214, /*Sym_3*/61, } ) ),
		sitepos(3, 'a', "32", vector<int>( {/*Sym_2*/210, /*Sym_2*/212, /*Sym_2*/214, /*Sym_3*/61, } ) ),
	},
	//Wyckoff Positions of Group 156
	{
		sitepos(6, 'e', "1", vector<int>( {} ) ),
		sitepos(3, 'd', ".m.", vector<int>( {/*Sym_m*/315, } ) ),
		sitepos(1, 'c', "3m.", vector<int>( {/*Sym_m*/310, /*Sym_m*/311, /*Sym_m*/315, /*Sym_3*/61, } ) ),
		sitepos(1, 'b', "3m.", vector<int>( {/*Sym_m*/310, /*Sym_m*/311, /*Sym_m*/315, /*Sym_3*/61, } ) ),
		sitepos(1, 'a', "3m.", vector<int>( {/*Sym_m*/310, /*Sym_m*/311, /*Sym_m*/315, /*Sym_3*/61, } ) ),
	},
	//Wyckoff Positions of Group 157
	{
		sitepos(6, 'd', "1", vector<int>( {} ) ),
		sitepos(3, 'c', "..m", vector<int>( {/*Sym_m*/39, } ) ),
		sitepos(2, 'b', "3..", vector<int>( {/*Sym_3*/61, } ) ),
		sitepos(1, 'a', "3.m", vector<int>( {/*Sym_m*/39, /*Sym_m*/312, /*Sym_m*/314, /*Sym_3*/61, } ) ),
	},
	//Wyckoff Positions of Group 158
	{
		sitepos(6, 'd', "1", vector<int>( {} ) ),
		sitepos(2, 'c', "3..", vector<int>( {/*Sym_3*/61, } ) ),
		sitepos(2, 'b', "3..", vector<int>( {/*Sym_3*/61, } ) ),
		sitepos(2, 'a', "3..", vector<int>( {/*Sym_3*/61, } ) ),
	},
	//Wyckoff Positions of Group 159
	{
		sitepos(6, 'c', "1", vector<int>( {} ) ),
		sitepos(2, 'b', "3..", vector<int>( {/*Sym_3*/61, } ) ),
		sitepos(2, 'a', "3..", vector<int>( {/*Sym_3*/61, } ) ),
	},
	//Wyckoff Positions of Group 160
	{
		sitepos(18, 'c', "1", vector<int>( {} ) ),
		sitepos(9, 'b', ".m", vector<int>( {/*Sym_m*/315, } ) ),
		sitepos(3, 'a', "3m", vector<int>( {/*Sym_m*/310, /*Sym_m*/311, /*Sym_m*/315, /*Sym_3*/61, } ) ),
	},
	//Wyckoff Positions of Group 161
	{
		sitepos(18, 'b', "1", vector<int>( {} ) ),
		sitepos(6, 'a', "3.", vector<int>( {/*Sym_3*/61, } ) ),
	},
	//Wyckoff Positions of Group 162
	{
		sitepos(12, 'l', "1", vector<int>( {} ) ),
		sitepos(6, 'k', "..m", vector<int>( {/*Sym_m*/39, } ) ),
		sitepos(6, 'j', "..2", vector<int>( {/*Sym_2*/215, } ) ),
		sitepos(6, 'i', "..2", vector<int>( {/*Sym_2*/215, } ) ),
		sitepos(4, 'h', "3..", vector<int>( {/*Sym_3*/61, } ) ),
		sitepos(3, 'g', "..2/m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/29, /*Sym_m*/39, } ) ),
		sitepos(3, 'f', "..2/m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/29, /*Sym_m*/39, } ) ),
		sitepos(2, 'e', "3.m", vector<int>( {/*Sym_m*/39, /*Sym_m*/312, /*Sym_m*/314, /*Sym_3*/61, } ) ),
		sitepos(2, 'd', "3.2", vector<int>( {/*Sym_2*/29, /*Sym_2*/211, /*Sym_2*/215, /*Sym_3*/61, } ) ),
		sitepos(2, 'c', "3.2", vector<int>( {/*Sym_2*/29, /*Sym_2*/211, /*Sym_2*/215, /*Sym_3*/61, } ) ),
		sitepos(1, 'b', "-3.m", vector<int>( {/*Sym_2*/29, /*Sym_2*/211, /*Sym_2*/215, /*Sym_m*/39, /*Sym_m*/312, /*Sym_m*/314, /*Sym_minus3*/71, } ) ),
		sitepos(1, 'a', "-3.m", vector<int>( {/*Sym_2*/29, /*Sym_2*/211, /*Sym_2*/215, /*Sym_m*/39, /*Sym_m*/312, /*Sym_m*/314, /*Sym_minus3*/71, } ) ),
	},
	//Wyckoff Positions of Group 163
	{
		sitepos(12, 'i', "1", vector<int>( {} ) ),
		sitepos(6, 'h', "..2", vector<int>( {/*Sym_2*/215, } ) ),
		sitepos(6, 'g', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(4, 'f', "3..", vector<int>( {/*Sym_3*/61, } ) ),
		sitepos(4, 'e', "3..", vector<int>( {/*Sym_3*/61, } ) ),
		sitepos(2, 'd', "3.2", vector<int>( {/*Sym_2*/29, /*Sym_2*/211, /*Sym_2*/215, /*Sym_3*/61, } ) ),
		sitepos(2, 'c', "3.2", vector<int>( {/*Sym_2*/29, /*Sym_2*/211, /*Sym_2*/215, /*Sym_3*/61, } ) ),
		sitepos(2, 'b', "-3..", vector<int>( {/*Sym_minus3*/71, } ) ),
		sitepos(2, 'a', "3.2", vector<int>( {/*Sym_2*/29, /*Sym_2*/211, /*Sym_2*/215, /*Sym_3*/61, } ) ),
	},
	//Wyckoff Positions of Group 164
	{
		sitepos(12, 'j', "1", vector<int>( {} ) ),
		sitepos(6, 'i', ".m.", vector<int>( {/*Sym_m*/315, } ) ),
		sitepos(6, 'h', ".2.", vector<int>( {/*Sym_2*/210, } ) ),
		sitepos(6, 'g', ".2.", vector<int>( {/*Sym_2*/210, } ) ),
		sitepos(3, 'f', ".2/m.", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/210, /*Sym_m*/310, } ) ),
		sitepos(3, 'e', ".2/m.", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/210, /*Sym_m*/310, } ) ),
		sitepos(2, 'd', "3m.", vector<int>( {/*Sym_m*/310, /*Sym_m*/311, /*Sym_m*/315, /*Sym_3*/61, } ) ),
		sitepos(2, 'c', "3m.", vector<int>( {/*Sym_m*/310, /*Sym_m*/311, /*Sym_m*/315, /*Sym_3*/61, } ) ),
		sitepos(1, 'b', "-3m.", vector<int>( {/*Sym_2*/210, /*Sym_2*/212, /*Sym_2*/214, /*Sym_m*/310, /*Sym_m*/311, /*Sym_m*/315, /*Sym_minus3*/71, } ) ),
		sitepos(1, 'a', "-3m.", vector<int>( {/*Sym_2*/210, /*Sym_2*/212, /*Sym_2*/214, /*Sym_m*/310, /*Sym_m*/311, /*Sym_m*/315, /*Sym_minus3*/71, } ) ),
	},
	//Wyckoff Positions of Group 165
	{
		sitepos(12, 'g', "1", vector<int>( {} ) ),
		sitepos(6, 'f', ".2.", vector<int>( {/*Sym_2*/210, } ) ),
		sitepos(6, 'e', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(4, 'd', "3..", vector<int>( {/*Sym_3*/61, } ) ),
		sitepos(4, 'c', "3..", vector<int>( {/*Sym_3*/61, } ) ),
		sitepos(2, 'b', "-3..", vector<int>( {/*Sym_minus3*/71, } ) ),
		sitepos(2, 'a', "32.", vector<int>( {/*Sym_2*/210, /*Sym_2*/212, /*Sym_2*/214, /*Sym_3*/61, } ) ),
	},
	//Wyckoff Positions of Group 166
	{
		sitepos(36, 'i', "1", vector<int>( {} ) ),
		sitepos(18, 'h', ".m", vector<int>( {/*Sym_m*/315, } ) ),
		sitepos(18, 'g', ".2", vector<int>( {/*Sym_2*/210, } ) ),
		sitepos(18, 'f', ".2", vector<int>( {/*Sym_2*/210, } ) ),
		sitepos(9, 'e', ".2/m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/210, /*Sym_m*/310, } ) ),
		sitepos(9, 'd', ".2/m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/210, /*Sym_m*/310, } ) ),
		sitepos(6, 'c', "3m", vector<int>( {/*Sym_m*/310, /*Sym_m*/311, /*Sym_m*/315, /*Sym_3*/61, } ) ),
		sitepos(3, 'b', "-3m", vector<int>( {/*Sym_2*/210, /*Sym_2*/212, /*Sym_2*/214, /*Sym_m*/310, /*Sym_m*/311, /*Sym_m*/315, /*Sym_minus3*/71, } ) ),
		sitepos(3, 'a', "-3m", vector<int>( {/*Sym_2*/210, /*Sym_2*/212, /*Sym_2*/214, /*Sym_m*/310, /*Sym_m*/311, /*Sym_m*/315, /*Sym_minus3*/71, } ) ),
	},
	//Wyckoff Positions of Group 167
	{
		sitepos(36, 'f', "1", vector<int>( {} ) ),
		sitepos(18, 'e', ".2", vector<int>( {/*Sym_2*/210, } ) ),
		sitepos(18, 'd', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(12, 'c', "3.", vector<int>( {/*Sym_3*/61, } ) ),
		sitepos(6, 'b', "-3.", vector<int>( {/*Sym_minus3*/71, } ) ),
		sitepos(6, 'a', "32", vector<int>( {/*Sym_2*/210, /*Sym_2*/212, /*Sym_2*/214, /*Sym_3*/61, } ) ),
	},
	//Wyckoff Positions of Group 168
	{
		sitepos(6, 'd', "1", vector<int>( {} ) ),
		sitepos(3, 'c', "2..", vector<int>( {/*Sym_2*/213, } ) ),
		sitepos(2, 'b', "3..", vector<int>( {/*Sym_3*/61, } ) ),
		sitepos(1, 'a', "6..", vector<int>( {/*Sym_6*/81, } ) ),
	},
	//Wyckoff Positions of Group 169
	{
		sitepos(6, 'a', "1", vector<int>( {} ) ),
	},
	//Wyckoff Positions of Group 170
	{
		sitepos(6, 'a', "1", vector<int>( {} ) ),
	},
	//Wyckoff Positions of Group 171
	{
		sitepos(6, 'c', "1", vector<int>( {} ) ),
		sitepos(3, 'b', "2..", vector<int>( {/*Sym_2*/213, } ) ),
		sitepos(3, 'a', "2..", vector<int>( {/*Sym_2*/213, } ) ),
	},
	//Wyckoff Positions of Group 172
	{
		sitepos(6, 'c', "1", vector<int>( {} ) ),
		sitepos(3, 'b', "2..", vector<int>( {/*Sym_2*/213, } ) ),
		sitepos(3, 'a', "2..", vector<int>( {/*Sym_2*/213, } ) ),
	},
	//Wyckoff Positions of Group 173
	{
		sitepos(6, 'c', "1", vector<int>( {} ) ),
		sitepos(2, 'b', "3..", vector<int>( {/*Sym_3*/61, } ) ),
		sitepos(2, 'a', "3..", vector<int>( {/*Sym_3*/61, } ) ),
	},
	//Wyckoff Positions of Group 174
	{
		sitepos(6, 'l', "1", vector<int>( {} ) ),
		sitepos(3, 'k', "m..", vector<int>( {/*Sym_m*/313, } ) ),
		sitepos(3, 'j', "m..", vector<int>( {/*Sym_m*/313, } ) ),
		sitepos(2, 'i', "3..", vector<int>( {/*Sym_3*/61, } ) ),
		sitepos(2, 'h', "3..", vector<int>( {/*Sym_3*/61, } ) ),
		sitepos(2, 'g', "3..", vector<int>( {/*Sym_3*/61, } ) ),
		sitepos(1, 'f', "-6..", vector<int>( {/*Sym_minus6*/91, } ) ),
		sitepos(1, 'e', "-6..", vector<int>( {/*Sym_minus6*/91, } ) ),
		sitepos(1, 'd', "-6..", vector<int>( {/*Sym_minus6*/91, } ) ),
		sitepos(1, 'c', "-6..", vector<int>( {/*Sym_minus6*/91, } ) ),
		sitepos(1, 'b', "-6..", vector<int>( {/*Sym_minus6*/91, } ) ),
		sitepos(1, 'a', "-6..", vector<int>( {/*Sym_minus6*/91, } ) ),
	},
	//Wyckoff Positions of Group 175
	{
		sitepos(12, 'l', "1", vector<int>( {} ) ),
		sitepos(6, 'k', "m..", vector<int>( {/*Sym_m*/313, } ) ),
		sitepos(6, 'j', "m..", vector<int>( {/*Sym_m*/313, } ) ),
		sitepos(6, 'i', "2..", vector<int>( {/*Sym_2*/213, } ) ),
		sitepos(4, 'h', "3..", vector<int>( {/*Sym_3*/61, } ) ),
		sitepos(3, 'g', "2/m..", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/213, /*Sym_m*/313, } ) ),
		sitepos(3, 'f', "2/m..", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/213, /*Sym_m*/313, } ) ),
		sitepos(2, 'e', "6..", vector<int>( {/*Sym_6*/81, } ) ),
		sitepos(2, 'd', "-6..", vector<int>( {/*Sym_minus6*/91, } ) ),
		sitepos(2, 'c', "-6..", vector<int>( {/*Sym_minus6*/91, } ) ),
		sitepos(1, 'b', "6/m..", vector<int>( {/*Sym_minus3*/71, /*Sym_6*/81, /*Sym_minus6*/91, } ) ),
		sitepos(1, 'a', "6/m..", vector<int>( {/*Sym_minus3*/71, /*Sym_6*/81, /*Sym_minus6*/91, } ) ),
	},
	//Wyckoff Positions of Group 176
	{
		sitepos(12, 'i', "1", vector<int>( {} ) ),
		sitepos(6, 'h', "m..", vector<int>( {/*Sym_m*/313, } ) ),
		sitepos(6, 'g', "-1", vector<int>( {/*Sym_minus1*/10, } ) ),
		sitepos(4, 'f', "3..", vector<int>( {/*Sym_3*/61, } ) ),
		sitepos(4, 'e', "3..", vector<int>( {/*Sym_3*/61, } ) ),
		sitepos(2, 'd', "-6..", vector<int>( {/*Sym_minus6*/91, } ) ),
		sitepos(2, 'c', "-6..", vector<int>( {/*Sym_minus6*/91, } ) ),
		sitepos(2, 'b', "-3..", vector<int>( {/*Sym_minus3*/71, } ) ),
		sitepos(2, 'a', "-6..", vector<int>( {/*Sym_minus6*/91, } ) ),
	},
	//Wyckoff Positions of Group 177
	{
		sitepos(12, 'n', "1", vector<int>( {} ) ),
		sitepos(6, 'm', "..2", vector<int>( {/*Sym_2*/215, } ) ),
		sitepos(6, 'l', "..2", vector<int>( {/*Sym_2*/215, } ) ),
		sitepos(6, 'k', ".2.", vector<int>( {/*Sym_2*/210, } ) ),
		sitepos(6, 'j', ".2.", vector<int>( {/*Sym_2*/210, } ) ),
		sitepos(6, 'i', "2..", vector<int>( {/*Sym_2*/213, } ) ),
		sitepos(4, 'h', "3..", vector<int>( {/*Sym_3*/61, } ) ),
		sitepos(3, 'g', "222", vector<int>( {/*Sym_2*/29, /*Sym_2*/210, /*Sym_2*/213, } ) ),
		sitepos(3, 'f', "222", vector<int>( {/*Sym_2*/29, /*Sym_2*/210, /*Sym_2*/213, } ) ),
		sitepos(2, 'e', "6..", vector<int>( {/*Sym_6*/81, } ) ),
		sitepos(2, 'd', "3.2", vector<int>( {/*Sym_2*/29, /*Sym_2*/211, /*Sym_2*/215, /*Sym_3*/61, } ) ),
		sitepos(2, 'c', "3.2", vector<int>( {/*Sym_2*/29, /*Sym_2*/211, /*Sym_2*/215, /*Sym_3*/61, } ) ),
		sitepos(1, 'b', "622", vector<int>( {/*Sym_2*/29, /*Sym_2*/210, /*Sym_2*/211, /*Sym_2*/212, /*Sym_2*/214, /*Sym_2*/215, /*Sym_6*/81, } ) ),
		sitepos(1, 'a', "622", vector<int>( {/*Sym_2*/29, /*Sym_2*/210, /*Sym_2*/211, /*Sym_2*/212, /*Sym_2*/214, /*Sym_2*/215, /*Sym_6*/81, } ) ),
	},
	//Wyckoff Positions of Group 178
	{
		sitepos(12, 'c', "1", vector<int>( {} ) ),
		sitepos(6, 'b', "..2", vector<int>( {/*Sym_2*/29,} ) ),
		sitepos(6, 'a', ".2.", vector<int>( {/*Sym_2*/210, } ) ),
	},
	//Wyckoff Positions of Group 179
	{
		sitepos(12, 'c', "1", vector<int>( {} ) ),
		sitepos(6, 'b', "..2", vector<int>( {/*Sym_2*/29,} ) ),
		sitepos(6, 'a', ".2.", vector<int>( {/*Sym_2*/210, } ) ),
	},
	//Wyckoff Positions of Group 180
	{
		sitepos(12, 'k', "1", vector<int>( {} ) ),
		sitepos(6, 'j', "..2", vector<int>( {/*Sym_2*/29,} ) ),
		sitepos(6, 'i', "..2", vector<int>( {/*Sym_2*/29,} ) ),
		sitepos(6, 'h', ".2.", vector<int>( {/*Sym_2*/210, } ) ),
		sitepos(6, 'g', ".2.", vector<int>( {/*Sym_2*/210, } ) ),
		sitepos(6, 'f', "2..", vector<int>( {/*Sym_2*/213, } ) ),
		sitepos(6, 'e', "2..", vector<int>( {/*Sym_2*/213, } ) ),
		sitepos(3, 'd', "222", vector<int>( {/*Sym_2*/29, /*Sym_2*/210, /*Sym_2*/213, } ) ),
		sitepos(3, 'c', "222", vector<int>( {/*Sym_2*/29, /*Sym_2*/210, /*Sym_2*/213, } ) ),
		sitepos(3, 'b', "222", vector<int>( {/*Sym_2*/29, /*Sym_2*/210, /*Sym_2*/213, } ) ),
		sitepos(3, 'a', "222", vector<int>( {/*Sym_2*/29, /*Sym_2*/210, /*Sym_2*/213, } ) ),
	},
	//Wyckoff Positions of Group 181
	{
		sitepos(12, 'k', "1", vector<int>( {} ) ),
		sitepos(6, 'j', "..2", vector<int>( {/*Sym_2*/29,} ) ),
		sitepos(6, 'i', "..2", vector<int>( {/*Sym_2*/29,} ) ),
		sitepos(6, 'h', ".2.", vector<int>( {/*Sym_2*/210, } ) ),
		sitepos(6, 'g', ".2.", vector<int>( {/*Sym_2*/210, } ) ),
		sitepos(6, 'f', "2..", vector<int>( {/*Sym_2*/213, } ) ),
		sitepos(6, 'e', "2..", vector<int>( {/*Sym_2*/213, } ) ),
		sitepos(3, 'd', "222", vector<int>( {/*Sym_2*/29, /*Sym_2*/210, /*Sym_2*/213, } ) ),
		sitepos(3, 'c', "222", vector<int>( {/*Sym_2*/29, /*Sym_2*/210, /*Sym_2*/213, } ) ),
		sitepos(3, 'b', "222", vector<int>( {/*Sym_2*/29, /*Sym_2*/210, /*Sym_2*/213, } ) ),
		sitepos(3, 'a', "222", vector<int>( {/*Sym_2*/29, /*Sym_2*/210, /*Sym_2*/213, } ) ),
	},
	//Wyckoff Positions of Group 182
	{
		sitepos(12, 'i', "1", vector<int>( {} ) ),
		sitepos(6, 'h', "..2", vector<int>( {/*Sym_2*/29,} ) ),
		sitepos(6, 'g', ".2.", vector<int>( {/*Sym_2*/210, } ) ),
		sitepos(4, 'f', "3..", vector<int>( {/*Sym_3*/61, } ) ),
		sitepos(4, 'e', "3..", vector<int>( {/*Sym_3*/61, } ) ),
		sitepos(2, 'd', "3.2", vector<int>( {/*Sym_2*/29, /*Sym_2*/211, /*Sym_2*/215, /*Sym_3*/61, } ) ),
		sitepos(2, 'c', "3.2", vector<int>( {/*Sym_2*/29, /*Sym_2*/211, /*Sym_2*/215, /*Sym_3*/61, } ) ),
		sitepos(2, 'b', "3.2", vector<int>( {/*Sym_2*/29, /*Sym_2*/211, /*Sym_2*/215, /*Sym_3*/61, } ) ),
		sitepos(2, 'a', "32.", vector<int>( {/*Sym_2*/210, /*Sym_2*/212, /*Sym_2*/214, /*Sym_3*/61, } ) ),
	},
	//Wyckoff Positions of Group 183
	{
		sitepos(12, 'f', "1", vector<int>( {} ) ),
		sitepos(6, 'e', ".m.", vector<int>( {/*Sym_m*/315, } ) ),
		sitepos(6, 'd', "..m", vector<int>( {/*Sym_m*/39, } ) ),
		sitepos(3, 'c', "2mm", vector<int>( {/*Sym_2*/213, /*Sym_m*/39, /*Sym_m*/310, } ) ),
		sitepos(2, 'b', "3m.", vector<int>( {/*Sym_m*/310, /*Sym_m*/311, /*Sym_m*/315, /*Sym_3*/61, } ) ),
		sitepos(1, 'a', "6mm", vector<int>( {/*Sym_m*/39, /*Sym_m*/310, /*Sym_m*/311, /*Sym_m*/312, /*Sym_m*/314, /*Sym_m*/315, /*Sym_6*/81, } ) ),
	},
	//Wyckoff Positions of Group 184
	{
		sitepos(12, 'd', "1", vector<int>( {} ) ),
		sitepos(6, 'c', "2..", vector<int>( {/*Sym_2*/213, } ) ),
		sitepos(4, 'b', "3..", vector<int>( {/*Sym_3*/61, } ) ),
		sitepos(2, 'a', "6..", vector<int>( {/*Sym_6*/81, } ) ),
	},
	//Wyckoff Positions of Group 185
	{
		sitepos(12, 'd', "1", vector<int>( {} ) ),
		sitepos(6, 'c', "..m", vector<int>( {/*Sym_m*/39, } ) ),
		sitepos(4, 'b', "3..", vector<int>( {/*Sym_3*/61, } ) ),
		sitepos(2, 'a', "3.m", vector<int>( {/*Sym_m*/39, /*Sym_m*/312, /*Sym_m*/314, /*Sym_3*/61, } ) ),
	},
	//Wyckoff Positions of Group 186
	{
		sitepos(12, 'd', "1", vector<int>( {} ) ),
		sitepos(6, 'c', ".m.", vector<int>( {/*Sym_m*/315, } ) ),
		sitepos(2, 'b', "3m.", vector<int>( {/*Sym_m*/310, /*Sym_m*/311, /*Sym_m*/315, /*Sym_3*/61, } ) ),
		sitepos(2, 'a', "3m.", vector<int>( {/*Sym_m*/310, /*Sym_m*/311, /*Sym_m*/315, /*Sym_3*/61, } ) ),
	},
	//Wyckoff Positions of Group 187
	{
		sitepos(12, 'o', "1", vector<int>( {} ) ),
		sitepos(6, 'n', ".m.", vector<int>( {/*Sym_m*/315, } ) ),
		sitepos(6, 'm', "m..", vector<int>( {/*Sym_m*/313, } ) ),
		sitepos(6, 'l', "m..", vector<int>( {/*Sym_m*/313, } ) ),
		sitepos(3, 'k', "mm2", vector<int>( {/*Sym_2*/215, /*Sym_m*/313, /*Sym_m*/315, } ) ),
		sitepos(3, 'j', "mm2", vector<int>( {/*Sym_2*/215, /*Sym_m*/313, /*Sym_m*/315, } ) ),
		sitepos(2, 'i', "3m.", vector<int>( {/*Sym_m*/310, /*Sym_m*/311, /*Sym_m*/315, /*Sym_3*/61, } ) ),
		sitepos(2, 'h', "3m.", vector<int>( {/*Sym_m*/310, /*Sym_m*/311, /*Sym_m*/315, /*Sym_3*/61, } ) ),
		sitepos(2, 'g', "3m.", vector<int>( {/*Sym_m*/310, /*Sym_m*/311, /*Sym_m*/315, /*Sym_3*/61, } ) ),
		sitepos(1, 'f', "-6m2", vector<int>( {/*Sym_2*/29, /*Sym_2*/211, /*Sym_2*/215, /*Sym_m*/310, /*Sym_m*/311, /*Sym_m*/315, /*Sym_minus6*/91, } ) ),
		sitepos(1, 'e', "-6m2", vector<int>( {/*Sym_2*/29, /*Sym_2*/211, /*Sym_2*/215, /*Sym_m*/310, /*Sym_m*/311, /*Sym_m*/315, /*Sym_minus6*/91, } ) ),
		sitepos(1, 'd', "-6m2", vector<int>( {/*Sym_2*/29, /*Sym_2*/211, /*Sym_2*/215, /*Sym_m*/310, /*Sym_m*/311, /*Sym_m*/315, /*Sym_minus6*/91, } ) ),
		sitepos(1, 'c', "-6m2", vector<int>( {/*Sym_2*/29, /*Sym_2*/211, /*Sym_2*/215, /*Sym_m*/310, /*Sym_m*/311, /*Sym_m*/315, /*Sym_minus6*/91, } ) ),
		sitepos(1, 'b', "-6m2", vector<int>( {/*Sym_2*/29, /*Sym_2*/211, /*Sym_2*/215, /*Sym_m*/310, /*Sym_m*/311, /*Sym_m*/315, /*Sym_minus6*/91, } ) ),
		sitepos(1, 'a', "-6m2", vector<int>( {/*Sym_2*/29, /*Sym_2*/211, /*Sym_2*/215, /*Sym_m*/310, /*Sym_m*/311, /*Sym_m*/315, /*Sym_minus6*/91, } ) ),
	},
	//Wyckoff Positions of Group 188
	{
		sitepos(12, 'l', "1", vector<int>( {} ) ),
		sitepos(6, 'k', "m..", vector<int>( {/*Sym_m*/313, } ) ),
		sitepos(6, 'j', "..2", vector<int>( {/*Sym_2*/215, } ) ),
		sitepos(4, 'i', "3..", vector<int>( {/*Sym_3*/61, } ) ),
		sitepos(4, 'h', "3..", vector<int>( {/*Sym_3*/61, } ) ),
		sitepos(4, 'g', "3..", vector<int>( {/*Sym_3*/61, } ) ),
		sitepos(2, 'f', "-6..", vector<int>( {/*Sym_minus6*/91, } ) ),
		sitepos(2, 'e', "3.2", vector<int>( {/*Sym_2*/29, /*Sym_2*/211, /*Sym_2*/215, /*Sym_3*/61, } ) ),
		sitepos(2, 'd', "-6..", vector<int>( {/*Sym_minus6*/91, } ) ),
		sitepos(2, 'c', "3.2", vector<int>( {/*Sym_2*/29, /*Sym_2*/211, /*Sym_2*/215, /*Sym_3*/61, } ) ),
		sitepos(2, 'b', "-6..", vector<int>( {/*Sym_minus6*/91, } ) ),
		sitepos(2, 'a', "3.2", vector<int>( {/*Sym_2*/29, /*Sym_2*/211, /*Sym_2*/215, /*Sym_3*/61, } ) ),
	},
	//Wyckoff Positions of Group 189
	{
		sitepos(12, 'l', "1", vector<int>( {} ) ),
		sitepos(6, 'k', "m..", vector<int>( {/*Sym_m*/313, } ) ),
		sitepos(6, 'j', "m..", vector<int>( {/*Sym_m*/313, } ) ),
		sitepos(6, 'i', "..m", vector<int>( {/*Sym_m*/39, } ) ),
		sitepos(4, 'h', "3..", vector<int>( {/*Sym_3*/61, } ) ),
		sitepos(3, 'g', "m2m", vector<int>( {/*Sym_2*/210, /*Sym_m*/39, /*Sym_m*/313, } ) ),
		sitepos(3, 'f', "m2m", vector<int>( {/*Sym_2*/210, /*Sym_m*/39, /*Sym_m*/313, } ) ),
		sitepos(2, 'e', "3.m", vector<int>( {/*Sym_m*/39, /*Sym_m*/312, /*Sym_m*/314, /*Sym_3*/61, } ) ),
		sitepos(2, 'd', "-6..", vector<int>( {/*Sym_minus6*/91, } ) ),
		sitepos(2, 'c', "-6..", vector<int>( {/*Sym_minus6*/91, } ) ),
		sitepos(1, 'b', "-62m", vector<int>( {/*Sym_2*/210, /*Sym_2*/212, /*Sym_2*/214, /*Sym_m*/39, /*Sym_m*/312, /*Sym_m*/314, /*Sym_minus6*/91, } ) ),
		sitepos(1, 'a', "-62m", vector<int>( {/*Sym_2*/210, /*Sym_2*/212, /*Sym_2*/214, /*Sym_m*/39, /*Sym_m*/312, /*Sym_m*/314, /*Sym_minus6*/91, } ) ),
	},
	//Wyckoff Positions of Group 190
	{
		sitepos(12, 'i', "1", vector<int>( {} ) ),
		sitepos(6, 'h', "m..", vector<int>( {/*Sym_m*/313, } ) ),
		sitepos(6, 'g', ".2.", vector<int>( {/*Sym_2*/210, } ) ),
		sitepos(4, 'f', "3..", vector<int>( {/*Sym_3*/61, } ) ),
		sitepos(4, 'e', "3..", vector<int>( {/*Sym_3*/61, } ) ),
		sitepos(2, 'd', "-6..", vector<int>( {/*Sym_minus6*/91, } ) ),
		sitepos(2, 'c', "-6..", vector<int>( {/*Sym_minus6*/91, } ) ),
		sitepos(2, 'b', "-6..", vector<int>( {/*Sym_minus6*/91, } ) ),
		sitepos(2, 'a', "32.", vector<int>( {/*Sym_2*/210, /*Sym_2*/212, /*Sym_2*/214, /*Sym_3*/61, } ) ),
	},
	//Wyckoff Positions of Group 191
	{
		sitepos(24, 'r', "1", vector<int>( {} ) ),
		sitepos(12, 'q', "m..", vector<int>( {/*Sym_m*/313, } ) ),
		sitepos(12, 'p', "m..", vector<int>( {/*Sym_m*/313, } ) ),
		sitepos(12, 'o', ".m.", vector<int>( {/*Sym_m*/310, } ) ),
		sitepos(12, 'n', "..m", vector<int>( {/*Sym_m*/39, } ) ),
		sitepos(6, 'm', "mm2", vector<int>( {/*Sym_2*/29, /*Sym_m*/313, /*Sym_m*/310, } ) ),
		sitepos(6, 'l', "mm2", vector<int>( {/*Sym_2*/29, /*Sym_m*/313, /*Sym_m*/310, } ) ),
		sitepos(6, 'k', "m2m", vector<int>( {/*Sym_2*/210, /*Sym_m*/39, /*Sym_m*/313, } ) ),
		sitepos(6, 'j', "m2m", vector<int>( {/*Sym_2*/210, /*Sym_m*/39, /*Sym_m*/313, } ) ),
		sitepos(6, 'i', "2mm", vector<int>( {/*Sym_2*/213, /*Sym_m*/39, /*Sym_m*/310, } ) ),
		sitepos(4, 'h', "3m.", vector<int>( {/*Sym_m*/310, /*Sym_m*/311, /*Sym_m*/315, /*Sym_3*/61, } ) ),
		sitepos(3, 'g', "mmm", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/29, /*Sym_2*/210, /*Sym_2*/213, /*Sym_m*/39, /*Sym_m*/310, /*Sym_m*/313, } ) ),
		sitepos(3, 'f', "mmm", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/29, /*Sym_2*/210, /*Sym_2*/213, /*Sym_m*/39, /*Sym_m*/310, /*Sym_m*/313, } ) ),
		sitepos(2, 'e', "6mm", vector<int>( {/*Sym_m*/39, /*Sym_m*/310, /*Sym_m*/311, /*Sym_m*/312, /*Sym_m*/314, /*Sym_m*/315, /*Sym_6*/81, } ) ),
		sitepos(2, 'd', "-6m2", vector<int>( {/*Sym_2*/29, /*Sym_2*/211, /*Sym_2*/215, /*Sym_m*/310, /*Sym_m*/311, /*Sym_m*/315, /*Sym_minus6*/91, } ) ),
		sitepos(2, 'c', "-6m2", vector<int>( {/*Sym_2*/29, /*Sym_2*/211, /*Sym_2*/215, /*Sym_m*/310, /*Sym_m*/311, /*Sym_m*/315, /*Sym_minus6*/91, } ) ),
		sitepos(1, 'b', "6/mmm", vector<int>( {/*Sym_2*/29, /*Sym_2*/210, /*Sym_2*/211, /*Sym_2*/212, /*Sym_2*/214, /*Sym_2*/215, /*Sym_m*/39, /*Sym_m*/310, /*Sym_m*/311, /*Sym_m*/312, /*Sym_m*/314, /*Sym_m*/315, /*Sym_minus3*/71, /*Sym_6*/81, /*Sym_minus6*/91, } ) ),
		sitepos(1, 'a', "6/mmm", vector<int>( {/*Sym_2*/29, /*Sym_2*/210, /*Sym_2*/211, /*Sym_2*/212, /*Sym_2*/214, /*Sym_2*/215, /*Sym_m*/39, /*Sym_m*/310, /*Sym_m*/311, /*Sym_m*/312, /*Sym_m*/314, /*Sym_m*/315, /*Sym_minus3*/71, /*Sym_6*/81, /*Sym_minus6*/91, } ) ),
	},
	//Wyckoff Positions of Group 192
	{
		sitepos(24, 'm', "1", vector<int>( {} ) ),
		sitepos(12, 'l', "m..", vector<int>( {/*Sym_m*/313, } ) ),
		sitepos(12, 'k', "..2", vector<int>( {/*Sym_2*/29, } ) ),
		sitepos(12, 'j', ".2.", vector<int>( {/*Sym_2*/210, } ) ),
		sitepos(12, 'i', "2..", vector<int>( {/*Sym_2*/213, } ) ),
		sitepos(8, 'h', "3..", vector<int>( {/*Sym_3*/61, } ) ),
		sitepos(6, 'g', "2/m..", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/213, /*Sym_m*/313, } ) ),
		sitepos(6, 'f', "222", vector<int>( {/*Sym_2*/29, /*Sym_2*/210, /*Sym_2*/213, } ) ),
		sitepos(4, 'e', "6..", vector<int>( {/*Sym_6*/81, } ) ),
		sitepos(4, 'd', "-6..", vector<int>( {/*Sym_minus6*/91, } ) ),
		sitepos(4, 'c', "3.2", vector<int>( {/*Sym_2*/29, /*Sym_2*/211, /*Sym_2*/215, /*Sym_3*/61, } ) ),
		sitepos(2, 'b', "6/m..", vector<int>( {/*Sym_minus3*/71, /*Sym_6*/81, /*Sym_minus6*/91, } ) ),
		sitepos(2, 'a', "622", vector<int>( {/*Sym_2*/29, /*Sym_2*/210, /*Sym_2*/211, /*Sym_2*/212, /*Sym_2*/214, /*Sym_2*/215, /*Sym_6*/81, } ) ),
	},
	//Wyckoff Positions of Group 193
	{
		sitepos(24, 'l', "1", vector<int>( {} ) ),
		sitepos(12, 'k', "..m", vector<int>( {/*Sym_m*/39, } ) ),
		sitepos(12, 'j', "m..", vector<int>( {/*Sym_m*/313, } ) ),
		sitepos(12, 'i', "..2", vector<int>( {/*Sym_2*/29, } ) ),
		sitepos(8, 'h', "3..", vector<int>( {/*Sym_3*/61, } ) ),
		sitepos(6, 'g', "m2m", vector<int>( {/*Sym_2*/210, /*Sym_m*/39, /*Sym_m*/313, } ) ),
		sitepos(6, 'f', "..2/m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/29, /*Sym_m*/39, } ) ),
		sitepos(4, 'e', "3.m", vector<int>( {/*Sym_m*/39, /*Sym_m*/312, /*Sym_m*/314, /*Sym_3*/61, } ) ),
		sitepos(4, 'd', "3.2", vector<int>( {/*Sym_2*/29, /*Sym_2*/211, /*Sym_2*/215, /*Sym_3*/61, } ) ),
		sitepos(4, 'c', "-6..", vector<int>( {/*Sym_minus6*/91, } ) ),
		sitepos(2, 'b', "-3.m", vector<int>( {/*Sym_2*/29, /*Sym_2*/211, /*Sym_2*/215, /*Sym_m*/39, /*Sym_m*/312, /*Sym_m*/314, /*Sym_minus3*/71, } ) ),
		sitepos(2, 'a', "-62m", vector<int>( {/*Sym_2*/210, /*Sym_2*/212, /*Sym_2*/214, /*Sym_m*/39, /*Sym_m*/312, /*Sym_m*/314, /*Sym_minus6*/91, } ) ),
	},
	//Wyckoff Positions of Group 194
	{
		sitepos(24, 'l', "1", vector<int>( {} ) ),
		sitepos(12, 'k', ".m.", vector<int>( {/*Sym_m*/310,} ) ),
		sitepos(12, 'j', "m..", vector<int>( {/*Sym_m*/313, } ) ),
		sitepos(12, 'i', ".2.", vector<int>( {/*Sym_2*/210, } ) ),
		sitepos(6, 'h', "mm2", vector<int>( {/*Sym_2*/29, /*Sym_m*/310, /*Sym_m*/313,} ) ),
		sitepos(6, 'g', ".2/m.", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/210, /*Sym_m*/310, } ) ),
		sitepos(4, 'f', "3m.", vector<int>( {/*Sym_m*/310, /*Sym_m*/311, /*Sym_m*/315, /*Sym_3*/61, } ) ),
		sitepos(4, 'e', "3m.", vector<int>( {/*Sym_m*/310, /*Sym_m*/311, /*Sym_m*/315, /*Sym_3*/61, } ) ),
		sitepos(2, 'd', "-6m2", vector<int>( {/*Sym_2*/29, /*Sym_2*/211, /*Sym_2*/215, /*Sym_m*/310, /*Sym_m*/311, /*Sym_m*/315, /*Sym_minus6*/91, } ) ),
		sitepos(2, 'c', "-6m2", vector<int>( {/*Sym_2*/29, /*Sym_2*/211, /*Sym_2*/215, /*Sym_m*/310, /*Sym_m*/311, /*Sym_m*/315, /*Sym_minus6*/91, } ) ),
		sitepos(2, 'b', "-6m2", vector<int>( {/*Sym_2*/29, /*Sym_2*/211, /*Sym_2*/215, /*Sym_m*/310, /*Sym_m*/311, /*Sym_m*/315, /*Sym_minus6*/91, } ) ),
		sitepos(2, 'a', "-3m.", vector<int>( {/*Sym_2*/210, /*Sym_2*/212, /*Sym_2*/214, /*Sym_m*/310, /*Sym_m*/311, /*Sym_m*/315, /*Sym_minus3*/71, } ) ),
	},
	//Wyckoff Positions of Group 195
	{
		sitepos(12, 'j', "1", vector<int>( {} ) ),
		sitepos(6, 'i', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(6, 'h', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(6, 'g', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(6, 'f', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(4, 'e', ".3.", vector<int>( {/*Sym_3*/63, } ) ),
		sitepos(3, 'd', "222 . .", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(3, 'c', "222 . .", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(1, 'b', "23.", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_3*/63, /*Sym_3*/65, /*Sym_3*/67, /*Sym_3*/69, } ) ),
		sitepos(1, 'a', "23.", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_3*/63, /*Sym_3*/65, /*Sym_3*/67, /*Sym_3*/69, } ) ),
	},
	//Wyckoff Positions of Group 196
	{
		sitepos(48, 'h', "1", vector<int>( {} ) ),
		sitepos(24, 'g', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(24, 'f', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(16, 'e', ".3.", vector<int>( {/*Sym_3*/63, } ) ),
		sitepos(4, 'd', "23.", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_3*/63, /*Sym_3*/65, /*Sym_3*/67, /*Sym_3*/69, } ) ),
		sitepos(4, 'c', "23.", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_3*/63, /*Sym_3*/65, /*Sym_3*/67, /*Sym_3*/69, } ) ),
		sitepos(4, 'b', "23.", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_3*/63, /*Sym_3*/65, /*Sym_3*/67, /*Sym_3*/69, } ) ),
		sitepos(4, 'a', "23.", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_3*/63, /*Sym_3*/65, /*Sym_3*/67, /*Sym_3*/69, } ) ),
	},
	//Wyckoff Positions of Group 197
	{
		sitepos(24, 'f', "1", vector<int>( {} ) ),
		sitepos(12, 'e', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(12, 'd', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(8, 'c', ".3.", vector<int>( {/*Sym_3*/63, } ) ),
		sitepos(6, 'b', "222 . .", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(2, 'a', "23.", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_3*/63, /*Sym_3*/65, /*Sym_3*/67, /*Sym_3*/69, } ) ),
	},
	//Wyckoff Positions of Group 198
	{
		sitepos(12, 'b', "1", vector<int>( {} ) ),
		sitepos(4, 'a', ".3.", vector<int>( {/*Sym_3*/63, } ) ),
	},
	//Wyckoff Positions of Group 199
	{
		sitepos(24, 'c', "1", vector<int>( {} ) ),
		sitepos(12, 'b', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(8, 'a', ".3.", vector<int>( {/*Sym_3*/63, } ) ),
	},
	//Wyckoff Positions of Group 200
	{
		sitepos(24, 'l', "1", vector<int>( {} ) ),
		sitepos(12, 'k', "m..", vector<int>( {/*Sym_m*/30, } ) ),
		sitepos(12, 'j', "m..", vector<int>( {/*Sym_m*/30, } ) ),
		sitepos(8, 'i', ".3.", vector<int>( {/*Sym_3*/63, } ) ),
		sitepos(6, 'h', "mm2 . .", vector<int>( {/*Sym_2*/20, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(6, 'g', "mm2 . .", vector<int>( {/*Sym_2*/20, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(6, 'f', "mm2 . .", vector<int>( {/*Sym_2*/20, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(6, 'e', "mm2 . .", vector<int>( {/*Sym_2*/20, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(3, 'd', "mmm . .", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(3, 'c', "mmm . .", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(1, 'b', "m-3.", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/32, /*Sym_minus3*/73, /*Sym_minus3*/75, /*Sym_minus3*/77, /*Sym_minus3*/79, } ) ),
		sitepos(1, 'a', "m-3.", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/32, /*Sym_minus3*/73, /*Sym_minus3*/75, /*Sym_minus3*/77, /*Sym_minus3*/79, } ) ),
	},
	//Wyckoff Positions of Group 201
	{
		sitepos(24, 'h', "1", vector<int>( {} ) ),
		sitepos(12, 'g', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(12, 'f', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(8, 'e', ".3.", vector<int>( {/*Sym_3*/63, } ) ),
		sitepos(6, 'd', "222 . .", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(4, 'c', ".-3.", vector<int>( {/*Sym_minus1*/10, /*Sym_minus3*/73, } ) ),
		sitepos(4, 'b', ".-3.", vector<int>( {/*Sym_minus1*/10, /*Sym_minus3*/73, } ) ),
		sitepos(2, 'a', "23.", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_3*/63, /*Sym_3*/65, /*Sym_3*/67, /*Sym_3*/69, } ) ),
	},
	//Wyckoff Positions of Group 202
	{
		sitepos(96, 'i', "1", vector<int>( {} ) ),
		sitepos(48, 'h', "m..", vector<int>( {/*Sym_m*/30, } ) ),
		sitepos(48, 'g', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(32, 'f', ".3.", vector<int>( {/*Sym_3*/63, } ) ),
		sitepos(24, 'e', "mm2 . .", vector<int>( {/*Sym_2*/20, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(24, 'd', "2/m..", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_m*/30, } ) ),
		sitepos(8, 'c', "23.", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_3*/63, /*Sym_3*/65, /*Sym_3*/67, /*Sym_3*/69, } ) ),
		sitepos(4, 'b', "m-3.", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/32, /*Sym_minus3*/73, /*Sym_minus3*/75, /*Sym_minus3*/77, /*Sym_minus3*/79, } ) ),
		sitepos(4, 'a', "m-3.", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/32, /*Sym_minus3*/73, /*Sym_minus3*/75, /*Sym_minus3*/77, /*Sym_minus3*/79, } ) ),
	},
	//Wyckoff Positions of Group 203
	{
		sitepos(96, 'g', "1", vector<int>( {} ) ),
		sitepos(48, 'f', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(32, 'e', ".3.", vector<int>( {/*Sym_3*/63, } ) ),
		sitepos(16, 'd', ".-3.", vector<int>( {/*Sym_minus1*/10, /*Sym_minus3*/73, } ) ),
		sitepos(16, 'c', ".-3.", vector<int>( {/*Sym_minus1*/10, /*Sym_minus3*/73, } ) ),
		sitepos(8, 'b', "23.", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_3*/63, /*Sym_3*/65, /*Sym_3*/67, /*Sym_3*/69, } ) ),
		sitepos(8, 'a', "23.", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_3*/63, /*Sym_3*/65, /*Sym_3*/67, /*Sym_3*/69, } ) ),
	},
	//Wyckoff Positions of Group 204
	{
		sitepos(48, 'h', "1", vector<int>( {} ) ),
		sitepos(24, 'g', "m..", vector<int>( {/*Sym_m*/30, } ) ),
		sitepos(16, 'f', ".3.", vector<int>( {/*Sym_3*/63, } ) ),
		sitepos(12, 'e', "mm2 . .", vector<int>( {/*Sym_2*/20, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(12, 'd', "mm2 . .", vector<int>( {/*Sym_2*/20, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(8, 'c', ".-3.", vector<int>( {/*Sym_minus1*/10, /*Sym_minus3*/73, } ) ),
		sitepos(6, 'b', "mmm . .", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(2, 'a', "m-3.", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/32, /*Sym_minus3*/73, /*Sym_minus3*/75, /*Sym_minus3*/77, /*Sym_minus3*/79, } ) ),
	},
	//Wyckoff Positions of Group 205
	{
		sitepos(24, 'd', "1", vector<int>( {} ) ),
		sitepos(8, 'c', ".3.", vector<int>( {/*Sym_3*/63, } ) ),
		sitepos(4, 'b', ".-3.", vector<int>( {/*Sym_minus1*/10, /*Sym_minus3*/73, } ) ),
		sitepos(4, 'a', ".-3.", vector<int>( {/*Sym_minus1*/10, /*Sym_minus3*/73, } ) ),
	},
	//Wyckoff Positions of Group 206
	{
		sitepos(48, 'e', "1", vector<int>( {} ) ),
		sitepos(24, 'd', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(16, 'c', ".3.", vector<int>( {/*Sym_3*/63, } ) ),
		sitepos(8, 'b', ".-3.", vector<int>( {/*Sym_minus1*/10, /*Sym_minus3*/73, } ) ),
		sitepos(8, 'a', ".-3.", vector<int>( {/*Sym_minus1*/10, /*Sym_minus3*/73, } ) ),
	},
	//Wyckoff Positions of Group 207
	{
		sitepos(24, 'k', "1", vector<int>( {} ) ),
		sitepos(12, 'j', "..2", vector<int>( {/*Sym_2*/24, } ) ),
		sitepos(12, 'i', "..2", vector<int>( {/*Sym_2*/24, } ) ),
		sitepos(12, 'h', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(8, 'g', ".3.", vector<int>( {/*Sym_3*/63, } ) ),
		sitepos(6, 'f', "4..", vector<int>( {/*Sym_4*/45, } ) ),
		sitepos(6, 'e', "4..", vector<int>( {/*Sym_4*/45, } ) ),
		sitepos(3, 'd', "42. 2", vector<int>( {/*Sym_2*/21, /*Sym_2*/22, /*Sym_2*/24, /*Sym_2*/27, /*Sym_4*/45, } ) ),
		sitepos(3, 'c', "42. 2", vector<int>( {/*Sym_2*/21, /*Sym_2*/22, /*Sym_2*/24, /*Sym_2*/27, /*Sym_4*/45, } ) ),
		sitepos(1, 'b', "432", vector<int>( {/*Sym_2*/23, /*Sym_2*/24, /*Sym_2*/25, /*Sym_2*/26, /*Sym_2*/27, /*Sym_2*/28, /*Sym_4*/41, /*Sym_4*/43, /*Sym_4*/45, /*Sym_3*/63, /*Sym_3*/65, /*Sym_3*/67, /*Sym_3*/69, } ) ),
		sitepos(1, 'a', "432", vector<int>( {/*Sym_2*/23, /*Sym_2*/24, /*Sym_2*/25, /*Sym_2*/26, /*Sym_2*/27, /*Sym_2*/28, /*Sym_4*/41, /*Sym_4*/43, /*Sym_4*/45, /*Sym_3*/63, /*Sym_3*/65, /*Sym_3*/67, /*Sym_3*/69, } ) ),
	},
	//Wyckoff Positions of Group 208
	{
		sitepos(24, 'm', "1", vector<int>( {} ) ),
		sitepos(12, 'l', "..2", vector<int>( {/*Sym_2*/24, } ) ),
		sitepos(12, 'k', "..2", vector<int>( {/*Sym_2*/27, } ) ),
		sitepos(12, 'j', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(12, 'i', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(12, 'h', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(8, 'g', ".3.", vector<int>( {/*Sym_3*/63, } ) ),
		sitepos(6, 'f', "2.2 2", vector<int>( {/*Sym_2*/20, /*Sym_2*/24, /*Sym_2*/27, } ) ),
		sitepos(6, 'e', "2.2 2", vector<int>( {/*Sym_2*/20, /*Sym_2*/24, /*Sym_2*/27, } ) ),
		sitepos(6, 'd', "222 . .", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(4, 'c', ".32", vector<int>( {/*Sym_2*/26, /*Sym_2*/27, /*Sym_2*/28, /*Sym_3*/63, } ) ),
		sitepos(4, 'b', ".32", vector<int>( {/*Sym_2*/26, /*Sym_2*/27, /*Sym_2*/28, /*Sym_3*/63, } ) ),
		sitepos(2, 'a', "23.", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_3*/63, /*Sym_3*/65, /*Sym_3*/67, /*Sym_3*/69, } ) ),
	},
	//Wyckoff Positions of Group 209
	{
		sitepos(96, 'j', "1", vector<int>( {} ) ),
		sitepos(48, 'i', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(48, 'h', "..2", vector<int>( {/*Sym_2*/24, } ) ),
		sitepos(48, 'g', "..2", vector<int>( {/*Sym_2*/24, } ) ),
		sitepos(32, 'f', ".3.", vector<int>( {/*Sym_3*/63, } ) ),
		sitepos(24, 'e', "4..", vector<int>( {/*Sym_4*/45, } ) ),
		sitepos(24, 'd', "2.2 2", vector<int>( {/*Sym_2*/20, /*Sym_2*/24, /*Sym_2*/27, } ) ),
		sitepos(8, 'c', "23.", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_3*/63, /*Sym_3*/65, /*Sym_3*/67, /*Sym_3*/69, } ) ),
		sitepos(4, 'b', "432", vector<int>( {/*Sym_2*/23, /*Sym_2*/24, /*Sym_2*/25, /*Sym_2*/26, /*Sym_2*/27, /*Sym_2*/28, /*Sym_4*/41, /*Sym_4*/43, /*Sym_4*/45, /*Sym_3*/63, /*Sym_3*/65, /*Sym_3*/67, /*Sym_3*/69, } ) ),
		sitepos(4, 'a', "432", vector<int>( {/*Sym_2*/23, /*Sym_2*/24, /*Sym_2*/25, /*Sym_2*/26, /*Sym_2*/27, /*Sym_2*/28, /*Sym_4*/41, /*Sym_4*/43, /*Sym_4*/45, /*Sym_3*/63, /*Sym_3*/65, /*Sym_3*/67, /*Sym_3*/69, } ) ),
	},
	//Wyckoff Positions of Group 210
	{
		sitepos(96, 'h', "1", vector<int>( {} ) ),
		sitepos(48, 'g', "..2", vector<int>( {/*Sym_2*/27, } ) ),
		sitepos(48, 'f', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(32, 'e', ".3.", vector<int>( {/*Sym_3*/63, } ) ),
		sitepos(16, 'd', ".32", vector<int>( {/*Sym_2*/26, /*Sym_2*/27, /*Sym_2*/28, /*Sym_3*/63, } ) ),
		sitepos(16, 'c', ".32", vector<int>( {/*Sym_2*/26, /*Sym_2*/27, /*Sym_2*/28, /*Sym_3*/63, } ) ),
		sitepos(8, 'b', "23.", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_3*/63, /*Sym_3*/65, /*Sym_3*/67, /*Sym_3*/69, } ) ),
		sitepos(8, 'a', "23.", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_3*/63, /*Sym_3*/65, /*Sym_3*/67, /*Sym_3*/69, } ) ),
	},
	//Wyckoff Positions of Group 211
	{
		sitepos(48, 'j', "1", vector<int>( {} ) ),
		sitepos(24, 'i', "..2", vector<int>( {/*Sym_2*/27, } ) ),
		sitepos(24, 'h', "..2", vector<int>( {/*Sym_2*/24, } ) ),
		sitepos(24, 'g', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(16, 'f', ".3.", vector<int>( {/*Sym_3*/63, } ) ),
		sitepos(12, 'e', "4..", vector<int>( {/*Sym_4*/45, } ) ),
		sitepos(12, 'd', "2.2 2", vector<int>( {/*Sym_2*/20, /*Sym_2*/24, /*Sym_2*/27, } ) ),
		sitepos(8, 'c', ".32", vector<int>( {/*Sym_2*/26, /*Sym_2*/27, /*Sym_2*/28, /*Sym_3*/63, } ) ),
		sitepos(6, 'b', "42. 2", vector<int>( {/*Sym_2*/21, /*Sym_2*/22, /*Sym_2*/24, /*Sym_2*/27, /*Sym_4*/45, } ) ),
		sitepos(2, 'a', "432", vector<int>( {/*Sym_2*/23, /*Sym_2*/24, /*Sym_2*/25, /*Sym_2*/26, /*Sym_2*/27, /*Sym_2*/28, /*Sym_4*/41, /*Sym_4*/43, /*Sym_4*/45, /*Sym_3*/63, /*Sym_3*/65, /*Sym_3*/67, /*Sym_3*/69, } ) ),
	},
	//Wyckoff Positions of Group 212
	{
		sitepos(24, 'e', "1", vector<int>( {} ) ),
		sitepos(12, 'd', "..2", vector<int>( {/*Sym_2*/27, } ) ),
		sitepos(8, 'c', ".3.", vector<int>( {/*Sym_3*/63, } ) ),
		sitepos(4, 'b', ".32", vector<int>( {/*Sym_2*/26, /*Sym_2*/27, /*Sym_2*/28, /*Sym_3*/63, } ) ),
		sitepos(4, 'a', ".32", vector<int>( {/*Sym_2*/26, /*Sym_2*/27, /*Sym_2*/28, /*Sym_3*/63, } ) ),
	},
	//Wyckoff Positions of Group 213
	{
		sitepos(24, 'e', "1", vector<int>( {} ) ),
		sitepos(12, 'd', "..2", vector<int>( {/*Sym_2*/24, } ) ),
		sitepos(8, 'c', ".3.", vector<int>( {/*Sym_3*/63, } ) ),
		sitepos(4, 'b', ".32", vector<int>( {/*Sym_2*/26, /*Sym_2*/27, /*Sym_2*/28, /*Sym_3*/63, } ) ),
		sitepos(4, 'a', ".32", vector<int>( {/*Sym_2*/26, /*Sym_2*/27, /*Sym_2*/28, /*Sym_3*/63, } ) ),
	},
	//Wyckoff Positions of Group 214
	{
		sitepos(48, 'i', "1", vector<int>( {} ) ),
		sitepos(24, 'h', "..2", vector<int>( {/*Sym_2*/27, } ) ),
		sitepos(24, 'g', "..2", vector<int>( {/*Sym_2*/24, } ) ),
		sitepos(24, 'f', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(16, 'e', ".3.", vector<int>( {/*Sym_3*/63, } ) ),
		sitepos(12, 'd', "2.2 2", vector<int>( {/*Sym_2*/20, /*Sym_2*/24, /*Sym_2*/27, } ) ),
		sitepos(12, 'c', "2.2 2", vector<int>( {/*Sym_2*/20, /*Sym_2*/24, /*Sym_2*/27, } ) ),
		sitepos(8, 'b', ".32", vector<int>( {/*Sym_2*/26, /*Sym_2*/27, /*Sym_2*/28, /*Sym_3*/63, } ) ),
		sitepos(8, 'a', ".32", vector<int>( {/*Sym_2*/26, /*Sym_2*/27, /*Sym_2*/28, /*Sym_3*/63, } ) ),
	},
	//Wyckoff Positions of Group 215
	{
		sitepos(24, 'j', "1", vector<int>( {} ) ),
		sitepos(12, 'i', "..m", vector<int>( {/*Sym_m*/33, } ) ),
		sitepos(12, 'h', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(6, 'g', "2.m m", vector<int>( {/*Sym_2*/20, /*Sym_m*/35, /*Sym_m*/38, } ) ),
		sitepos(6, 'f', "2.m m", vector<int>( {/*Sym_2*/20, /*Sym_m*/35, /*Sym_m*/38, } ) ),
		sitepos(4, 'e', ".3m", vector<int>( {/*Sym_m*/33, /*Sym_m*/34, /*Sym_m*/35, /*Sym_3*/63, } ) ),
		sitepos(3, 'd', "-42. m", vector<int>( {/*Sym_2*/21, /*Sym_2*/22, /*Sym_m*/35, /*Sym_m*/38, /*Sym_minus4*/55, } ) ),
		sitepos(3, 'c', "-42. m", vector<int>( {/*Sym_2*/21, /*Sym_2*/22, /*Sym_m*/35, /*Sym_m*/38, /*Sym_minus4*/55, } ) ),
		sitepos(1, 'b', "-43m", vector<int>( {/*Sym_m*/33, /*Sym_m*/34, /*Sym_m*/35, /*Sym_m*/36, /*Sym_m*/37, /*Sym_m*/38, /*Sym_minus4*/51, /*Sym_minus4*/53, /*Sym_minus4*/55, /*Sym_3*/63, /*Sym_3*/65, /*Sym_3*/67, /*Sym_3*/69, } ) ),
		sitepos(1, 'a', "-43m", vector<int>( {/*Sym_m*/33, /*Sym_m*/34, /*Sym_m*/35, /*Sym_m*/36, /*Sym_m*/37, /*Sym_m*/38, /*Sym_minus4*/51, /*Sym_minus4*/53, /*Sym_minus4*/55, /*Sym_3*/63, /*Sym_3*/65, /*Sym_3*/67, /*Sym_3*/69, } ) ),
	},
	//Wyckoff Positions of Group 216
	{
		sitepos(96, 'i', "1", vector<int>( {} ) ),
		sitepos(48, 'h', "..m", vector<int>( {/*Sym_m*/33, } ) ),
		sitepos(24, 'g', "2.m m", vector<int>( {/*Sym_2*/20, /*Sym_m*/35, /*Sym_m*/38, } ) ),
		sitepos(24, 'f', "2.m m", vector<int>( {/*Sym_2*/20, /*Sym_m*/35, /*Sym_m*/38, } ) ),
		sitepos(16, 'e', ".3m", vector<int>( {/*Sym_m*/33, /*Sym_m*/34, /*Sym_m*/35, /*Sym_3*/63, } ) ),
		sitepos(4, 'd', "-43m", vector<int>( {/*Sym_m*/33, /*Sym_m*/34, /*Sym_m*/35, /*Sym_m*/36, /*Sym_m*/37, /*Sym_m*/38, /*Sym_minus4*/51, /*Sym_minus4*/53, /*Sym_minus4*/55, /*Sym_3*/63, /*Sym_3*/65, /*Sym_3*/67, /*Sym_3*/69, } ) ),
		sitepos(4, 'c', "-43m", vector<int>( {/*Sym_m*/33, /*Sym_m*/34, /*Sym_m*/35, /*Sym_m*/36, /*Sym_m*/37, /*Sym_m*/38, /*Sym_minus4*/51, /*Sym_minus4*/53, /*Sym_minus4*/55, /*Sym_3*/63, /*Sym_3*/65, /*Sym_3*/67, /*Sym_3*/69, } ) ),
		sitepos(4, 'b', "-43m", vector<int>( {/*Sym_m*/33, /*Sym_m*/34, /*Sym_m*/35, /*Sym_m*/36, /*Sym_m*/37, /*Sym_m*/38, /*Sym_minus4*/51, /*Sym_minus4*/53, /*Sym_minus4*/55, /*Sym_3*/63, /*Sym_3*/65, /*Sym_3*/67, /*Sym_3*/69, } ) ),
		sitepos(4, 'a', "-43m", vector<int>( {/*Sym_m*/33, /*Sym_m*/34, /*Sym_m*/35, /*Sym_m*/36, /*Sym_m*/37, /*Sym_m*/38, /*Sym_minus4*/51, /*Sym_minus4*/53, /*Sym_minus4*/55, /*Sym_3*/63, /*Sym_3*/65, /*Sym_3*/67, /*Sym_3*/69, } ) ),
	},
	//Wyckoff Positions of Group 217
	{
		sitepos(48, 'h', "1", vector<int>( {} ) ),
		sitepos(24, 'g', "..m", vector<int>( {/*Sym_m*/33, } ) ),
		sitepos(24, 'f', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(12, 'e', "2.m m", vector<int>( {/*Sym_2*/20, /*Sym_m*/35, /*Sym_m*/38, } ) ),
		sitepos(12, 'd', "-4..", vector<int>( {/*Sym_minus4*/55, } ) ),
		sitepos(8, 'c', ".3m", vector<int>( {/*Sym_m*/33, /*Sym_m*/34, /*Sym_m*/35, /*Sym_3*/63, } ) ),
		sitepos(6, 'b', "-42. m", vector<int>( {/*Sym_2*/21, /*Sym_2*/22, /*Sym_m*/35, /*Sym_m*/38, /*Sym_minus4*/55, } ) ),
		sitepos(2, 'a', "-43m", vector<int>( {/*Sym_m*/33, /*Sym_m*/34, /*Sym_m*/35, /*Sym_m*/36, /*Sym_m*/37, /*Sym_m*/38, /*Sym_minus4*/51, /*Sym_minus4*/53, /*Sym_minus4*/55, /*Sym_3*/63, /*Sym_3*/65, /*Sym_3*/67, /*Sym_3*/69, } ) ),
	},
	//Wyckoff Positions of Group 218
	{
		sitepos(24, 'i', "1", vector<int>( {} ) ),
		sitepos(12, 'h', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(12, 'g', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(12, 'f', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(8, 'e', ".3.", vector<int>( {/*Sym_3*/63, } ) ),
		sitepos(6, 'd', "-4..", vector<int>( {/*Sym_minus4*/55, } ) ),
		sitepos(6, 'c', "-4..", vector<int>( {/*Sym_minus4*/55, } ) ),
		sitepos(6, 'b', "222 . .", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, } ) ),
		sitepos(2, 'a', "23.", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_3*/63, /*Sym_3*/65, /*Sym_3*/67, /*Sym_3*/69, } ) ),
	},
	//Wyckoff Positions of Group 219
	{
		sitepos(96, 'h', "1", vector<int>( {} ) ),
		sitepos(48, 'g', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(48, 'f', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(32, 'e', ".3.", vector<int>( {/*Sym_3*/63, } ) ),
		sitepos(24, 'd', "-4..", vector<int>( {/*Sym_minus4*/55, } ) ),
		sitepos(24, 'c', "-4..", vector<int>( {/*Sym_minus4*/55, } ) ),
		sitepos(8, 'b', "23.", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_3*/63, /*Sym_3*/65, /*Sym_3*/67, /*Sym_3*/69, } ) ),
		sitepos(8, 'a', "23.", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_3*/63, /*Sym_3*/65, /*Sym_3*/67, /*Sym_3*/69, } ) ),
	},
	//Wyckoff Positions of Group 220
	{
		sitepos(48, 'e', "1", vector<int>( {} ) ),
		sitepos(24, 'd', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(16, 'c', ".3.", vector<int>( {/*Sym_3*/63, } ) ),
		sitepos(12, 'b', "-4..", vector<int>( {/*Sym_minus4*/55, } ) ),
		sitepos(12, 'a', "-4..", vector<int>( {/*Sym_minus4*/55, } ) ),
	},
	//Wyckoff Positions of Group 221
	{
		sitepos(48, 'n', "1", vector<int>( {} ) ),
		sitepos(24, 'm', "..m", vector<int>( {/*Sym_m*/33, } ) ),
		sitepos(24, 'l', "m..", vector<int>( {/*Sym_m*/30, } ) ),
		sitepos(24, 'k', "m..", vector<int>( {/*Sym_m*/30, } ) ),
		sitepos(12, 'j', "m.m 2", vector<int>( {/*Sym_2*/24, /*Sym_m*/30, /*Sym_m*/35, } ) ),
		sitepos(12, 'i', "m.m 2", vector<int>( {/*Sym_2*/24, /*Sym_m*/30, /*Sym_m*/35, } ) ),
		sitepos(12, 'h', "mm2 . .", vector<int>( {/*Sym_2*/20, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(8, 'g', ".3m", vector<int>( {/*Sym_m*/33, /*Sym_m*/34, /*Sym_m*/35, /*Sym_3*/63, } ) ),
		sitepos(6, 'f', "4m. m", vector<int>( {/*Sym_m*/31, /*Sym_m*/32, /*Sym_m*/35, /*Sym_m*/38, /*Sym_4*/45, } ) ),
		sitepos(6, 'e', "4m. m", vector<int>( {/*Sym_m*/31, /*Sym_m*/32, /*Sym_m*/35, /*Sym_m*/38, /*Sym_4*/45, } ) ),
		sitepos(3, 'd', "4/mm. m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/21, /*Sym_2*/22, /*Sym_2*/24, /*Sym_2*/27, /*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/32, /*Sym_m*/35, /*Sym_m*/38, /*Sym_4*/45, /*Sym_minus4*/55, } ) ),
		sitepos(3, 'c', "4/mm. m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/21, /*Sym_2*/22, /*Sym_2*/24, /*Sym_2*/27, /*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/32, /*Sym_m*/35, /*Sym_m*/38, /*Sym_4*/45, /*Sym_minus4*/55, } ) ),
		sitepos(1, 'b', "m-3m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/23, /*Sym_2*/24, /*Sym_2*/25, /*Sym_2*/26, /*Sym_2*/27, /*Sym_2*/28, /*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/32, /*Sym_m*/33, /*Sym_m*/34, /*Sym_m*/35, /*Sym_m*/36, /*Sym_m*/37, /*Sym_m*/38, /*Sym_4*/41, /*Sym_4*/43, /*Sym_4*/45, /*Sym_minus4*/51, /*Sym_minus4*/53, /*Sym_minus4*/55, /*Sym_minus3*/73, /*Sym_minus3*/75, /*Sym_minus3*/77, /*Sym_minus3*/79, } ) ),
		sitepos(1, 'a', "m-3m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/23, /*Sym_2*/24, /*Sym_2*/25, /*Sym_2*/26, /*Sym_2*/27, /*Sym_2*/28, /*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/32, /*Sym_m*/33, /*Sym_m*/34, /*Sym_m*/35, /*Sym_m*/36, /*Sym_m*/37, /*Sym_m*/38, /*Sym_4*/41, /*Sym_4*/43, /*Sym_4*/45, /*Sym_minus4*/51, /*Sym_minus4*/53, /*Sym_minus4*/55, /*Sym_minus3*/73, /*Sym_minus3*/75, /*Sym_minus3*/77, /*Sym_minus3*/79, } ) ),
	},
	//Wyckoff Positions of Group 222
	{
		sitepos(48, 'i', "1", vector<int>( {} ) ),
		sitepos(24, 'h', "..2", vector<int>( {/*Sym_2*/24, } ) ),
		sitepos(24, 'g', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(16, 'f', ".3.", vector<int>( {/*Sym_3*/63, } ) ),
		sitepos(12, 'e', "4..", vector<int>( {/*Sym_4*/45, } ) ),
		sitepos(12, 'd', "-4..", vector<int>( {/*Sym_minus4*/55, } ) ),
		sitepos(8, 'c', ".-3.", vector<int>( {/*Sym_minus1*/10, /*Sym_minus3*/73, } ) ),
		sitepos(6, 'b', "42. 2", vector<int>( {/*Sym_2*/21, /*Sym_2*/22, /*Sym_2*/24, /*Sym_2*/27, /*Sym_4*/45, } ) ),
		sitepos(2, 'a', "432", vector<int>( {/*Sym_2*/23, /*Sym_2*/24, /*Sym_2*/25, /*Sym_2*/26, /*Sym_2*/27, /*Sym_2*/28, /*Sym_4*/41, /*Sym_4*/43, /*Sym_4*/45, /*Sym_3*/63, /*Sym_3*/65, /*Sym_3*/67, /*Sym_3*/69, } ) ),
	},
	//Wyckoff Positions of Group 223
	{
		sitepos(48, 'l', "1", vector<int>( {} ) ),
		sitepos(24, 'k', "m..", vector<int>( {/*Sym_m*/30, } ) ),
		sitepos(24, 'j', "..2", vector<int>( {/*Sym_2*/24, } ) ),
		sitepos(16, 'i', ".3.", vector<int>( {/*Sym_3*/63, } ) ),
		sitepos(12, 'h', "mm2 . .", vector<int>( {/*Sym_2*/20, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(12, 'g', "mm2 . .", vector<int>( {/*Sym_2*/20, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(12, 'f', "mm2 . .", vector<int>( {/*Sym_2*/20, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(8, 'e', ".32", vector<int>( {/*Sym_2*/26, /*Sym_2*/27, /*Sym_2*/28, /*Sym_3*/63, } ) ),
		sitepos(6, 'd', "-4m. 2", vector<int>( {/*Sym_2*/24, /*Sym_2*/27, /*Sym_m*/31, /*Sym_m*/32, /*Sym_minus4*/55, } ) ),
		sitepos(6, 'c', "-4m. 2", vector<int>( {/*Sym_2*/24, /*Sym_2*/27, /*Sym_m*/31, /*Sym_m*/32, /*Sym_minus4*/55, } ) ),
		sitepos(6, 'b', "mmm . .", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(2, 'a', "m-3.", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/32, /*Sym_minus3*/73, /*Sym_minus3*/75, /*Sym_minus3*/77, /*Sym_minus3*/79, } ) ),
	},
	//Wyckoff Positions of Group 224
	{
		sitepos(48, 'l', "1", vector<int>( {} ) ),
		sitepos(24, 'k', "..m", vector<int>( {/*Sym_m*/33, } ) ),
		sitepos(24, 'j', "..2", vector<int>( {/*Sym_2*/27, } ) ),
		sitepos(24, 'i', "..2", vector<int>( {/*Sym_2*/24, } ) ),
		sitepos(24, 'h', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(12, 'g', "2.m m", vector<int>( {/*Sym_2*/20, /*Sym_m*/35, /*Sym_m*/38, } ) ),
		sitepos(12, 'f', "2.2 2", vector<int>( {/*Sym_2*/20, /*Sym_2*/24, /*Sym_2*/27, } ) ),
		sitepos(8, 'e', ".3m", vector<int>( {/*Sym_m*/33, /*Sym_m*/34, /*Sym_m*/35, /*Sym_3*/63, } ) ),
		sitepos(6, 'd', "-42. m", vector<int>( {/*Sym_2*/21, /*Sym_2*/22, /*Sym_m*/35, /*Sym_m*/38, /*Sym_minus4*/55, } ) ),
		sitepos(4, 'c', ".-3m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/26, /*Sym_2*/27, /*Sym_2*/28, /*Sym_m*/33, /*Sym_m*/34, /*Sym_m*/35, /*Sym_minus3*/73, } ) ),
		sitepos(4, 'b', ".-3m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/26, /*Sym_2*/27, /*Sym_2*/28, /*Sym_m*/33, /*Sym_m*/34, /*Sym_m*/35, /*Sym_minus3*/73, } ) ),
		sitepos(2, 'a', "-43m", vector<int>( {/*Sym_m*/33, /*Sym_m*/34, /*Sym_m*/35, /*Sym_m*/36, /*Sym_m*/37, /*Sym_m*/38, /*Sym_minus4*/51, /*Sym_minus4*/53, /*Sym_minus4*/55, /*Sym_3*/63, /*Sym_3*/65, /*Sym_3*/67, /*Sym_3*/69, } ) ),
	},
	//Wyckoff Positions of Group 225
	{
		sitepos(192, 'l', "1", vector<int>( {} ) ),
		sitepos(96, 'k', "..m", vector<int>( {/*Sym_m*/33, } ) ),
		sitepos(96, 'j', "m..", vector<int>( {/*Sym_m*/30, } ) ),
		sitepos(48, 'i', "m.m 2", vector<int>( {/*Sym_2*/24, /*Sym_m*/30, /*Sym_m*/35, } ) ),
		sitepos(48, 'h', "m.m 2", vector<int>( {/*Sym_2*/24, /*Sym_m*/30, /*Sym_m*/35, } ) ),
		sitepos(48, 'g', "2.m m", vector<int>( {/*Sym_2*/20, /*Sym_m*/35, /*Sym_m*/38, } ) ),
		sitepos(32, 'f', ".3m", vector<int>( {/*Sym_m*/33, /*Sym_m*/34, /*Sym_m*/35, /*Sym_3*/63, } ) ),
		sitepos(24, 'e', "4m. m", vector<int>( {/*Sym_m*/31, /*Sym_m*/32, /*Sym_m*/35, /*Sym_m*/38, /*Sym_4*/45, } ) ),
		sitepos(24, 'd', "m.m m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_2*/24, /*Sym_2*/27, /*Sym_m*/30, /*Sym_m*/35, /*Sym_m*/38, } ) ),
		sitepos(8, 'c', "-43m", vector<int>( {/*Sym_m*/33, /*Sym_m*/34, /*Sym_m*/35, /*Sym_m*/36, /*Sym_m*/37, /*Sym_m*/38, /*Sym_minus4*/51, /*Sym_minus4*/53, /*Sym_minus4*/55, /*Sym_3*/63, /*Sym_3*/65, /*Sym_3*/67, /*Sym_3*/69, } ) ),
		sitepos(4, 'b', "m-3m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/23, /*Sym_2*/24, /*Sym_2*/25, /*Sym_2*/26, /*Sym_2*/27, /*Sym_2*/28, /*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/32, /*Sym_m*/33, /*Sym_m*/34, /*Sym_m*/35, /*Sym_m*/36, /*Sym_m*/37, /*Sym_m*/38, /*Sym_4*/41, /*Sym_4*/43, /*Sym_4*/45, /*Sym_minus4*/51, /*Sym_minus4*/53, /*Sym_minus4*/55, /*Sym_minus3*/73, /*Sym_minus3*/75, /*Sym_minus3*/77, /*Sym_minus3*/79, } ) ),
		sitepos(4, 'a', "m-3m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/23, /*Sym_2*/24, /*Sym_2*/25, /*Sym_2*/26, /*Sym_2*/27, /*Sym_2*/28, /*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/32, /*Sym_m*/33, /*Sym_m*/34, /*Sym_m*/35, /*Sym_m*/36, /*Sym_m*/37, /*Sym_m*/38, /*Sym_4*/41, /*Sym_4*/43, /*Sym_4*/45, /*Sym_minus4*/51, /*Sym_minus4*/53, /*Sym_minus4*/55, /*Sym_minus3*/73, /*Sym_minus3*/75, /*Sym_minus3*/77, /*Sym_minus3*/79, } ) ),
	},
	//Wyckoff Positions of Group 226
	{
		sitepos(192, 'j', "1", vector<int>( {} ) ),
		sitepos(96, 'i', "m..", vector<int>( {/*Sym_m*/30, } ) ),
		sitepos(96, 'h', "..2", vector<int>( {/*Sym_2*/24, } ) ),
		sitepos(64, 'g', ".3.", vector<int>( {/*Sym_3*/63, } ) ),
		sitepos(48, 'f', "4..", vector<int>( {/*Sym_4*/45, } ) ),
		sitepos(48, 'e', "mm2 . .", vector<int>( {/*Sym_2*/20, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(24, 'd', "4/m..", vector<int>( {/*Sym_minus1*/10, /*Sym_m*/30, /*Sym_4*/45, /*Sym_minus4*/55, } ) ),
		sitepos(24, 'c', "-4m. 2", vector<int>( {/*Sym_2*/24, /*Sym_2*/27, /*Sym_m*/31, /*Sym_m*/32, /*Sym_minus4*/55, } ) ),
		sitepos(8, 'b', "m-3.", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/32, /*Sym_minus3*/73, /*Sym_minus3*/75, /*Sym_minus3*/77, /*Sym_minus3*/79, } ) ),
		sitepos(8, 'a', "432", vector<int>( {/*Sym_2*/23, /*Sym_2*/24, /*Sym_2*/25, /*Sym_2*/26, /*Sym_2*/27, /*Sym_2*/28, /*Sym_4*/41, /*Sym_4*/43, /*Sym_4*/45, /*Sym_3*/63, /*Sym_3*/65, /*Sym_3*/67, /*Sym_3*/69, } ) ),
	},
	//Wyckoff Positions of Group 227
	{
		sitepos(192, 'i', "1", vector<int>( {} ) ),
		sitepos(96, 'h', "..2", vector<int>( {/*Sym_2*/27, } ) ),
		sitepos(96, 'g', "..m", vector<int>( {/*Sym_m*/33, } ) ),
		sitepos(48, 'f', "2.m m", vector<int>( {/*Sym_2*/20, /*Sym_m*/35, /*Sym_m*/38, } ) ),
		sitepos(32, 'e', ".3m", vector<int>( {/*Sym_m*/33, /*Sym_m*/34, /*Sym_m*/35, /*Sym_3*/63, } ) ),
		sitepos(16, 'd', ".-3m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/26, /*Sym_2*/27, /*Sym_2*/28, /*Sym_m*/33, /*Sym_m*/34, /*Sym_m*/35, /*Sym_minus3*/73, } ) ),
		sitepos(16, 'c', ".-3m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/26, /*Sym_2*/27, /*Sym_2*/28, /*Sym_m*/33, /*Sym_m*/34, /*Sym_m*/35, /*Sym_minus3*/73, } ) ),
		sitepos(8, 'b', "-43m", vector<int>( {/*Sym_m*/33, /*Sym_m*/34, /*Sym_m*/35, /*Sym_m*/36, /*Sym_m*/37, /*Sym_m*/38, /*Sym_minus4*/51, /*Sym_minus4*/53, /*Sym_minus4*/55, /*Sym_3*/63, /*Sym_3*/65, /*Sym_3*/67, /*Sym_3*/69, } ) ),
		sitepos(8, 'a', "-43m", vector<int>( {/*Sym_m*/33, /*Sym_m*/34, /*Sym_m*/35, /*Sym_m*/36, /*Sym_m*/37, /*Sym_m*/38, /*Sym_minus4*/51, /*Sym_minus4*/53, /*Sym_minus4*/55, /*Sym_3*/63, /*Sym_3*/65, /*Sym_3*/67, /*Sym_3*/69, } ) ),
	},
	//Wyckoff Positions of Group 228
	{
		sitepos(192, 'h', "1", vector<int>( {} ) ),
		sitepos(96, 'g', "..2", vector<int>( {/*Sym_2*/27, } ) ),
		sitepos(96, 'f', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(64, 'e', ".3.", vector<int>( {/*Sym_3*/63, } ) ),
		sitepos(48, 'd', "-4..", vector<int>( {/*Sym_minus4*/55, } ) ),
		sitepos(32, 'c', ".-3.", vector<int>( {/*Sym_minus1*/10, /*Sym_minus3*/73, } ) ),
		sitepos(32, 'b', ".32", vector<int>( {/*Sym_2*/26, /*Sym_2*/27, /*Sym_2*/28, /*Sym_3*/63, } ) ),
		sitepos(16, 'a', "23.", vector<int>( {/*Sym_2*/20, /*Sym_2*/21, /*Sym_2*/22, /*Sym_3*/63, /*Sym_3*/65, /*Sym_3*/67, /*Sym_3*/69, } ) ),
	},
	//Wyckoff Positions of Group 229
	{
		sitepos(96, 'l', "1", vector<int>( {} ) ),
		sitepos(48, 'k', "..m", vector<int>( {/*Sym_m*/33, } ) ),
		sitepos(48, 'j', "m..", vector<int>( {/*Sym_m*/30, } ) ),
		sitepos(48, 'i', "..2", vector<int>( {/*Sym_2*/27, } ) ),
		sitepos(24, 'h', "m.m 2", vector<int>( {/*Sym_2*/24, /*Sym_m*/30, /*Sym_m*/35, } ) ),
		sitepos(24, 'g', "mm2 . .", vector<int>( {/*Sym_2*/20, /*Sym_m*/31, /*Sym_m*/32, } ) ),
		sitepos(16, 'f', ".3m", vector<int>( {/*Sym_m*/33, /*Sym_m*/34, /*Sym_m*/35, /*Sym_3*/63, } ) ),
		sitepos(12, 'e', "4m. m", vector<int>( {/*Sym_m*/31, /*Sym_m*/32, /*Sym_m*/35, /*Sym_m*/38, /*Sym_4*/45, } ) ),
		sitepos(12, 'd', "-4m. 2", vector<int>( {/*Sym_2*/24, /*Sym_2*/27, /*Sym_m*/31, /*Sym_m*/32, /*Sym_minus4*/55, } ) ),
		sitepos(8, 'c', ".-3m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/26, /*Sym_2*/27, /*Sym_2*/28, /*Sym_m*/33, /*Sym_m*/34, /*Sym_m*/35, /*Sym_minus3*/73, } ) ),
		sitepos(6, 'b', "4/mm. m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/21, /*Sym_2*/22, /*Sym_2*/24, /*Sym_2*/27, /*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/32, /*Sym_m*/35, /*Sym_m*/38, /*Sym_4*/45, /*Sym_minus4*/55, } ) ),
		sitepos(2, 'a', "m-3m", vector<int>( {/*Sym_minus1*/10, /*Sym_2*/23, /*Sym_2*/24, /*Sym_2*/25, /*Sym_2*/26, /*Sym_2*/27, /*Sym_2*/28, /*Sym_m*/30, /*Sym_m*/31, /*Sym_m*/32, /*Sym_m*/33, /*Sym_m*/34, /*Sym_m*/35, /*Sym_m*/36, /*Sym_m*/37, /*Sym_m*/38, /*Sym_4*/41, /*Sym_4*/43, /*Sym_4*/45, /*Sym_minus4*/51, /*Sym_minus4*/53, /*Sym_minus4*/55, /*Sym_minus3*/73, /*Sym_minus3*/75, /*Sym_minus3*/77, /*Sym_minus3*/79, } ) ),
	},
	//Wyckoff Positions of Group 230
	{
		sitepos(96, 'h', "1", vector<int>( {} ) ),
		sitepos(48, 'g', "..2", vector<int>( {/*Sym_2*/27, } ) ),
		sitepos(48, 'f', "2..", vector<int>( {/*Sym_2*/20, } ) ),
		sitepos(32, 'e', ".3.", vector<int>( {/*Sym_3*/63, } ) ),
		sitepos(24, 'd', "-4..", vector<int>( {/*Sym_minus4*/55, } ) ),
		sitepos(24, 'c', "2.2 2", vector<int>( {/*Sym_2*/20, /*Sym_2*/24, /*Sym_2*/27, } ) ),
		sitepos(16, 'b', ".32", vector<int>( {/*Sym_2*/26, /*Sym_2*/27, /*Sym_2*/28, /*Sym_3*/63, } ) ),
		sitepos(16, 'a', ".-3.", vector<int>( {/*Sym_minus1*/10, /*Sym_minus3*/73, } ) ),
	},
};
