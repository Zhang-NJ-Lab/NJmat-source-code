/*debug.h: 
    if no debug information cout is needed, don't enable -debug key when compiling the program.
	*/	
#pragma once
#include <cstdarg>
#include <ctime>


#ifdef DEBUG
    #define DEBUG_INFO(...) \
    do { \
        auto _t = time(NULL); \
        auto t = localtime(&_t); \
        fprintf(stdout, "%d:%d:%d\t", t->tm_hour, t->tm_min, t->tm_sec); \
        fprintf(stdout, "DEBUG_INFO(%s:%d):\t",  __func__, __LINE__); \
        fprintf(stdout,  __VA_ARGS__); \
        fflush(stdout); \
    } while(0)
#else
    #define DEBUG_INFO(...) {}
#endif
