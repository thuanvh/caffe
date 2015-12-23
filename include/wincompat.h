#if !defined _HEADER_WIN_COMPAT_20140627_INCLUDED_
#define _HEADER_WIN_COMPAT_20140627_INCLUDED_

typedef unsigned int uint;
#define snprintf _snprintf

#include <process.h>
#define getpid _getpid
//#define signbit(x) ((x)<0?true:false)
#include <direct.h>
#define mkdir(x,a) _mkdir((x))
#include <io.h>
#endif //_HEADER_WIN_COMPAT_20140627_INCLUDED_
