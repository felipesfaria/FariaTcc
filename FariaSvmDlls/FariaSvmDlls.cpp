// FariaSvmDlls.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include "FariaSvmDlls.h"


// This is an example of an exported variable
FARIASVMDLLS_API int nFariaSvmDlls=0;

// This is an example of an exported function.
FARIASVMDLLS_API int fnFariaSvmDlls(void)
{
	return 42;
}

// This is the constructor of a class that has been exported.
// see FariaSvmDlls.h for the class definition
CFariaSvmDlls::CFariaSvmDlls()
{
	return;
}

int CFariaSvmDlls::fnFariaSvmDlls()
{
	return 42;
}
