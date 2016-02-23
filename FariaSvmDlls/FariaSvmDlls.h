// The following ifdef block is the standard way of creating macros which make exporting 
// from a DLL simpler. All files within this DLL are compiled with the FARIASVMDLLS_EXPORTS
// symbol defined on the command line. This symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see 
// FARIASVMDLLS_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#ifdef FARIASVMDLLS_EXPORTS
#define FARIASVMDLLS_API __declspec(dllexport)
#else
#define FARIASVMDLLS_API __declspec(dllimport)
#endif

// This class is exported from the FariaSvmDlls.dll
class FARIASVMDLLS_API CFariaSvmDlls {
public:
	CFariaSvmDlls(void);
	int fnFariaSvmDlls();
	// TODO: add your methods here.
};

extern FARIASVMDLLS_API int nFariaSvmDlls;

FARIASVMDLLS_API int fnFariaSvmDlls(void);
