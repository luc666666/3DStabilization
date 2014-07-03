#ifndef ROFMODEL_H
#define ROFMODEL_H

// includes
#include <list>

#include <iudefs.h>

#ifdef WIN32
#pragma warning( disable : 4251 ) // disable the warning about exported template code from stl
#pragma warning( disable : 4231 ) // disable the warning about nonstandard extension in e.g. istream

  #ifdef ROF_USE_STATIC
    #define ROF_DLLAPI
  #else
    #ifdef ROF_DLL_EXPORTS
      #define ROF_DLLAPI __declspec(dllexport)
    #else
      #define ROF_DLLAPI __declspec(dllimport)
    #endif
  #endif
#else
  #define ROF_DLLAPI
#endif

class ROF_DLLAPI ROFModel
{

public:
  ROFModel();
  ~ROFModel();

  void solve();

  bool setF(iu::ImageGpu_32f_C1* f);
  bool setU(iu::ImageGpu_32f_C1* u);

  void setLambda(float val) {m_lambda = val;}
  void setDiscr(int val)  {m_discr = val;}
  void setMaxIter(int val){m_max_iter = val;}
  void setCheck(int val)  {m_check = val;}
  void setTol(float val)    {m_tol = val;}
  void setVerbose(bool val) {m_verbose = val;}

  std::list<double> getPrimalEnergyLog(){return m_log_primal_energy;};
  std::list<double> getDualEnergyLog()  {return m_log_dual_energy;};
  std::list<int>    getIterationLog()   {return m_log_iterations;};
  std::list<double> getTimeLog()        {return m_log_time;};

protected:
  // IO images
  iu::ImageGpu_32f_C1* m_f;
  iu::ImageGpu_32f_C1* m_u;

  // internal images
  iu::ImageGpu_32f_C1* m_u_;
  iu::ImageGpu_32f_C2* m_p;
  iu::ImageGpu_32f_C4* m_p_acc;

  // parameters
  float m_lambda;
  int   m_discr;
  int   m_max_iter;
  int   m_check;
  float m_tol;
  bool  m_verbose;

  // logs
  std::list<double> m_log_primal_energy;
  std::list<double> m_log_dual_energy;
  std::list<int>    m_log_iterations;
  std::list<double> m_log_time;

  // internal variables
  bool m_realloc;
};

#endif // ROFMODEL_H
