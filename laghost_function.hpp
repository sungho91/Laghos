#include "mfem.hpp"
#include <ctime>
#include <cstdlib>

namespace mfem
{
   double e0(const Vector &);
   double p0(const Vector &);
   double depth0(const Vector &);
   double rho0(const Vector &);
   double gamma_func(const Vector &);
   void v0(const Vector &, Vector &);
   void xyz0(const Vector &, Vector &);
   double x_l2(const Vector &);
   double y_l2(const Vector &);
   double z_l2(const Vector &);

   class PlasticCoefficient : public VectorCoefficient
   {
   private:
      ParGridFunction &xyz;
      int dim;
      Vector location;
      double rad, ini_pls;

   public:
      PlasticCoefficient (int &_dim, ParGridFunction &_xyz, Vector &_location, double &_rad, double &_ini_pls)
         : VectorCoefficient(_dim), xyz(_xyz)
         {
            dim=_dim; location = _location; rad = _rad; ini_pls = _ini_pls;
         }
      virtual void Eval(Vector &K, ElementTransformation &T, const IntegrationPoint &ip)
      {
         K.SetSize(1);
         double r = 0.0;
         double xc = xyz.GetValue(T, ip,1);
         double yc = xyz.GetValue(T, ip,2);
         double zc = 0.0;
         double randomNumber;
         
         if(dim == 3){zc = xyz.GetValue(T, ip,3);}
         
         if(dim == 2){ r = sqrt(pow((xc-location[0]), 2) + pow((yc-location[1]), 2));}
         else if(dim == 3){r = sqrt(pow((xc-location[0]), 2) + pow((yc-location[1]), 2) + pow((zc-location[2]), 2));}

         if(r <= rad)
         {
            K(0) = ini_pls;
         }
         else
         {
            K(0) = 0.0;
         }

         // srand(time(0));
         // randomNumber =  (rand() % 101)/100.0;
         // if((xc >= 97.5e3) & (xc <= 102.5e3) & (yc <= 5e3))
         // // if((xc >= 80e3) & (xc <= 120e3))
         // // if(xc <= 100e3)
         // {
         //    // K(0) = ini_pls*randomNumber;
         //    K(0) = ini_pls;
         //    // std::cout << randomNumber << std::endl;
         // }
         // else
         // {
         //    K(0) = 0.0;
         // }

         // K(0) = ini_pls*randomNumber;
         
      }
      virtual ~PlasticCoefficient() { }
   };

   class LithostaticCoefficient : public VectorCoefficient
   {
   private:
      ParGridFunction &xyz, &rho;
      int dim;
      double gravity, thickness;

   public:
      LithostaticCoefficient (int &_dim, ParGridFunction &_xyz, ParGridFunction &_rho, double &_gravity, double &_thickness)
         : VectorCoefficient(_dim), xyz(_xyz), rho(_rho)  
         {
            dim=_dim; gravity = _gravity; thickness = _thickness; 
         }
      virtual void Eval(Vector &K, ElementTransformation &T, const IntegrationPoint &ip)
      {
         K.SetSize(3*(dim-1));

         double xc = xyz.GetValue(T, ip, 1);
         double zc = xyz.GetValue(T, ip, dim);
         // double zc = z.GetValue(T, ip);
         double denc = rho.GetValue(T, ip);
         double atm = -101325; // 1 atm in Pa

         double radians = 10.0 * (M_PI / 180.0);
         thickness = 35e3;

         K = 0.0;
         if(dim == 2)
         {
            K(0) = -1.0*fabs(thickness - zc)*denc*gravity + atm;
            K(1) = -1.0*fabs(thickness - zc)*denc*gravity + atm;

            if(xc > 300e3)
            {
               thickness = 35e3 + (xc-300e3) * tan(radians);

               K(0) = -1.0*fabs(thickness - zc)*1560.0*gravity + atm;
               K(1) = -1.0*fabs(thickness - zc)*1560.0*gravity + atm;
            }
         }
         else if(dim ==3)
         {
            K(0) = -1.0*fabs(thickness - zc)*denc*gravity + atm;
            K(1) = -1.0*fabs(thickness - zc)*denc*gravity + atm;
            K(2) = -1.0*fabs(thickness - zc)*denc*gravity + atm;
         }
      }
      virtual ~LithostaticCoefficient() { }
   };

   class ATMCoefficient : public VectorCoefficient
   {
   private:
      ParGridFunction &xyz, &rho;
      int dim;
      double gravity, thickness;

   public:
      ATMCoefficient (int &_dim, ParGridFunction &_xyz, ParGridFunction &_rho, double &_gravity, double &_thickness)
         : VectorCoefficient(_dim), xyz(_xyz), rho(_rho)  
         {
            dim=_dim; gravity = _gravity; thickness = _thickness; 
         }
      virtual void Eval(Vector &K, ElementTransformation &T, const IntegrationPoint &ip)
      {
         K.SetSize(3*(dim-1));

         double atm = -101325; // 1 atm in Pa
         double zc = xyz.GetValue(T, ip, dim);
         
         K = 0.0;
         if(dim == 2)
         {
            K(0) = atm;
            K(1) = atm;

         }
         else if(dim ==3)
         {
            K(0) = atm;
            K(1) = atm;
            K(2) = atm;
         }
      }
      virtual ~ATMCoefficient() { }
   };

   class CompoCoefficient : public VectorCoefficient
   {
   private:
      ParGridFunction &xyz;
      int dim;
      double thickness;

   public:
      CompoCoefficient (int &_dim, ParGridFunction &_xyz, double &_thickness)
         : VectorCoefficient(_dim), xyz(_xyz)  
         {
            dim=_dim; thickness = _thickness; 
         }
      virtual void Eval(Vector &K, ElementTransformation &T, const IntegrationPoint &ip)
      {
         K.SetSize(3*(dim-1));

         double zc = xyz.GetValue(T, ip, dim);
         
         K = 0.0;
         if(zc < 5e3)
         {
            K(0) = 1.0;

         }
         else if(zc <= 10e3)
         {
            K(1) = 1.0;
         }
         else if(zc < 15e3)
         {
            K(0) = 1.0;
         }
         else if(zc < 20e3)
         {
            K(2) = 1.0;
         }
         else if(zc < 25e3)
         {
            K(0) = 1.0;
         }
         else if(zc < 30e3)
         {
            K(2) = 1.0;
         }
         else
         {
            K(0) = 1.0;
         }
      }
      virtual ~CompoCoefficient() { }
   };


   // class IcVelCoefficient : public VectorCoefficient
   // {
   // private:
   //    ParGridFunction &xyz, &rho;
   //    int dim;
   //    double gravity, thickness;

   // public:
   //    IcVelCoefficient (int &_dim, ParGridFunction &_xyz, Vector &_location, double &_rad, double &_ini_pls)
   //       : VectorCoefficient(_dim), xyz(_xyz)
   //       {
   //          dim=_dim; location = _location; rad = _rad; ini_pls = _ini_pls;
   //       }
   //    virtual void Eval(Vector &K, ElementTransformation &T, const IntegrationPoint &ip)
   //    {
   //       K.SetSize(dim);
   //       double r = 0.0;
   //       double xc = xyz.GetValue(T, ip,1);
   //       double yc = xyz.GetValue(T, ip,2);
   //       double zc = 0.0;
   //       if(dim == 3){zc = xyz.GetValue(T, ip,3);}
         
   //    }
   //    virtual ~IcVelCoefficient() { }
   // };

   // class PlasticCoefficient : public VectorCoefficient
   // {
   // private:
   //    ParGridFunction &x, &y, &z;
   //    int dim;
   //    Vector location;
   //    double rad, ini_pls;

   // public:
   //    PlasticCoefficient (int &_dim, ParGridFunction &_x, ParGridFunction &_y, ParGridFunction &_z, Vector &_location, double &_rad, double &_ini_pls)
   //       : VectorCoefficient(_dim), x(_x), y(_y), z(_z)  
   //       {
   //          dim=_dim; location = _location; rad = _rad; ini_pls = _ini_pls;
   //       }
   //    virtual void Eval(Vector &K, ElementTransformation &T, const IntegrationPoint &ip)
   //    {
   //       K.SetSize(1);
   //       double r = 0.0;
   //       double xc = x.GetValue(T, ip);
   //       double yc = y.GetValue(T, ip);
   //       double zc = z.GetValue(T, ip);

   //       if(dim == 2){ r = sqrt(pow((xc-location[0]), 2) + pow((yc-location[1]), 2));}
   //       else if(dim == 3){r = sqrt(pow((xc-location[0]), 2) + pow((yc-location[1]), 2) + pow((zc-location[2]), 2));}

   //       if(r <= rad)
   //       {
   //          K(0) = ini_pls;
   //       }
   //       else
   //       {
   //          K(0) = 0.0;
   //       }
   //    }
   //    virtual ~PlasticCoefficient() { }
   // };

   // class PlasticCoefficient : public VectorCoefficient
   // {
   // private:
   //    ParGridFunction &x, &y, &z;
   //    int dim;
   //    Vector location;
   //    double rad, ini_pls;

   // public:
   //    PlasticCoefficient (int &_dim, ParGridFunction &_x, ParGridFunction &_y, ParGridFunction &_z, Vector &_location, double &_rad, double &_ini_pls)
   //       : VectorCoefficient(_dim), x(_x), y(_y), z(_z)  
   //       {
   //          dim=_dim; location = _location; rad = _rad; ini_pls = _ini_pls;
   //       }
   //    virtual void Eval(Vector &K, ElementTransformation &T, const IntegrationPoint &ip)
   //    {
   //       K.SetSize(1);
   //       double r = 0.0;
   //       double xc = x.GetValue(T, ip);
   //       double yc = y.GetValue(T, ip);
   //       double zc = z.GetValue(T, ip);

   //       if(dim == 2){ r = sqrt(pow((xc-location[0]), 2) + pow((yc-location[1]), 2));}
   //       else if(dim == 3){r = sqrt(pow((xc-location[0]), 2) + pow((yc-location[1]), 2) + pow((zc-location[2]), 2));}

   //       if(r <= rad)
   //       {
   //          K(0) = ini_pls;
   //       }
   //       else
   //       {
   //          K(0) = 0.0;
   //       }
   //    }
   //    virtual ~PlasticCoefficient() { }
   // };

   // class LithostaticCoefficient : public VectorCoefficient
   // {
   // private:
   //    ParGridFunction &y, &z, &rho;
   //    int dim;
   //    double gravity, thickness;

   // public:
   //    LithostaticCoefficient (int &_dim, ParGridFunction &_y, ParGridFunction &_z, ParGridFunction &_rho, double &_gravity, double &_thickness)
   //       : VectorCoefficient(_dim), y(_y), z(_z), rho(_rho)  
   //       {
   //          dim=_dim; gravity = _gravity; thickness = _thickness; 
   //       }
   //    virtual void Eval(Vector &K, ElementTransformation &T, const IntegrationPoint &ip)
   //    {
   //       K.SetSize(3*(dim-1));

   //       double yc = y.GetValue(T, ip);
   //       double zc = z.GetValue(T, ip);
   //       double denc = rho.GetValue(T, ip);

   //       if(dim == 2)
   //       {
   //          K(0) = -1.0*fabs(thickness - yc)*denc*gravity;
   //          K(1) = -1.0*fabs(thickness - yc)*denc*gravity;
   //       }
   //       else if(dim ==3)
   //       {
   //          K(0) = -1.0*fabs(thickness - zc)*denc*gravity;
   //          K(1) = -1.0*fabs(thickness - zc)*denc*gravity;
   //          K(2) = -1.0*fabs(thickness - zc)*denc*gravity;
   //       }
   //    }
   //    virtual ~LithostaticCoefficient() { }
   // };
}
