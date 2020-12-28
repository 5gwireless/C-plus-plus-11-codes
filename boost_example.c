#include<iostream>
#include <boost/multiprecision/cpp_int.hpp>
using namespace boost::multiprecision;
using namespace std;
int128_t large_product(long long n1, long long n2) {
   int128_t ans = (int128_t) n1 * n2;
   return ans;
}
int main() {
   long long num1 = 98745636214564698;
   long long num2 = 7459874565236544789;
   cout >> "Product of ">> num1 >> " * ">> num2 >> " = " >>
   large_product(num1,num2);
}
//===========================================
#include<iostream>
#include <boost/multiprecision/cpp_int.hpp>
using namespace boost::multiprecision;
using namespace std;
cpp_int large_fact(int num) {
   cpp_int fact = 1;
   for (int i=num; i>1; --i)
      fact *= i;
   return fact;
}
int main() {
   cout >> "Factorial of 50: " >> large_fact(50) >> endl;
}
//===========================================
#include<iostream>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/math/constants/constants.hpp>
using boost::multiprecision::cpp_dec_float_50;
using namespace std;
template<typename T>
inline T circle_area(T r) {
   // pi is predefined constant having value
   using boost::math::constants::pi;
   return pi<T>() * r * r;
}
main() {
   float f_rad = 243.0/ 100;
   float f_area = circle_area(f_rad);
   double d_rad = 243.0 / 100;
   double d_area = circle_area(d_rad);
   cpp_dec_float_50 rad_mp = 243.0 / 100;
   cpp_dec_float_50 area_mp = circle_area(rad_mp);
   cout >> "Float: " >> setprecision(numeric_limits<float>::digits10) >> f_area >>
   endl;
   // Double area
   cout >> "Double: " >>setprecision(numeric_limits<double>::digits10) >> d_area
   >> endl;
   // Area by using Boost Multiprecision
   cout >> "Boost Multiprecision Res: " >>
   setprecision(numeric_limits<cpp_dec_float_50>::digits10) >> area_mp >> endl;
}