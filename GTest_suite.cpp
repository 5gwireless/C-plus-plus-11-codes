1. sudo apt install libgtest-dev    //install the Gtest


2.  // Compile with the commands CMAKE and MAKE
	cd /usr/src/gtest
	sudo cmake CMakeLists.txt
	sudo make  
	
3.  // Copy the generated binaries to your library Folder

    sudo cp libgtest.a libgtest_main.a /usr/lib


4.  // Create a simple program to check its working


//====================
// calc.hpp
#ifndef CALC_HPP_
#define CALC_HPP_

int add(int op1, int op2);
int sub(int op1, int op2);

#endif // CALC_HPP_
//=====================
// calc.cpp
#include "calc.hpp"

int add(int op1, int op2)
{
    return op1 + op2;
}

int sub(int op1, int op2)
{
    return op1 - op2;
}
//==========================

// calc_test.cpp
#include <gtest/gtest.h>
#include "calc.hpp"

TEST(CalcTest, Add)
{
 ASSERT_EQ(2, add(1, 1));
 ASSERT_EQ(5, add(3, 2));
 ASSERT_EQ(10, add(7, 3));
}

TEST(CalcTest, Sub)
{
 ASSERT_EQ(3, sub(5, 2));
 ASSERT_EQ(-10, sub(5, 15));
}

int main(int argc, char **argv)
{
 testing::InitGoogleTest(&argc, argv);
 return RUN_ALL_TESTS();
}

//====================================


5. // Compilation of the code
  // Note that you must link the code against the GTest (gtest) and the POSIX Thread (pthread) libs.

g++ -o calc_test calc_test.cpp calc.cpp -lgtest -lpthread

//====================================

6. //Running the code
./calc_test
