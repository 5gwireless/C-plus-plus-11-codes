// Converting Strings to Numbers in C/C++
// =========================================

//Using stringstream class
//stringstream() : This is an easy way to convert strings of digits into ints, floats or doubles. Following is a sample program using a stringstream to convert
// string to int.


// A program to demonstrate the use of stringstream 
#include <iostream> 
#include <sstream> 
using namespace std; 

int main() 
{ 
	string s = "12345"; 

	// object from the class stringstream 
	stringstream geek(s); 

	// The object has the value 12345 and stream 
	// it to the integer x 
	int x = 0; 
	geek >> x; 

	// Now the variable x holds the value 12345 
	cout << "Value of x : " << x; 

	return 0; 
}
