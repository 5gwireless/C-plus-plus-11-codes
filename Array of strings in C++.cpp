// https://www.geeksforgeeks.org/array-strings-c-3-different-ways-create/ 


//Array of Strings in C++ (5 Different Ways to Create)


/*1. Using Pointers: 
We actually create an array of string literals by creating an array of pointers.
This is supported by both C and C++. 
*/

// C++ program to demonstrate array of strings using
// 2D character array
#include <iostream>

int main()
{
	// Initialize array of pointer
	const char *colour[4] = { "Blue", "Red", 
							"Orange", "Yellow" };

	// Printing Strings stored in 2D array
	for (int i = 0; i < 4; i++)
		std::cout << colour[i] << "\n";

	return 0;
}

//====================================================================================

/*2. Using the string class:
The STL string class may be used to create an array of mutable strings. In this method, the size of the string is not fixed, and the strings can be changed. 

This is supported only in C++, as C does not have classes.
*/

// C++ program to demonstrate array of strings using
// array of strings.
#include <iostream>
#include <string>

int main()
{
	// Initialize String Array
	std::string colour[4] = { "Blue", "Red",
							"Orange", "Yellow" };

	// Print Strings
	for (int i = 0; i < 4; i++)
		std::cout << colour[i] << "\n";
}

//====================================================================================
/*3. Using the vector class:
The STL container Vector can be used to dynamically allocate an array that can vary in size.

This is only usable in C++, as C does not have classes. Note that the initializer-list syntax here requires a compiler that 
supports the 2011 C++ standard, and though it is quite likely your compiler does, it is something to be aware of. */

// C++ program to demonstrate vector of strings using
#include <iostream>
#include <vector>
#include <string>

int main()
{
	// Declaring Vector of String type
	// Values can be added here using initializer-list syntax
	std::vector<std::string> colour {"Blue", "Red", "Orange"};

	// Strings can be added at any time with push_back
	colour.push_back("Yellow");

	// Print Strings stored in Vector
	for (int i = 0; i < colour.size(); i++)
		std::cout << colour[i] << "\n";
}
