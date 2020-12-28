// https://www.geeksforgeeks.org/stdstring-class-in-c/

/* Operations on strings
==============================
1. getline() :- This function is used to store a stream of characters as entered by the user in the object memory.
2. push_back() :- This function is used to input a character at the end of the string.
3. pop_back() :- Introduced from C++11(for strings), this function is used to delete the last character from the string.
 */


// C++ code to demonstrate the working of 
// getline(), push_back() and pop_back() 

#include<iostream> 
#include<string> // for string class 
using namespace std; 
int main() 
{ 
	// Declaring string 
	string str; 

	// Taking string input using getline() 
	// "geeksforgeek" in givin output 
	getline(cin,str); 

	// Displaying string 
	cout << "The initial string is : "; 
	cout << str << endl; 

	// Using push_back() to insert a character 
	// at end 
	// pushes 's' in this case 
	str.push_back('s'); 

	// Displaying string 
	cout << "The string after push_back operation is : "; 
	cout << str << endl; 

	// Using pop_back() to delete a character 
	// from end 
	// pops 's' in this case 
	str.pop_back(); 

	// Displaying string 
	cout << "The string after pop_back operation is : "; 
	cout << str << endl; 

	return 0; 

} 

The initial string is : geeksforgeek
The string after push_back operation is : geeksforgeeks
The string after pop_back operation is : geeksforgeek

////////////////////////////////////////////////////////////////////////////////////////////////////
/*
Capacity Functions
=====================
4. capacity() :- This function returns the capacity allocated to the string, which can be equal to or more than the size of the string. Additional space is allocated so that when the new characters are added to the string, the operations can be done efficiently.
5. resize() :- This function changes the size of string, the size can be increased or decreased.
6.length():-This function finds the length of the string
7.shrink_to_fit() :- This function decreases the capacity of the string and makes it equal to the minimum capacity of the string. This operation is useful to save additional memory if we are sure that no further addition of characters have to be made.
*/


// C++ code to demonstrate the working of 
// capacity(), resize() and shrink_to_fit() 
#include<iostream> 
#include<string> // for string class 
using namespace std; 
int main() 
{ 
	// Initializing string 
	string str = "geeksforgeeks is for geeks"; 

	// Displaying string 
	cout << "The initial string is : "; 
	cout << str << endl; 

	// Resizing string using resize() 
	str.resize(13); 

	// Displaying string 
	cout << "The string after resize operation is : "; 
	cout << str << endl; 

	// Displaying capacity of string 
	cout << "The capacity of string is : "; 
	cout << str.capacity() << endl; 

	//Displaying length of the string 
	cout<<"The length of the string is :"<<str.length()<<endl; 

	// Decreasing the capacity of string 
	// using shrink_to_fit() 
	str.shrink_to_fit(); 

	// Displaying string 
	cout << "The new capacity after shrinking is : "; 
	cout << str.capacity() << endl; 

	return 0; 

} 
//////////////////////////////////////////////////////////////////////////////////

/*Iterator Functions

8. begin() :- This function returns an iterator to beginning of the string.
9. end() :- This function returns an iterator to end of the string.
10. rbegin() :- This function returns a reverse iterator pointing at the end of string.
11. rend() :- This function returns a reverse iterator pointing at beginning of string.
*/


// C++ code to demonstrate the working of 
// begin(), end(), rbegin(), rend() 
#include<iostream> 
#include<string> // for string class 
using namespace std; 
int main() 
{ 
	// Initializing string` 
	string str = "geeksforgeeks"; 

	// Declaring iterator 
	std::string::iterator it; 

	// Declaring reverse iterator 
	std::string::reverse_iterator it1; 

	// Displaying string 
	cout << "The string using forward iterators is : "; 
	for (it=str.begin(); it!=str.end(); it++) 
	cout << *it; 
	cout << endl; 

	// Displaying reverse string 
	cout << "The reverse string using reverse iterators is : "; 
	for (it1=str.rbegin(); it1!=str.rend(); it1++) 
	cout << *it1; 
	cout << endl; 

	return 0; 

} 
