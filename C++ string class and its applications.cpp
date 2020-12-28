 //C++ string class and its applications

/*
In C++ we can store string by one of the two ways –

1- C style strings
2- string class (discussed in this post)


In this post, the second method is discussed. string class is part of C++ library that supports a lot much functionality over C style strings.
*/


As string class is a container class, we can iterate over all its characters using an iterator similar to other containers like vector, set and maps, but 
generally, we use a simple for loop for iterating over the characters and index them using [] operator.
C++ string class has a lot of functions to handle string easily. Most useful of them are demonstrated in below code.



// C++ program to demonstrate various function string class 
#include <bits/stdc++.h> 
using namespace std; 

int main() 
{ 
	// various constructor of string class 

	// initialization by raw string 
	string str1("first string"); 

	// initialization by another string 
	string str2(str1); 

	// initialization by character with number of occurrence 
	string str3(5, '#'); 

	// initialization by part of another string 
	string str4(str1, 6, 6); // from 6th index (second parameter) 
							// 6 characters (third parameter) 

	// initialization by part of another string : iteartor version 
	string str5(str2.begin(), str2.begin() + 5); 

	cout << str1 << endl; 
	cout << str2 << endl; 
	cout << str3 << endl; 
	cout << str4 << endl; 
	cout << str5 << endl; 

	// assignment operator 
	string str6 = str4; 

	// clear function deletes all character from string 
	str4.clear(); 

	// both size() and length() return length of string and 
	// they work as synonyms 
	int len = str6.length(); // Same as "len = str6.size();" 

	cout << "Length of string is : " << len << endl; 

	// a particular character can be accessed using at / 
	// [] operator 
	char ch = str6.at(2); // Same as "ch = str6[2];" 


	cout << "third character of string is : " << ch << endl; 

	// front return first character and back returns last character 
	// of string 

	char ch_f = str6.front(); // Same as "ch_f = str6[0];" 
	char ch_b = str6.back(); // Same as below 
							// "ch_b = str6[str6.length() - 1];" 

	cout << "First char is : " << ch_f << ", Last char is : "
		<< ch_b << endl; 

	// c_str returns null terminated char array version of string 
	const char* charstr = str6.c_str(); 
	printf("%s\n", charstr); 

	// append add the argument string at the end 
	str6.append(" extension"); 
	// same as str6 += " extension" 

	// another version of append, which appends part of other 
	// string 
	str4.append(str6, 0, 6); // at 0th position 6 character 

	cout << str6 << endl; 
	cout << str4 << endl; 

	// find returns index where pattern is found. 
	// If pattern is not there it returns predefined 
	// constant npos whose value is -1 

	if (str6.find(str4) != string::npos) 
		cout << "str4 found in str6 at " << str6.find(str4) 
			<< " pos" << endl; 
	else
		cout << "str4 not found in str6" << endl; 

	// substr(a, b) function returns a substring of b length 
	// starting from index a 
	cout << str6.substr(7, 3) << endl; 

	// if second argument is not passed, string till end is 
	// taken as substring 
	cout << str6.substr(7) << endl; 

	// erase(a, b) deletes b characters at index a 
	str6.erase(7, 4); 
	cout << str6 << endl; 

	// iterator version of erase 
	str6.erase(str6.begin() + 5, str6.end() - 3); 
	cout << str6 << endl; 

	str6 = "This is a examples"; 

	// replace(a, b, str) replaces b characters from a index by str 
	str6.replace(2, 7, "ese are test"); 

	cout << str6 << endl; 

	return 0; 
} 


/*
As seen in the above code, we can get the length of the string by size() as well as length() but length() is preferred for strings. We can concat a string
to another string by += or by append(), but += is slightly slower than append() because each time + is called a new string (creation of new buffer) is made 
which is returned that is a bit overhead in case of many append operation.
*/


/////////////////////////////////////////////////////////////////

//https://www.tutorialspoint.com/cplusplus/cpp_strings.htm

//The String Class in C++
//The standard C++ library provides a string class type that supports all the operations mentioned above, additionally much more functionality. Let us check the following example −

Live Demo
#include <iostream>
#include <string>

using namespace std;

int main () {

   string str1 = "Hello";
   string str2 = "World";
   string str3;
   int  len ;

   // copy str1 into str3
   str3 = str1;
   cout << "str3 : " << str3 << endl;

   // concatenates str1 and str2
   str3 = str1 + str2;
   cout << "str1 + str2 : " << str3 << endl;

   // total length of str3 after concatenation
   len = str3.size();
   cout << "str3.size() :  " << len << endl;

   return 0;
}