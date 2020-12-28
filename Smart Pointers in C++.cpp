#Importance of pointer in C/C++:
#================================
Pointers are used for accessing the resources which are external to the program like heap memory. 
So for accessing the heap memory if anything is created inside heap memory Pointers are used.


#Problem with normal pointer:
#================================
Let’s understand what’s the main problem with the normal pointer by taking a small C++ program using this illustration.


#include <iostream> 
using namespace std; 

class Rectangle { 
private: 
	int length; 
	int breadth; 
}; 

void fun() 
{ 
	// By taking a pointer p and 
	// dynamically creating object 
	// of class rectangle 
	Rectangle* p = new Rectangle(); 
} 

int main() 
{ 
	// Infinite Loop 
	while (1) { 
		fun(); 
	} 
} 

So what happens is it’ll have a pointer ‘p’ and this will be pointing to an object of type rectangle which will have length and breadth. 
Once the function ends this ‘p’ is deleted because p is a local variable to the function which will end but a new rectangle that is allocated 
inside heap that will not be deallocated.
So every time it will create an object but not it’s deleting so this is causing leakage of memory from the heap memory.
So at the last of the fun() we should use ‘delete p’ if we do not mention this, this will cause a very severe problem. so because of the laziness or carelessness of the programmer this type of problem may arise. 
So to help the programmer C++ 11 takes responsibility and introduces smart pointers.

#=====================================

#Introduction of Smart Pointers
#	The problem with heap memory is that when you don’t need it you must deallocate itself. So mostly the programmers are too lazy in writing the 
#	code for deallocation of objects and that causes severe problem like memory leak which will cause the program to crash. So the languages like Java, 
#	C#, .Net Framework they provide a garbage collection mechanism to deallocate the object which is not in use. So in C++ 11, it introduces smart pointers
#	that automatically manage memory and they will deallocate the object when they are not in use when the pointer is going out of scope automatically it’ll 
#	deallocate the memory.

MyClass* ptr = new MyClass(); 
ptr->doSomething(); 
// We must do delete(ptr) to avoid memory leak 


# Since the destructor is automatically called when an object goes out of scope, the dynamically allocated memory would automatically be deleted 
#(or reference count can be decremented). 
# Consider the following simple smart ptr class.
# Here, we are writing our own Smart Pointer class for demonstaration

#include <iostream> 
using namespace std; 

class SmartPtr { 
	int* ptr; // Actual pointer 
public: 
	// Constructor: Refer https:// www.geeksforgeeks.org/g-fact-93/ 
	// for use of explicit keyword 
	explicit SmartPtr(int* p = NULL) { ptr = p; } 

	// Destructor 
	~SmartPtr() { delete (ptr); } 

	// Overloading dereferencing operator 
	int& operator*() { return *ptr; } 
}; 

int main() 
{ 
	SmartPtr ptr(new int()); 
	*ptr = 20; 
	cout << *ptr; 

	// We don't need to call delete ptr: when the object 
	// ptr goes out of scope, the destructor for it is automatically 
	// called and destructor does delete ptr. 

	return 0; 
}
#=================================================================

#Types of Smart Pointer:

#unique_ptr
#-------------------
#If you are using a unique pointer then if one object is created and pointer P1 is pointing to this one them only one pointer can point this one 
#at one time. So we can’t share with another pointer, but we can transfer the control to P2 by removing P1.


#include <iostream> 
using namespace std; 
#include <memory> 

class Rectangle { 
	int length; 
	int breadth; 

public: 
	Rectangle(int l, int b) 
	{ 
		length = l; 
		breadth = b; 
	} 

	int area() 
	{ 
		return length * breadth; 
	} 
}; 

int main() 
{ 

	unique_ptr<Rectangle> P1(new Rectangle(10, 5)); 
	cout << P1->area() << endl; // This'll print 50 

	// unique_ptr<Rectangle> P2(P1); 

	unique_ptr<Rectangle> P2; 
	P2 = move(P1); 

	// This'll print 50 
	cout << P2->area() << endl; 

	// cout<<P1->area()<<endl; 
	return 0; 
} 

#========================================================================
# shared_ptr
# If you are using shared_ptr then more than one pointer can point to this one object at a time and it’ll maintain a Reference Counter 
# using use_count() method.

#include <iostream> 
using namespace std; 
#include <memory> 

class Rectangle { 
	int length; 
	int breadth; 

public: 
	Rectangle(int l, int b) 
	{ 
		length = l; 
		breadth = b; 
	} 

	int area() 
	{ 
		return length * breadth; 
	} 
}; 

int main() 
{ 

	shared_ptr<Rectangle> P1(new Rectangle(10, 5)); 
	// This'll print 50 
	cout << P1->area() << endl; 

	shared_ptr<Rectangle> P2; 
	P2 = P1; 

	// This'll print 50 
	cout << P2->area() << endl; 

	// This'll now not give an error, 
	cout << P1->area() << endl; 

	// This'll also print 50 now 
	// This'll print 2 as Reference Counter is 2 
	cout << P1.use_count() << endl; 
	return 0; 
} 

