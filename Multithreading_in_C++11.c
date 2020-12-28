
// Multithreading in C++ https://www.geeksforgeeks.org/multithreading-in-cpp/
// Multithreading support was introduced in C+11. Prior to C++11, we had to use POSIX threads or p threads library in C. While this library did the job the lack of any standard language provided feature-set caused serious portability issues. C++ 11 did away with all that and gave us std::thread. The thread classes and related functions are defined in the thread header file.

// CPP program to demonstrate multithreading using three different callables. 

// Note: To compile programs with std::thread support use 
// g++ -std=c++11 -pthread


#include <iostream> 
#include <thread> 
using namespace std; 

// A dummy function 
void foo(int Z) 
{ 
	for (int i = 0; i < Z; i++) { 
		cout << "Thread using function"
			" pointer as callable\n"; 
	} 
} 

// A callable object 
class thread_obj { 
public: 
	void operator()(int x) 
	{ 
		for (int i = 0; i < x; i++) 
			cout << "Thread using function"
				" object as callable\n"; 
	} 
}; 

int main() 
{ 
	cout << "Threads 1 and 2 and 3 "
		"operating independently" << endl; 

	// This thread is launched by using 
	// function pointer as callable 
	thread th1(foo, 3); 

	// This thread is launched by using 
	// function object as callable 
	thread th2(thread_obj(), 3); 

	// Define a Lambda Expression 
	auto f = [](int x) { 
		for (int i = 0; i < x; i++) 
			cout << "Thread using lambda"
			" expression as callable\n"; 
	}; 

	// This thread is launched by using 
	// lamda expression as callable 
	thread th3(f, 3); 

	// Wait for the threads to finish 
	// Wait for thread t1 to finish 
	th1.join(); 

	// Wait for thread t2 to finish 
	th2.join(); 

	// Wait for thread t3 to finish 
	th3.join(); 

	return 0; 
} 




/* Output (Machine Dependent)

Threads 1 and 2 and 3 operating independently                                                       
Thread using function pointer as callable                                                           
Thread using lambda expression as callable                                                          
Thread using function pointer as callable                                                           
Thread using lambda expression as callable                                                          
Thread using function object as  callable                                                          
Thread using lambda expression as callable                                                          
Thread using function pointer as callable                                                          
Thread using function object as  callable                                                           
Thread using function object as  callable */