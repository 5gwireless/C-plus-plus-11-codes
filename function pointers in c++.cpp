/* Function Pointer in C
=================================================

In C, like normal data pointers (int *, char *, etc), we can have pointers to functions. Following is a simple example that shows declaration and function call
using function pointer */



#include <stdio.h> 
// A normal function with an int parameter 
// and void return type 
void fun(int a) 
{ 
	printf("Value of a is %d\n", a); 
} 

int main() 
{ 
	void (*fun_ptr)(int) = fun;  
	fun_ptr(10);  
	return 0; 
}


//================================================
//Like normal pointers, we can have an array of function pointers. Below example in point 5 shows syntax for array of pointers.

#include <stdio.h> 
void add(int a, int b) 
{ 
	printf("Addition is %d\n", a+b); 
} 
void subtract(int a, int b) 
{ 
	printf("Subtraction is %d\n", a-b); 
} 
void multiply(int a, int b) 
{ 
	printf("Multiplication is %d\n", a*b); 
} 

int main() 
{ 
	// fun_ptr_arr is an array of function pointers 
	void (*fun_ptr_arr[])(int, int) = {add, subtract, multiply}; 
	unsigned int ch, a = 15, b = 10; 

	printf("Enter Choice: 0 for add, 1 for subtract and 2 "
			"for multiply\n"); 
	scanf("%d", &ch); 

	if (ch > 2) return 0; 

	(*fun_ptr_arr[ch])(a, b); 

	return 0; 
} 

//======================================================


// An example for qsort and comparator 
#include <stdio.h> 
#include <stdlib.h> 

// A sample comparator function that is used 
// for sorting an integer array in ascending order. 
// To sort any array for any other data type and/or 
// criteria, all we need to do is write more compare 
// functions. And we can use the same qsort() 
int compare (const void * a, const void * b) 
{ 
return ( *(int*)a - *(int*)b ); 
} 

int main () 
{ 
int arr[] = {10, 5, 15, 12, 90, 80}; 
int n = sizeof(arr)/sizeof(arr[0]), i; 

qsort (arr, n, sizeof(int), compare); 

for (i=0; i<n; i++) 
	printf ("%d ", arr[i]); 
return 0; 
} 
