//Initialize a vector in C++ (5 different ways)


// CPP program to create an empty vector 
// and push values one by one. 
#include <bits/stdc++.h> 
using namespace std; 

int main() 
{ 
	// Create an empty vector 
	vector<int> vect; 

	vect.push_back(10); 
	vect.push_back(20); 
	vect.push_back(30); 

	for (int x : vect) 
		cout << x << " "; 

	return 0; 
} 

//==========================================
// CPP program to initialize a vector like 
// an array. 
#include <bits/stdc++.h> 
using namespace std; 
  
int main() 
{ 
    vector<int> vect{ 10, 20, 30 }; 
  
    for (int x : vect) 
        cout << x << " "; 
  
    return 0; 
} 