#include <iostream>

#include "mylib/array_utils.hpp"

using namespace std;

int main()
{
    cout << "App started\n";
    const unsigned int &n = 6;

    cout << "Test 1\n";
    bmath::test_1d_array_dynamic_allocation(n);
    
    cout << "Test 2\n";
    bmath::test_vector_allocation(n);
    return 0;
}
