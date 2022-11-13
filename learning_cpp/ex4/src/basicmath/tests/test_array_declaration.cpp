#include <iostream>

#include "mylib/array_utils.hpp"

int main()
{
    std::cout << "App started\n";
    const unsigned int &n = 6;

    std::cout << "Test 1\n";
    bmath::test_1d_array_dynamic_allocation(n);
    
    std::cout << "Test 2\n";
    bmath::test_vector_allocation(n);
    return 0;
}
