#include <iostream>

#include "mylib/adder.hpp"


namespace bmath{
    int add(const int& a, const int& b){
        std::cout << a << ", " << b << "\n";
        std::cout << &a << ", " << &b << "\n";
        return a + b;
    }
}

