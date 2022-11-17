#include <iostream>

#include "mylib/adder.hpp"

namespace bmath
{
    auto add(const int &a, const int &b) -> int
    {
        std::cout << a << ", " << b << "\n";
        std::cout << &a << ", " << &b << "\n";
        return a + b;
    }
}
