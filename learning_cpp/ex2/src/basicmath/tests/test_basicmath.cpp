#include <iostream>
#include <cassert>

#include "mylib/adder.hpp"

int main()
{
    assert(bmath::add(2, 2) == 4);
    assert(bmath::add(4, 0) == 4);
    assert(bmath::add(9, 4) == 13);
    assert(bmath::add(4, 5) == 9);

    std::cout << "All tests passed.\n";
}
