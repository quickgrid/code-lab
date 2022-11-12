#include <iostream>
#include <cassert>

#include "mylib/adder.hpp"

using namespace std;

int main()
{
    assert(bmath::add(2, 2) == 4);
    // assert(bmath::add(5.5, 1.2) == 7.7);
    // assert(bmath::add(2, 2.3) == 4.3);
    assert(bmath::add(4, 0) == 4);
    assert(bmath::add(9, 4) == 13);
    assert(bmath::add(4, 5) == 9);

    cout << "All tests passed.\n";
}
