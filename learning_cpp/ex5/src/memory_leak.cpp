/*
 * Example of memory leak and unused variables.
 */

#include <iostream>

auto main() -> int
{
    // Memory leak and unused variables.
    int *d = new int[5]{1, 2, 3, 4, 5};
    int *e;
    const int a = 5;
    const int b = 6;
    const uint8_t c = 3;
    std::cout << "HELLO"
              << "\n";
    // delete d;
    // delete e;
    return 0;
}
