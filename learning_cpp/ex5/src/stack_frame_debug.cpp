/*
 * Example of memory leak and function call.
 */

#include <iostream>

auto func1(int x) -> int
{
    std::cout << __FUNCTION__ << " started\n";
    int a = 1;
    return a;
}

auto func2(int x) -> int
{
    std::cout << __FUNCTION__ << " started\n";
    int a = 2;
    return func1(11);
}

auto func3() -> int
{
    std::cout << __FUNCTION__ << " started\n";
    int a = 3;
    a = func2(7);
    return a;
    std::cout << __FUNCTION__ << " not reached here\n";
}

auto func4() -> int
{
    std::cout << __FUNCTION__ << " started\n";
    int a = 4;
    func3();
    // Unhandled exception here if trying to step over on continue in debugging.
    // Goes away if returned.
    std::cout << __FUNCTION__ << " ended\n";
}

auto main() -> int
{
    int a = 5;
    func4();
    return 0;
}
