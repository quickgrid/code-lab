/*
 * Example of memory leak and function call.
 */

#include <iostream>

int func1()
{
    std::cout << __FUNCTION__ << " started\n";
    int a = 1;
    return a;
}

int func2()
{
    std::cout << __FUNCTION__ << " started\n";
    int a = 2;
    return func1();
}

int func3()
{
    std::cout << __FUNCTION__ << " started\n";
    int a = 3;
    a = func2();
    return a;
    std::cout << __FUNCTION__ << " not reached here\n";
}

int func4()
{
    std::cout << __FUNCTION__ << " started\n";
    int a = 4;
    func3();
    // Unhandled exception here if trying to step over on continue in debugging.
    // Goes away if returned.
    std::cout << __FUNCTION__ << " ended\n";
}

int main()
{
    int a = 5;
    func4();
    return 0;
}
