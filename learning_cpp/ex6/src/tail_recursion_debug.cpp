/**
 * @file tail_recursion_debug.cpp
 * @author your name (you@domain.com)
 * @brief Tail recursion example.
 * @version 0.1
 * @date 2022-11-14
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <iostream>

int func1(int x)
{
    if (!x)
    {
        return 0;
    }
    std::cout << x << "\n";
    return func1(x - 1);
}

// Infinite loops.
int func2(int x)
{
    if (!x)
    {
        return 0;
    }
    std::cout << x << "\n";
    return func2(x) - 1;
}

int main()
{
    // func1(6);
    func2(6);
    return 0;
}
