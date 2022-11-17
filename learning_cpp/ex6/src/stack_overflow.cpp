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

// Infinite loops stack overflow.
// Can be seen on debug call stack keeps adding func2.
auto func2(int x) -> int
{
    if (!x)
    {
        return 0;
    }
    return func2(x) - 1;
}

auto main() -> int
{
    const int val = 6;
    func2(val);
    return 0;
}
