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
#include <chrono>
#include <functional>
#include <variant>

// Non tail recursive factorial as control return to this function.
// Also previous values are needed to calculate final result.
int factorial_recursive(int x)
{
    if (x <= 0)
    {
        return 1;
    }
    return x * factorial_recursive(x - 1);
}

// Tail recursive factorial the last statement in return.
// Returns calculated value without further operation on caller stack frame.
int factorial_tail_recursive(int x, int result = 1)
{
    if (x <= 0)
    {
        return result;
    }
    return factorial_tail_recursive(x - 1, x * result);
}

void timing_func_reference(int (*func)(int), int val)
{
    std::cout << "START\n";
    auto start = std::chrono::high_resolution_clock::now();
    int result = func(val);
    std::cout << result << "\n";
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << duration.count() << "\n";
    std::cout << "DONE\n";
}

// @todo fix error
// template <typename... Args>
void timing_func_functional(
    // std::variant<std::function<int(int)>, std::function<int(int, int)>> func,
    auto func,
    // Args &&...val)
    int val)
{
    std::cout << "START\n";
    auto start = std::chrono::high_resolution_clock::now();
    int result = func(val);
    std::cout << result << "\n";
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << duration.count() << "\n";
    std::cout << "DONE\n";
}

int main()
{
    int test_val = 10;
    timing_func_functional(factorial_recursive, test_val);
    timing_func_functional(factorial_tail_recursive, test_val);
    timing_func_reference(&factorial_recursive, test_val);
    // timing_func_reference(&factorial_tail_recursive, test_val);
    return 0;
}
