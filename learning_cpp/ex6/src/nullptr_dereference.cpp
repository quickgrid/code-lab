// https://hackingcpp.com/cpp/tools/asan.html
#include <iostream>

auto main() -> int
{
    int *p = nullptr;
    std::cout << p << " " << *p << "\n";
}
