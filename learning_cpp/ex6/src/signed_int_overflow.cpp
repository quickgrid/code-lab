// https://hackingcpp.com/cpp/tools/ubsan.html
#include <iostream>

auto main() -> int
{
    int i = std::numeric_limits<int>::max();
    std::cout << "i = " << i << "\n";
    i += 1;
    std::cout << "i = " << i << "\n";
}
