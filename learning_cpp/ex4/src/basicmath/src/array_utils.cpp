#include <iostream>
#include <typeinfo>
#include <vector>

namespace bmath
{
    void test_1d_array_dynamic_allocation(const unsigned int &array_size)
    {
        int *arr = new int[array_size];
        for (uint_fast8_t k = 0; k < array_size; ++k)
        {
            arr[k] = k * k;
        }

        for (uint_fast8_t k = 0; k < array_size; ++k)
        {
            std::cout << *(arr + k) << "\n";
            std::cout << (arr + k) << "\n";
            std::cout << typeid(k).name() << ", " << sizeof(k) << "\n";
            std::cout << "\n";
        }

        delete[] arr;
        std::cout << "Done\n";
    }

    void test_vector_allocation(const unsigned int &array_size)
    {
        std::vector<int> arr;
        for (uint_fast8_t k = 0; k < array_size; ++k)
        {
            arr.push_back(k * k);
        }

        for (const auto &k : arr)
        {
            std::cout << k << "\n";
            std::cout << &k << "\n";
            std::cout << typeid(k).name() << ", " << sizeof(k) << "\n";
            std::cout << "\n";
        }

        std::cout << "Done\n";
    }
}
