#include <iostream>
#include <typeinfo>
#include <vector>

using namespace std;

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
            cout << *(arr + k) << "\n";
            cout << (arr + k) << "\n";
            cout << typeid(k).name() << ", " << sizeof(k) << "\n";
            cout << "\n";
        }

        delete[] arr;
        cout << "Done\n";
    }

    void test_vector_allocation(const unsigned int &array_size)
    {
        vector<int> arr;
        for (uint_fast8_t k = 0; k < array_size; ++k)
        {
            arr.push_back(k * k);
        }

        for (const auto &k : arr)
        {
            cout << k << "\n";
            cout << &k << "\n";
            cout << typeid(k).name() << ", " << sizeof(k) << "\n";
            cout << "\n";
        }

        cout << "Done\n";
    }
}
