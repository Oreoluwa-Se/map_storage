#ifndef TIMER_HPP
#define TIMER_HPP
#include <chrono>
#include <functional>
#include <type_traits>
#include <utility>
#include <iostream>

template <typename Func, typename Obj, typename... Args>
auto invoke(Func Obj::*func, Obj *obj, Args &&...args) -> decltype((obj->*func)(std::forward<Args>(args)...))
{
    return (obj->*func)(std::forward<Args>(args)...);
}

template <typename Func, typename... Args>
auto invoke(Func &&func, Args &&...args) -> decltype(std::forward<Func>(func)(std::forward<Args>(args)...))
{
    return std::forward<Func>(func)(std::forward<Args>(args)...);
}
// Timing function templates
template <typename Func, typename... Args>
typename std::enable_if<std::is_void<typename std::result_of<Func(Args...)>::type>::value>::type
measure_time(Func func, Args &&...args)
{
    auto start = std::chrono::high_resolution_clock::now();
    invoke(func, std::forward<Args>(args)...);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << 1000 * duration.count() << " milliseconds\n";
}

// Helper function for non-void return type
template <typename Func, typename... Args>
typename std::enable_if<!std::is_void<typename std::result_of<Func(Args...)>::type>::value, typename std::result_of<Func(Args...)>::type>::type
measure_time(Func func, Args &&...args)
{
    auto start = std::chrono::high_resolution_clock::now();
    auto result = invoke(func, std::forward<Args>(args)...);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << 1000 * duration.count() << " milliseconds\n";

    return result;
}

// Overload for member function pointers
template <typename Obj, typename Ret, typename... Params, typename... Args>
typename std::enable_if<std::is_void<Ret>::value>::type
measure_time(Ret (Obj::*func)(Params...), Obj *obj, Args &&...args)
{
    auto start = std::chrono::high_resolution_clock::now();
    invoke(func, obj, std::forward<Args>(args)...);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << 1000 * duration.count() << " milliseconds\n";
}

// Helper function for non-void return type
template <typename Obj, typename Ret, typename... Params, typename... Args>
typename std::enable_if<!std::is_void<Ret>::value, Ret>::type
measure_time(Ret (Obj::*func)(Params...), Obj *obj, Args &&...args)
{
    auto start = std::chrono::high_resolution_clock::now();
    Ret result = invoke(func, obj, std::forward<Args>(args)...);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << 1000 * duration.count() << " milliseconds\n";

    return result;
}

#endif