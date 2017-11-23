#ifndef VSB_SEMESTRAL_PROJECT_TIMER_H
#define VSB_SEMESTRAL_PROJECT_TIMER_H

#include <iostream>
#include <chrono>

class Timer {
private:
    typedef std::chrono::high_resolution_clock clock;
    typedef std::chrono::duration<double, std::ratio<1>> second;
    std::chrono::time_point<clock> beginning;
public:
    Timer() : beginning(clock::now()) {}

    double elapsed() const;
    void reset();
};


#endif
