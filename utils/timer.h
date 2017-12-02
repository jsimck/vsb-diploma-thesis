#ifndef VSB_SEMESTRAL_PROJECT_TIMER_H
#define VSB_SEMESTRAL_PROJECT_TIMER_H

#include <iostream>
#include <chrono>

namespace tless {
    /**
     * @brief Utility class used to measure computing time
     *
     * To use first create object -> Timer t; then print time using
     * t.elapsed(), optionally reset time again t.reset(). Results are in seconds
     */
    class Timer {
    private:
        typedef std::chrono::high_resolution_clock clock;
        typedef std::chrono::duration<double, std::ratio<1>> second;
        std::chrono::time_point<clock> beginning;

    public:
        Timer();

        /**
         * @brief Returns time that elapsed from last reset or construction in seconds
         *
         * @return Elapsed time from last reset [seconds]
         */
        double elapsed() const;

        /**
         * @brief Sets start timer to current system time
         */
        void reset();
    };
}


#endif
