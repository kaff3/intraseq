#ifndef PMPH_TIMING
#define PMPH_TIMING

#include<time.h>
#include<sys/time.h>

#include "helper.cu.h"

struct Timer
{
private:
    struct timeval t_start, t_end, t_diff;

public:
    void Start() {
        gettimeofday(&this->t_start, NULL);
    }

    void Stop() {
        gettimeofday(&this->t_end, NULL);
        timeval_subtract(&this->t_diff, &this->t_end, &this->t_start);
    }

    int Get() {
        return this->t_diff.tv_sec * 1e6 + this->t_diff.tv_usec;
    }

};

#endif // PMPH_TIMING