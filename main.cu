#include<stdio.h>
#include<stdlib.h>

#include"radix.cuh"

int main(int argc, char* argv[]) {

    if (argc < 2) {
        printf("Usage: ./radix <gpu runs>\n");
        return 0;
    }

    int gpu_runs = atoi(argv[1]);

    return 0;
}