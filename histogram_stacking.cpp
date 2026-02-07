#include <iostream>
#include <vector>
#include <immintrin.h> // AVX2
#include <cstdint>
#include <algorithm>
#include <fstream>
#include <string>

// Definiamo l'implementazione di STB prima dell'include
#define STB_IMAGE_WRITE_IMPLEMENTATION

extern "C" {

    uint8_t* histogramStack(uint16_t* x, uint16_t* y, uint8_t* p, 
        int numEvents, int width, int height) {
        
        const int SIZE = width * height * 2;
        uint8_t* accBuffer = (uint8_t*)malloc(SIZE);

        std::fill(accBuffer, accBuffer + SIZE, 0);

        for (int i  = 0; i < numEvents; ++i) {
            
            uint16_t curr_x = x[i];
            uint16_t curr_y = y[i];
            uint8_t curr_p = p[i];
            int pos = (curr_y * width + curr_x)*2 +(1-curr_p);
            
            if (accBuffer[pos] < 255) {
                ++accBuffer[pos];
            }
            
        }

        return accBuffer;
    }
}

// Compila con:
// g++ -O3 -mavx2 -shared -fPIC histogram_stacking.cpp -o histogram_stacking.so