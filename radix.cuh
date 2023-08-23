
// Kernel A
// Load and sort tile.
// Write histogram and sorted data to global mem
__global__ void localSort();

// Kernel B
// Scan global histogram
__global__ void histogramScan();

// Kernel C
// Copy elements to correct output
__global__ void swapBuffers();

/* 
Kan man ikke gøre det hele i en enkelt kernel ved at bruge nogle smarte CUB ting til at skanne
det "globale" histogram på tværs af thread blocks?
*/