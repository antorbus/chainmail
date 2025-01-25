#include "../include/interface.h"

__attribute__((constructor))
void library_init() {
    // int num_threads = 8;
    // omp_set_num_threads(num_threads); 
    // printf("num_threads %d \n", num_threads);
     omp_set_dynamic(1);

   
}