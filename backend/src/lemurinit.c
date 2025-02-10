#include "../include/interface.h"

__attribute__((constructor))
void library_init() {
    int num_cores = omp_get_num_procs();
    printf("(Alleged) physical cores: %d\n", num_cores);
    int num_threads = num_cores;
    omp_set_num_threads(num_threads); 
    printf("Set num_threads to: %d\n", num_threads);
    init_random();
    omp_set_dynamic(1);

   
}