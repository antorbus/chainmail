#include "../../include/tensor.h"


inline lemur_float is_close(lemur_float a, lemur_float b, lemur_float rtol, lemur_float atol) {
    return (fabsf(a - b) <= (atol + rtol * fabsf(b))) ? 1.0 : 0;
}

tensor *isclose(tensor *a, tensor *b, lemur_float rtol, lemur_float atol){
    if (rtol < 0.0){
        fprintf(stderr, "Error: rtol cannot be less than 0.\n");
        return NULL;
    }
    if (atol < 0.0){
    fprintf(stderr, "Error: atol cannot be less than 0.\n");
        return NULL; 
    }
    if (a->k->length != b->k->length){
        fprintf(stderr, "Error: Tensors a and b must have the same memory length.\n");
        return NULL;
    }
    
    if (a->k->shape[0] != b->k->shape[0] || 
        a->k->shape[1] != b->k->shape[1] ||
        a->k->shape[2] != b->k->shape[2] ||
        a->k->shape[3] != b->k->shape[3] ||
        a->k->shape[4] != b->k->shape[4]){
        fprintf(stderr, "Error: Tensors a and b must have the same shape.\n");
        return NULL;
    }
    tensor *c = empty_tensor(a->k->shape, false, false);
    
    if (a->k->length > 1<<17) {                                                 
        #pragma omp parallel for simd                              
        for (size_t _i = 0; _i < a->k->length; _i++) {                       
            c->k->array[_i] = is_close(a->k->array[_i], b->k->array[_i], rtol, atol);   
        }                                                                    
    } else {         
        #pragma omp simd                                                            
        for (size_t _i = 0; _i < a->k->length; _i++) {                           
            c->k->array[_i] = is_close(a->k->array[_i], b->k->array[_i], rtol, atol);   
        }                                                                        
    }          
    
    

    return c;

}