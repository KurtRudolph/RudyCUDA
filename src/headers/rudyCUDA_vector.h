/**
  * @fiel rudyCUDA_vector.h
  * @author Kurt Robert Rudolph
  * @description this is the header file for
  * a set of functions which do basic vecotor 
  * operations.
  */

#ifndef __RUDY_VECTOR_H__
#define __RUDY_VECTOR_H__

#include <rudyCUDA.h>
/**
  * @struct rudyCUDA_vector_element_t
	* @brif holds a vector element
  * @param element, the elment being stored
  */
typedef struct rudyCUDA_vector_element_t {
	void * element;
} rudyCUDA_vector_element;

typedef struct rudyCUDA_vector_t {
	int elementCount;
	rudyCUDA_vector_element **element;
} rudyCUDA_vector; 

/**
  * @struct rudyCUDA_vector_int_t
  * @brief stores a vector
  * @param elementCount The number of elements
  * in the vector
  * @param vector A double pointer array containaing 
  * #elementCount number of int* (s).
  */
typedef struct rudyCUDA_vector_int_t {
	int elementCount;
	int **vector;
} rudyCUDA_vector_int;

/**
  * @struct rudyCUDA_vector_double_t
  * @brief stores a vector
  * @param elementCount The number of elements
  * in the vector
  * @param vector A double pointer array containaing 
  * #elementCount number of double* (s).
  */
typedef struct rudyCUDA_vector_double_t {
	int elementCount;
	double **vector;
} rudyCUDA_vector_double;

/**
  * @struct rudyCUDA_vector_double_t
  * @brief stores a vector
  * @param elementCount The number of elements
  * in the vector
  * @param vector A double pointer array containaing 
  * #elementCount number of float* (s).
  */
typedef struct rudyCUDA_vector_float_t {
	int elementCount;
	float **vector;
} rudyCUDA_vector_float;

/**
  * @brief Free's the memory of a rudyCUDAVecotor
  * @param vecotor the vecotor to be freed and 
  * set to NULL.
  */
__global__ void rudyCUDA_vector_freeInt( rudyCUDA_vector_int * vector);

/**
  * @brief Free's the memory of a rudyCUDAVecotor
  * @param vecotor the vecotor to be freed and 
  * set to NULL.
  */
__global__ void rudyCUDA_vector_freeDouble( rudyCUDA_vector_double * vector);

/**
  * @brief Sums vectors.  
  * @param sumVectors The vectros to be summed.
  * @param storeSumVector The vector where the sums are to be stored,
  * NOTE: it is valid to store have a store vector also a sum vector
  * @param vectorCount The number of vectors in the array.
  * stored.  
  */
__global__ void rudyCUDA_vector_sumInt( rudyCUDA_vector_int ** sumVectors, rudyCUDA_vector_int * storeSumVector, int * vectorCount);

/**
  * @brief Sums vectors.  
  * @param sumVectors The vectros to be summed.
  * @param storeSumVector The vector where the sums are to be stored,
  * NOTE: it is valid to store have a store vector also a sum vector
  * @param vectorCount The number of vectors in the array.
  * stored.  
  */
__global__ void rudyCUDA_vector_sumDouble( rudyCUDA_vector_int ** sumVectors, rudyCUDA_vector_int * storeSumVector, int * vectorCount);

/**
  * @brief Sums vectors.  
  * @param sumVectors The vectros to be summed.
  * @param storeSumVector The vector where the sums are to be stored,
  * NOTE: it is valid to store have a store vector also a sum vector
  * @param vectorCount The number of vectors in the array.
  * stored.  
  */
__global__ void rudyCUDA_vector_sumFloat( rudyCUDA_vector_int ** sumVectors, rudyCUDA_vector_int * storeSumVector, int * vectorCount);

typedef enum rudyCUDA_vecotor_reduceOp_t {
	rudyCUDA_vector_reduceSum =0,
	rudyCUDA_vector_reduceMin,
	rudyCUDA_vector_reduceMax,
	rudyCUDA_vector_reduceMean,
	rudyCUDA_vector_reduceMedian,
  rudyCUDA_vector_reduceTwoNorm
} rudyCUDA_vector_reduceOP;
	
/**
  * @brief Ruduces a vector  
  * @param vectors The vectros to be reduced.
  * @param storeRduce Where the reduced vector is to be stored,
  * NOTE: it is valid to store have a store vector also a sum vector
  * @param vectorCount The number of vectors in the array.
  * stored.  
  */
//__global__ void rudyCUDA_vector_reduceInt( rudyCUDA_vector_int *vector, int *storeRduce , rudyCUDA_vector_reduceOp_t *op);

/**
  * @brief Ruduces a vector  
  * @param vectors The vectros to be reduced.
  * @param storeRduce Where the reduced vector is to be stored,
  * NOTE: it is valid to store a store the reduced vector in the 
  * zero element of the #vector
	* @param op the operation to be completed 
  */
//__global__ void rudyCUDA_vector_reduceDouble( rudyCUDA_vector_int *vector, int *storeRduce , rudyCUDA_vector_reduceOp *op);

/**
  * @brief Ruduces a vector  
  * @param vectors The vectros to be reduced.
  * @param storeRduce Where the reduced vector is to be stored,
  * NOTE: it is valid to store have a store vector also a sum vector
  * @param vectorCount The number of vectors in the array.
  * stored.  
  */
//__global__ void rudyCUDA_vector_reduceInt( rudyCUDA_vector_int *vector, int *storeRduce , rudyCUDA_vector_reduceOp *op);



/**
  * @brief Computes the dotProduct of two equal length vectors.
  * @param vector1 The first input vector.
  * @param vector2 The second input vector.
  * @param storeDotProductElement The result of the dotProduct
  */
//__global__ void rudyCUDA_vector_dotProduct( rudyCUDAVector * vector1, rudyCUDAVector * vector2, rudyCUDAVectorElement * storeDotProductElement);

#endif
