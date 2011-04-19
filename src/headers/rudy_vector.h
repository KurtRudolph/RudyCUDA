/**
  * @fiel rudy_vector.h
  * @author Kurt Robert Rudolph
  * @description this is the header file for
  * a set of functions which do basic vecotor 
  * operations.
  */

#ifndef __RUDY_VECTOR_H__
#define __RUDY_VECTOR_H__

#include <rudyCUDA.h>

typedef struct rudyVectorElement_t {
	double * element;
} rudyVectorElement;

/**
  * @struct rudyVector
  * @brief stores a vector
  * @param elementCount The number of elements
  * in the vector
  * @param vector A double pointer array containaing 
  * #elementCount number of #rudyVecotrElement (s)
  */
typedef struct rudyVector_t {
	int64_t elementCount;
	rudyVectorElement ** vecotor;
} rudyVector;

/**
  * @brief Inializes a vector
  * @param numElements the number of elements to set 
  * the vector to.
  * @param zeroOut a bool value which indicates weather 
  * the vecotor should be initialized to zero.
  */
void rudy_vector_initialize( int64_t * numElements, bool zeroOut);

/**
  * @brief Free's the memory of a rudyVecotor
  * @param vecotor the vecotor to be freed and 
  * set to NULL.
  */
void rudy_vector_free( rudyVector * vector);

/**
  * @brief Sums the values of an array of vectors.  
  * @param sumVectors An array of the vectros to be summed.
  * @param vectorCount The number of vectors in the array.
  * @param storeSumVector The vector where the sums are to be
  * stored.  
  */
void rudy_vector_sum( rudyVector ** sumVectors, int64_t * vectorCount, rudyVector * storeSumVector);

/**
  * @brief Sums the values of a vector.
  * @param vector The vector whos elements are to be summed.
  * @param sum The sum of the vector elements.
  */
void rudy_vector_sumElements( rudyVector * vector, rudyVectorElement * storeSumElement);

/**
  * @brief Computes the dotProduct of two equal length vectors.
  * @param vector1 The first input vector.
  * @param vector2 The second input vector.
  * @param storeDotProductElement The result of the dotProduct
  */
void rudy_vector_dotProduct( rudyVector * vector1, rudyVector * vector2, rudyVectorElement * storeDotProductElement);

#endif
