/*
 * ------------------------------------
 * General utility functions
 * ------------------------------------
 */

#ifndef UTIL_HPP_
#define UTIL_HPP_

#define BOOST_DYNAMIC_BITSET_DONT_USE_FRIENDS

#include <limits>
#include <boost/dynamic_bitset.hpp>


/**
 * -------------------------------------------------------------
 * Objective function type
 * -------------------------------------------------------------
 */

#define ObjType int



/**
 * -------------------------------------------------------------
 * Macros
 * -------------------------------------------------------------
 */

#define MIN(_a_,_b_)              ((_a_ < _b_) ? _a_ : _b_)
#define MAX(_a_,_b_)              ((_a_ > _b_) ? _a_ : _b_)
#define FLOOR(_a_)                ( (int)_a_ )
#define CEIL(_a_)                 ( ((int)_a_ == _a_) ? _a_ : (((int)_a_)+1) )
//#define ABS(_a_)                    ( _a_ < 0 ? (-1)*_a_ : _a_ )



/**
 * -------------------------------------------------------------
 * Constants
 * -------------------------------------------------------------
 */

const int   INF_WIDTH   = -1;        							 // infinite width
const int   INF   		= std::numeric_limits<int>::max();		 // infinite *


/**
 * -------------------------------------------------------------
 * Random integers 
 * -------------------------------------------------------------
 */

//
// Generate integer random number between min and max (inclusive)
//
inline int generate_random_int(int min, int max) {
    int n = max - min + 1;
    // Chop off all of the values that would cause skew...
    long end = RAND_MAX / n; // truncate skew
    end *= n;

    // ... and ignore results from rand() that fall above that limit.
    // (Worst case the loop condition should succeed 50% of the time,
    // so we can expect to bail out of this loop pretty quickly.)
    int r;
    while ((r = rand()) >= end);
    return (r % n) + min;
}



/**
 * -------------------------------------------------------------
 * Comparator with respect to auxiliary vector
 * -------------------------------------------------------------
 */

struct ComparatorAuxIntVectorDescending {
	const std::vector<int> &v;
	ComparatorAuxIntVectorDescending(const std::vector<int> &_v) : v(_v) { }
	bool operator()(int i, int j) {
		return (v[i] > v[j]);
	}
};


struct ComparatorAuxIntVectorAscending {
	const std::vector<int> &v;
	ComparatorAuxIntVectorAscending(const std::vector<int> &_v) : v(_v) { }
	bool operator()(int i, int j) {
		return (v[i] > v[j]);
	}
};


/**
 * -------------------------------------------------------------
 * Boost auxiliaries
 * -------------------------------------------------------------
 */


//
// Equality functions to dynamic_bitset pointer
//
struct bitset_equal_to 
	: std::binary_function<boost::dynamic_bitset<>*, boost::dynamic_bitset<>*, bool>
{
    bool operator()(const boost::dynamic_bitset<>* const x, 
					const boost::dynamic_bitset<>* const y) const;
};


//
// Hash functions of dynamic_bitset pointer
//
struct bitset_hash : std::unary_function<boost::dynamic_bitset<>*, std::size_t>
{
    std::size_t operator()(const boost::dynamic_bitset<>* const x) const;
};


#endif /* UTIL_HPP_ */
