#include "util.hpp"
#include <boost/unordered_map.hpp>

using namespace boost;


//
// Equality functions to dynamic_bitset pointer
//
bool bitset_equal_to::operator()(const boost::dynamic_bitset<>* const x, 
			const boost::dynamic_bitset<>* const y) const
{
        return (*x == *y);
}


//
// Hash functions to dynamic_bitset pointer
//
std::size_t bitset_hash::operator()(const boost::dynamic_bitset<>* const x) const
{
    return boost::hash_value(x->m_bits);
}
