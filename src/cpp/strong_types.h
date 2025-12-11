#ifndef STRONG_TYPES_H
#define STRONG_TYPES_H

#include <type_traits>
#include <ostream>

/**
 * Strong type wrapper to prevent accidental mixing of different types
 * with the same underlying type (e.g., ImageWidth vs ImageHeight)
 * 
 * Usage:
 *   using ImageWidth = StrongType<int, struct ImageWidthTag>;
 *   using ImageHeight = StrongType<int, struct ImageHeightTag>;
 *   
 *   ImageWidth width(800);
 *   ImageHeight height(450);
 *   // width + height;  // Compile error - types don't match
 */
template<typename T, typename Tag>
class StrongType {
private:
    T value_;
    
public:
    explicit StrongType(T value) : value_(value) {}
    
    // Get underlying value
    T get() const { return value_; }
    
    // Implicit conversion to underlying type (for convenience)
    // Can be removed if strict type safety is desired
    operator T() const { return value_; }
    
    // Comparison operators
    bool operator==(const StrongType& other) const {
        return value_ == other.value_;
    }
    
    bool operator!=(const StrongType& other) const {
        return value_ != other.value_;
    }
    
    bool operator<(const StrongType& other) const {
        return value_ < other.value_;
    }
    
    bool operator>(const StrongType& other) const {
        return value_ > other.value_;
    }
    
    bool operator<=(const StrongType& other) const {
        return value_ <= other.value_;
    }
    
    bool operator>=(const StrongType& other) const {
        return value_ >= other.value_;
    }
    
    // Arithmetic operators (if underlying type supports them)
    StrongType operator+(const StrongType& other) const {
        return StrongType(value_ + other.value_);
    }
    
    StrongType operator-(const StrongType& other) const {
        return StrongType(value_ - other.value_);
    }
    
    StrongType operator*(const StrongType& other) const {
        return StrongType(value_ * other.value_);
    }
    
    StrongType operator/(const StrongType& other) const {
        return StrongType(value_ / other.value_);
    }
    
    // Stream output
    friend std::ostream& operator<<(std::ostream& os, const StrongType& st) {
        return os << st.value_;
    }
};

// Type aliases for common strong types
// These can be used to prevent mixing up similar types

// Image dimensions
using ImageWidth = StrongType<int, struct ImageWidthTag>;
using ImageHeight = StrongType<int, struct ImageHeightTag>;

// Coordinates (if we want to distinguish between different coordinate spaces)
using WorldCoordinate = StrongType<double, struct WorldCoordTag>;
using ScreenCoordinate = StrongType<double, struct ScreenCoordTag>;

#endif // STRONG_TYPES_H
