#pragma once

struct Vector
{
    double x;
    double y;
    double z;

    Vector() : x(0.0), y(0.0), z(0.0) {}
    Vector(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
    
    Vector operator+(const Vector &other) const { return Vector(x + other.x, y + other.y, z + other.z); }
    Vector operator-(const Vector &other) const { return Vector(x - other.x, y - other.y, z - other.z); }
    Vector operator*(double scalar) const { return Vector(x * scalar, y * scalar, z * scalar); }
    
    double dot(const Vector &other) const { return x * other.x + y * other.y + z * other.z; }
    double lengthSquared() const { return x * x + y * y + z * z; }
    double length() const { return sqrt(lengthSquared()); }
};

struct Body
{
    Vector position;
    Vector velocity;
    Vector acceleration;
    double mass;
    double radius;
    bool isDynamic;

    Body() : mass(1.0), radius(1.0), isDynamic(true) {}
    Body(const Vector &pos, const Vector &vel, double m, double r = 1.0, bool dynamic = true)
        : position(pos), velocity(vel), acceleration(), mass(m), radius(r), isDynamic(dynamic) {}
}; 