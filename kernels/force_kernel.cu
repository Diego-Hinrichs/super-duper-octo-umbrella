#include "./types.cuh"
#include "./constants.cuh"
#include <stdio.h>

// Apply periodic boundary conditions to a vector difference
__device__ Vector applyPeriodicBoundary(Vector rij, double domainSize)
{
    Vector result = rij;
    double halfDomain = domainSize * 0.5;
    
    // Apply minimum image convention
    if (result.x > halfDomain) result.x -= domainSize;
    else if (result.x < -halfDomain) result.x += domainSize;
    
    if (result.y > halfDomain) result.y -= domainSize;
    else if (result.y < -halfDomain) result.y += domainSize;
    
    if (result.z > halfDomain) result.z -= domainSize;
    else if (result.z < -halfDomain) result.z += domainSize;
    
    return result;
}

// Apply periodic boundary conditions to position
__device__ Vector wrapPosition(Vector pos, double domainSize)
{
    Vector result = pos;
    
    // Wrap coordinates to [0, domainSize)
    result.x = fmod(result.x + domainSize, domainSize);
    if (result.x < 0) result.x += domainSize;
    
    result.y = fmod(result.y + domainSize, domainSize);
    if (result.y < 0) result.y += domainSize;
    
    result.z = fmod(result.z + domainSize, domainSize);
    if (result.z < 0) result.z += domainSize;
    
    return result;
}

__device__ double getDistance(Vector pos1, Vector pos2)
{
    return sqrt(pow(pos1.x - pos2.x, 2) + pow(pos1.y - pos2.y, 2) + pow(pos1.z - pos2.z, 2));
}

__device__ double getPeriodicDistance(Vector pos1, Vector pos2, double domainSize)
{
    Vector rij = {pos2.x - pos1.x, pos2.y - pos1.y, pos2.z - pos1.z};
    rij = applyPeriodicBoundary(rij, domainSize);
    return sqrt(rij.x * rij.x + rij.y * rij.y + rij.z * rij.z);
}

__device__ bool isCollide(Body &b1, Vector cm, double domainSize)
{
    double d = getPeriodicDistance(b1.position, cm, domainSize);
    double threshold = b1.radius * 2 + COLLISION_TH;
    return threshold > d;
}

__device__ void ComputeForce(Node *node, Body *bodies, int nodeIndex, int bodyIndex, int nNodes, int nBodies, int leafLimit, double width, double theta, double domainSize)
{
    if (nodeIndex >= nNodes)
        return;

    Node curNode = node[nodeIndex];
    Body bi = bodies[bodyIndex];
    
    // Caso de nodo hoja: usar el centro de masa para aproximar la fuerza
    if (curNode.isLeaf)
    {   
        if (curNode.centerMass.x != -1 && !isCollide(bi, curNode.centerMass, domainSize))
        // if (curNode.centerMass.x != -1)
        {
            // printf("Leaf node force calculation: Body %d, Node %d\n", bodyIndex, nodeIndex);
            Vector rij = {
                curNode.centerMass.x - bi.position.x,
                curNode.centerMass.y - bi.position.y,
                curNode.centerMass.z - bi.position.z};
            
            // Apply periodic boundary conditions
            rij = applyPeriodicBoundary(rij, domainSize);
            
            // Calcular r² sin suavizado
            double r2 = (rij.x * rij.x) + (rij.y * rij.y) + (rij.z * rij.z);
            // Usar la fórmula: (r^2 + E^2)^(3/2)
            double r = sqrt(r2 + (E * E));
            double f = (GRAVITY * bi.mass * curNode.totalMass) / (r * r * r + (E * E));
            Vector force = {rij.x * f, rij.y * f, rij.z * f};

            bodies[bodyIndex].acceleration.x += (force.x / bi.mass);
            bodies[bodyIndex].acceleration.y += (force.y / bi.mass);
            bodies[bodyIndex].acceleration.z += (force.z / bi.mass);
        }
        return;
    }

    // Caso de aproximación multipolo
    double distance = getPeriodicDistance(bi.position, curNode.centerMass, domainSize);
    double sd = width / distance; // TAMANIO DE LA REGION / DISTANCIA
    if (sd < theta)
    {
        if (!isCollide(bi, curNode.centerMass, domainSize))
        {
            Vector rij = {
                curNode.centerMass.x - bi.position.x,
                curNode.centerMass.y - bi.position.y,
                curNode.centerMass.z - bi.position.z};

            // Apply periodic boundary conditions
            rij = applyPeriodicBoundary(rij, domainSize);

            double r2 = (rij.x * rij.x) + (rij.y * rij.y) + (rij.z * rij.z);
            double r = sqrt(r2 + (E * E));
            double f = (GRAVITY * bi.mass * curNode.totalMass) / (r * r * r + (E * E));
            Vector force = {rij.x * f, rij.y * f, rij.z * f};

            bodies[bodyIndex].acceleration.x += (force.x / bi.mass);
            bodies[bodyIndex].acceleration.y += (force.y / bi.mass);
            bodies[bodyIndex].acceleration.z += (force.z / bi.mass);
        }
        return;
    }

    // Si no se cumple la condición de aproximación, se recorre recursivamente a los 8 hijos.
    for (int i = 1; i <= 8; i++)
    {
        ComputeForce(node, bodies, (nodeIndex * 8) + i, bodyIndex, nNodes, nBodies, leafLimit, width / 2, theta, domainSize);
    }
}


__global__ void ComputeForceKernel(Node *node, Body *bodies, int nNodes, int nBodies, int leafLimit, double theta, double domainSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double width = domainSize; // Use fixed domain size instead of calculated width
    if (i < nBodies)
    {
        Body &bi = bodies[i];
        if (bi.isDynamic)
        {
            // Reiniciar la aceleración
            bi.acceleration = {0.0, 0.0, 0.0};

            // Compute the force recursively
            ComputeForce(node, bodies, 0, i, nNodes, nBodies, leafLimit, width, theta, domainSize);

            // Update velocity and position with integration (Euler)
            bi.velocity.x += bi.acceleration.x * DT;
            bi.velocity.y += bi.acceleration.y * DT;
            bi.velocity.z += bi.acceleration.z * DT;

            bi.position.x += bi.velocity.x * DT;
            bi.position.y += bi.velocity.y * DT;
            bi.position.z += bi.velocity.z * DT;

            // Apply periodic boundary conditions to position
            bi.position = wrapPosition(bi.position, domainSize);
        }
    }
}
