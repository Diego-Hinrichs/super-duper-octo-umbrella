#pragma once

#include <cstdint>
#include <algorithm>
#include <vector>
#include <memory>
#include <functional>
#include <omp.h>
#include <iostream>
#include "../../morton/morton.h"

// No incluimos tipos externos, el archivo main.cpp ya define Vector y Body

namespace sfc {

enum class CurveType
{
    MORTON,
    HILBERT
};

template<typename BodyType>
class BodySorter
{
public:
    BodySorter(int numBodies, CurveType type) : nBodies(numBodies), curveType(type)
    {
        keys = new uint64_t[numBodies];
        indices = new int[numBodies];
    }

    ~BodySorter()
    {
        if (keys)
            delete[] keys;
        if (indices)
            delete[] indices;
    }

    void setCurveType(CurveType type)
    {
        curveType = type;
    }

    // Implementación exacta del calculateSFCKey de BarnesHutGPU/main.cu
    uint64_t calculateSFCKey(const Vector &pos, const Vector &minBound, const Vector &maxBound)
    {
        double normalizedX = (pos.x - minBound.x) / (maxBound.x - minBound.x);
        double normalizedY = (pos.y - minBound.y) / (maxBound.y - minBound.y);
        double normalizedZ = (pos.z - minBound.z) / (maxBound.z - minBound.z);

        normalizedX = std::max(0.0, std::min(0.999999, normalizedX));
        normalizedY = std::max(0.0, std::min(0.999999, normalizedY));
        normalizedZ = std::max(0.0, std::min(0.999999, normalizedZ));

        uint32_t x = static_cast<uint32_t>(normalizedX * ((1 << 20) - 1));
        uint32_t y = static_cast<uint32_t>(normalizedY * ((1 << 20) - 1));
        uint32_t z = static_cast<uint32_t>(normalizedZ * ((1 << 20) - 1));

        if (curveType == CurveType::MORTON)
        {
            return mortonEncode(x, y, z);
        }
        else
        {
            // Para Hilbert usamos implementación adaptada del algoritmo en GPU
            return hilbertEncode(x, y, z);
        }
    }

    // Implementación del mortonEncode como en el original
    uint64_t mortonEncode(uint32_t x, uint32_t y, uint32_t z)
    {
        return libmorton::morton3D_64_encode(x, y, z);
    }

    // Implementación para Hilbert 
    uint64_t hilbertEncode(uint32_t x, uint32_t y, uint32_t z)
    {
        // Implementación simple adaptada de la idea en GPU
        const uint32_t n = 1 << 10; // 10 bits por dimensión
        return hilbertXYZToIndex(n, x, y, z);
    }

    // Función adaptada de rotateHilbert
    void rotateHilbert(uint32_t n, uint32_t *x, uint32_t *y, uint32_t *z, uint32_t rx, uint32_t ry, uint32_t rz)
    {
        if (ry == 0)
        {
            if (rx == 1)
            {
                *x = n - 1 - *x;
                *z = n - 1 - *z;
            }
            std::swap(*x, *z);
        }
    }

    // Función auxiliar para Hilbert
    uint64_t hilbertXYZToIndex(uint32_t n, uint32_t x, uint32_t y, uint32_t z)
    {
        uint64_t index = 0;
        uint32_t rx, ry, rz, s;
        for (s = n/2; s > 0; s >>= 1)
        {
            rx = (x & s) > 0;
            ry = (y & s) > 0;
            rz = (z & s) > 0;
            
            index += s * s * s * ((3 * rx) ^ ry ^ rz);
            
            rotateHilbert(s, &x, &y, &z, rx, ry, rz);
        }
        return index;
    }

    // Ordenar cuerpos usando SFC
    int* sortBodies(std::vector<BodyType> &bodies, const Vector &minBound, const Vector &maxBound)
    {
        // Calcular las claves SFC para todos los cuerpos
        #pragma omp parallel for if(nBodies > 10000)
        for (int i = 0; i < nBodies; i++)
        {
            keys[i] = calculateSFCKey(bodies[i].position, minBound, maxBound);
            indices[i] = i;
        }

        // Ordenar índices basados en las claves SFC
        if (curveType == CurveType::MORTON) {
            sortByMorton(bodies);
        } else {
            sortByHilbert(bodies);
        }

        return indices;
    }

private:
    int nBodies;
    CurveType curveType;
    uint64_t *keys = nullptr;
    int *indices = nullptr;

    // Ordenamiento usando claves Morton
    void sortByMorton(std::vector<BodyType> &bodies)
    {
        // Ordenar indices basados en claves
        std::sort(indices, indices + nBodies, [this](int a, int b) {
            return keys[a] < keys[b];
        });

        // Reordenar cuerpos
        std::vector<BodyType> reorderedBodies(nBodies);
        #pragma omp parallel for if(nBodies > 10000)
        for (int i = 0; i < nBodies; i++)
        {
            reorderedBodies[i] = bodies[indices[i]];
        }

        // Copiar de vuelta al arreglo original
        bodies = reorderedBodies;
    }

    // Ordenamiento usando claves Hilbert (usa el mismo mecanismo de Morton por simplicidad)
    void sortByHilbert(std::vector<BodyType> &bodies)
    {
        // Usamos el mismo método de ordenamiento que Morton ya que las claves ya son Hilbert
        sortByMorton(bodies);
    }
};

} // namespace sfc 