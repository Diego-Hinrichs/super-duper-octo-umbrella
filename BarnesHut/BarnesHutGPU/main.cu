#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <string>
#include <random>
#include <limits>
#include <algorithm>
#include <cfloat>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <sys/stat.h>
#include <deque>
#include <numeric>
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include "../../argparse.hpp"
#include "../../morton/morton.h"

// Include the kernel headers
#include "../../kernels/constants.cuh"
#include "../../kernels/types.cuh"
#include "../../kernels/reset_kernel.cu"
#include "../../kernels/octree_kernel.cu"
#include "../../kernels/force_kernel.cu"

// Implementación de atomicAdd para double si no está disponible
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
__device__ double atomicAdd(double *address, double val)
{
    unsigned long long int *address_as_ull = (unsigned long long int *)address;
    unsigned long long int old = *address_as_ull, assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                      __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

__global__ void CalculateEnergiesKernel(Body *bodies, int nBodies, double *d_potentialEnergy, double *d_kineticEnergy)
{
    extern __shared__ double sharedEnergy[];
    double *sharedPotential = sharedEnergy;
    double *sharedKinetic = &sharedEnergy[blockDim.x];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;

    sharedPotential[tx] = 0.0;
    sharedKinetic[tx] = 0.0;

    if (i < nBodies)
    {
        if (bodies[i].isDynamic)
        {
            double vSquared = bodies[i].velocity.lengthSquared();
            sharedKinetic[tx] = 0.5 * bodies[i].mass * vSquared;
        }

        for (int j = i + 1; j < nBodies; j++)
        {
            Vector r = bodies[j].position - bodies[i].position;

            double distSqr = r.lengthSquared() + (E * E);
            double dist = sqrt(distSqr);

            if (dist < COLLISION_TH)
                continue;

            sharedPotential[tx] -= GRAVITY * bodies[i].mass * bodies[j].mass / dist;
        }
    }

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tx < s)
        {
            sharedPotential[tx] += sharedPotential[tx + s];
            sharedKinetic[tx] += sharedKinetic[tx + s];
        }
        __syncthreads();
    }

    if (tx == 0)
    {
        atomicAdd(d_potentialEnergy, sharedPotential[0]);
        atomicAdd(d_kineticEnergy, sharedKinetic[0]);
    }
}

namespace sfc
{
    __global__ void extractPositionsAndCalculateKeysKernel(
        Body *bodies,
        uint64_t *keys,
        int nBodies,
        Vector minBound,
        Vector maxBound,
        bool isHilbert)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= nBodies)
            return;

        // Extraer posición directamente del cuerpo
        Vector pos = bodies[i].position;

        // Precalcular los rangos inversos para evitar divisiones
        double invRangeX = 1.0 / (maxBound.x - minBound.x);
        double invRangeY = 1.0 / (maxBound.y - minBound.y);
        double invRangeZ = 1.0 / (maxBound.z - minBound.z);

        // Normalizar las coordenadas al rango [0,1)
        double normalizedX = (pos.x - minBound.x) * invRangeX;
        double normalizedY = (pos.y - minBound.y) * invRangeY;
        double normalizedZ = (pos.z - minBound.z) * invRangeZ;

        // Asegurar que están dentro del rango para evitar desbordamientos
        normalizedX = fmax(0.0, fmin(0.999999, normalizedX));
        normalizedY = fmax(0.0, fmin(0.999999, normalizedY));
        normalizedZ = fmax(0.0, fmin(0.999999, normalizedZ));

        // Convertir a enteros para el cálculo de claves
        // Usar 20 bits para mayor precisión (1M valores)
        const uint32_t MAX_COORD = (1 << 20) - 1;

        uint32_t x = static_cast<uint32_t>(normalizedX * MAX_COORD);
        uint32_t y = static_cast<uint32_t>(normalizedY * MAX_COORD);
        uint32_t z = static_cast<uint32_t>(normalizedZ * MAX_COORD);

        uint64_t key = 0;

        if (!isHilbert)
        {
            // Define magic bit masks for 64-bit Morton encoding
            const uint64_t masks64[6] = { 
                0x1fffff, 
                0x1f00000000ffff, 
                0x1f0000ff0000ff, 
                0x100f00f00f00f00f, 
                0x10c30c30c30c30c3, 
                0x1249249249249249 
            };
            
            // Magic bits implementation for Morton encoding
            // Split the bits for each coordinate using magic bits method
            uint64_t morton_x = ((uint64_t)x) & masks64[0];
            morton_x = (morton_x | (morton_x << 32)) & masks64[1];
            morton_x = (morton_x | (morton_x << 16)) & masks64[2];
            morton_x = (morton_x | (morton_x << 8))  & masks64[3];
            morton_x = (morton_x | (morton_x << 4))  & masks64[4];
            morton_x = (morton_x | (morton_x << 2))  & masks64[5];
            
            uint64_t morton_y = ((uint64_t)y) & masks64[0];
            morton_y = (morton_y | (morton_y << 32)) & masks64[1];
            morton_y = (morton_y | (morton_y << 16)) & masks64[2];
            morton_y = (morton_y | (morton_y << 8))  & masks64[3];
            morton_y = (morton_y | (morton_y << 4))  & masks64[4];
            morton_y = (morton_y | (morton_y << 2))  & masks64[5];
            
            uint64_t morton_z = ((uint64_t)z) & masks64[0];
            morton_z = (morton_z | (morton_z << 32)) & masks64[1];
            morton_z = (morton_z | (morton_z << 16)) & masks64[2];
            morton_z = (morton_z | (morton_z << 8))  & masks64[3];
            morton_z = (morton_z | (morton_z << 4))  & masks64[4];
            morton_z = (morton_z | (morton_z << 2))  & masks64[5];
            
            // Combine the encoded coordinates
            key = morton_x | (morton_y << 1) | (morton_z << 2);
        }

        keys[i] = key;
    }

    enum class CurveType
    {
        MORTON,
        HILBERT
    };

    __device__ int g_hilbert_counters[2];

    __global__ void fastHilbertPartitionKernel(
        const Body *bodies, int *indices, int numBodies,
        int axis, int bitPos, bool direction, int *leftCount)
    {
        extern __shared__ int s_flags[];
        
        int tid = threadIdx.x;
        int gid = blockIdx.x * blockDim.x + tid;
        
        if (tid < blockDim.x) {
            s_flags[tid] = 0;
        }
        __syncthreads();
        
        int flag = 0;
        if (gid < numBodies) {
            int bodyIdx = indices[gid];
            
            double value;
            switch(axis) {
                case 0: value = bodies[bodyIdx].position.x; break;
                case 1: value = bodies[bodyIdx].position.y; break;
                case 2: value = bodies[bodyIdx].position.z; break;
            }
            
            uint32_t intValue = *((uint32_t*)&value);
            bool bit = (intValue >> bitPos) & 1;
            
            if (direction) bit = !bit;
            
            flag = (bit == 0) ? 1 : 0;
        }
        
        s_flags[tid] = flag;
        __syncthreads();
        
        for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                s_flags[tid] += s_flags[tid + stride];
            }
            __syncthreads();
        }
        
        if (tid == 0 && s_flags[0] > 0) {
            atomicAdd(leftCount, s_flags[0]);
        }
    }
    
    __global__ void fastHilbertReorderKernel(
        const Body *bodies, int *oldIndices, int *newIndices, int numBodies,
        int axis, int bitPos, bool direction, int leftSize)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= numBodies) return;
        
        int bodyIdx = oldIndices[idx];
        
        double value;
        switch(axis) {
            case 0: value = bodies[bodyIdx].position.x; break;
            case 1: value = bodies[bodyIdx].position.y; break;
            case 2: value = bodies[bodyIdx].position.z; break;
        }
        
        uint32_t intValue = *((uint32_t*)&value);
        bool bit = (intValue >> bitPos) & 1;
        
        if (direction) bit = !bit;
        
        int destIdx;
        if (bit == 0) {
            destIdx = atomicAdd(&g_hilbert_counters[0], 1);
        } else {
            destIdx = leftSize + atomicAdd(&g_hilbert_counters[1], 1);
        }
        
        newIndices[destIdx] = bodyIdx;
    }

    class BodySorter
    {
    public:
        BodySorter(int numBodies, CurveType type) : nBodies(numBodies), curveType(type),
                                                    d_temp_storage_initialized(false)
        {
            cudaMalloc(&d_orderedIndices, numBodies * sizeof(int));
            cudaMalloc(&d_keys, numBodies * sizeof(uint64_t));

            cudaStreamCreate(&stream1);
            cudaStreamCreate(&stream2);
        }

        ~BodySorter()
        {
            if (d_orderedIndices)
                cudaFree(d_orderedIndices);
            if (d_keys)
                cudaFree(d_keys);
            if (d_temp_storage && d_temp_storage_initialized)
                cudaFree(d_temp_storage);

            cudaStreamDestroy(stream1);
            cudaStreamDestroy(stream2);
        }

        void setCurveType(CurveType type)
        {
            curveType = type;
        }

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
                // Para mantener compatibilidad con API, pero no se usa en el nuevo algoritmo
                return 0;
            }
        }

        uint64_t mortonEncode(uint32_t x, uint32_t y, uint32_t z)
        {
            return libmorton::morton3D_64_encode(x, y, z);
        }

        int *sortBodies(Body *d_bodies, const Vector &minBound, const Vector &maxBound)
        {
            int blockSize = 256;
            int gridSize = (nBodies + blockSize - 1) / blockSize;

            if (curveType == CurveType::MORTON) {
                extractPositionsAndCalculateKeysKernel<<<gridSize, blockSize, 0, stream1>>>(
                    d_bodies, d_keys, nBodies, minBound, maxBound, false);

                thrust::counting_iterator<int> first(0);
                thrust::copy(first, first + nBodies, thrust::device_pointer_cast(d_orderedIndices));

                if (!d_temp_storage_initialized)
                {
                    void *d_temp_storage = NULL;
                    size_t temp_storage_bytes = 0;

                    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                                d_keys, d_keys,
                                                d_orderedIndices, d_orderedIndices,
                                                nBodies);

                    cudaMalloc(&d_temp_storage, temp_storage_bytes);
                    d_temp_storage_size = temp_storage_bytes;
                    d_temp_storage_initialized = true;
                }

                cub::DeviceRadixSort::SortPairs(d_temp_storage, d_temp_storage_size,
                                            d_keys, d_keys,
                                            d_orderedIndices, d_orderedIndices,
                                            nBodies);

                return d_orderedIndices;
            } 
            else {
                return sortBodiesFastHilbert(d_bodies, minBound, maxBound);
            }
        }

    private:
        int nBodies;
        CurveType curveType;
        int *d_orderedIndices = nullptr;
        uint64_t *d_keys = nullptr;
        uint64_t *h_keys = nullptr;
        int *h_indices = nullptr;
        void *d_temp_storage = nullptr;
        size_t d_temp_storage_size = 0;
        bool d_temp_storage_initialized = false;
        Vector *d_positions = nullptr;
        Vector *h_positions = nullptr;
        cudaStream_t stream;
        cudaStream_t stream1, stream2;

        int* sortBodiesFastHilbert(Body* d_bodies, const Vector &minBound, const Vector &maxBound)
        {
            thrust::counting_iterator<int> first(0);
            thrust::copy(first, first + nBodies, thrust::device_pointer_cast(d_orderedIndices));
            
            int* d_tempIndices;
            cudaMalloc(&d_tempIndices, nBodies * sizeof(int));
            
            int maxBits = 30;
            
            hilbertSortRecursive(d_bodies, d_orderedIndices, d_tempIndices, 0, nBodies - 1, 
                                maxBits - 1, 0, false, minBound, maxBound);
            
            cudaFree(d_tempIndices);
            return d_orderedIndices;
        }

        void hilbertSortRecursive(Body* d_bodies, int* d_indices, int* d_tempIndices,
                                int start, int end, int bitPos, int axis, bool invertDir,
                                const Vector &minBound, const Vector &maxBound)
        {
            if (start >= end || bitPos < 0) return;
            
            int numElements = end - start + 1;
            
            int h_leftCount = 0;
            int *d_leftCount;
            cudaMalloc(&d_leftCount, sizeof(int));
            cudaMemset(d_leftCount, 0, sizeof(int));
            
            int blockSize = 256;
            int gridSize = (numElements + blockSize - 1) / blockSize;
            
            fastHilbertPartitionKernel<<<gridSize, blockSize, blockSize * sizeof(int), stream1>>>(
                d_bodies, d_indices + start, numElements, axis, bitPos, invertDir, d_leftCount);
            
            cudaMemcpy(&h_leftCount, d_leftCount, sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(d_leftCount);
            
            if (h_leftCount == 0 || h_leftCount == numElements) {
                int nextAxis = (axis + 1) % 3;
                hilbertSortRecursive(d_bodies, d_indices, d_tempIndices, 
                                   start, end, bitPos - 1, nextAxis, !invertDir,
                                   minBound, maxBound);
                return;
            }
            
            int h_counters[2] = {0, 0};
            cudaMemcpyToSymbol(g_hilbert_counters, h_counters, 2 * sizeof(int));
            
            fastHilbertReorderKernel<<<gridSize, blockSize, 0, stream1>>>(
                d_bodies, d_indices + start, d_tempIndices, numElements, 
                axis, bitPos, invertDir, h_leftCount);
            
            cudaMemcpy(d_indices + start, d_tempIndices, numElements * sizeof(int), cudaMemcpyDeviceToDevice);
            
            int mid = start + h_leftCount - 1;
            
            int nextAxisLeft = (axis + 1) % 3;
            hilbertSortRecursive(d_bodies, d_indices, d_tempIndices,
                               start, mid, bitPos - 1, nextAxisLeft, invertDir,
                               minBound, maxBound);
            
            int nextAxisRight = (axis + 1) % 3;
            hilbertSortRecursive(d_bodies, d_indices, d_tempIndices,
                               mid + 1, end, bitPos - 1, nextAxisRight, !invertDir,
                               minBound, maxBound);
        }
    };
}

enum class BodyDistribution
{
    RANDOM,
    SOLAR_SYSTEM,
    GALAXY,
    UNIFORM_SPHERE
};

enum class MassDistribution
{
    UNIFORM,
    NORMAL
};

constexpr int DEFAULT_BLOCK_SIZE = 256;
constexpr double DEFAULT_THETA = 0.5;

int g_blockSize = DEFAULT_BLOCK_SIZE;
double g_theta = DEFAULT_THETA;

bool dirExists(const std::string &dirName)
{
    struct stat info;
    return stat(dirName.c_str(), &info) == 0 && (info.st_mode & S_IFDIR);
}

inline void checkCudaError(cudaError_t err, const char *const func, const char *const file, int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

inline void checkLastCudaError(const char *const file, int line)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_CUDA_ERROR(err) checkCudaError(err, __func__, __FILE__, __LINE__)
#define CHECK_LAST_CUDA_ERROR() checkLastCudaError(__FILE__, __LINE__)

#define CUDA_KERNEL_CALL(kernel, gridSize, blockSize, sharedMem, stream, ...) \
    do                                                                        \
    {                                                                         \
        kernel<<<gridSize, blockSize, sharedMem, stream>>>(__VA_ARGS__);      \
        CHECK_LAST_CUDA_ERROR();                                              \
    } while (0)

class CudaTimer
{
public:
    CudaTimer(float &outputMs) : output(outputMs)
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    }

    ~CudaTimer()
    {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&output, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

private:
    float &output;
    cudaEvent_t start, stop;
};

class SFCDynamicReorderingStrategy
{
private:
    double reorderTime;
    double postReorderSimTime;
    double lastSimTime;
    double updateTime;
    double degradationRate;

    int iterationsSinceReorder;
    int currentOptimalFrequency;
    int iterationCounter;
    int consecutiveSkips;

    int metricsWindowSize;
    std::deque<double> reorderTimeHistory;
    std::deque<double> postReorderSimTimeHistory;
    std::deque<double> simulationTimeHistory;

    bool isInitialStageDone;
    int initialSampleSize;
    double initialPerformanceGain;

    double reorderCostScale;
    double performanceGainScale;

    double movingAverageRatio;
    int stableIterations;
    bool hasPerformancePlateau;

    int computeOptimalFrequency(int totalIterations)
    {

        if (simulationTimeHistory.size() < 3)
        {
            return 15;
        }

        double recent = simulationTimeHistory.front();
        double oldest = simulationTimeHistory.back();
        int historySize = simulationTimeHistory.size();
        double measuredDegradation = (recent - oldest) / std::max(1, historySize - 1);

        if (measuredDegradation > 0.001 && historySize > 5)
        {

            degradationRate = degradationRate * 0.7 + measuredDegradation * 0.3;
        }

        double effectiveDegradationRate = std::max(0.001, degradationRate);

        double effectiveReorderTime = reorderTime * reorderCostScale;

        if (hasPerformancePlateau && stableIterations > 20)
        {
            effectiveReorderTime *= 1.5;
        }

        double determinant = 1.0 - 2.0 * (updateTime - effectiveReorderTime) / (effectiveDegradationRate + 0.00001);

        if (determinant < 0)
            return 15;

        double optNu = -1.0 + sqrt(determinant);

        optNu = std::max(5.0, std::min(100.0, optNu));

        int optimalFreq = static_cast<int>(optNu);

        if (reorderTime > 5.0 * postReorderSimTime && optimalFreq < 20)
        {
            optimalFreq = 20;
        }

        if (movingAverageRatio > 1.1)
        {

            optimalFreq = std::max(5, static_cast<int>(optimalFreq * 0.9));
        }
        else if (movingAverageRatio < 1.02 && optimalFreq < 50)
        {

            optimalFreq = std::min(100, static_cast<int>(optimalFreq * 1.1));
        }

        return optimalFreq;
    }

    bool isReorderingBeneficial()
    {
        if (simulationTimeHistory.size() < 3)
            return true;

        double avgPerformanceBeforeReorder = 0.0;
        double avgPerformanceAfterReorder = 0.0;
        int countBefore = 0;
        int countAfter = 0;

        for (int i = std::min(3, (int)simulationTimeHistory.size() - 1); i < simulationTimeHistory.size(); i++)
        {
            avgPerformanceBeforeReorder += simulationTimeHistory[i];
            countBefore++;
        }

        for (int i = 0; i < std::min(3, (int)simulationTimeHistory.size()); i++)
        {
            avgPerformanceAfterReorder += simulationTimeHistory[i];
            countAfter++;
        }

        if (countBefore > 0)
            avgPerformanceBeforeReorder /= countBefore;
        if (countAfter > 0)
            avgPerformanceAfterReorder /= countAfter;

        if (countBefore == 0 || countAfter == 0)
            return true;

        double benefitRatio = avgPerformanceBeforeReorder / avgPerformanceAfterReorder;

        movingAverageRatio = movingAverageRatio * 0.8 + benefitRatio * 0.2;

        if (std::abs(benefitRatio - 1.0) < 0.03)
        {
            stableIterations++;
            if (stableIterations > 10)
            {
                hasPerformancePlateau = true;
            }
        }
        else
        {
            stableIterations = std::max(0, stableIterations - 1);
            if (stableIterations < 5)
            {
                hasPerformancePlateau = false;
            }
        }

        if (benefitRatio > 1.05)
        {

            reorderCostScale = std::max(0.6, reorderCostScale * 0.95);
            performanceGainScale = std::min(1.1, performanceGainScale * 1.05);
            return true;
        }
        else if (benefitRatio < 0.95)
        {

            reorderCostScale = std::min(1.5, reorderCostScale * 1.05);
            performanceGainScale = std::max(0.9, performanceGainScale * 0.95);
            return false;
        }

        return benefitRatio >= 1.0;
    }

public:
    SFCDynamicReorderingStrategy(int windowSize = 10)
        : reorderTime(0.0),
          postReorderSimTime(0.0),
          lastSimTime(0.0),
          updateTime(0.0),
          degradationRate(0.0005),
          iterationsSinceReorder(0),
          currentOptimalFrequency(15),
          iterationCounter(0),
          consecutiveSkips(0),
          metricsWindowSize(windowSize),
          isInitialStageDone(false),
          initialSampleSize(5),
          initialPerformanceGain(0.0),
          reorderCostScale(0.8),
          performanceGainScale(1.0),
          movingAverageRatio(1.0),
          stableIterations(0),
          hasPerformancePlateau(false)
    {
    }

    void updateMetrics(double newReorderTime, double newSimTime)
    {

        lastSimTime = newSimTime;
        iterationCounter++;

        if (newReorderTime > 0)
        {
            reorderTime = reorderTime * 0.7 + newReorderTime * 0.3;
            postReorderSimTime = newSimTime;
            iterationsSinceReorder = 0;
        }

        simulationTimeHistory.push_front(newSimTime);
        while (simulationTimeHistory.size() > metricsWindowSize)
            simulationTimeHistory.pop_back();

        if (newReorderTime > 0 && simulationTimeHistory.size() > 1)
        {

            if (!isInitialStageDone && iterationCounter >= initialSampleSize * 2)
            {
                double avgBefore = 0.0, avgAfter = 0.0;
                int beforeCount = 0, afterCount = 0;

                for (int i = initialSampleSize; i < std::min(initialSampleSize * 2, (int)simulationTimeHistory.size()); i++)
                {
                    avgBefore += simulationTimeHistory[i];
                    beforeCount++;
                }

                for (int i = 0; i < initialSampleSize && i < simulationTimeHistory.size(); i++)
                {
                    avgAfter += simulationTimeHistory[i];
                    afterCount++;
                }

                if (beforeCount > 0 && afterCount > 0)
                {
                    avgBefore /= beforeCount;
                    avgAfter /= afterCount;
                    initialPerformanceGain = (avgBefore - avgAfter) / avgBefore;

                    if (initialPerformanceGain > 0.01)
                    {
                        degradationRate = initialPerformanceGain / (initialSampleSize * 4.0);
                    }

                    isInitialStageDone = true;
                }
            }
        }

        if (simulationTimeHistory.size() > 3 && iterationsSinceReorder > 1)
        {
            double recent = simulationTimeHistory[0];
            double previous = simulationTimeHistory[1];
            double diff = recent - previous;

            if (diff > 0.01 && diff < previous * 0.3)
            {
                double newRate = diff;

                degradationRate = degradationRate * 0.9 + newRate * 0.1;
            }
        }

        iterationsSinceReorder++;
    }

    bool shouldReorder(double lastSimTime, double predictedReorderTime)
    {
        iterationsSinceReorder++;

        updateMetrics(0.0, lastSimTime);

        if (!isInitialStageDone)
        {
            bool shouldReorder = (iterationsSinceReorder >= initialSampleSize);
            if (shouldReorder)
                iterationsSinceReorder = 0;
            return shouldReorder;
        }

        if (iterationsSinceReorder % 5 == 0)
        {
            currentOptimalFrequency = computeOptimalFrequency(1000);
            currentOptimalFrequency = std::max(5, std::min(150, currentOptimalFrequency));
        }

        bool shouldReorder = false;

        if (iterationsSinceReorder >= currentOptimalFrequency)
        {

            if (isReorderingBeneficial())
            {
                shouldReorder = true;
                consecutiveSkips = 0;
            }
            else
            {

                consecutiveSkips++;
                if (consecutiveSkips >= 3)
                {
                    shouldReorder = true;
                    consecutiveSkips = 0;
                }
            }
        }

        if (!shouldReorder && iterationsSinceReorder > currentOptimalFrequency / 2)
        {

            if (simulationTimeHistory.size() >= 3)
            {
                double recent = simulationTimeHistory[0];
                double previous = simulationTimeHistory[std::min(2, (int)simulationTimeHistory.size() - 1)];
                if (recent > previous * 1.3)
                {
                    shouldReorder = true;
                }
            }
        }

        if (shouldReorder)
        {
            iterationsSinceReorder = 0;
        }

        return shouldReorder;
    }

    void updateMetrics(double sortTime)
    {

        updateMetrics(sortTime, lastSimTime);
    }

    void setWindowSize(int windowSize)
    {
        if (windowSize > 0)
        {
            metricsWindowSize = windowSize;
        }
    }

    int getOptimalFrequency() const
    {
        return currentOptimalFrequency;
    }

    double getDegradationRate() const
    {
        return degradationRate;
    }

    double getPerformanceGain() const
    {
        return initialPerformanceGain;
    }

    double getPerformanceRatio() const
    {
        return movingAverageRatio;
    }

    bool isPerformancePlateau() const
    {
        return hasPerformancePlateau;
    }

    void reset()
    {
        iterationsSinceReorder = 0;
        iterationCounter = 0;
        consecutiveSkips = 0;
        reorderTimeHistory.clear();
        postReorderSimTimeHistory.clear();
        simulationTimeHistory.clear();
        isInitialStageDone = false;
        movingAverageRatio = 1.0;
        stableIterations = 0;
        hasPerformancePlateau = false;
    }
};

class SFCBarnesHutGPU
{
private:
    Body *h_bodies = nullptr;
    Body *d_bodies = nullptr;
    Body *d_bodiesBuffer = nullptr;
    Node *h_nodes = nullptr;
    Node *d_nodes = nullptr;
    int *d_mutex = nullptr;
    int nBodies;
    int nNodes;
    int leafLimit;
    bool useSFC;
    sfc::BodySorter *sorter;
    int *d_orderedIndices;
    sfc::CurveType curveType;
    int reorderFrequency;
    int iterationCounter;
    Vector minBound;
    Vector maxBound;
    SimulationMetrics metrics;

    bool useDynamicReordering;
    SFCDynamicReorderingStrategy reorderingStrategy;

    double potentialEnergy;
    double kineticEnergy;
    double totalEnergyAvg;
    double potentialEnergyAvg;
    double kineticEnergyAvg;

    float minTimeMs;
    float maxTimeMs;

    double *d_potentialEnergy;
    double *d_kineticEnergy;
    double *h_potentialEnergy;
    double *h_kineticEnergy;

    void initializeDistribution(BodyDistribution dist, MassDistribution massDist, unsigned int seed)
    {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<double> pos_dist(-MAX_DIST, MAX_DIST);
        std::uniform_real_distribution<double> vel_dist(-1.0e3, 1.0e3);
        std::normal_distribution<double> normal_pos_dist(0.0, MAX_DIST / 2.0);
        std::normal_distribution<double> normal_vel_dist(0.0, 5.0e2);

        for (int i = 0; i < nBodies; i++)
        {
            if (massDist == MassDistribution::UNIFORM)
            {

                h_bodies[i].position = Vector(
                    CENTERX + pos_dist(gen),
                    CENTERY + pos_dist(gen),
                    CENTERZ + pos_dist(gen));

                h_bodies[i].velocity = Vector(
                    vel_dist(gen),
                    vel_dist(gen),
                    vel_dist(gen));
            }
            else
            {

                h_bodies[i].position = Vector(
                    CENTERX + normal_pos_dist(gen),
                    CENTERY + normal_pos_dist(gen),
                    CENTERZ + normal_pos_dist(gen));

                h_bodies[i].velocity = Vector(
                    normal_vel_dist(gen),
                    normal_vel_dist(gen),
                    normal_vel_dist(gen));
            }

            h_bodies[i].mass = 1.0;
            h_bodies[i].radius = pow(h_bodies[i].mass / EARTH_MASS, 1.0 / 3.0) * (EARTH_DIA / 2.0);
            h_bodies[i].isDynamic = true;
            h_bodies[i].acceleration = Vector(0, 0, 0);
        }
    }

    void updateBoundingBox()
    {

        if (useSFC)
        {

            Vector *h_positions = new Vector[nBodies];

            for (int i = 0; i < nBodies; i++)
            {
                size_t offset = i * sizeof(Body) + offsetof(Body, position);
                cudaMemcpy(&h_positions[i], (char *)d_bodies + offset, sizeof(Vector), cudaMemcpyDeviceToHost);
            }

            minBound = Vector(INFINITY, INFINITY, INFINITY);
            maxBound = Vector(-INFINITY, -INFINITY, -INFINITY);

#pragma omp parallel for reduction(min : minBound.x, minBound.y, minBound.z) reduction(max : maxBound.x, maxBound.y, maxBound.z) if (nBodies > 50000)
            for (int i = 0; i < nBodies; i++)
            {
                Vector pos = h_positions[i];

                minBound.x = std::min(minBound.x, pos.x);
                minBound.y = std::min(minBound.y, pos.y);
                minBound.z = std::min(minBound.z, pos.z);

                maxBound.x = std::max(maxBound.x, pos.x);
                maxBound.y = std::max(maxBound.y, pos.y);
                maxBound.z = std::max(maxBound.z, pos.z);
            }

            delete[] h_positions;

            double padding = std::max(1.0e10, (maxBound.x - minBound.x) * 0.01);
            minBound.x -= padding;
            minBound.y -= padding;
            minBound.z -= padding;
            maxBound.x += padding;
            maxBound.y += padding;
            maxBound.z += padding;
        }
    }

    void updateBoundsFromRoot()
    {

        if (d_nodes)
        {

            Node rootNode;
            cudaMemcpy(&rootNode, d_nodes, sizeof(Node), cudaMemcpyDeviceToHost);

            if (rootNode.bodyCount > 0 &&
                !std::isinf(rootNode.min.x) && !std::isinf(rootNode.max.x))
            {

                minBound = rootNode.min;
                maxBound = rootNode.max;

                Vector padding = (maxBound - minBound) * 0.05;
                minBound = minBound - padding;
                maxBound = maxBound + padding;

                return;
            }
        }

        updateBoundingBox();
    }

    void orderBodiesBySFC()
    {
        CudaTimer timer(metrics.reorderTimeMs);

        if (!useSFC || !sorter)
        {
            d_orderedIndices = nullptr;
            return;
        }

        updateBoundsFromRoot();

        d_orderedIndices = sorter->sortBodies(d_bodies, minBound, maxBound);
    }

    void checkInitialization()
    {
        if (!d_bodies || !d_nodes || !d_mutex || !d_bodiesBuffer)
        {
            std::cerr << "Error: Device memory not allocated." << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    void calculateEnergies()
    {

        if (d_bodies == nullptr)
        {
            std::cerr << "Error: Device bodies not initialized in calculateEnergies" << std::endl;
            return;
        }

        if (nBodies <= 0)
        {
            potentialEnergy = 0.0;
            kineticEnergy = 0.0;
            return;
        }

        cudaDeviceSynchronize();

        CudaTimer timer(metrics.simTimeMs);

        CHECK_CUDA_ERROR(cudaMemset(d_potentialEnergy, 0, sizeof(double)));
        CHECK_CUDA_ERROR(cudaMemset(d_kineticEnergy, 0, sizeof(double)));

        int blockSize = g_blockSize;

        blockSize = std::min(blockSize, nBodies);

        blockSize = (blockSize / 32) * 32;
        if (blockSize < 32)
            blockSize = 32;
        if (blockSize > 1024)
            blockSize = 1024;

        int gridSize = (nBodies + blockSize - 1) / blockSize;
        if (gridSize < 1)
            gridSize = 1;

        size_t sharedMemSize = 2 * blockSize * sizeof(double);

        int device;
        cudaGetDevice(&device);
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, device);

        if (sharedMemSize > props.sharedMemPerBlock)
        {

            blockSize = (props.sharedMemPerBlock / (2 * sizeof(double))) & ~0x1F;
            if (blockSize < 32)
            {
                std::cerr << "Error: Insufficient shared memory for energy calculation" << std::endl;
                return;
            }

            gridSize = (nBodies + blockSize - 1) / blockSize;
            sharedMemSize = 2 * blockSize * sizeof(double);
        }

        CUDA_KERNEL_CALL(CalculateEnergiesKernel, gridSize, blockSize, sharedMemSize, 0,
                         d_bodies, nBodies, d_potentialEnergy, d_kineticEnergy);

        CHECK_CUDA_ERROR(cudaMemcpy(h_potentialEnergy, d_potentialEnergy, sizeof(double), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(h_kineticEnergy, d_kineticEnergy, sizeof(double), cudaMemcpyDeviceToHost));

        potentialEnergy = *h_potentialEnergy;
        kineticEnergy = *h_kineticEnergy;
    }

    void initializeEnergyData()
    {

        h_potentialEnergy = new double[1];
        h_kineticEnergy = new double[1];

        *h_potentialEnergy = 0.0;
        *h_kineticEnergy = 0.0;

        CHECK_CUDA_ERROR(cudaMalloc(&d_potentialEnergy, sizeof(double)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_kineticEnergy, sizeof(double)));

        CHECK_CUDA_ERROR(cudaMemset(d_potentialEnergy, 0, sizeof(double)));
        CHECK_CUDA_ERROR(cudaMemset(d_kineticEnergy, 0, sizeof(double)));
    }

    void cleanupEnergyData()
    {

        if (d_potentialEnergy)
        {
            cudaFree(d_potentialEnergy);
            d_potentialEnergy = nullptr;
        }

        if (d_kineticEnergy)
        {
            cudaFree(d_kineticEnergy);
            d_kineticEnergy = nullptr;
        }

        if (h_potentialEnergy)
        {
            delete[] h_potentialEnergy;
            h_potentialEnergy = nullptr;
        }

        if (h_kineticEnergy)
        {
            delete[] h_kineticEnergy;
            h_kineticEnergy = nullptr;
        }
    }

public:
    SFCBarnesHutGPU(
        int numBodies,
        int numNodes,
        int leafNodeLimit,
        BodyDistribution bodyDist = BodyDistribution::GALAXY,
        MassDistribution massDist = MassDistribution::UNIFORM,
        bool useSpaceFillingCurve = false,
        sfc::CurveType curve = sfc::CurveType::MORTON,
        bool dynamicReordering = true,
        unsigned int seed = 1234)
        : nBodies(numBodies),
          nNodes(numNodes),
          leafLimit(std::min(numNodes - 1, leafNodeLimit)),
          useSFC(useSpaceFillingCurve),
          sorter(nullptr),
          d_orderedIndices(nullptr),
          curveType(curve),
          reorderFrequency(10),
          iterationCounter(0),
          useDynamicReordering(dynamicReordering),
          reorderingStrategy(10),
          potentialEnergy(0.0),
          kineticEnergy(0.0),
          totalEnergyAvg(0.0),
          potentialEnergyAvg(0.0),
          kineticEnergyAvg(0.0),
          minTimeMs(FLT_MAX),
          maxTimeMs(0.0f)
    {
        if (numBodies < 1)
            numBodies = 1;

        std::cout << "SFC Barnes-Hut GPU Simulation created with " << numBodies << " bodies." << std::endl;

        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        std::cout << "Using GPU: " << deviceProp.name << std::endl;

        if (useSFC)
        {
            std::string curveTypeStr = (curveType == sfc::CurveType::MORTON) ? "MORTON" : "HILBERT";
            std::cout << "SFC Mode: PARTICLES with "
                      << (useDynamicReordering ? "dynamic" : "fixed") << " reordering"
                      << ", Curve: " << curveTypeStr << std::endl;
        }

        h_bodies = new Body[nBodies];
        h_nodes = new Node[nNodes];

        CHECK_CUDA_ERROR(cudaMalloc(&d_bodies, nBodies * sizeof(Body)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_bodiesBuffer, nBodies * sizeof(Body)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_nodes, nNodes * sizeof(Node)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_mutex, nNodes * sizeof(int)));

        if (useSFC)
        {
            sorter = new sfc::BodySorter(numBodies, curveType);
        }

        initializeDistribution(bodyDist, massDist, seed);

        CHECK_CUDA_ERROR(cudaMemcpy(d_bodies, h_bodies, nBodies * sizeof(Body), cudaMemcpyHostToDevice));

        minBound = Vector(INFINITY, INFINITY, INFINITY);
        maxBound = Vector(-INFINITY, -INFINITY, -INFINITY);

        initializeEnergyData();
    }

    ~SFCBarnesHutGPU()
    {

        delete[] h_bodies;
        delete[] h_nodes;
        if (sorter)
            delete sorter;

        if (d_bodies)
            CHECK_CUDA_ERROR(cudaFree(d_bodies));
        if (d_bodiesBuffer)
            CHECK_CUDA_ERROR(cudaFree(d_bodiesBuffer));
        if (d_nodes)
            CHECK_CUDA_ERROR(cudaFree(d_nodes));
        if (d_mutex)
            CHECK_CUDA_ERROR(cudaFree(d_mutex));

        cleanupEnergyData();
    }

    void setCurveType(sfc::CurveType type)
    {
        if (type != curveType)
        {
            curveType = type;

            if (sorter)
                sorter->setCurveType(type);

            iterationCounter = reorderFrequency;
        }
    }

    void resetOctree()
    {
        CudaTimer timer(metrics.resetTimeMs);

        int blockSize = g_blockSize;
        int numBlocks = (nNodes + blockSize - 1) / blockSize;

        ResetKernel<<<numBlocks, blockSize>>>(d_nodes, d_mutex, nNodes, nBodies);
        CHECK_LAST_CUDA_ERROR();
    }

    void computeBoundingBox()
    {
        CudaTimer timer(metrics.bboxTimeMs);

        // Compute node bounds on CPU side
        if (h_bodies == nullptr || nBodies <= 0)
            return;

        Vector minPos = Vector(INFINITY, INFINITY, INFINITY);
        Vector maxPos = Vector(-INFINITY, -INFINITY, -INFINITY);

        // Copy positions from device to host
        for (int i = 0; i < nBodies; i++) {
            size_t offset = i * sizeof(Body) + offsetof(Body, position);
            Vector pos;
            cudaMemcpy(&pos, (char *)d_bodies + offset, sizeof(Vector), cudaMemcpyDeviceToHost);
            
            minPos.x = std::min(minPos.x, pos.x);
            minPos.y = std::min(minPos.y, pos.y);
            minPos.z = std::min(minPos.z, pos.z);
            
            maxPos.x = std::max(maxPos.x, pos.x);
            maxPos.y = std::max(maxPos.y, pos.y);
            maxPos.z = std::max(maxPos.z, pos.z);
        }

        // Add padding
        Vector padding = (maxPos - minPos) * 0.01;
        minPos = minPos - padding;
        maxPos = maxPos + padding;

        // Create a temporary node structure to copy to device
        Node rootNode;
        rootNode.min = minPos;
        rootNode.max = maxPos;
        rootNode.bodyIndex = 0;
        rootNode.bodyCount = nBodies;
        rootNode.isLeaf = true;
        rootNode.position = (minPos + maxPos) * 0.5;
        rootNode.mass = 0.0;
        
        // Calculate the radius
        Vector dimensions = maxPos - minPos;
        rootNode.radius = max(max(dimensions.x, dimensions.y), dimensions.z) * 0.5;

        // Copy root node to device
        cudaMemcpy(d_nodes, &rootNode, sizeof(Node), cudaMemcpyHostToDevice);
    }

    void constructOctree()
    {
        CudaTimer timer(metrics.octreeTimeMs);

        int blockSize = g_blockSize;
        
        // Calculate required shared memory size for the kernel
        size_t sharedMemSize = blockSize * sizeof(double) + blockSize * sizeof(double3);

        // Launch the kernel with a single block
        ConstructOctTreeKernel<<<1, blockSize, sharedMemSize>>>(
            d_nodes, d_bodies, d_bodiesBuffer, 0, nNodes, nBodies, leafLimit);
        
        CHECK_LAST_CUDA_ERROR();
    }

    void computeForces()
    {
        CudaTimer timer(metrics.forceTimeMs);

        int blockSize = g_blockSize;
        int gridSize = (nBodies + blockSize - 1) / blockSize;

        // Launch the force calculation kernel
        ComputeForceKernel<<<gridSize, blockSize>>>(
            d_nodes, d_bodies, nNodes, nBodies, leafLimit, g_theta);
            
        CHECK_LAST_CUDA_ERROR();
    }

    void update()
    {
        CudaTimer timer(metrics.totalTimeMs);

        checkInitialization();

        bool shouldReorder = false;
        if (useSFC)
        {

            shouldReorder = reorderingStrategy.shouldReorder(metrics.forceTimeMs, metrics.reorderTimeMs);

            if (shouldReorder)
            {
                orderBodiesBySFC();
                iterationCounter = 0;
            }
        }

        resetOctree();

        constructOctree();

        computeForces();

        calculateEnergies();

        iterationCounter++;

        if (useSFC)
        {
            reorderingStrategy.updateMetrics(shouldReorder ? metrics.reorderTimeMs : 0.0, metrics.forceTimeMs);
        }
    }

    void printPerformance()
    {
        printf("Performance Metrics:\n");
        printf("  Reset:       %.2f ms\n", metrics.resetTimeMs);
        printf("  Bounding box: %.2f ms\n", metrics.bboxTimeMs);
        printf("  Tree build:   %.2f ms\n", metrics.octreeTimeMs);
        printf("  Force calc:   %.2f ms\n", metrics.forceTimeMs);
        if (useSFC)
        {
            printf("  Reordering:   %.2f ms\n", metrics.reorderTimeMs);
            printf("  Optimal reorder freq: %d\n", reorderingStrategy.getOptimalFrequency());
            printf("  Degradation rate: %.6f ms/iter\n", reorderingStrategy.getDegradationRate());
        }
        printf("  Total update: %.2f ms\n", metrics.totalTimeMs);
    }

    void runSimulation(int numIterations, int printFreq = 10)
    {

        double totalTime = 0.0;
        minTimeMs = FLT_MAX;
        maxTimeMs = 0.0f;
        potentialEnergyAvg = 0.0;
        kineticEnergyAvg = 0.0;
        totalEnergyAvg = 0.0;

        std::cout << "Starting SFC Barnes-Hut GPU simulation..." << std::endl;
        std::cout << "Bodies: " << nBodies << ", Nodes: " << nNodes << std::endl;
        std::cout << "Theta parameter: " << g_theta << std::endl;
        if (useSFC)
            std::cout << "Using SFC ordering, curve type: " << (curveType == sfc::CurveType::MORTON ? "MORTON" : "HILBERT") << std::endl;
        else
            std::cout << "No SFC ordering used." << std::endl;

        auto startTime = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < numIterations; i++)
        {

            update();

            potentialEnergyAvg += potentialEnergy;
            kineticEnergyAvg += kineticEnergy;
            totalEnergyAvg += (potentialEnergy + kineticEnergy);

            minTimeMs = std::min(minTimeMs, metrics.totalTimeMs);
            maxTimeMs = std::max(maxTimeMs, metrics.totalTimeMs);

            totalTime += metrics.totalTimeMs;

            cudaDeviceSynchronize();
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        double simTimeMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();

        if (numIterations > 0)
        {
            potentialEnergyAvg /= numIterations;
            kineticEnergyAvg /= numIterations;
            totalEnergyAvg /= numIterations;
        }

        std::cout << "Simulation completed in " << simTimeMs << " ms." << std::endl;
        printSummary(numIterations);
    }

    void run(int steps)
    {
        std::cout << "Running SFC Barnes-Hut GPU simulation for " << steps << " steps..." << std::endl;

        float totalTime = 0.0f;
        float totalForceTime = 0.0f;
        float totalBuildTime = 0.0f;
        float totalBboxTime = 0.0f;
        float totalReorderTime = 0.0f;
        float minTime = std::numeric_limits<float>::max();
        float maxTime = 0.0f;
        double totalPotentialEnergy = 0.0;
        double totalKineticEnergy = 0.0;

        for (int step = 0; step < steps; step++)
        {
            update();

            totalTime += metrics.totalTimeMs;
            totalForceTime += metrics.forceTimeMs;
            totalBuildTime += metrics.octreeTimeMs;
            totalBboxTime += metrics.bboxTimeMs;
            totalReorderTime += metrics.reorderTimeMs;
            minTime = std::min(minTime, metrics.totalTimeMs);
            maxTime = std::max(maxTime, metrics.totalTimeMs);
            totalPotentialEnergy += potentialEnergy;
            totalKineticEnergy += kineticEnergy;
        }

        potentialEnergyAvg = totalPotentialEnergy / steps;
        kineticEnergyAvg = totalKineticEnergy / steps;
        totalEnergyAvg = potentialEnergyAvg + kineticEnergyAvg;

        std::cout << "Simulation complete." << std::endl;
        std::cout << "Performance Summary:" << std::endl;
        std::cout << "  Average time per step: " << std::fixed << std::setprecision(2) << totalTime / steps << " ms" << std::endl;
        std::cout << "  Min time: " << std::fixed << std::setprecision(2) << minTime << " ms" << std::endl;
        std::cout << "  Max time: " << std::fixed << std::setprecision(2) << maxTime << " ms" << std::endl;
        std::cout << "  Build tree: " << std::fixed << std::setprecision(2) << totalBuildTime / steps << " ms" << std::endl;
        std::cout << "  Bounding box: " << std::fixed << std::setprecision(2) << totalBboxTime / steps << " ms" << std::endl;
        std::cout << "  Compute forces: " << std::fixed << std::setprecision(2) << totalForceTime / steps << " ms" << std::endl;
        if (useSFC)
        {
            std::cout << "  Reordering: " << std::fixed << std::setprecision(2) << totalReorderTime / steps << " ms" << std::endl;
        }
        std::cout << "Average Energy Values:" << std::endl;
        std::cout << "  Potential Energy: " << std::scientific << std::setprecision(6) << potentialEnergyAvg << std::endl;
        std::cout << "  Kinetic Energy:   " << std::scientific << std::setprecision(6) << kineticEnergyAvg << std::endl;
        std::cout << "  Total Energy:     " << std::scientific << std::setprecision(6) << totalEnergyAvg << std::endl;

        if (useSFC && reorderingStrategy.getOptimalFrequency() > 0)
        {
            std::cout << "SFC Configuration:" << std::endl;
            std::cout << "  Optimal reorder freq: " << reorderingStrategy.getOptimalFrequency() << std::endl;
            std::cout << "  Degradation rate: " << std::fixed << std::setprecision(6)
                      << reorderingStrategy.getDegradationRate() << " ms/iter" << std::endl;
        }
    }

    float getTotalTime() const { return metrics.totalTimeMs; }
    float getForceCalculationTime() const { return metrics.forceTimeMs; }
    float getBuildTime() const { return metrics.octreeTimeMs; }
    float getBboxTime() const { return metrics.bboxTimeMs; }
    float getReorderTime() const { return metrics.reorderTimeMs; }
    float getEnergyCalculationTime() const { return metrics.simTimeMs; }
    double getPotentialEnergy() const { return potentialEnergy; }
    double getKineticEnergy() const { return kineticEnergy; }
    double getTotalEnergy() const { return potentialEnergy + kineticEnergy; }
    double getPotentialEnergyAvg() const { return potentialEnergyAvg; }
    double getKineticEnergyAvg() const { return kineticEnergyAvg; }
    double getTotalEnergyAvg() const { return totalEnergyAvg; }
    int getBlockSize() const { return g_blockSize; }
    int getNumBodies() const { return nBodies; }
    sfc::CurveType getCurveType() const { return curveType; }
    bool isDynamicReordering() const { return true; }
    double getTheta() const { return g_theta; }
    const char *getSortType() const { return useSFC ? (curveType == sfc::CurveType::MORTON ? "MORTON" : "HILBERT") : "NONE"; }
    float getTreeBuildTime() const { return metrics.octreeTimeMs; }
    float getSortTime() const { return metrics.reorderTimeMs; }

    void printSummary(int steps)
    {
        double totalTime = metrics.totalTimeMs;
        double totalBuildTime = metrics.octreeTimeMs;
        double totalBboxTime = metrics.bboxTimeMs;
        double totalForceTime = metrics.forceTimeMs;
        double totalReorderTime = metrics.reorderTimeMs;

        std::cout << "Performance Summary:" << std::endl;
        std::cout << "  Average time per step: " << std::fixed << std::setprecision(2) << totalTime / steps << " ms" << std::endl;
        std::cout << "  Min time: " << std::fixed << std::setprecision(2) << minTimeMs << " ms" << std::endl;
        std::cout << "  Max time: " << std::fixed << std::setprecision(2) << maxTimeMs << " ms" << std::endl;
        std::cout << "  Build tree: " << std::fixed << std::setprecision(2) << totalBuildTime / steps << " ms" << std::endl;
        std::cout << "  Bounding box: " << std::fixed << std::setprecision(2) << totalBboxTime / steps << " ms" << std::endl;
        std::cout << "  Compute forces: " << std::fixed << std::setprecision(2) << totalForceTime / steps << " ms" << std::endl;
        if (useSFC)
        {
            std::cout << "  Reordering: " << std::fixed << std::setprecision(2) << totalReorderTime / steps << " ms" << std::endl;
        }
        std::cout << "Average Energy Values:" << std::endl;
        std::cout << "  Potential Energy: " << std::scientific << std::setprecision(6) << potentialEnergyAvg << std::endl;
        std::cout << "  Kinetic Energy:   " << std::scientific << std::setprecision(6) << kineticEnergyAvg << std::endl;
        std::cout << "  Total Energy:     " << std::scientific << std::setprecision(6) << totalEnergyAvg << std::endl;

        if (useSFC)
        {
            std::cout << "SFC Configuration:" << std::endl;
            std::cout << "  Optimal reorder freq: " << reorderingStrategy.getOptimalFrequency() << std::endl;
            std::cout << "  Degradation rate: " << std::fixed << std::setprecision(6)
                      << reorderingStrategy.getDegradationRate() << " ms/iter" << std::endl;
            std::cout << "  Performance ratio: " << std::fixed << std::setprecision(3)
                      << reorderingStrategy.getPerformanceRatio() << std::endl;
            std::cout << "  Performance plateau: " << (reorderingStrategy.isPerformancePlateau() ? "yes" : "no") << std::endl;
        }
    }
};

int main(int argc, char **argv)
{
    ArgumentParser parser("BarnesHut GPU Simulation");
    
    parser.add_argument("n", "Number of bodies", 10000);
    parser.add_flag("nosfc", "Disable Space-Filling Curve ordering");
    parser.add_argument("freq", "Reordering frequency for fixed mode", 10);
    parser.add_argument("dist", "Body distribution (galaxy, solar, uniform, random)", std::string("galaxy"));
    parser.add_argument("mass", "Mass distribution (uniform, normal)", std::string("normal"));
    parser.add_argument("seed", "Random seed", 42);
    parser.add_argument("curve", "SFC curve type (morton, hilbert)", std::string("morton"));
    parser.add_argument("iter", "Number of iterations", 100);
    parser.add_argument("block", "CUDA block size", DEFAULT_BLOCK_SIZE);
    parser.add_argument("theta", "Barnes-Hut opening angle parameter", DEFAULT_THETA);
    parser.add_argument("sm", "Shared memory size per block (bytes, 0=auto)", 0);
    parser.add_argument("leaf", "Maximum bodies per leaf node", 1);
    parser.add_flag("energy", "Calculate system energy");
    
    // Parse command line arguments
    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        parser.print_help();
        return 1;
    }
    
    // Extract parsed arguments
    int nBodies = parser.get<int>("n");
    bool useSFC = !parser.get<bool>("nosfc");
    int reorderFreq = parser.get<int>("freq");
    
    // Parse distribution type
    std::string distStr = parser.get<std::string>("dist");
    BodyDistribution bodyDist = BodyDistribution::GALAXY;
    if (distStr == "solar") {
        bodyDist = BodyDistribution::SOLAR_SYSTEM;
    } else if (distStr == "uniform") {
        bodyDist = BodyDistribution::UNIFORM_SPHERE;
    } else if (distStr == "random") {
        bodyDist = BodyDistribution::RANDOM;
    }
    
    // Parse mass distribution
    std::string massStr = parser.get<std::string>("mass");
    MassDistribution massDist = MassDistribution::NORMAL;
    if (massStr == "uniform") {
        massDist = MassDistribution::UNIFORM;
    }
    
    unsigned int seed = parser.get<int>("seed");
    
    // Parse curve type
    std::string curveStr = parser.get<std::string>("curve");
    sfc::CurveType curveType = sfc::CurveType::MORTON;
    if (curveStr == "hilbert") {
        curveType = sfc::CurveType::HILBERT;
    }
    
    int numIterations = parser.get<int>("iter");
    int blockSize = parser.get<int>("block");
    double theta = parser.get<double>("theta");
    
    bool useDynamicReordering = true;

    g_blockSize = blockSize;
    g_theta = theta;

    SFCBarnesHutGPU simulation(
        nBodies,
        nBodies * 16, // Increased from 8 to 16 to provide more nodes
        8,
        bodyDist,
        massDist,
        useSFC,
        curveType,
        useDynamicReordering,
        seed);

    simulation.runSimulation(numIterations);

    return 0;
}
