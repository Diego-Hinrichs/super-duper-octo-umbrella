#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <string>
#include <random>
#include <limits>
#include <algorithm>
#include <vector>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <deque>
#include <numeric>
#include "../../argparse.hpp"

class SFCDynamicReorderingStrategy
{
private:
    double reorderTime;
    double postReorderSimTime;
    double updateTime;
    double degradationRate;

    int iterationsSinceReorder;
    int currentOptimalFrequency;

    int metricsWindowSize;
    std::deque<double> reorderTimeHistory;
    std::deque<double> postReorderSimTimeHistory;
    std::deque<double> simulationTimeHistory;

    int computeOptimalFrequency(int totalIterations)
    {

        double determinant = 1.0 - 2.0 * (updateTime - reorderTime) / degradationRate;

        if (determinant < 0)
            return 10;

        double optNu = -1.0 + sqrt(determinant);

        int nu1 = static_cast<int>(optNu);
        int nu2 = nu1 + 1;

        if (nu1 <= 0)
            return 1;

        double time1 = ((nu1 * nu1 * degradationRate / 2.0) + nu1 * (updateTime + postReorderSimTime) +
                        (reorderTime + postReorderSimTime)) *
                       totalIterations / (nu1 + 1.0);
        double time2 = ((nu2 * nu2 * degradationRate / 2.0) + nu2 * (updateTime + postReorderSimTime) +
                        (reorderTime + postReorderSimTime)) *
                       totalIterations / (nu2 + 1.0);

        return time1 < time2 ? nu1 : nu2;
    }

    void updateMetrics(double newReorderTime, double newSimTime)
    {

        if (newReorderTime > 0)
        {
            reorderTimeHistory.push_back(newReorderTime);
            if (reorderTimeHistory.size() > metricsWindowSize)
            {
                reorderTimeHistory.pop_front();
            }

            reorderTime = std::accumulate(reorderTimeHistory.begin(), reorderTimeHistory.end(), 0.0) /
                          reorderTimeHistory.size();
        }

        simulationTimeHistory.push_back(newSimTime);
        if (simulationTimeHistory.size() > metricsWindowSize)
        {
            simulationTimeHistory.pop_front();
        }

        if (iterationsSinceReorder == 1)
        {
            postReorderSimTimeHistory.push_back(newSimTime);
            if (postReorderSimTimeHistory.size() > metricsWindowSize)
            {
                postReorderSimTimeHistory.pop_front();
            }

            postReorderSimTime = std::accumulate(postReorderSimTimeHistory.begin(),
                                                 postReorderSimTimeHistory.end(), 0.0) /
                                 postReorderSimTimeHistory.size();
        }

        if (simulationTimeHistory.size() >= 3)
        {

            int n = std::min(5, static_cast<int>(simulationTimeHistory.size()));
            double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;

            for (int i = 0; i < n; i++)
            {
                double x = i;
                double y = simulationTimeHistory[simulationTimeHistory.size() - n + i];
                sumX += x;
                sumY += y;
                sumXY += x * y;
                sumX2 += x * x;
            }

            double slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
            if (slope > 0)
            {
                degradationRate = slope;
            }
        }
    }

public:
    SFCDynamicReorderingStrategy(int windowSize = 10)
        : reorderTime(0.0),
          postReorderSimTime(0.0),
          updateTime(0.0),
          degradationRate(0.001),
          iterationsSinceReorder(0),
          currentOptimalFrequency(10),
          metricsWindowSize(windowSize)
    {
    }

    bool shouldReorder(double lastSimTime = 0.0, double lastReorderTime = 0.0)
    {
        iterationsSinceReorder++;

        updateMetrics(lastReorderTime, lastSimTime);

        if (iterationsSinceReorder % 10 == 0)
        {
            currentOptimalFrequency = computeOptimalFrequency(1000);

            currentOptimalFrequency = std::max(1, std::min(100, currentOptimalFrequency));
        }

        bool shouldReorder = iterationsSinceReorder >= currentOptimalFrequency;

        if (shouldReorder)
        {
            iterationsSinceReorder = 0;
        }

        return shouldReorder;
    }

    void updateMetrics(double sortTime)
    {

        updateMetrics(sortTime, 0.0);
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

    void reset()
    {
        iterationsSinceReorder = 0;
        reorderTimeHistory.clear();
        postReorderSimTimeHistory.clear();
        simulationTimeHistory.clear();
    }
};

struct Vector
{
    double x;
    double y;
    double z;

    __host__ __device__ Vector() : x(0.0), y(0.0), z(0.0) {}

    __host__ __device__ Vector(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    __host__ __device__ Vector operator+(const Vector &other) const
    {
        return Vector(x + other.x, y + other.y, z + other.z);
    }

    __host__ __device__ Vector operator-(const Vector &other) const
    {
        return Vector(x - other.x, y - other.y, z - other.z);
    }

    __host__ __device__ Vector operator*(double scalar) const
    {
        return Vector(x * scalar, y * scalar, z * scalar);
    }

    __host__ __device__ double dot(const Vector &other) const
    {
        return x * other.x + y * other.y + z * other.z;
    }

    __host__ __device__ double lengthSquared() const
    {
        return x * x + y * y + z * z;
    }

    __host__ __device__ double length() const
    {
        return sqrt(lengthSquared());
    }

    __host__ __device__ Vector normalize() const
    {
        double len = length();
        if (len > 0.0)
        {
            return Vector(x / len, y / len, z / len);
        }
        return *this;
    }

    __host__ __device__ static double distance(const Vector &a, const Vector &b)
    {
        return (a - b).length();
    }

    __host__ __device__ static double distanceSquared(const Vector &a, const Vector &b)
    {
        return (a - b).lengthSquared();
    }
};

struct Body
{
    bool isDynamic;
    double mass;
    double radius;
    Vector position;
    Vector velocity;
    Vector acceleration;

    __host__ __device__ Body() : isDynamic(true),
                                 mass(0.0),
                                 radius(0.0),
                                 position(),
                                 velocity(),
                                 acceleration() {}
};

struct SimulationMetrics
{
    float forceTimeMs;
    float reorderTimeMs;
    float totalTimeMs;
    float energyCalculationTimeMs;

    SimulationMetrics() : forceTimeMs(0.0f),
                          reorderTimeMs(0.0f),
                          totalTimeMs(0.0f),
                          energyCalculationTimeMs(0.0f) {}
};

namespace sfc
{
    enum class CurveType
    {
        MORTON,
        HILBERT
    };

    class BodySorter
    {
    public:
        BodySorter(int numBodies, CurveType type) : nBodies(numBodies), curveType(type)
        {

            cudaMalloc(&d_orderedIndices, numBodies * sizeof(int));

            h_keys = new uint64_t[numBodies];
            h_indices = new int[numBodies];
        }

        ~BodySorter()
        {
            if (d_orderedIndices)
                cudaFree(d_orderedIndices);
            if (h_keys)
                delete[] h_keys;
            if (h_indices)
                delete[] h_indices;
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

            normalizedX = std::max(0.0, std::min(1.0, normalizedX));
            normalizedY = std::max(0.0, std::min(1.0, normalizedY));
            normalizedZ = std::max(0.0, std::min(1.0, normalizedZ));

            uint32_t x = static_cast<uint32_t>(normalizedX * ((1 << 21) - 1));
            uint32_t y = static_cast<uint32_t>(normalizedY * ((1 << 21) - 1));
            uint32_t z = static_cast<uint32_t>(normalizedZ * ((1 << 21) - 1));

            if (curveType == CurveType::MORTON)
            {
                return mortonEncode(x, y, z);
            }
            else
            {
                return hilbertEncode(x, y, z);
            }
        }

        uint64_t mortonEncode(uint32_t x, uint32_t y, uint32_t z)
        {

            uint64_t answer = 0;
            for (uint64_t i = 0; i < 21; ++i)
            {
                answer |= ((x & ((uint64_t)1 << i)) << (2 * i)) |
                          ((y & ((uint64_t)1 << i)) << (2 * i + 1)) |
                          ((z & ((uint64_t)1 << i)) << (2 * i + 2));
            }
            return answer;
        }

        uint64_t hilbertEncode(uint32_t x, uint32_t y, uint32_t z)
        {

            uint64_t result = 0;

            for (int i = 20; i >= 0; i--)
            {
                uint8_t bitPos = i * 3;
                uint8_t bitX = (x >> i) & 1;
                uint8_t bitY = (y >> i) & 1;
                uint8_t bitZ = (z >> i) & 1;

                uint8_t idx = (bitX << 2) | (bitY << 1) | bitZ;
                result = (result << 3) | idx;
            }

            return result;
        }

        int *sortBodies(Body *d_bodies, const Vector &minBound, const Vector &maxBound)
        {

            Body *h_bodies = new Body[nBodies];
            cudaMemcpy(h_bodies, d_bodies, nBodies * sizeof(Body), cudaMemcpyDeviceToHost);

            for (int i = 0; i < nBodies; i++)
            {
                h_keys[i] = calculateSFCKey(h_bodies[i].position, minBound, maxBound);
                h_indices[i] = i;
            }

            std::sort(h_indices, h_indices + nBodies,
                      [this](int a, int b)
                      { return h_keys[a] < h_keys[b]; });

            cudaMemcpy(d_orderedIndices, h_indices, nBodies * sizeof(int), cudaMemcpyHostToDevice);

            delete[] h_bodies;

            return d_orderedIndices;
        }

    private:
        int nBodies;
        CurveType curveType;
        int *d_orderedIndices = nullptr;
        uint64_t *h_keys = nullptr;
        int *h_indices = nullptr;
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

constexpr double GRAVITY = 6.67430e-11;
constexpr double SOFTENING_FACTOR = 0.5;
constexpr double TIME_STEP = 25000.0;
constexpr double COLLISION_THRESHOLD = 1.0e10;

constexpr double MAX_DIST = 5.0e11;
constexpr double MIN_DIST = 2.0e10;
constexpr double EARTH_MASS = 5.974e24;
constexpr double EARTH_DIA = 12756.0;
constexpr double SUN_MASS = 1.989e30;
constexpr double SUN_DIA = 1.3927e6;
constexpr double CENTERX = 0;
constexpr double CENTERY = 0;
constexpr double CENTERZ = 0;

constexpr int DEFAULT_BLOCK_SIZE = 256;
constexpr double DEFAULT_DOMAIN_SIZE = 1.0e6; // Default domain size for periodic boundaries

#define E SOFTENING_FACTOR
#define DT TIME_STEP
#define COLLISION_TH COLLISION_THRESHOLD

int g_blockSize = DEFAULT_BLOCK_SIZE;
double g_domainSize = DEFAULT_DOMAIN_SIZE;

// Periodic boundary condition functions
__device__ Vector applyPeriodicBoundary_device(Vector rij, double domainSize)
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

__device__ Vector applyPeriodicPosition_device(Vector position, double domainSize)
{
    Vector result = position;
    
    // Wrap positions to stay within [0, L)
    result.x = fmod(result.x + domainSize, domainSize);
    if (result.x < 0) result.x += domainSize;
    
    result.y = fmod(result.y + domainSize, domainSize);
    if (result.y < 0) result.y += domainSize;
    
    result.z = fmod(result.z + domainSize, domainSize);
    if (result.z < 0) result.z += domainSize;
    
    return result;
}

// Host versions for initialization
Vector applyPeriodicPosition(Vector position, double domainSize)
{
    Vector result = position;
    
    // Wrap positions to stay within [0, L)
    result.x = fmod(result.x + domainSize, domainSize);
    if (result.x < 0) result.x += domainSize;
    
    result.y = fmod(result.y + domainSize, domainSize);
    if (result.y < 0) result.y += domainSize;
    
    result.z = fmod(result.z + domainSize, domainSize);
    if (result.z < 0) result.z += domainSize;
    
    return result;
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

#define CHECK_CUDA_ERROR(val) checkCudaError((val), #val, __FILE__, __LINE__)
#define CHECK_LAST_CUDA_ERROR() checkLastCudaError(__FILE__, __LINE__)

#define CUDA_KERNEL_CALL(kernel, gridSize, blockSize, sharedMem, stream, ...) \
    do                                                                        \
    {                                                                         \
        kernel<<<gridSize, blockSize, sharedMem, stream>>>(__VA_ARGS__);      \
        CHECK_LAST_CUDA_ERROR();                                              \
    } while (0)

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

__global__ void SFCDirectSumForceKernel(Body *bodies, int *orderedIndices, bool useSFC, int nBodies, double domainSize)
{

    extern __shared__ char sharedMemory[];
    Vector *sharedPos = (Vector *)sharedMemory;
    double *sharedMass = (double *)(sharedPos + blockDim.x);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;

    int realBodyIndex = (useSFC && orderedIndices != nullptr) ? orderedIndices[i] : i;

    Vector myPos = Vector(0, 0, 0);
    Vector myVel = Vector(0, 0, 0);
    Vector myAcc = Vector(0, 0, 0);
    double myMass = 0.0;
    bool isDynamic = false;

    if (i < nBodies)
    {
        myPos = bodies[realBodyIndex].position;
        myVel = bodies[realBodyIndex].velocity;
        myMass = bodies[realBodyIndex].mass;
        isDynamic = bodies[realBodyIndex].isDynamic;
    }

    const int tileSize = blockDim.x;

    for (int tile = 0; tile < (nBodies + tileSize - 1) / tileSize; ++tile)
    {

        int idx = tile * tileSize + tx;

        if (tx < tileSize)
        {
            if (idx < nBodies)
            {

                int tileBodyIndex = (useSFC && orderedIndices != nullptr) ? orderedIndices[idx] : idx;
                sharedPos[tx] = bodies[tileBodyIndex].position;
                sharedMass[tx] = bodies[tileBodyIndex].mass;
            }
            else
            {
                sharedPos[tx] = Vector(0, 0, 0);
                sharedMass[tx] = 0.0;
            }
        }

        __syncthreads();

        if (i < nBodies && isDynamic)
        {

            int tileLimit = min(tileSize, nBodies - tile * tileSize);

            for (int j = 0; j < tileLimit; ++j)
            {
                int jBody = tile * tileSize + j;

                if (jBody != i)
                {

                    Vector diff = sharedPos[j] - myPos;
                    
                    // Apply periodic boundary conditions
                    diff = applyPeriodicBoundary_device(diff, domainSize);

                    double distSqr = diff.lengthSquared() + E * E;
                    double dist = sqrt(distSqr);

                    if (dist >= COLLISION_TH)
                    {
                        double forceMag = (GRAVITY * myMass * sharedMass[j]) / (dist * distSqr);

                        myAcc.x += diff.x * forceMag / myMass;
                        myAcc.y += diff.y * forceMag / myMass;
                        myAcc.z += diff.z * forceMag / myMass;
                    }
                }
            }
        }

        __syncthreads();
    }

    if (i < nBodies && isDynamic)
    {

        bodies[realBodyIndex].acceleration = myAcc;

        myVel.x += myAcc.x * DT;
        myVel.y += myAcc.y * DT;
        myVel.z += myAcc.z * DT;
        bodies[realBodyIndex].velocity = myVel;

        myPos.x += myVel.x * DT;
        myPos.y += myVel.y * DT;
        myPos.z += myVel.z * DT;
        
        // Apply periodic boundary conditions to positions
        myPos = applyPeriodicPosition_device(myPos, domainSize);
        bodies[realBodyIndex].position = myPos;
    }
}

__global__ void CalculateEnergiesKernel(Body *bodies, int nBodies, double *d_potentialEnergy, double *d_kineticEnergy, double domainSize)
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
            
            // Apply periodic boundary conditions
            r = applyPeriodicBoundary_device(r, domainSize);

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

class SFCDirectSumGPU
{
private:
    Body *h_bodies = nullptr;
    Body *d_bodies = nullptr;
    int nBodies;
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

    double *d_potentialEnergy;
    double *d_kineticEnergy;
    double *h_potentialEnergy;
    double *h_kineticEnergy;

    void updateBoundingBox()
    {
        // Use fixed domain bounds for periodic boundary conditions
        minBound = Vector(0.0, 0.0, 0.0);
        maxBound = Vector(g_domainSize, g_domainSize, g_domainSize);
    }

    void orderBodiesBySFC()
    {
        CudaTimer timer(metrics.reorderTimeMs);

        if (!useSFC || !sorter)
        {
            d_orderedIndices = nullptr;
            return;
        }

        CHECK_CUDA_ERROR(cudaMemcpy(h_bodies, d_bodies, nBodies * sizeof(Body), cudaMemcpyDeviceToHost));

        updateBoundingBox();

        d_orderedIndices = sorter->sortBodies(d_bodies, minBound, maxBound);
    }

    void initializeDistribution(BodyDistribution dist, MassDistribution massDist, unsigned int seed)
    {
        std::mt19937 gen(seed);
        // Use domain-relative distributions for periodic boundaries
        std::uniform_real_distribution<double> pos_dist(0.0, g_domainSize);
        std::uniform_real_distribution<double> vel_dist(-1.0e3, 1.0e3);
        std::normal_distribution<double> normal_pos_dist(g_domainSize * 0.5, g_domainSize * 0.25);
        std::normal_distribution<double> normal_vel_dist(0.0, 5.0e2);

        for (int i = 0; i < nBodies; i++)
        {
            if (massDist == MassDistribution::UNIFORM)
            {
                // Uniform distribution within the domain [0, L]
                h_bodies[i].position = Vector(
                    pos_dist(gen),
                    pos_dist(gen),
                    pos_dist(gen));

                h_bodies[i].velocity = Vector(
                    vel_dist(gen),
                    vel_dist(gen),
                    vel_dist(gen));
            }
            else
            {
                // Normal distribution centered in the domain
                h_bodies[i].position = Vector(
                    normal_pos_dist(gen),
                    normal_pos_dist(gen),
                    normal_pos_dist(gen));

                h_bodies[i].velocity = Vector(
                    normal_vel_dist(gen),
                    normal_vel_dist(gen),
                    normal_vel_dist(gen));
            }

            // Ensure positions are within domain bounds [0, L)
            h_bodies[i].position = applyPeriodicPosition(h_bodies[i].position, g_domainSize);

            h_bodies[i].mass = 1.0;
            h_bodies[i].radius = pow(h_bodies[i].mass / EARTH_MASS, 1.0 / 3.0) * (EARTH_DIA / 2.0);
            h_bodies[i].isDynamic = true;
            h_bodies[i].acceleration = Vector(0, 0, 0);
        }
    }

    void calculateEnergies()
    {
        cudaDeviceSynchronize();

        CudaTimer timer(metrics.energyCalculationTimeMs);

        CHECK_CUDA_ERROR(cudaMemset(d_potentialEnergy, 0, sizeof(double)));
        CHECK_CUDA_ERROR(cudaMemset(d_kineticEnergy, 0, sizeof(double)));

        int blockSize = g_blockSize;

        blockSize = (blockSize / 32) * 32;
        if (blockSize < 32)
            blockSize = 32;
        if (blockSize > 1024)
            blockSize = 1024;

        int gridSize = (nBodies + blockSize - 1) / blockSize;
        if (gridSize < 1)
            gridSize = 1;

        size_t sharedMemSize = 2 * blockSize * sizeof(double);

        CUDA_KERNEL_CALL(CalculateEnergiesKernel, gridSize, blockSize, sharedMemSize, 0,
                         d_bodies, nBodies, d_potentialEnergy, d_kineticEnergy, g_domainSize);

        CHECK_CUDA_ERROR(cudaMemcpy(h_potentialEnergy, d_potentialEnergy, sizeof(double), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(h_kineticEnergy, d_kineticEnergy, sizeof(double), cudaMemcpyDeviceToHost));

        potentialEnergy = *h_potentialEnergy;
        kineticEnergy = *h_kineticEnergy;
    }

    void initializeEnergyData()
    {
        h_potentialEnergy = new double[1];
        h_kineticEnergy = new double[1];

        CHECK_CUDA_ERROR(cudaMalloc((void **)&d_potentialEnergy, sizeof(double)));
        CHECK_CUDA_ERROR(cudaMalloc((void **)&d_kineticEnergy, sizeof(double)));
    }

    void cleanupEnergyData()
    {
        if (h_potentialEnergy != nullptr)
        {
            delete[] h_potentialEnergy;
            h_potentialEnergy = nullptr;
        }

        if (h_kineticEnergy != nullptr)
        {
            delete[] h_kineticEnergy;
            h_kineticEnergy = nullptr;
        }

        if (d_potentialEnergy != nullptr)
        {
            cudaFree(d_potentialEnergy);
            d_potentialEnergy = nullptr;
        }

        if (d_kineticEnergy != nullptr)
        {
            cudaFree(d_kineticEnergy);
            d_kineticEnergy = nullptr;
        }
    }

public:
    SFCDirectSumGPU(int numBodies, bool enableSFC = true, int reorderFreq = 10, sfc::CurveType type = sfc::CurveType::MORTON, bool dynamicReordering = true)
        : nBodies(numBodies),
          useSFC(enableSFC),
          d_orderedIndices(nullptr),
          curveType(type),
          reorderFrequency(reorderFreq),
          iterationCounter(0),
          useDynamicReordering(dynamicReordering),
          reorderingStrategy(10),
          potentialEnergy(0.0), kineticEnergy(0.0),
          totalEnergyAvg(0.0), potentialEnergyAvg(0.0), kineticEnergyAvg(0.0),
          d_potentialEnergy(nullptr), d_kineticEnergy(nullptr),
          h_potentialEnergy(nullptr), h_kineticEnergy(nullptr)
    {

        h_bodies = new Body[nBodies];
        CHECK_CUDA_ERROR(cudaMalloc(&d_bodies, nBodies * sizeof(Body)));

        if (useSFC)
        {
            sorter = new sfc::BodySorter(nBodies, curveType);
        }
        else
        {
            sorter = nullptr;
        }

        initializeDistribution(BodyDistribution::RANDOM, MassDistribution::UNIFORM, 42);

        CHECK_CUDA_ERROR(cudaMemcpy(d_bodies, h_bodies, nBodies * sizeof(Body), cudaMemcpyHostToDevice));

        if (useSFC)
        {
            orderBodiesBySFC();
        }

        initializeEnergyData();

        std::cout << "SFC Direct Sum GPU simulation created with " << nBodies << " bodies." << std::endl;
        if (useSFC)
        {
            std::cout << "Space-Filling Curve ordering enabled with "
                      << (useDynamicReordering ? "dynamic" : "fixed") << " reordering";
            if (!useDynamicReordering)
            {
                std::cout << " frequency " << reorderFrequency;
            }
            std::cout << " and curve type "
                      << (curveType == sfc::CurveType::MORTON ? "MORTON" : "HILBERT") << std::endl;
        }
    }

    ~SFCDirectSumGPU()
    {

        cudaDeviceSynchronize();

        cleanupEnergyData();

        delete[] h_bodies;
        if (d_bodies)
            CHECK_CUDA_ERROR(cudaFree(d_bodies));
        if (sorter)
            delete sorter;
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

    void computeForces()
    {

        CudaTimer timer(metrics.forceTimeMs);

        int blockSize = g_blockSize;
        int gridSize = (nBodies + blockSize - 1) / blockSize;

        size_t sharedMemSize = blockSize * sizeof(Vector) + blockSize * sizeof(double);
        SFCDirectSumForceKernel<<<gridSize, blockSize, sharedMemSize>>>(d_bodies, d_orderedIndices, useSFC, nBodies, g_domainSize);
        CHECK_LAST_CUDA_ERROR();
    }

    void update()
    {
        {
            CudaTimer timer(metrics.totalTimeMs);

            bool shouldReorder = false;
            if (useSFC)
            {
                if (useDynamicReordering)
                {

                    shouldReorder = reorderingStrategy.shouldReorder(metrics.forceTimeMs, metrics.reorderTimeMs);
                }
                else
                {

                    shouldReorder = (iterationCounter >= reorderFrequency);
                }

                if (shouldReorder)
                {
                    orderBodiesBySFC();
                    iterationCounter = 0;
                }
            }

            computeForces();
            iterationCounter++;

            if (useSFC && useDynamicReordering)
            {
                reorderingStrategy.updateMetrics(shouldReorder ? metrics.reorderTimeMs : 0.0);
            }
        }

        // Calcular energías por separado (no afecta el tiempo de simulación)
        calculateEnergies();
    }

    void printPerformance()
    {
        printf("Performance Metrics:\n");
        printf("  Force calculation: %.2f ms\n", metrics.forceTimeMs);
        if (useSFC)
        {
            printf("  Reordering: %.2f ms\n", metrics.reorderTimeMs);
            if (useDynamicReordering)
            {
                printf("  Optimal reorder freq: %d\n", reorderingStrategy.getOptimalFrequency());
                printf("  Degradation rate: %.6f ms/iter\n", reorderingStrategy.getDegradationRate());
            }
        }
        printf("  Total update: %.2f ms\n", metrics.totalTimeMs);
    }

    void runSimulation(int numIterations, int printFreq = 10)
    {
        printf("Starting simulation with %d bodies for %d iterations\n", nBodies, numIterations);
        printf("Using SFC: %s\n", useSFC ? "Yes" : "No");
        if (useSFC)
            printf("Reorder frequency: %d iterations\n", reorderFrequency);

        float totalTime = 0.0f;

        for (int i = 0; i < numIterations; i++)
        {
            update();
            totalTime += metrics.totalTimeMs;

            if (i % printFreq == 0 || i == numIterations - 1)
            {
                printf("Iteration %d/%d (%.1f%%)\n", i + 1, numIterations, (i + 1) * 100.0f / numIterations);
                printPerformance();
            }
        }
    }

    int getOptimalReorderFrequency() const
    {
        if (useDynamicReordering)
        {
            return reorderingStrategy.getOptimalFrequency();
        }
        return reorderFrequency;
    }

    double getTotalTime() const { return metrics.totalTimeMs; }
    double getForceCalculationTime() const { return metrics.forceTimeMs; }
    double getReorderTime() const { return metrics.reorderTimeMs; }
    double getEnergyCalculationTime() const { return metrics.energyCalculationTimeMs; }
    int getNumBodies() const { return nBodies; }
    int getBlockSize() const { return g_blockSize; }
    int getSortType() const { return static_cast<int>(curveType); }
    bool isDynamicReordering() const { return useDynamicReordering; }

    double getPotentialEnergy() const { return potentialEnergy; }
    double getKineticEnergy() const { return kineticEnergy; }
    double getTotalEnergy() const { return potentialEnergy + kineticEnergy; }
    double getPotentialEnergyAvg() const { return potentialEnergyAvg; }
    double getKineticEnergyAvg() const { return kineticEnergyAvg; }
    double getTotalEnergyAvg() const { return totalEnergyAvg; }


};

bool dirExists(const std::string &dirName)
{
    struct stat info;
    return stat(dirName.c_str(), &info) == 0 && (info.st_mode & S_IFDIR);
}

bool createDir(const std::string &dirName)
{
#ifdef _WIN32
    int status = mkdir(dirName.c_str());
#else
    int status = mkdir(dirName.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
#endif
    return status == 0;
}

bool ensureDirExists(const std::string &dirPath)
{
    if (dirExists(dirPath))
    {
        return true;
    }

    std::cout << "Creando directorio: " << dirPath << std::endl;
    if (createDir(dirPath))
    {
        return true;
    }
    else
    {
        std::cerr << "Error: No se pudo crear el directorio " << dirPath << std::endl;
        return false;
    }
}

int main(int argc, char **argv)
{
    ArgumentParser parser("DirectSum GPU Simulation");
    
    // Add arguments with help messages and default values
    parser.add_argument("n", "Number of bodies", 10000);
    parser.add_flag("nosfc", "Disable Space-Filling Curve ordering");
    parser.add_argument("freq", "Reordering frequency for fixed mode", 10);
    parser.add_argument("dist", "Body distribution (galaxy, solar, uniform, random)", std::string("galaxy"));
    parser.add_argument("mass", "Mass distribution (uniform, normal)", std::string("normal"));
    parser.add_argument("seed", "Random seed", 42);
    parser.add_argument("curve", "SFC curve type (morton, hilbert)", std::string("morton"));
    parser.add_argument("s", "Number of iterations", 100);
    parser.add_argument("block", "CUDA block size", DEFAULT_BLOCK_SIZE);
    parser.add_argument("l", "Domain size L for periodic boundary conditions", DEFAULT_DOMAIN_SIZE);
    
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
    
    int numIterations = parser.get<int>("s");
    int blockSize = parser.get<int>("block");
    bool useDynamicReordering = true; // Always use dynamic reordering
    double domainSize = parser.get<double>("l");

    g_blockSize = blockSize;
    g_domainSize = domainSize;

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0)
    {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    std::cout << "Using GPU: " << deviceProp.name << std::endl;
    std::cout << "DirectSum GPU Simulation" << std::endl;
    std::cout << "Bodies: " << nBodies << std::endl;
    std::cout << "Iterations: " << numIterations << std::endl;
    std::cout << "Domain size (periodic boundaries): " << std::scientific << domainSize << std::endl;
    std::cout << "Using SFC: " << (useSFC ? "Yes" : "No") << std::endl;
    if (useSFC) {
        std::cout << "Curve type: " << (curveType == sfc::CurveType::HILBERT ? "HILBERT" : "MORTON") << std::endl;
    }

    SFCDirectSumGPU simulation(
        nBodies,
        useSFC,
        reorderFreq,
        curveType,
        useDynamicReordering);

    simulation.runSimulation(numIterations);

    return 0;
}