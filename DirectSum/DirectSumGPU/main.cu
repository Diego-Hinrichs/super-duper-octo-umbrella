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

#define E SOFTENING_FACTOR
#define DT TIME_STEP
#define COLLISION_TH COLLISION_THRESHOLD

int g_blockSize = DEFAULT_BLOCK_SIZE;

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

__global__ void SFCDirectSumForceKernel(Body *bodies, int *orderedIndices, bool useSFC, int nBodies)
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

                    double rx = sharedPos[j].x - myPos.x;
                    double ry = sharedPos[j].y - myPos.y;
                    double rz = sharedPos[j].z - myPos.z;

                    double distSqr = rx * rx + ry * ry + rz * rz + E * E;
                    double dist = sqrt(distSqr);

                    if (dist >= COLLISION_TH)
                    {
                        double forceMag = (GRAVITY * myMass * sharedMass[j]) / (dist * distSqr);

                        myAcc.x += rx * forceMag / myMass;
                        myAcc.y += ry * forceMag / myMass;
                        myAcc.z += rz * forceMag / myMass;
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
        bodies[realBodyIndex].position = myPos;
    }
}

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

        minBound = Vector(INFINITY, INFINITY, INFINITY);
        maxBound = Vector(-INFINITY, -INFINITY, -INFINITY);

        for (int i = 0; i < nBodies; i++)
        {
            Vector pos = h_bodies[i].position;

            minBound.x = std::min(minBound.x, pos.x);
            minBound.y = std::min(minBound.y, pos.y);
            minBound.z = std::min(minBound.z, pos.z);

            maxBound.x = std::max(maxBound.x, pos.x);
            maxBound.y = std::max(maxBound.y, pos.y);
            maxBound.z = std::max(maxBound.z, pos.z);
        }

        double padding = std::max(1.0e10, (maxBound.x - minBound.x) * 0.01);
        minBound.x -= padding;
        minBound.y -= padding;
        minBound.z -= padding;
        maxBound.x += padding;
        maxBound.y += padding;
        maxBound.z += padding;
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

    void calculateEnergies();
    void initializeEnergyData();
    void cleanupEnergyData();

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

        initializeDistribution(BodyDistribution::RANDOM, MassDistribution::UNIFORM);

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
        SFCDirectSumForceKernel<<<gridSize, blockSize, sharedMemSize>>>(d_bodies, d_orderedIndices, useSFC, nBodies);
        CHECK_LAST_CUDA_ERROR();
    }

    void update()
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
            reorderingStrategy.updateMetrics(shouldReorder ? metrics.reorderTimeMs : 0.0, metrics.forceTimeMs);
        }

        calculateEnergies();
    }

    void printPerformance()
    {
        printf("Performance Metrics:\n");
        printf("  Force calculation: %.3f ms\n", metrics.forceTimeMs);
        if (useSFC)
        {
            printf("  Reordering: %.3f ms\n", metrics.reorderTimeMs);
            if (useDynamicReordering)
            {
                printf("  Optimal reorder freq: %d\n", reorderingStrategy.getOptimalFrequency());
                printf("  Degradation rate: %.6f ms/iter\n", reorderingStrategy.getDegradationRate());
            }
        }
        printf("  Total update: %.3f ms\n", metrics.totalTimeMs);
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

        printf("Simulation completed in %.3f ms (avg %.3f ms per iteration)\n",
               totalTime, totalTime / numIterations);
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

    void calculateEnergies()
    {
        Body *d_bodies = bodySystem->getDeviceBodies();
        int nBodies = bodySystem->getNumBodies();

        if (d_bodies == nullptr)
        {
            std::cerr << "Error: Device bodies not initialized in calculateEnergies" << std::endl;
            return;
        }

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

void initializeCsv(const std::string &filename, bool append = false)
{

    size_t pos = filename.find_last_of('/');
    if (pos != std::string::npos)
    {
        std::string dirPath = filename.substr(0, pos);
        if (!ensureDirExists(dirPath))
        {
            std::cerr << "Error: No se puede crear el directorio para el archivo " << filename << std::endl;
            return;
        }
    }

    std::ofstream file;
    if (append)
    {
        file.open(filename, std::ios::app);
    }
    else
    {
        file.open(filename);
    }

    if (!file.is_open())
    {
        std::cerr << "Error: No se pudo abrir el archivo " << filename << " para escritura" << std::endl;
        return;
    }

    if (!append)
    {
        file << "timestamp,method,bodies,steps,block_size,sort_type,total_time_ms,avg_step_time_ms,force_calculation_time_ms,sort_time_ms,potential_energy,kinetic_energy,total_energy" << std::endl;
    }

    file.close();
    std::cout << "Archivo CSV " << (append ? "actualizado" : "inicializado") << ": " << filename << std::endl;
}

void saveMetrics(const std::string &filename,
                 int bodies,
                 int steps,
                 int blockSize,
                 int sortType,
                 float totalTime,
                 float forceCalculationTime,
                 float sortTime,
                 double potentialEnergy,
                 double kineticEnergy,
                 double totalEnergy)
{
    std::ofstream file(filename, std::ios::app);
    if (!file.is_open())
    {
        std::cerr << "Error: No se pudo abrir el archivo " << filename << " para escritura." << std::endl;
        return;
    }

    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    std::stringstream timestamp;
    timestamp << std::put_time(std::localtime(&now_c), "%Y-%m-%d %H:%M:%S");

    float avgTimePerStep = totalTime / steps;

    file << timestamp.str() << ","
         << "GPU_SFC_Direct_Sum" << ","
         << bodies << ","
         << steps << ","
         << blockSize << ","
         << sortType << ","
         << totalTime << ","
         << avgTimePerStep << ","
         << forceCalculationTime << ","
         << sortTime << ","
         << potentialEnergy << ","
         << kineticEnergy << ","
         << totalEnergy << std::endl;

    file.close();
    std::cout << "Métricas guardadas en: " << filename << std::endl;
}

int main(int argc, char **argv)
{

    int nBodies = 10000;
    bool useSFC = true;
    int reorderFreq = 10;
    BodyDistribution bodyDist = BodyDistribution::GALAXY;
    MassDistribution massDist = MassDistribution::NORMAL;
    unsigned int seed = 42;
    sfc::CurveType curveType = sfc::CurveType::MORTON;
    int numIterations = 100;
    int blockSize = DEFAULT_BLOCK_SIZE;
    bool useDynamicReordering = true;

    bool saveMetricsToFile = false;
    std::string metricsFile = "./SFCDirectSumGPU_metrics.csv";

    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "-n" && i + 1 < argc)
            nBodies = std::stoi(argv[++i]);
        else if (arg == "-nosfc")
            useSFC = false;
        else if (arg == "-freq" && i + 1 < argc)
            reorderFreq = std::stoi(argv[++i]);
        else if (arg == "-dist" && i + 1 < argc)
        {
            std::string distType = argv[++i];
            if (distType == "galaxy")
                bodyDist = BodyDistribution::GALAXY;
            else if (distType == "solar")
                bodyDist = BodyDistribution::SOLAR_SYSTEM;
            else if (distType == "uniform")
                bodyDist = BodyDistribution::UNIFORM_SPHERE;
            else if (distType == "random")
                bodyDist = BodyDistribution::RANDOM;
        }
        else if (arg == "-mass" && i + 1 < argc)
        {
            std::string massType = argv[++i];
            if (massType == "uniform")
                massDist = MassDistribution::UNIFORM;
            else if (massType == "normal")
                massDist = MassDistribution::NORMAL;
        }
        else if (arg == "-seed" && i + 1 < argc)
            seed = std::stoi(argv[++i]);
        else if (arg == "-curve" && i + 1 < argc)
        {
            std::string curveStr = argv[++i];
            if (curveStr == "morton")
                curveType = sfc::CurveType::MORTON;
            else if (curveStr == "hilbert")
                curveType = sfc::CurveType::HILBERT;
        }
        else if (arg == "-iter" && i + 1 < argc)
            numIterations = std::stoi(argv[++i]);
        else if (arg == "-block" && i + 1 < argc)
            blockSize = std::stoi(argv[++i]);
        else if (arg == "-h" || arg == "--help")
        {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  -n <num>          Number of bodies (default: 10000)" << std::endl;
            std::cout << "  -nosfc            Disable Space-Filling Curve ordering" << std::endl;
            std::cout << "  -freq <num>       Reordering frequency (default: 10)" << std::endl;
            std::cout << "  -dist <type>      Body distribution: galaxy, solar, uniform, random (default: galaxy)" << std::endl;
            std::cout << "  -mass <type>      Mass distribution: uniform, normal (default: normal)" << std::endl;
            std::cout << "  -seed <num>       Random seed (default: 42)" << std::endl;
            std::cout << "  -curve <type>     SFC curve type: morton, hilbert (default: morton)" << std::endl;
            std::cout << "  -iter <num>       Number of iterations (default: 100)" << std::endl;
            std::cout << "  -block <num>      CUDA block size (default: 256)" << std::endl;
            return 0;
        }
        else if (arg == "--save-metrics")
        {
            saveMetricsToFile = true;
        }
        else if (arg == "--metrics-file" && i + 1 < argc)
        {
            metricsFile = argv[++i];
        }
        else if (arg == "--dynamic-reordering")
        {
        }
    }

    g_blockSize = blockSize;

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

    SFCDirectSumGPU simulation(
        nBodies,
        useSFC,
        reorderFreq,
        curveType,
        useDynamicReordering);

    simulation.runSimulation(numIterations);

    if (saveMetricsToFile)
    {

        bool fileExists = false;
        std::ifstream checkFile(metricsFile);
        if (checkFile.good())
        {
            fileExists = true;
        }
        checkFile.close();

        initializeCsv(metricsFile, fileExists);
        saveMetrics(
            metricsFile,
            simulation.getNumBodies(),
            numIterations,
            simulation.getBlockSize(),
            simulation.getSortType(),
            simulation.getTotalTime(),
            simulation.getForceCalculationTime(),
            simulation.getReorderTime(),
            simulation.getPotentialEnergy(),
            simulation.getKineticEnergy(),
            simulation.getTotalEnergy());

        std::cout << "Métricas guardadas en: " << metricsFile << std::endl;
    }

    return 0;
}