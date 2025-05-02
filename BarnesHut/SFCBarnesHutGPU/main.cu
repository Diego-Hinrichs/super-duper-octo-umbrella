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

__global__ void extractPositionsKernel(struct Body *bodies, struct Vector *positions, int n);
__global__ void calculateSFCKeysKernel(struct Vector *positions, uint64_t *keys, int nBodies,
                                       struct Vector minBound, struct Vector maxBound, bool isHilbert);
__global__ void extractPositionsAndCalculateKeysKernel(Body *bodies, uint64_t *keys, int nBodies, Vector minBound, Vector maxBound, bool isHilbert);

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

struct Node
{
    bool isLeaf;
    int firstChildIndex;
    int bodyIndex;
    int bodyCount;
    Vector position;
    double mass;
    double radius;
    Vector min;
    Vector max;

    __host__ __device__ Node()
        : isLeaf(true),
          firstChildIndex(-1),
          bodyIndex(-1),
          bodyCount(0),
          position(),
          mass(0.0),
          radius(0.0),
          min(),
          max() {}
};

struct SimulationMetrics
{
    float resetTimeMs;
    float bboxTimeMs;
    float buildTimeMs;
    float forceTimeMs;
    float reorderTimeMs;
    float totalTimeMs;
    float energyCalculationTimeMs;

    SimulationMetrics() : resetTimeMs(0.0f),
                          bboxTimeMs(0.0f),
                          buildTimeMs(0.0f),
                          forceTimeMs(0.0f),
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
                return hilbertEncode(x, y, z);
            }
        }

        uint64_t mortonEncode(uint32_t x, uint32_t y, uint32_t z)
        {

            x = (x | (x << 16)) & 0x0000FFFF0000FFFF;
            x = (x | (x << 8)) & 0x00FF00FF00FF00FF;
            x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0F;
            x = (x | (x << 2)) & 0x3333333333333333;
            x = (x | (x << 1)) & 0x5555555555555555;

            y = (y | (y << 16)) & 0x0000FFFF0000FFFF;
            y = (y | (y << 8)) & 0x00FF00FF00FF00FF;
            y = (y | (y << 4)) & 0x0F0F0F0F0F0F0F0F;
            y = (y | (y << 2)) & 0x3333333333333333;
            y = (y | (y << 1)) & 0x5555555555555555;

            z = (z | (z << 16)) & 0x0000FFFF0000FFFF;
            z = (z | (z << 8)) & 0x00FF00FF00FF00FF;
            z = (z | (z << 4)) & 0x0F0F0F0F0F0F0F0F;
            z = (z | (z << 2)) & 0x3333333333333333;
            z = (z | (z << 1)) & 0x5555555555555555;

            return x | (y << 1) | (z << 2);
        }

        uint64_t hilbertEncode(uint32_t x, uint32_t y, uint32_t z)
        {

            x &= 0xFFFF;
            y &= 0xFFFF;
            z &= 0xFFFF;

            uint64_t result = 0;
            uint8_t state = 0;

            static const uint8_t hilbertMap[8][8] = {
                {0, 1, 3, 2, 7, 6, 4, 5},
                {4, 5, 7, 6, 0, 1, 3, 2},
                {6, 7, 5, 4, 2, 3, 1, 0},
                {2, 3, 1, 0, 6, 7, 5, 4},
                {0, 7, 1, 6, 3, 4, 2, 5},
                {6, 1, 7, 0, 5, 2, 4, 3},
                {2, 5, 3, 4, 1, 6, 0, 7},
                {4, 3, 5, 2, 7, 0, 6, 1}};

            static const uint8_t nextState[8][8] = {
                {0, 1, 3, 2, 7, 6, 4, 5},
                {1, 0, 2, 3, 4, 5, 7, 6},
                {2, 3, 1, 0, 5, 4, 6, 7},
                {3, 2, 0, 1, 6, 7, 5, 4},
                {4, 5, 7, 6, 0, 1, 3, 2},
                {5, 4, 6, 7, 1, 0, 2, 3},
                {6, 7, 5, 4, 2, 3, 1, 0},
                {7, 6, 4, 5, 3, 2, 0, 1}};

            for (int i = 15; i >= 0; i--)
            {
                uint8_t octant = 0;
                if (x & (1 << i))
                    octant |= 1;
                if (y & (1 << i))
                    octant |= 2;
                if (z & (1 << i))
                    octant |= 4;

                uint8_t position = hilbertMap[state][octant];
                result = (result << 3) | position;
                state = nextState[state][octant];
            }

            return result;
        }

        // int *sortBodies(Body *d_bodies, const Vector &minBound, const Vector &maxBound)
        // {

        //     int blockSize = 256;
        //     int gridSize = (nBodies + blockSize - 1) / blockSize;

        //     extractPositionsKernel<<<gridSize, blockSize, 0, stream1>>>(d_bodies, d_positions, nBodies);

        //     calculateSFCKeysKernel<<<gridSize, blockSize, 0, stream1>>>(
        //         d_positions, d_keys, nBodies, minBound, maxBound, curveType == CurveType::HILBERT);

        //     thrust::counting_iterator<int> first(0);
        //     thrust::copy(first, first + nBodies, thrust::device_pointer_cast(d_orderedIndices));

        //     void *d_temp_storage = NULL;
        //     size_t temp_storage_bytes = 0;

        //     cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
        //                                     d_keys, d_keys,
        //                                     d_orderedIndices, d_orderedIndices,
        //                                     nBodies);

        //     cudaMalloc(&d_temp_storage, temp_storage_bytes);

        //     cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
        //                                     d_keys, d_keys,
        //                                     d_orderedIndices, d_orderedIndices,
        //                                     nBodies);

        //     cudaFree(d_temp_storage);

        //     return d_orderedIndices;
        // }

        int *sortBodies(Body *d_bodies, const Vector &minBound, const Vector &maxBound)
        {
            int blockSize = 256;
            int gridSize = (nBodies + blockSize - 1) / blockSize;

            // Usar el kernel combinado en lugar de los dos kernels separados
            extractPositionsAndCalculateKeysKernel<<<gridSize, blockSize, 0, stream1>>>(
                d_bodies, d_keys, nBodies, minBound, maxBound, curveType == CurveType::HILBERT);

            // Inicializar índices secuenciales
            thrust::counting_iterator<int> first(0);
            thrust::copy(first, first + nBodies, thrust::device_pointer_cast(d_orderedIndices));

            // Verificar si ya tenemos memoria temporal asignada
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

            // Realizar el ordenamiento con la memoria ya asignada
            cub::DeviceRadixSort::SortPairs(d_temp_storage, d_temp_storage_size,
                                            d_keys, d_keys,
                                            d_orderedIndices, d_orderedIndices,
                                            nBodies);

            return d_orderedIndices;
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
constexpr int MAX_NODES = 1000000;
constexpr int N_LEAF = 8;
constexpr double DEFAULT_THETA = 0.5;

#define E SOFTENING_FACTOR
#define DT TIME_STEP
#define COLLISION_TH COLLISION_THRESHOLD

int g_blockSize = DEFAULT_BLOCK_SIZE;
double g_theta = DEFAULT_THETA;

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
        file << "timestamp,method,bodies,steps,block_size,theta,sort_type,total_time_ms,avg_step_time_ms,force_calculation_time_ms,tree_build_time_ms,sort_time_ms,potential_energy,kinetic_energy,total_energy" << std::endl;
    }

    file.close();
    std::cout << "Archivo CSV " << (append ? "actualizado" : "inicializado") << ": " << filename << std::endl;
}

void saveMetrics(const std::string &filename,
                 int bodies,
                 int steps,
                 int blockSize,
                 float theta,
                 const char *sortType,
                 float totalTime,
                 float forceCalculationTime,
                 float treeBuildTime,
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
         << "GPU_SFC_Barnes_Hut" << ","
         << bodies << ","
         << steps << ","
         << blockSize << ","
         << theta << ","
         << sortType << ","
         << totalTime << ","
         << avgTimePerStep << ","
         << forceCalculationTime << ","
         << treeBuildTime << ","
         << sortTime << ","
         << potentialEnergy << ","
         << kineticEnergy << ","
         << totalEnergy << std::endl;

    file.close();
    std::cout << "Métricas guardadas en: " << filename << std::endl;
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

__global__ void ResetKernel(Node *nodes, int *mutex, int nNodes, int nBodies)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nNodes)
    {

        nodes[i].isLeaf = (i < nBodies);
        nodes[i].firstChildIndex = -1;
        nodes[i].bodyIndex = (i < nBodies) ? i : -1;
        nodes[i].bodyCount = (i < nBodies) ? 1 : 0;
        nodes[i].position = Vector(0, 0, 0);
        nodes[i].mass = 0.0;
        nodes[i].radius = 0.0;
        nodes[i].min = Vector(0, 0, 0);
        nodes[i].max = Vector(0, 0, 0);

        if (i < nNodes)
            mutex[i] = 0;
    }
}

__global__ void ComputeBoundingBoxKernel(Node *nodes, Body *bodies, int *orderedIndices, bool useSFC, int *mutex, int nBodies)
{

    extern __shared__ double sharedMem[];
    double *minX = &sharedMem[0];
    double *minY = &sharedMem[blockDim.x];
    double *minZ = &sharedMem[2 * blockDim.x];
    double *maxX = &sharedMem[3 * blockDim.x];
    double *maxY = &sharedMem[4 * blockDim.x];
    double *maxZ = &sharedMem[5 * blockDim.x];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;

    int realBodyIndex = (useSFC && orderedIndices != nullptr && i < nBodies) ? orderedIndices[i] : i;

    minX[tx] = (i < nBodies) ? bodies[realBodyIndex].position.x : DBL_MAX;
    minY[tx] = (i < nBodies) ? bodies[realBodyIndex].position.y : DBL_MAX;
    minZ[tx] = (i < nBodies) ? bodies[realBodyIndex].position.z : DBL_MAX;
    maxX[tx] = (i < nBodies) ? bodies[realBodyIndex].position.x : -DBL_MAX;
    maxY[tx] = (i < nBodies) ? bodies[realBodyIndex].position.y : -DBL_MAX;
    maxZ[tx] = (i < nBodies) ? bodies[realBodyIndex].position.z : -DBL_MAX;

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tx < s)
        {
            minX[tx] = min(minX[tx], minX[tx + s]);
            minY[tx] = min(minY[tx], minY[tx + s]);
            minZ[tx] = min(minZ[tx], minZ[tx + s]);
            maxX[tx] = max(maxX[tx], maxX[tx + s]);
            maxY[tx] = max(maxY[tx], maxY[tx + s]);
            maxZ[tx] = max(maxZ[tx], maxZ[tx + s]);
        }
        __syncthreads();
    }

    if (tx == 0)
    {

        nodes[0].isLeaf = false;
        nodes[0].bodyCount = nBodies;
        nodes[0].min = Vector(minX[0], minY[0], minZ[0]);
        nodes[0].max = Vector(maxX[0], maxY[0], maxZ[0]);

        Vector padding = (nodes[0].max - nodes[0].min) * 0.01;
        nodes[0].min = nodes[0].min - padding;
        nodes[0].max = nodes[0].max + padding;

        Vector dimensions = nodes[0].max - nodes[0].min;
        nodes[0].radius = max(max(dimensions.x, dimensions.y), dimensions.z) * 0.5;

        nodes[0].position = (nodes[0].min + nodes[0].max) * 0.5;
    }
}

__device__ bool InsertBody(Node *nodes, Body *bodies, int bodyIdx, int nodeIdx, int nNodes, int leafLimit)
{

    Node &node = nodes[nodeIdx];

    if (node.isLeaf && node.bodyCount == 0)
    {
        node.bodyIndex = bodyIdx;
        node.bodyCount = 1;
        node.position = bodies[bodyIdx].position;
        node.mass = bodies[bodyIdx].mass;
        return true;
    }

    if (node.isLeaf && node.bodyCount > 0)
    {

        if (nodeIdx >= leafLimit)
            return false;

        node.isLeaf = false;
        int firstChildIdx = atomicAdd(&nodes[nNodes - 1].bodyCount, 8);

        if (firstChildIdx + 7 >= nNodes)
            return false;

        node.firstChildIndex = firstChildIdx;

        int existingBodyIdx = node.bodyIndex;
        node.bodyIndex = -1;

        Vector center = node.position;

        Vector pos = bodies[existingBodyIdx].position;
        int childIdx = ((pos.x >= center.x) ? 1 : 0) |
                       ((pos.y >= center.y) ? 2 : 0) |
                       ((pos.z >= center.z) ? 4 : 0);

        for (int i = 0; i < 8; i++)
        {
            Node &child = nodes[firstChildIdx + i];
            child.isLeaf = true;
            child.firstChildIndex = -1;
            child.bodyIndex = -1;
            child.bodyCount = 0;
            child.mass = 0.0;

            Vector min = node.min;
            Vector max = node.max;

            if (i & 1)
                min.x = center.x;
            else
                max.x = center.x;
            if (i & 2)
                min.y = center.y;
            else
                max.y = center.y;
            if (i & 4)
                min.z = center.z;
            else
                max.z = center.z;

            child.min = min;
            child.max = max;
            child.position = (min + max) * 0.5;

            double dx = max.x - min.x;
            double dy = max.y - min.y;
            double dz = max.z - min.z;
            double maxDim = dx > dy ? dx : dy;
            maxDim = maxDim > dz ? maxDim : dz;
            child.radius = maxDim * 0.5;
        }

        InsertBody(nodes, bodies, existingBodyIdx, firstChildIdx + childIdx, nNodes, leafLimit);
    }

    Vector pos = bodies[bodyIdx].position;
    Vector center = node.position;

    int childIdx = ((pos.x >= center.x) ? 1 : 0) |
                   ((pos.y >= center.y) ? 2 : 0) |
                   ((pos.z >= center.z) ? 4 : 0);

    if (node.firstChildIndex >= 0 && node.firstChildIndex + childIdx < nNodes)
    {
        InsertBody(nodes, bodies, bodyIdx, node.firstChildIndex + childIdx, nNodes, leafLimit);
    }

    double totalMass = node.mass + bodies[bodyIdx].mass;
    Vector weightedPos = node.position * node.mass + bodies[bodyIdx].position * bodies[bodyIdx].mass;

    if (totalMass > 0.0)
    {
        node.position = weightedPos * (1.0 / totalMass);
        node.mass = totalMass;
    }

    node.bodyCount++;

    return true;
}

__global__ void ConstructOctTreeKernel(Node *nodes, Body *bodies, Body *bodyBuffer, int *orderedIndices, bool useSFC,
                                       int rootIdx, int nNodes, int nBodies, int leafLimit)
{

    extern __shared__ double sharedMem[];
    double *totalMass = &sharedMem[0];
    double3 *centerMass = (double3 *)(totalMass + blockDim.x);

    int i = threadIdx.x;

    for (int bodyIdx = i; bodyIdx < nBodies; bodyIdx += blockDim.x)
    {

        int realBodyIdx = (useSFC && orderedIndices != nullptr) ? orderedIndices[bodyIdx] : bodyIdx;

        bodyBuffer[bodyIdx] = bodies[realBodyIdx];

        InsertBody(nodes, bodies, realBodyIdx, rootIdx, nNodes, leafLimit);
    }
}

__global__ void ComputeForceKernel(Node *nodes, Body *bodies, int *orderedIndices, bool useSFC,
                                   int nNodes, int nBodies, int leafLimit, double theta)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int realBodyIdx = (useSFC && orderedIndices != nullptr && i < nBodies) ? orderedIndices[i] : i;

    if (realBodyIdx >= nBodies || !bodies[realBodyIdx].isDynamic)
        return;

    __shared__ Node nodeCache[32];
    __shared__ int cacheIndices[32];
    __shared__ int cacheSize;

    if (threadIdx.x == 0)
    {
        cacheSize = 0;

        nodeCache[0] = nodes[0];
        cacheIndices[0] = 0;
        cacheSize = 1;
    }
    __syncthreads();

    Vector acc(0.0, 0.0, 0.0);
    Vector pos = bodies[realBodyIdx].position;
    double bodyMass = bodies[realBodyIdx].mass;

    extern __shared__ int sharedData[];
    const int MAX_STACK_SIZE = 32;
    int *localStack = &sharedData[threadIdx.x * MAX_STACK_SIZE];
    int stackSize = 0;

    if (stackSize < MAX_STACK_SIZE)
    {
        localStack[stackSize++] = 0;
    }

    double adaptiveTheta = theta;
    if (useSFC)
    {

        int sectionSize = nBodies / 4;
        int section = i / sectionSize;

        if (section == 0 || section == 3)
        {

            adaptiveTheta = theta * 0.85;
        }
        else
        {

            adaptiveTheta = theta * 1.15;
        }

        adaptiveTheta = max(0.3, min(0.9, adaptiveTheta));
    }

    while (stackSize > 0)
    {
        int nodeIdx = localStack[--stackSize];

        Node node;
        bool foundInCache = false;

#pragma unroll
        for (int c = 0; c < 32; c++)
        {
            if (c < cacheSize && cacheIndices[c] == nodeIdx)
            {
                node = nodeCache[c];
                foundInCache = true;
                break;
            }
        }

        if (!foundInCache)
        {
            node = nodes[nodeIdx];

            if (cacheSize < 32)
            {
                int cacheIdx = atomicAdd(&cacheSize, 1);
                if (cacheIdx < 32)
                {
                    nodeCache[cacheIdx] = node;
                    cacheIndices[cacheIdx] = nodeIdx;
                }
            }
        }

        Vector nodePos = node.position;
        Vector dir = nodePos - pos;
        double distSqr = dir.lengthSquared();

        if (distSqr < 1e-10)
            continue;

        if (!node.isLeaf)
        {
            double nodeSizeSqr = node.radius * node.radius * 4.0;

            if (nodeSizeSqr < distSqr * adaptiveTheta * adaptiveTheta)
            {

                double dist = sqrt(distSqr);
                double invDist3 = 1.0 / (dist * distSqr);
                acc = acc + dir * (node.mass * invDist3);
            }
            else
            {

                int firstChildIdx = node.firstChildIndex;

                if (useSFC)
                {

                    static const int childOrder[8] = {0, 1, 3, 2, 6, 7, 5, 4};

#pragma unroll
                    for (int c = 0; c < 8; c++)
                    {
                        int childIdx = firstChildIdx + childOrder[c];
                        if (childIdx < nNodes && stackSize < MAX_STACK_SIZE - 1)
                        {
                            localStack[stackSize++] = childIdx;
                        }
                    }
                }
                else
                {

#pragma unroll
                    for (int c = 0; c < 8; c++)
                    {
                        int childIdx = firstChildIdx + c;
                        if (childIdx < nNodes && stackSize < MAX_STACK_SIZE - 1)
                        {
                            localStack[stackSize++] = childIdx;
                        }
                    }
                }
            }
        }
        else
        {

            double dist = sqrt(distSqr);
            double invDist3 = 1.0 / (dist * distSqr);
            acc = acc + dir * (node.mass * invDist3);
        }
    }

    bodies[realBodyIdx].acceleration = acc;

    bodies[realBodyIdx].velocity = bodies[realBodyIdx].velocity + acc * DT;

    bodies[realBodyIdx].position = bodies[realBodyIdx].position + bodies[realBodyIdx].velocity * DT;
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

__global__ void extractPositionsKernel(Body *bodies, Vector *positions, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        positions[idx] = bodies[idx].position;
    }
}

__global__ void calculateSFCKeysKernel(Vector *positions, uint64_t *keys, int nBodies,
                                       Vector minBound, Vector maxBound, bool isHilbert)
{

    __shared__ uint8_t hilbertMap[8][8];
    __shared__ uint8_t nextState[8][8];

    if (threadIdx.x == 0)
    {

        hilbertMap[0][0] = 0;
        hilbertMap[0][1] = 1;
        hilbertMap[0][2] = 3;
        hilbertMap[0][3] = 2;
        hilbertMap[0][4] = 7;
        hilbertMap[0][5] = 6;
        hilbertMap[0][6] = 4;
        hilbertMap[0][7] = 5;

        hilbertMap[1][0] = 4;
        hilbertMap[1][1] = 5;
        hilbertMap[1][2] = 7;
        hilbertMap[1][3] = 6;
        hilbertMap[1][4] = 0;
        hilbertMap[1][5] = 1;
        hilbertMap[1][6] = 3;
        hilbertMap[1][7] = 2;

        hilbertMap[2][0] = 6;
        hilbertMap[2][1] = 7;
        hilbertMap[2][2] = 5;
        hilbertMap[2][3] = 4;
        hilbertMap[2][4] = 2;
        hilbertMap[2][5] = 3;
        hilbertMap[2][6] = 1;
        hilbertMap[2][7] = 0;

        hilbertMap[3][0] = 2;
        hilbertMap[3][1] = 3;
        hilbertMap[3][2] = 1;
        hilbertMap[3][3] = 0;
        hilbertMap[3][4] = 6;
        hilbertMap[3][5] = 7;
        hilbertMap[3][6] = 5;
        hilbertMap[3][7] = 4;

        hilbertMap[4][0] = 0;
        hilbertMap[4][1] = 7;
        hilbertMap[4][2] = 1;
        hilbertMap[4][3] = 6;
        hilbertMap[4][4] = 3;
        hilbertMap[4][5] = 4;
        hilbertMap[4][6] = 2;
        hilbertMap[4][7] = 5;

        hilbertMap[5][0] = 6;
        hilbertMap[5][1] = 1;
        hilbertMap[5][2] = 7;
        hilbertMap[5][3] = 0;
        hilbertMap[5][4] = 5;
        hilbertMap[5][5] = 2;
        hilbertMap[5][6] = 4;
        hilbertMap[5][7] = 3;

        hilbertMap[6][0] = 2;
        hilbertMap[6][1] = 5;
        hilbertMap[6][2] = 3;
        hilbertMap[6][3] = 4;
        hilbertMap[6][4] = 1;
        hilbertMap[6][5] = 6;
        hilbertMap[6][6] = 0;
        hilbertMap[6][7] = 7;

        hilbertMap[7][0] = 4;
        hilbertMap[7][1] = 3;
        hilbertMap[7][2] = 5;
        hilbertMap[7][3] = 2;
        hilbertMap[7][4] = 7;
        hilbertMap[7][5] = 0;
        hilbertMap[7][6] = 6;
        hilbertMap[7][7] = 1;

        nextState[0][0] = 0;
        nextState[0][1] = 1;
        nextState[0][2] = 3;
        nextState[0][3] = 2;
        nextState[0][4] = 7;
        nextState[0][5] = 6;
        nextState[0][6] = 4;
        nextState[0][7] = 5;

        nextState[1][0] = 1;
        nextState[1][1] = 0;
        nextState[1][2] = 2;
        nextState[1][3] = 3;
        nextState[1][4] = 4;
        nextState[1][5] = 5;
        nextState[1][6] = 7;
        nextState[1][7] = 6;

        nextState[2][0] = 2;
        nextState[2][1] = 3;
        nextState[2][2] = 1;
        nextState[2][3] = 0;
        nextState[2][4] = 5;
        nextState[2][5] = 4;
        nextState[2][6] = 6;
        nextState[2][7] = 7;

        nextState[3][0] = 3;
        nextState[3][1] = 2;
        nextState[3][2] = 0;
        nextState[3][3] = 1;
        nextState[3][4] = 6;
        nextState[3][5] = 7;
        nextState[3][6] = 5;
        nextState[3][7] = 4;

        nextState[4][0] = 4;
        nextState[4][1] = 5;
        nextState[4][2] = 7;
        nextState[4][3] = 6;
        nextState[4][4] = 0;
        nextState[4][5] = 1;
        nextState[4][6] = 3;
        nextState[4][7] = 2;

        nextState[5][0] = 5;
        nextState[5][1] = 4;
        nextState[5][2] = 6;
        nextState[5][3] = 7;
        nextState[5][4] = 1;
        nextState[5][5] = 0;
        nextState[5][6] = 2;
        nextState[5][7] = 3;

        nextState[6][0] = 6;
        nextState[6][1] = 7;
        nextState[6][2] = 5;
        nextState[6][3] = 4;
        nextState[6][4] = 2;
        nextState[6][5] = 3;
        nextState[6][6] = 1;
        nextState[6][7] = 0;

        nextState[7][0] = 7;
        nextState[7][1] = 6;
        nextState[7][2] = 4;
        nextState[7][3] = 5;
        nextState[7][4] = 3;
        nextState[7][5] = 2;
        nextState[7][6] = 0;
        nextState[7][7] = 1;
    }

    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nBodies)
        return;

    Vector pos = positions[i];

    double invRangeX = 1.0 / (maxBound.x - minBound.x);
    double invRangeY = 1.0 / (maxBound.y - minBound.y);
    double invRangeZ = 1.0 / (maxBound.z - minBound.z);

    double normalizedX = (pos.x - minBound.x) * invRangeX;
    double normalizedY = (pos.y - minBound.y) * invRangeY;
    double normalizedZ = (pos.z - minBound.z) * invRangeZ;

    normalizedX = max(0.0, min(0.999999, normalizedX));
    normalizedY = max(0.0, min(0.999999, normalizedY));
    normalizedZ = max(0.0, min(0.999999, normalizedZ));

    const uint32_t MAX_COORD = 0xFFFF;

    uint32_t x = static_cast<uint32_t>(normalizedX * MAX_COORD);
    uint32_t y = static_cast<uint32_t>(normalizedY * MAX_COORD);
    uint32_t z = static_cast<uint32_t>(normalizedZ * MAX_COORD);

    uint64_t key = 0;

    if (!isHilbert)
    {

        uint32_t xx = x;
        uint32_t yy = y;
        uint32_t zz = z;

        xx = (xx | (xx << 8)) & 0x00FF00FF;
        xx = (xx | (xx << 4)) & 0x0F0F0F0F;
        xx = (xx | (xx << 2)) & 0x33333333;
        xx = (xx | (xx << 1)) & 0x55555555;

        yy = (yy | (yy << 8)) & 0x00FF00FF;
        yy = (yy | (yy << 4)) & 0x0F0F0F0F;
        yy = (yy | (yy << 2)) & 0x33333333;
        yy = (yy | (yy << 1)) & 0x55555555;

        zz = (zz | (zz << 8)) & 0x00FF00FF;
        zz = (zz | (zz << 4)) & 0x0F0F0F0F;
        zz = (zz | (zz << 2)) & 0x33333333;
        zz = (zz | (zz << 1)) & 0x55555555;

        key = xx | (yy << 1) | (zz << 2);
    }

    else
    {
        uint8_t state = 0;

        {
            uint8_t octant = 0;
            if (x & 0x8000)
                octant |= 1;
            if (y & 0x8000)
                octant |= 2;
            if (z & 0x8000)
                octant |= 4;

            uint8_t position = hilbertMap[state][octant];
            key = (key << 3) | position;
            state = nextState[state][octant];
        }

        {
            uint8_t octant = 0;
            if (x & 0x4000)
                octant |= 1;
            if (y & 0x4000)
                octant |= 2;
            if (z & 0x4000)
                octant |= 4;

            uint8_t position = hilbertMap[state][octant];
            key = (key << 3) | position;
            state = nextState[state][octant];
        }

        {
            uint8_t octant = 0;
            if (x & 0x2000)
                octant |= 1;
            if (y & 0x2000)
                octant |= 2;
            if (z & 0x2000)
                octant |= 4;

            uint8_t position = hilbertMap[state][octant];
            key = (key << 3) | position;
            state = nextState[state][octant];
        }

        {
            uint8_t octant = 0;
            if (x & 0x1000)
                octant |= 1;
            if (y & 0x1000)
                octant |= 2;
            if (z & 0x1000)
                octant |= 4;

            uint8_t position = hilbertMap[state][octant];
            key = (key << 3) | position;
            state = nextState[state][octant];
        }

        for (int j = 0; j < 3; j++)
        {
            for (int i = 3; i >= 0; i--)
            {
                uint8_t octant = 0;
                uint32_t mask = 1 << (i + j * 4);
                if (x & mask)
                    octant |= 1;
                if (y & mask)
                    octant |= 2;
                if (z & mask)
                    octant |= 4;

                uint8_t position = hilbertMap[state][octant];
                key = (key << 3) | position;
                state = nextState[state][octant];
            }
        }
    }

    keys[i] = key;
}

__constant__ uint8_t d_hilbertMap[8][8] = {
    {0, 1, 3, 2, 7, 6, 4, 5},
    {4, 5, 7, 6, 0, 1, 3, 2},
    {6, 7, 5, 4, 2, 3, 1, 0},
    {2, 3, 1, 0, 6, 7, 5, 4},
    {0, 7, 1, 6, 3, 4, 2, 5},
    {6, 1, 7, 0, 5, 2, 4, 3},
    {2, 5, 3, 4, 1, 6, 0, 7},
    {4, 3, 5, 2, 7, 0, 6, 1}};

__constant__ uint8_t d_nextState[8][8] = {
    {0, 1, 3, 2, 7, 6, 4, 5},
    {1, 0, 2, 3, 4, 5, 7, 6},
    {2, 3, 1, 0, 5, 4, 6, 7},
    {3, 2, 0, 1, 6, 7, 5, 4},
    {4, 5, 7, 6, 0, 1, 3, 2},
    {5, 4, 6, 7, 1, 0, 2, 3},
    {6, 7, 5, 4, 2, 3, 1, 0},
    {7, 6, 4, 5, 3, 2, 0, 1}};

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

    uint64_t key;

    if (!isHilbert)
    {
        // Codificación Morton (Z-order) optimizada
        // Método de dispersión de bits
        x = (x | (x << 16)) & 0x0000FFFF0000FFFF;
        x = (x | (x << 8)) & 0x00FF00FF00FF00FF;
        x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0F;
        x = (x | (x << 2)) & 0x3333333333333333;
        x = (x | (x << 1)) & 0x5555555555555555;

        y = (y | (y << 16)) & 0x0000FFFF0000FFFF;
        y = (y | (y << 8)) & 0x00FF00FF00FF00FF;
        y = (y | (y << 4)) & 0x0F0F0F0F0F0F0F0F;
        y = (y | (y << 2)) & 0x3333333333333333;
        y = (y | (y << 1)) & 0x5555555555555555;

        z = (z | (z << 16)) & 0x0000FFFF0000FFFF;
        z = (z | (z << 8)) & 0x00FF00FF00FF00FF;
        z = (z | (z << 4)) & 0x0F0F0F0F0F0F0F0F;
        z = (z | (z << 2)) & 0x3333333333333333;
        z = (z | (z << 1)) & 0x5555555555555555;

        // Combinar los bits esparcidos
        key = x | (y << 1) | (z << 2);
    }
    else
    {
        // Codificación de curva Hilbert usando tablas en memoria constante
        uint8_t state = 0;
        key = 0;

        // Procesamiento por bits para los 16 bits más significativos
        // para mantener la misma resolución que en tu implementación original
        // pero optimizando con tablas precalculadas en memoria constante

        // Comienza con el bit más significativo
        {
            uint8_t octant = 0;
            if (x & 0x80000)
                octant |= 1;
            if (y & 0x80000)
                octant |= 2;
            if (z & 0x80000)
                octant |= 4;

            uint8_t position = d_hilbertMap[state][octant];
            key = (key << 3) | position;
            state = d_nextState[state][octant];
        }

        {
            uint8_t octant = 0;
            if (x & 0x40000)
                octant |= 1;
            if (y & 0x40000)
                octant |= 2;
            if (z & 0x40000)
                octant |= 4;

            uint8_t position = d_hilbertMap[state][octant];
            key = (key << 3) | position;
            state = d_nextState[state][octant];
        }

        {
            uint8_t octant = 0;
            if (x & 0x20000)
                octant |= 1;
            if (y & 0x20000)
                octant |= 2;
            if (z & 0x20000)
                octant |= 4;

            uint8_t position = d_hilbertMap[state][octant];
            key = (key << 3) | position;
            state = d_nextState[state][octant];
        }

        {
            uint8_t octant = 0;
            if (x & 0x10000)
                octant |= 1;
            if (y & 0x10000)
                octant |= 2;
            if (z & 0x10000)
                octant |= 4;

            uint8_t position = d_hilbertMap[state][octant];
            key = (key << 3) | position;
            state = d_nextState[state][octant];
        }

        // Procesar los siguientes bits en grupos de 4
        // Para mantener un código más compacto y eficiente
        for (int j = 0; j < 4; j++)
        {
            int shift = 12 - j * 4;

            for (int b = 0; b < 4; b++)
            {
                uint8_t octant = 0;
                uint32_t mask = 1 << (shift + b);

                if (x & mask)
                    octant |= 1;
                if (y & mask)
                    octant |= 2;
                if (z & mask)
                    octant |= 4;

                uint8_t position = d_hilbertMap[state][octant];
                key = (key << 3) | position;
                state = d_nextState[state][octant];
            }
        }
    }

    // Guardar la clave calculada
    keys[i] = key;
}

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

        CudaTimer timer(metrics.energyCalculationTimeMs);

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

        int blockSize = g_blockSize;
        int gridSize = (nBodies + blockSize - 1) / blockSize;

        size_t sharedMemSize = 6 * blockSize * sizeof(double);
        ComputeBoundingBoxKernel<<<gridSize, blockSize, sharedMemSize>>>(d_nodes, d_bodies, d_orderedIndices, useSFC, d_mutex, nBodies);
        CHECK_LAST_CUDA_ERROR();
    }

    void constructOctree()
    {
        CudaTimer timer(metrics.buildTimeMs);

        int blockSize = g_blockSize;

        size_t sharedMemSize = blockSize * sizeof(double) +
                               blockSize * sizeof(double3);

        ConstructOctTreeKernel<<<1, blockSize, sharedMemSize>>>(d_nodes, d_bodies, d_bodiesBuffer, d_orderedIndices, useSFC, 0, nNodes, nBodies, leafLimit);
        CHECK_LAST_CUDA_ERROR();
    }

    void computeForces()
    {
        CudaTimer timer(metrics.forceTimeMs);

        int blockSize = g_blockSize;
        int gridSize = (nBodies + blockSize - 1) / blockSize;

        constexpr int MAX_STACK = 32;

        size_t sharedMemSize = blockSize * MAX_STACK * sizeof(int);

        int device;
        cudaGetDevice(&device);
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, device);

        if (sharedMemSize > props.sharedMemPerBlock)
        {

            int maxThreads = props.sharedMemPerBlock / (MAX_STACK * sizeof(int));
            maxThreads = (maxThreads / 32) * 32;
            if (maxThreads < 32)
            {

                blockSize = 32;

                sharedMemSize = props.sharedMemPerBlock / 32 * 32;
            }
            else
            {
                blockSize = maxThreads;
                sharedMemSize = blockSize * MAX_STACK * sizeof(int);
            }

            gridSize = (nBodies + blockSize - 1) / blockSize;
        }

        ComputeForceKernel<<<gridSize, blockSize, sharedMemSize>>>(
            d_nodes, d_bodies, d_orderedIndices, useSFC, nNodes, nBodies, leafLimit, g_theta);
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
        printf("  Reset:       %.3f ms\n", metrics.resetTimeMs);
        printf("  Bounding box: %.3f ms\n", metrics.bboxTimeMs);
        printf("  Tree build:   %.3f ms\n", metrics.buildTimeMs);
        printf("  Force calc:   %.3f ms\n", metrics.forceTimeMs);
        if (useSFC)
        {
            printf("  Reordering:   %.3f ms\n", metrics.reorderTimeMs);
            printf("  Optimal reorder freq: %d\n", reorderingStrategy.getOptimalFrequency());
            printf("  Degradation rate: %.6f ms/iter\n", reorderingStrategy.getDegradationRate());
            printf("  Performance ratio: %.3f\n", reorderingStrategy.getPerformanceRatio());
            printf("  Performance plateau: %s\n", reorderingStrategy.isPerformancePlateau() ? "yes" : "no");
        }
        printf("  Total update: %.3f ms\n", metrics.totalTimeMs);
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
            totalBuildTime += metrics.buildTimeMs;
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
        std::cout << "  Average time per step: " << totalTime / steps << " ms" << std::endl;
        std::cout << "  Min time: " << minTime << " ms" << std::endl;
        std::cout << "  Max time: " << maxTime << " ms" << std::endl;
        std::cout << "  Build tree: " << totalBuildTime / steps << " ms" << std::endl;
        std::cout << "  Bounding box: " << totalBboxTime / steps << " ms" << std::endl;
        std::cout << "  Compute forces: " << totalForceTime / steps << " ms" << std::endl;
        if (useSFC)
        {
            std::cout << "  Reordering: " << totalReorderTime / steps << " ms" << std::endl;
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
    float getBuildTime() const { return metrics.buildTimeMs; }
    float getBboxTime() const { return metrics.bboxTimeMs; }
    float getReorderTime() const { return metrics.reorderTimeMs; }
    float getEnergyCalculationTime() const { return metrics.energyCalculationTimeMs; }
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
    float getTreeBuildTime() const { return metrics.buildTimeMs; }
    float getSortTime() const { return metrics.reorderTimeMs; }

    void printSummary(int steps)
    {
        double totalTime = metrics.totalTimeMs;
        double totalBuildTime = metrics.buildTimeMs;
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

    int nBodies = 10000;
    bool useSFC = true;
    int reorderFreq = 10;
    BodyDistribution bodyDist = BodyDistribution::GALAXY;
    MassDistribution massDist = MassDistribution::NORMAL;
    unsigned int seed = 42;
    sfc::CurveType curveType = sfc::CurveType::MORTON;
    int numIterations = 100;
    int blockSize = DEFAULT_BLOCK_SIZE;
    double theta = DEFAULT_THETA;

    bool saveMetricsToFile = false;
    std::string metricsFile = "./SFCBarnesHutGPU_metrics.csv";

    bool useDynamicReordering = true;

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
        else if (arg == "-theta" && i + 1 < argc)
            theta = std::stod(argv[++i]);
        else if (arg == "--save-metrics")
            saveMetricsToFile = true;
        else if (arg == "--metrics-file" && i + 1 < argc)
            metricsFile = argv[++i];
        else if (arg == "-h" || arg == "--help")
        {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  -n <num>          Number of bodies (default: 10000)" << std::endl;
            std::cout << "  -nosfc            Disable Space-Filling Curve ordering" << std::endl;
            std::cout << "  -freq <num>       Reordering frequency for fixed mode (default: 10)" << std::endl;
            std::cout << "  -dist <type>      Body distribution: galaxy, solar, uniform, random (default: galaxy)" << std::endl;
            std::cout << "  -mass <type>      Mass distribution: uniform, normal (default: normal)" << std::endl;
            std::cout << "  -seed <num>       Random seed (default: 42)" << std::endl;
            std::cout << "  -curve <type>     SFC curve type: morton, hilbert (default: morton)" << std::endl;
            std::cout << "  -iter <num>       Number of iterations (default: 100)" << std::endl;
            std::cout << "  -block <num>      CUDA block size (default: 256)" << std::endl;
            std::cout << "  -theta <float>    Barnes-Hut opening angle parameter (default: 0.5)" << std::endl;
            std::cout << "  --save-metrics    Save metrics to CSV file" << std::endl;
            std::cout << "  --metrics-file <filename>  Name of CSV metrics file (default: SFCBarnesHutGPU_metrics.csv)" << std::endl;
            return 0;
        }
    }

    g_blockSize = blockSize;
    g_theta = theta;

    SFCBarnesHutGPU simulation(
        nBodies,
        nBodies * 8,
        8,
        bodyDist,
        massDist,
        useSFC,
        curveType,
        useDynamicReordering,
        seed);

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
            simulation.getTheta(),
            simulation.getSortType(),
            simulation.getTotalTime(),
            simulation.getForceCalculationTime(),
            simulation.getTreeBuildTime(),
            simulation.getSortTime(),
            simulation.getPotentialEnergy(),
            simulation.getKineticEnergy(),
            simulation.getTotalEnergy());

        std::cout << "Métricas guardadas en: " << metricsFile << std::endl;
    }

    return 0;
}