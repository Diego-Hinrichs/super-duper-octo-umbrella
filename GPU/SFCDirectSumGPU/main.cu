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
#include <sys/stat.h>  // Para verificar/crear directorios
#include <deque>
#include <numeric>

// =============================================================================
// DYNAMIC REORDERING STRATEGY
// =============================================================================

class SFCDynamicReorderingStrategy
{
private:
    // Parameters for the optimization formula
    double reorderTime;        // Time to reorder (equivalent to Rt)
    double postReorderSimTime; // Simulation time right after reordering (equivalent to Rq)
    double updateTime;         // Time to update without reordering (equivalent to Ut, typically 0 for SFC)
    double degradationRate;    // Average performance degradation per iteration (equivalent to dQ)

    int iterationsSinceReorder;  // Counter for iterations since last reorder
    int currentOptimalFrequency; // Current calculated optimal frequency

    // Tracking metrics for dynamic calculation
    int metricsWindowSize;
    std::deque<double> reorderTimeHistory;
    std::deque<double> postReorderSimTimeHistory;
    std::deque<double> simulationTimeHistory;

    // Calculate the optimal reordering frequency
    int computeOptimalFrequency(int totalIterations)
    {
        // Using the formula: ((nU*nU*dQ/2) + nU*(Ut+Rq) + (Rt+Rq)) * Nit/(nU+1)
        // The optimal frequency is where the derivative = 0

        double determinant = 1.0 - 2.0 * (updateTime - reorderTime) / degradationRate;

        // If determinant is negative, use a default value
        if (determinant < 0)
            return 10; // Default to 10 as a reasonable value

        double optNu = -1.0 + sqrt(determinant);

        // Convert to integer values and check which one is better
        int nu1 = static_cast<int>(optNu);
        int nu2 = nu1 + 1;

        if (nu1 <= 0)
            return 1; // Avoid negative or zero values

        // Calculate total time with nu1 and nu2
        double time1 = ((nu1 * nu1 * degradationRate / 2.0) + nu1 * (updateTime + postReorderSimTime) +
                        (reorderTime + postReorderSimTime)) *
                       totalIterations / (nu1 + 1.0);
        double time2 = ((nu2 * nu2 * degradationRate / 2.0) + nu2 * (updateTime + postReorderSimTime) +
                        (reorderTime + postReorderSimTime)) *
                       totalIterations / (nu2 + 1.0);

        return time1 < time2 ? nu1 : nu2;
    }

    // Update metrics with new timing information
    void updateMetrics(double newReorderTime, double newSimTime)
    {
        // Update reorder time if available
        if (newReorderTime > 0)
        {
            reorderTimeHistory.push_back(newReorderTime);
            if (reorderTimeHistory.size() > metricsWindowSize)
            {
                reorderTimeHistory.pop_front();
            }

            // Recalculate average reorder time
            reorderTime = std::accumulate(reorderTimeHistory.begin(), reorderTimeHistory.end(), 0.0) /
                          reorderTimeHistory.size();
        }

        // Track simulation times to calculate degradation
        simulationTimeHistory.push_back(newSimTime);
        if (simulationTimeHistory.size() > metricsWindowSize)
        {
            simulationTimeHistory.pop_front();
        }

        // If first simulation after reorder, update postReorderSimTime
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

        // Calculate degradation rate if we have enough data
        if (simulationTimeHistory.size() >= 3)
        {
            // Simple linear regression on the most recent simulation times
            // to estimate the degradation rate
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

            // Calculate slope (degradation rate)
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
          degradationRate(0.001), // Initial small degradation assumption
          iterationsSinceReorder(0),
          currentOptimalFrequency(10), // Start with a reasonable default
          metricsWindowSize(windowSize)
    {
    }

    // Check if reordering is needed based on current metrics
    bool shouldReorder(double lastSimTime = 0.0, double lastReorderTime = 0.0)
    {
        iterationsSinceReorder++;

        // Update metrics with new timing information
        updateMetrics(lastReorderTime, lastSimTime);

        // Recalculate optimal frequency periodically
        if (iterationsSinceReorder % 10 == 0)
        {
            currentOptimalFrequency = computeOptimalFrequency(1000); // Assuming 1000 total iterations

            // Ensure frequency is reasonable
            currentOptimalFrequency = std::max(1, std::min(100, currentOptimalFrequency));
        }

        // Decide if we should reorder based on current counter and optimal frequency
        bool shouldReorder = iterationsSinceReorder >= currentOptimalFrequency;

        // Reset counter if reordering
        if (shouldReorder)
        {
            iterationsSinceReorder = 0;
        }

        return shouldReorder;
    }

    // Public method for updating metrics with just sort time
    void updateMetrics(double sortTime)
    {
        // Call the internal method with proper defaults
        updateMetrics(sortTime, 0.0);
    }

    // Set the window size for metrics tracking
    void setWindowSize(int windowSize)
    {
        if (windowSize > 0) {
            metricsWindowSize = windowSize;
        }
    }

    // Get the current optimal frequency
    int getOptimalFrequency() const
    {
        return currentOptimalFrequency;
    }

    // Get the current degradation rate estimate
    double getDegradationRate() const
    {
        return degradationRate;
    }

    // Reset the strategy
    void reset()
    {
        iterationsSinceReorder = 0;
        reorderTimeHistory.clear();
        postReorderSimTimeHistory.clear();
        simulationTimeHistory.clear();
    }
};

// =============================================================================
// DEFINICIÓN DE TIPOS
// =============================================================================

/**
 * @brief 3D vector structure with basic operations
 */
struct Vector
{
    double x;
    double y;
    double z;

    // Default constructor
    __host__ __device__ Vector() : x(0.0), y(0.0), z(0.0) {}

    // Constructor with initial values
    __host__ __device__ Vector(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    // Vector addition
    __host__ __device__ Vector operator+(const Vector &other) const
    {
        return Vector(x + other.x, y + other.y, z + other.z);
    }

    // Vector subtraction
    __host__ __device__ Vector operator-(const Vector &other) const
    {
        return Vector(x - other.x, y - other.y, z - other.z);
    }

    // Scalar multiplication
    __host__ __device__ Vector operator*(double scalar) const
    {
        return Vector(x * scalar, y * scalar, z * scalar);
    }

    // Dot product
    __host__ __device__ double dot(const Vector &other) const
    {
        return x * other.x + y * other.y + z * other.z;
    }

    // Vector length squared
    __host__ __device__ double lengthSquared() const
    {
        return x * x + y * y + z * z;
    }

    // Vector length
    __host__ __device__ double length() const
    {
        return sqrt(lengthSquared());
    }

    // Vector normalization
    __host__ __device__ Vector normalize() const
    {
        double len = length();
        if (len > 0.0)
        {
            return Vector(x / len, y / len, z / len);
        }
        return *this;
    }

    // Distance between two vectors
    __host__ __device__ static double distance(const Vector &a, const Vector &b)
    {
        return (a - b).length();
    }

    // Distance squared between two vectors (more efficient)
    __host__ __device__ static double distanceSquared(const Vector &a, const Vector &b)
    {
        return (a - b).lengthSquared();
    }
};

/**
 * @brief Body structure representing a celestial body
 */
struct Body
{
    bool isDynamic;      // Whether the body moves or is static
    double mass;         // Mass of the body
    double radius;       // Radius of the body
    Vector position;     // Position in 3D space
    Vector velocity;     // Velocity vector
    Vector acceleration; // Acceleration vector

    // Default constructor
    __host__ __device__ Body() : isDynamic(true),
                                mass(0.0),
                                radius(0.0),
                                position(),
                                velocity(),
                                acceleration() {}
};

/**
 * @brief Performance metrics for simulation timing
 */
struct SimulationMetrics
{
    float forceTimeMs;
    float reorderTimeMs;
    float totalTimeMs;

    SimulationMetrics() : forceTimeMs(0.0f),
                          reorderTimeMs(0.0f),
                          totalTimeMs(0.0f) {}
};

// Space-filling curve types
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
            // Allocate memory for ordered indices
            cudaMalloc(&d_orderedIndices, numBodies * sizeof(int));
            
            // Allocate host memory for ordering
            h_keys = new uint64_t[numBodies];
            h_indices = new int[numBodies];
        }
        
        ~BodySorter() 
        {
            if (d_orderedIndices) cudaFree(d_orderedIndices);
            if (h_keys) delete[] h_keys;
            if (h_indices) delete[] h_indices;
        }
        
        void setCurveType(CurveType type) 
        {
            curveType = type;
        }
        
        // Calculate Morton/Hilbert code for a 3D position
        uint64_t calculateSFCKey(const Vector& pos, const Vector& minBound, const Vector& maxBound) 
        {
            // Normalize position to [0,1] range
            double normalizedX = (pos.x - minBound.x) / (maxBound.x - minBound.x);
            double normalizedY = (pos.y - minBound.y) / (maxBound.y - minBound.y);
            double normalizedZ = (pos.z - minBound.z) / (maxBound.z - minBound.z);
            
            // Clamp to [0,1]
            normalizedX = std::max(0.0, std::min(1.0, normalizedX));
            normalizedY = std::max(0.0, std::min(1.0, normalizedY));
            normalizedZ = std::max(0.0, std::min(1.0, normalizedZ));
            
            // Convert to integer coordinates with 21 bits precision (3*21=63 bits)
            uint32_t x = static_cast<uint32_t>(normalizedX * ((1 << 21) - 1));
            uint32_t y = static_cast<uint32_t>(normalizedY * ((1 << 21) - 1));
            uint32_t z = static_cast<uint32_t>(normalizedZ * ((1 << 21) - 1));
            
            if (curveType == CurveType::MORTON) {
                return mortonEncode(x, y, z);
            } else {
                return hilbertEncode(x, y, z);
            }
        }
        
        uint64_t mortonEncode(uint32_t x, uint32_t y, uint32_t z) 
        {
            // Interleave bits of x, y, and z
            uint64_t answer = 0;
            for (uint64_t i = 0; i < 21; ++i) {
                answer |= ((x & ((uint64_t)1 << i)) << (2*i)) |
                          ((y & ((uint64_t)1 << i)) << (2*i + 1)) |
                          ((z & ((uint64_t)1 << i)) << (2*i + 2));
            }
            return answer;
        }
        
        uint64_t hilbertEncode(uint32_t x, uint32_t y, uint32_t z) 
        {
            // Simplified Hilbert encoding (an approximation)
            // In a real implementation, this would be more complex
            uint64_t result = 0;
            
            // Simplified approach that preserves some spatial locality
            for (int i = 20; i >= 0; i--) {
                uint8_t bitPos = i * 3;
                uint8_t bitX = (x >> i) & 1;
                uint8_t bitY = (y >> i) & 1;
                uint8_t bitZ = (z >> i) & 1;
                
                // Combine bits with a simple pattern that provides some spatial locality
                uint8_t idx = (bitX << 2) | (bitY << 1) | bitZ;
                result = (result << 3) | idx;
            }
            
            return result;
        }
        
        // Sort bodies by their SFC position
        int* sortBodies(Body* d_bodies, const Vector& minBound, const Vector& maxBound) 
        {
            // Copy bodies to host for key calculation
            Body* h_bodies = new Body[nBodies];
            cudaMemcpy(h_bodies, d_bodies, nBodies * sizeof(Body), cudaMemcpyDeviceToHost);
            
            // Calculate SFC keys and initialize indices
            for (int i = 0; i < nBodies; i++) {
                h_keys[i] = calculateSFCKey(h_bodies[i].position, minBound, maxBound);
                h_indices[i] = i;
            }
            
            // Sort indices by keys
            std::sort(h_indices, h_indices + nBodies, 
                     [this](int a, int b) { return h_keys[a] < h_keys[b]; });
            
            // Copy sorted indices to device
            cudaMemcpy(d_orderedIndices, h_indices, nBodies * sizeof(int), cudaMemcpyHostToDevice);
            
            // Cleanup
            delete[] h_bodies;
            
            return d_orderedIndices;
        }
        
    private:
        int nBodies;
        CurveType curveType;
        int* d_orderedIndices = nullptr;
        uint64_t* h_keys = nullptr;
        int* h_indices = nullptr;
    };
}

// Enumeraciones para tipos de distribución
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

// =============================================================================
// CONSTANTES
// =============================================================================

// Constantes físicas
constexpr double GRAVITY = 6.67430e-11;        // Gravitational constant
constexpr double SOFTENING_FACTOR = 0.5;       // Softening factor for avoiding div by 0
constexpr double TIME_STEP = 25000.0;          // Time step in seconds
constexpr double COLLISION_THRESHOLD = 1.0e10; // Collision threshold distance

// Constantes astronómicas
constexpr double MAX_DIST = 5.0e11;     // Maximum distance for initial distribution
constexpr double MIN_DIST = 2.0e10;     // Minimum distance for initial distribution
constexpr double EARTH_MASS = 5.974e24; // Mass of Earth in kg
constexpr double EARTH_DIA = 12756.0;   // Diameter of Earth in km
constexpr double SUN_MASS = 1.989e30;   // Mass of Sun in kg
constexpr double SUN_DIA = 1.3927e6;    // Diameter of Sun in km
constexpr double CENTERX = 0;           // Center of simulation X coordinate
constexpr double CENTERY = 0;           // Center of simulation Y coordinate
constexpr double CENTERZ = 0;           // Center of simulation Z coordinate

// Constantes de implementación
constexpr int DEFAULT_BLOCK_SIZE = 256; // Default CUDA block size

// Definir macros para simplificar el código
#define E SOFTENING_FACTOR
#define DT TIME_STEP
#define COLLISION_TH COLLISION_THRESHOLD

// Variable global para el tamaño de bloque
int g_blockSize = DEFAULT_BLOCK_SIZE;

// =============================================================================
// MANEJO DE ERRORES CUDA
// =============================================================================

/**
 * @brief Check CUDA error and output diagnostic information
 * @param err CUDA error code to check
 * @param func Name of the function that returned the error
 * @param file Source file name
 * @param line Line number in the source file
 */
inline void checkCudaError(cudaError_t err, const char *const func, const char *const file, int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

/**
 * @brief Check last CUDA error and output diagnostic information
 * @param file Source file name
 * @param line Line number in the source file
 */
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

// Macros to simplify error checking
#define CHECK_CUDA_ERROR(err) checkCudaError(err, __func__, __FILE__, __LINE__)
#define CHECK_LAST_CUDA_ERROR() checkLastCudaError(__FILE__, __LINE__)

// =============================================================================
// TIMER CLASS FOR PERFORMANCE MEASUREMENTS
// =============================================================================

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

// =============================================================================
// KERNEL FUNCTIONS
// =============================================================================

__global__ void SFCDirectSumForceKernel(Body *bodies, int *orderedIndices, bool useSFC, int nBodies)
{
    // Use dynamic shared memory
    extern __shared__ char sharedMemory[];
    Vector *sharedPos = (Vector*)sharedMemory;
    double *sharedMass = (double*)(sharedPos + blockDim.x);

    // Get global thread ID
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;

    // Get the real body index when using SFC ordering
    int realBodyIndex = (useSFC && orderedIndices != nullptr) ? orderedIndices[i] : i;

    // Load data only if the index is valid
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

    // Use block size as tile size for better memory access patterns
    const int tileSize = blockDim.x;

    // Process all tiles
    for (int tile = 0; tile < (nBodies + tileSize - 1) / tileSize; ++tile)
    {
        // Load this tile to shared memory
        int idx = tile * tileSize + tx;

        // Only load valid data to shared memory
        if (tx < tileSize)
        { // Ensure we don't exceed array size
            if (idx < nBodies)
            {
                // When using SFC ordering, get the real body index
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

        // Calculate force only for valid and dynamic bodies
        if (i < nBodies && isDynamic)
        {
            // Limit the loop to the real tile size
            int tileLimit = min(tileSize, nBodies - tile * tileSize);

            for (int j = 0; j < tileLimit; ++j)
            {
                int jBody = tile * tileSize + j;

                // Avoid self-interaction
                if (jBody != i)
                {
                    // Distance vector
                    double rx = sharedPos[j].x - myPos.x;
                    double ry = sharedPos[j].y - myPos.y;
                    double rz = sharedPos[j].z - myPos.z;

                    // Distance squared with softening
                    double distSqr = rx * rx + ry * ry + rz * rz + E * E;
                    double dist = sqrt(distSqr);

                    // Apply force only if above collision threshold
                    if (dist >= COLLISION_TH)
                    {
                        double forceMag = (GRAVITY * myMass * sharedMass[j]) / (dist * distSqr);

                        // Accumulate acceleration
                        myAcc.x += rx * forceMag / myMass;
                        myAcc.y += ry * forceMag / myMass;
                        myAcc.z += rz * forceMag / myMass;
                    }
                }
            }
        }

        __syncthreads();
    }

    // Update the body only if valid and dynamic
    if (i < nBodies && isDynamic)
    {
        // Save acceleration
        bodies[realBodyIndex].acceleration = myAcc;

        // Update velocity
        myVel.x += myAcc.x * DT;
        myVel.y += myAcc.y * DT;
        myVel.z += myAcc.z * DT;
        bodies[realBodyIndex].velocity = myVel;

        // Update position
        myPos.x += myVel.x * DT;
        myPos.y += myVel.y * DT;
        myPos.z += myVel.z * DT;
        bodies[realBodyIndex].position = myPos;
    }
}

// =============================================================================
// SIMULATION CLASS
// =============================================================================

class SFCDirectSumGPU
{
private:
    Body *h_bodies = nullptr;        // Host bodies
    Body *d_bodies = nullptr;        // Device bodies
    int nBodies;                     // Number of bodies
    bool useSFC;                     // Whether to use SFC ordering
    sfc::BodySorter *sorter;         // SFC sorter
    int *d_orderedIndices;           // Device ordered indices
    sfc::CurveType curveType;        // Type of SFC curve
    int reorderFrequency;            // How often to reorder bodies
    int iterationCounter;            // Counter for reordering
    Vector minBound;                 // Minimum bounds for SFC calculation
    Vector maxBound;                 // Maximum bounds for SFC calculation
    SimulationMetrics metrics;       // Performance metrics
    
    // Dynamic reordering strategy
    bool useDynamicReordering;
    SFCDynamicReorderingStrategy reorderingStrategy;

    void updateBoundingBox()
    {
        // Update bounds for SFC calculation
        minBound = Vector(INFINITY, INFINITY, INFINITY);
        maxBound = Vector(-INFINITY, -INFINITY, -INFINITY);

        for (int i = 0; i < nBodies; i++)
        {
            Vector pos = h_bodies[i].position;

            // Update minimum bounds
            minBound.x = std::min(minBound.x, pos.x);
            minBound.y = std::min(minBound.y, pos.y);
            minBound.z = std::min(minBound.z, pos.z);

            // Update maximum bounds
            maxBound.x = std::max(maxBound.x, pos.x);
            maxBound.y = std::max(maxBound.y, pos.y);
            maxBound.z = std::max(maxBound.z, pos.z);
        }

        // Add padding to avoid edge issues
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

        // Copy bodies to host
        CHECK_CUDA_ERROR(cudaMemcpy(h_bodies, d_bodies, nBodies * sizeof(Body), cudaMemcpyDeviceToHost));
        
        // Update bounds for SFC calculation
        updateBoundingBox();

        // Get indices ordered by SFC
        d_orderedIndices = sorter->sortBodies(d_bodies, minBound, maxBound);
    }

    void initializeDistribution(BodyDistribution dist, MassDistribution massDist, unsigned int seed)
    {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<double> pos_dist(-MAX_DIST, MAX_DIST);
        std::uniform_real_distribution<double> vel_dist(-1.0e3, 1.0e3);
        std::normal_distribution<double> normal_pos_dist(0.0, MAX_DIST/2.0);
        std::normal_distribution<double> normal_vel_dist(0.0, 5.0e2);
        
        // Initialize with random uniform distribution regardless of requested distribution type
        // This simplified version only supports random initialization for now
        for (int i = 0; i < nBodies; i++) {
            if (massDist == MassDistribution::UNIFORM) {
                // Position
                h_bodies[i].position = Vector(
                    CENTERX + pos_dist(gen),
                    CENTERY + pos_dist(gen),
                    CENTERZ + pos_dist(gen)
                );
                
                // Velocity
                h_bodies[i].velocity = Vector(
                    vel_dist(gen),
                    vel_dist(gen),
                    vel_dist(gen)
                );
            } else { // NORMAL distribution
                // Position
                h_bodies[i].position = Vector(
                    CENTERX + normal_pos_dist(gen),
                    CENTERY + normal_pos_dist(gen),
                    CENTERZ + normal_pos_dist(gen)
                );
                
                // Velocity
                h_bodies[i].velocity = Vector(
                    normal_vel_dist(gen),
                    normal_vel_dist(gen),
                    normal_vel_dist(gen)
                );
            }
            
            // Mass always 1.0
            h_bodies[i].mass = 1.0;
            h_bodies[i].radius = pow(h_bodies[i].mass / EARTH_MASS, 1.0/3.0) * (EARTH_DIA/2.0);
            h_bodies[i].isDynamic = true;
            h_bodies[i].acceleration = Vector(0, 0, 0);
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
          reorderingStrategy(10) // Start with window size of 10
    {
        // Allocate host and device memory
        h_bodies = new Body[nBodies];
        CHECK_CUDA_ERROR(cudaMalloc(&d_bodies, nBodies * sizeof(Body)));

        // Initialize SFC sorter if needed
        if (useSFC)
        {
            sorter = new sfc::BodySorter(nBodies, curveType);
        }
        else
        {
            sorter = nullptr;
        }

        // Initialize with random uniform distribution
        initializeDistribution(BodyDistribution::RANDOM_UNIFORM, MassDistribution::UNIFORM);

        // Copy bodies to device
        CHECK_CUDA_ERROR(cudaMemcpy(d_bodies, h_bodies, nBodies * sizeof(Body), cudaMemcpyHostToDevice));

        // Initial ordering
        if (useSFC)
        {
            orderBodiesBySFC();
        }

        std::cout << "SFC Direct Sum GPU simulation created with " << nBodies << " bodies." << std::endl;
        if (useSFC)
        {
            std::cout << "Space-Filling Curve ordering enabled with "
                     << (useDynamicReordering ? "dynamic" : "fixed") << " reordering"; 
            if (!useDynamicReordering) {
                std::cout << " frequency " << reorderFrequency;
            }
            std::cout << " and curve type " 
                     << (curveType == sfc::CurveType::MORTON ? "MORTON" : "HILBERT") << std::endl;
        }
    }

    ~SFCDirectSumGPU()
    {
        // Free resources
        delete[] h_bodies;
        if (d_bodies) CHECK_CUDA_ERROR(cudaFree(d_bodies));
        if (sorter) delete sorter;
    }

    void setCurveType(sfc::CurveType type)
    {
        if (type != curveType)
        {
            curveType = type;

            // Update sorter with new curve type
            if (sorter)
                sorter->setCurveType(type);

            // Force reordering on next update
            iterationCounter = reorderFrequency;
        }
    }

    void computeForces()
    {
        // Measure execution time
        CudaTimer timer(metrics.forceTimeMs);

        // Launch kernel with SFC support
        int blockSize = g_blockSize;
        int gridSize = (nBodies + blockSize - 1) / blockSize;

        size_t sharedMemSize = blockSize * sizeof(Vector) + blockSize * sizeof(double);
        SFCDirectSumForceKernel<<<gridSize, blockSize, sharedMemSize>>>(d_bodies, d_orderedIndices, useSFC, nBodies);
        CHECK_LAST_CUDA_ERROR();
    }

    void update()
    {
        CudaTimer timer(metrics.totalTimeMs);

        // Check if reordering is needed
        bool shouldReorder = false;
        if (useSFC) {
            if (useDynamicReordering) {
                // Use dynamic strategy to decide if reordering is needed
                shouldReorder = reorderingStrategy.shouldReorder(metrics.forceTimeMs, metrics.reorderTimeMs);
            } else {
                // Use fixed frequency
                shouldReorder = (iterationCounter >= reorderFrequency);
            }
            
            if (shouldReorder) {
                orderBodiesBySFC();
                iterationCounter = 0;
            }
        }

        computeForces();
        iterationCounter++;
        
        // If using dynamic strategy, update it with the latest timings
        if (useSFC && useDynamicReordering) {
            reorderingStrategy.updateMetrics(shouldReorder ? metrics.reorderTimeMs : 0.0, metrics.forceTimeMs);
        }
    }

    void printPerformance()
    {
        printf("Performance Metrics:\n");
        printf("  Force calculation: %.3f ms\n", metrics.forceTimeMs);
        if (useSFC) {
            printf("  Reordering: %.3f ms\n", metrics.reorderTimeMs);
            if (useDynamicReordering) {
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

    int getOptimalReorderFrequency() const {
        if (useDynamicReordering) {
            return reorderingStrategy.getOptimalFrequency();
        }
        return reorderFrequency;
    }
    
    double getTotalTime() const { return metrics.totalTimeMs; }
    double getForceCalculationTime() const { return metrics.forceTimeMs; }
    double getSortTime() const { return metrics.reorderTimeMs; }
    int getNumBodies() const { return nBodies; }
    int getBlockSize() const { return g_blockSize; }
    int getSortType() const { return static_cast<int>(curveType); }
    bool isDynamicReordering() const { return useDynamicReordering; }
};

// =============================================================================
// GUARDADO DE MÉTRICAS EN CSV
// =============================================================================

// Función para verificar si un directorio existe
bool dirExists(const std::string& dirName) {
    struct stat info;
    return stat(dirName.c_str(), &info) == 0 && (info.st_mode & S_IFDIR);
}

// Función para crear un directorio
bool createDir(const std::string& dirName) {
    #ifdef _WIN32
    int status = mkdir(dirName.c_str());
    #else
    int status = mkdir(dirName.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    #endif
    return status == 0;
}

// Función para asegurar que el directorio existe
bool ensureDirExists(const std::string& dirPath) {
    if (dirExists(dirPath)) {
        return true;
    }
    
    std::cout << "Creando directorio: " << dirPath << std::endl;
    if (createDir(dirPath)) {
        return true;
    } else {
        std::cerr << "Error: No se pudo crear el directorio " << dirPath << std::endl;
        return false;
    }
}

void initializeCsv(const std::string& filename, bool append = false) {
    // Extraer el directorio del nombre de archivo
    size_t pos = filename.find_last_of('/');
    if (pos != std::string::npos) {
        std::string dirPath = filename.substr(0, pos);
        if (!ensureDirExists(dirPath)) {
            std::cerr << "Error: No se puede crear el directorio para el archivo " << filename << std::endl;
            return;
        }
    }
    
    std::ofstream file;
    if (append) {
        file.open(filename, std::ios::app);
    } else {
        file.open(filename);
    }
    
    if (!file.is_open()) {
        std::cerr << "Error: No se pudo abrir el archivo " << filename << " para escritura" << std::endl;
        return;
    }
    
    // Solo escribimos el encabezado si estamos creando un nuevo archivo
    if (!append) {
        file << "timestamp,method,bodies,steps,block_size,sort_type,total_time_ms,avg_step_time_ms,force_calculation_time_ms,sort_time_ms" << std::endl;
    }
    
    file.close();
    std::cout << "Archivo CSV " << (append ? "actualizado" : "inicializado") << ": " << filename << std::endl;
}

void saveMetrics(const std::string& filename, 
                int bodies, 
                int steps, 
                int blockSize,
                int sortType,
                double totalTime, 
                double forceCalculationTime,
                double sortTime) {
    std::ofstream file(filename, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Error: No se pudo abrir el archivo " << filename << " para escritura." << std::endl;
        return;
    }
    
    // Obtener timestamp actual
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    std::stringstream timestamp;
    timestamp << std::put_time(std::localtime(&now_c), "%Y-%m-%d %H:%M:%S");
    
    double avgTimePerStep = totalTime / steps;
    
    file << timestamp.str() << ","
         << "GPU_SFC_Direct_Sum" << ","
         << bodies << ","
         << steps << ","
         << blockSize << ","
         << sortType << ","
         << totalTime << ","
         << avgTimePerStep << ","
         << forceCalculationTime << ","
         << sortTime << std::endl;
    
    file.close();
    std::cout << "Métricas guardadas en: " << filename << std::endl;
}

// =============================================================================
// MAIN FUNCTION
// =============================================================================

int main(int argc, char **argv)
{
    // Default parameters
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

    // Añadir nuevas variables para métricas
    bool saveMetricsToFile = false;
    std::string metricsFile = "./SFCDirectSumGPU_metrics.csv";

    // Parse command line arguments
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
        else if (arg == "--save-metrics") {
            saveMetricsToFile = true;
        } else if (arg == "--metrics-file" && i + 1 < argc) {
            metricsFile = argv[++i];
        } else if (arg == "--dynamic-reordering") {
            // Parameter removed, kept as no-op for backward compatibility since dynamic reordering is enabled by default
        }
    }

    // Update global block size
    g_blockSize = blockSize;

    // Check CUDA device
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0)
    {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }

    // Print device info
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    std::cout << "Using GPU: " << deviceProp.name << std::endl;

    // Create simulation
    SFCDirectSumGPU simulation(
        nBodies,
        useSFC,
        reorderFreq,
        curveType,
        useDynamicReordering);

    // Run simulation
    simulation.runSimulation(numIterations);

    // Al final del main, añadir guardado de métricas
    // Si se solicitó guardar métricas, inicializar el archivo CSV y guardar
    if (saveMetricsToFile) {
        // Verificar si el archivo existe para decidir si añadir encabezados
        bool fileExists = false;
        std::ifstream checkFile(metricsFile);
        if (checkFile.good()) {
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
            simulation.getSortTime()
        );
        
        std::cout << "Métricas guardadas en: " << metricsFile << std::endl;
    }

    return 0;
} 