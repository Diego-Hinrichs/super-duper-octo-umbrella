#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <string>
#include <random>
#include <limits>
#include <algorithm>
#include <cfloat>  // Added for DBL_MAX
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <sys/stat.h>  // Para verificar/crear directorios
#include <deque>
#include <numeric>
#include <cub/cub.cuh>  // Added for GPU Radix Sort
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

// Prototipos de kernels CUDA
__global__ void extractPositionsKernel(struct Body* bodies, struct Vector* positions, int n);
__global__ void calculateSFCKeysKernel(struct Vector *positions, uint64_t *keys, int nBodies,
                                     struct Vector minBound, struct Vector maxBound, bool isHilbert);

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
 * @brief Node structure for Barnes-Hut octree
 */
struct Node
{
    bool isLeaf;        // Is this a leaf node?
    int firstChildIndex; // Index of first child (other 7 children follow consecutively)
    int bodyIndex;      // Index of the body (if leaf)
    int bodyCount;      // Number of bodies in this node
    Vector position;    // Center of mass position
    double mass;        // Total mass of the node
    double radius;      // Half the width of the node
    Vector min;         // Minimum coordinates of the node
    Vector max;         // Maximum coordinates of the node

    // Default constructor initializes to default values
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

/**
 * @brief Performance metrics for simulation timing
 */
struct SimulationMetrics
{
    float resetTimeMs;
    float bboxTimeMs;
    float buildTimeMs;
    float forceTimeMs;
    float reorderTimeMs;
    float totalTimeMs;
    float energyCalculationTimeMs;  // Added time for energy calculation

    SimulationMetrics() : resetTimeMs(0.0f),
                          bboxTimeMs(0.0f),
                          buildTimeMs(0.0f),
                          forceTimeMs(0.0f),
                          reorderTimeMs(0.0f),
                          totalTimeMs(0.0f),
                          energyCalculationTimeMs(0.0f) {}
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
            
            // Allocate device memory for keys
            cudaMalloc(&d_keys, numBodies * sizeof(uint64_t));
            
            // Allocate host memory for ordering (reduced memory footprint)
            h_keys = new uint64_t[numBodies];
            h_indices = new int[numBodies];
            
            // For position data copying
            cudaMalloc(&d_positions, numBodies * sizeof(Vector));
            h_positions = new Vector[numBodies];
            
            // Create streams for overlapping operations
            cudaStreamCreate(&stream1);
            cudaStreamCreate(&stream2);
        }
        
        ~BodySorter() 
        {
            if (d_orderedIndices) cudaFree(d_orderedIndices);
            if (d_keys) cudaFree(d_keys);
            if (d_positions) cudaFree(d_positions);
            
            if (h_keys) delete[] h_keys;
            if (h_indices) delete[] h_indices;
            if (h_positions) delete[] h_positions;
            
            cudaStreamDestroy(stream1);
            cudaStreamDestroy(stream2);
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
            normalizedX = std::max(0.0, std::min(0.999999, normalizedX));
            normalizedY = std::max(0.0, std::min(0.999999, normalizedY));
            normalizedZ = std::max(0.0, std::min(0.999999, normalizedZ));
            
            // Reduce precision from 21 to 20 bits for faster calculation
            uint32_t x = static_cast<uint32_t>(normalizedX * ((1 << 20) - 1));
            uint32_t y = static_cast<uint32_t>(normalizedY * ((1 << 20) - 1));
            uint32_t z = static_cast<uint32_t>(normalizedZ * ((1 << 20) - 1));
            
            if (curveType == CurveType::MORTON) {
                return mortonEncode(x, y, z);
            } else {
                return hilbertEncode(x, y, z);
            }
        }
        
        uint64_t mortonEncode(uint32_t x, uint32_t y, uint32_t z) 
        {
            // Optimized bit-spreading technique
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
            // Simplified Hilbert implementation - focus on performance
            // Uses only 16 bits for faster computation
            x &= 0xFFFF;
            y &= 0xFFFF;
            z &= 0xFFFF;
            
            uint64_t result = 0;
            uint8_t state = 0;
            
            // Pre-computed tables
            static const uint8_t hilbertMap[8][8] = {
                {0, 1, 3, 2, 7, 6, 4, 5}, // state 0
                {4, 5, 7, 6, 0, 1, 3, 2}, // state 1
                {6, 7, 5, 4, 2, 3, 1, 0}, // state 2
                {2, 3, 1, 0, 6, 7, 5, 4}, // state 3
                {0, 7, 1, 6, 3, 4, 2, 5}, // state 4
                {6, 1, 7, 0, 5, 2, 4, 3}, // state 5
                {2, 5, 3, 4, 1, 6, 0, 7}, // state 6
                {4, 3, 5, 2, 7, 0, 6, 1}  // state 7
            };
            
            // Simplified state transition table
            static const uint8_t nextState[8][8] = {
                {0, 1, 3, 2, 7, 6, 4, 5}, // state 0
                {1, 0, 2, 3, 4, 5, 7, 6}, // state 1
                {2, 3, 1, 0, 5, 4, 6, 7}, // state 2
                {3, 2, 0, 1, 6, 7, 5, 4}, // state 3
                {4, 5, 7, 6, 0, 1, 3, 2}, // state 4
                {5, 4, 6, 7, 1, 0, 2, 3}, // state 5
                {6, 7, 5, 4, 2, 3, 1, 0}, // state 6
                {7, 6, 4, 5, 3, 2, 0, 1}  // state 7
            };
            
            // Process bits from most significant to least - using fewer bits for speed
            for (int i = 15; i >= 0; i--) { // Reduced from 20 to 16 bits
                uint8_t octant = 0;
                if (x & (1 << i)) octant |= 1;
                if (y & (1 << i)) octant |= 2;
                if (z & (1 << i)) octant |= 4;
                
                uint8_t position = hilbertMap[state][octant];
                result = (result << 3) | position;
                state = nextState[state][octant];
            }
            
            return result;
        }
        
        // Sort bodies by their SFC position - optimized version
        int* sortBodies(Body* d_bodies, const Vector& minBound, const Vector& maxBound) 
        {
            // Extract positions in separate stream
            int blockSize = 256;
            int gridSize = (nBodies + blockSize - 1) / blockSize;
            
            // Extract positions to device buffer
            extractPositionsKernel<<<gridSize, blockSize, 0, stream1>>>(d_bodies, d_positions, nBodies);
            
            // Calculate SFC keys directly on GPU - much faster than CPU
            calculateSFCKeysKernel<<<gridSize, blockSize, 0, stream1>>>(
                d_positions, d_keys, nBodies, minBound, maxBound, curveType == CurveType::HILBERT);
            
            // Initialize index array (1, 2, 3, ..., nBodies)
            thrust::counting_iterator<int> first(0);
            thrust::copy(first, first + nBodies, thrust::device_pointer_cast(d_orderedIndices));
            
            // Create device memory for radix sort
            void *d_temp_storage = NULL;
            size_t temp_storage_bytes = 0;
            
            // Use GPU Radix Sort instead of CPU sort (much more efficient)
            // First determine required storage
            cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                          d_keys, d_keys, 
                                          d_orderedIndices, d_orderedIndices,
                                          nBodies);
            
            // Allocate temporary storage
            cudaMalloc(&d_temp_storage, temp_storage_bytes);
            
            // Perform the sort
            cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                          d_keys, d_keys,
                                          d_orderedIndices, d_orderedIndices,
                                          nBodies);
            
            // Free temporary storage
            cudaFree(d_temp_storage);
            
            return d_orderedIndices;
        }
        
    private:
        int nBodies;
        CurveType curveType;
        int* d_orderedIndices = nullptr;
        uint64_t* d_keys = nullptr;
        uint64_t* h_keys = nullptr;
        int* h_indices = nullptr;
        
        // Separate position data for faster transfers
        Vector* d_positions = nullptr;
        Vector* h_positions = nullptr;
        
        // CUDA streams for overlapping operations
        cudaStream_t stream1, stream2;
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
constexpr int MAX_NODES = 1000000;      // Maximum number of octree nodes
constexpr int N_LEAF = 8;               // Bodies per leaf before subdividing
constexpr double DEFAULT_THETA = 0.5;   // Default opening angle theta

// Definir macros para simplificar el código
#define E SOFTENING_FACTOR
#define DT TIME_STEP
#define COLLISION_TH COLLISION_THRESHOLD

// Variable global para el tamaño de bloque y theta
int g_blockSize = DEFAULT_BLOCK_SIZE;
double g_theta = DEFAULT_THETA;

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
        file << "timestamp,method,bodies,steps,block_size,theta,sort_type,total_time_ms,avg_step_time_ms,force_calculation_time_ms,tree_build_time_ms,sort_time_ms,potential_energy,kinetic_energy,total_energy" << std::endl;
    }
    
    file.close();
    std::cout << "Archivo CSV " << (append ? "actualizado" : "inicializado") << ": " << filename << std::endl;
}

void saveMetrics(const std::string& filename, 
                int bodies, 
                int steps, 
                int blockSize,
                float theta,
                const char* sortType,
                float totalTime, 
                float forceCalculationTime,
                float treeBuildTime,
                float sortTime,
                double potentialEnergy,
                double kineticEnergy,
                double totalEnergy) {
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

// Macro for calling a kernel CUDA with verification of errors
#define CUDA_KERNEL_CALL(kernel, gridSize, blockSize, sharedMem, stream, ...) \
    do                                                                        \
    {                                                                         \
        kernel<<<gridSize, blockSize, sharedMem, stream>>>(__VA_ARGS__);      \
        CHECK_LAST_CUDA_ERROR();                                              \
    } while (0)

// Custom atomicAdd for double precision is only needed for compute capability < 6.0
// CUDA 12.8 already includes this for newer architectures, so we need to conditionally compile
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

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
// DYNAMIC REORDERING STRATEGY
// =============================================================================

class SFCDynamicReorderingStrategy
{
private:
    // Parameters for the optimization formula
    double reorderTime;        // Time to reorder (equivalent to Rt)
    double postReorderSimTime; // Simulation time right after reordering (equivalent to Rq)
    double lastSimTime;        // Last simulation time
    double updateTime;         // Time to update without reordering (equivalent to Ut, typically 0 for SFC)
    double degradationRate;    // Average performance degradation per iteration (equivalent to dQ)

    int iterationsSinceReorder;  // Counter for iterations since last reorder
    int currentOptimalFrequency; // Current calculated optimal frequency
    int iterationCounter;        // Counter for total iterations
    int consecutiveSkips;        // Count how many times we've skipped reordering

    // Tracking metrics for dynamic calculation
    int metricsWindowSize;
    std::deque<double> reorderTimeHistory;
    std::deque<double> postReorderSimTimeHistory;
    std::deque<double> simulationTimeHistory;
    
    // Early stage tuning parameters
    bool isInitialStageDone;
    int initialSampleSize;
    double initialPerformanceGain;
    
    // Adaptive scaling factors
    double reorderCostScale;
    double performanceGainScale;
    
    // Performance trend tracking
    double movingAverageRatio;
    int stableIterations;
    bool hasPerformancePlateau;

    // Calculate the optimal reordering frequency
    int computeOptimalFrequency(int totalIterations)
    {
        // Bail out if we don't have enough history
        if (simulationTimeHistory.size() < 3) {
            return 15; // Conservative default
        }
        
        // Calculate more accurate degradation rate based on recent history
        double recent = simulationTimeHistory.front();
        double oldest = simulationTimeHistory.back();
        int historySize = simulationTimeHistory.size();
        double measuredDegradation = (recent - oldest) / std::max(1, historySize - 1);
        
        // Only update degradation rate if we have a statistically significant measurement
        if (measuredDegradation > 0.001 && historySize > 5) {
            // Use weighted average to avoid oscillation
            degradationRate = degradationRate * 0.7 + measuredDegradation * 0.3;
        }
        
        // Add a minimum degradation rate to avoid too small values causing always=1 results
        double effectiveDegradationRate = std::max(0.001, degradationRate);
        
        // Scale the reorder time based on our analysis of whether reordering is beneficial
        double effectiveReorderTime = reorderTime * reorderCostScale;
        
        // If reordering provides negligible benefit, make it less frequent
        if (hasPerformancePlateau && stableIterations > 20) {
            effectiveReorderTime *= 1.5;
        }
        
        // Calculate using a modified formula that limits the impact of measurement noise
        double determinant = 1.0 - 2.0 * (updateTime - effectiveReorderTime) / (effectiveDegradationRate + 0.00001);

        // If determinant is negative, use a default value
        if (determinant < 0)
            return 15;

        double optNu = -1.0 + sqrt(determinant);
        
        // Enforce reasonable bounds
        optNu = std::max(5.0, std::min(100.0, optNu));

        // Convert to integer values
        int optimalFreq = static_cast<int>(optNu);
        
        // If reordering appears to be too costly, use larger frequency
        if (reorderTime > 5.0 * postReorderSimTime && optimalFreq < 20) {
            optimalFreq = 20;
        }
        
        // Adaptive frequency based on workload characteristics
        if (movingAverageRatio > 1.1) {
            // Significant benefit from reordering, potentially reduce frequency
            optimalFreq = std::max(5, static_cast<int>(optimalFreq * 0.9));
        } else if (movingAverageRatio < 1.02 && optimalFreq < 50) {
            // Little benefit, increase frequency to reduce overhead
            optimalFreq = std::min(100, static_cast<int>(optimalFreq * 1.1));
        }
        
        return optimalFreq;
    }
    
    // Evaluate if reordering is beneficial overall based on performance history
    bool isReorderingBeneficial() {
        if (simulationTimeHistory.size() < 3) return true;
        
        // Calculate average performance gain from reordering
        double avgPerformanceBeforeReorder = 0.0;
        double avgPerformanceAfterReorder = 0.0;
        int countBefore = 0;
        int countAfter = 0;
        
        // Get samples before last reorder
        for (int i = std::min(3, (int)simulationTimeHistory.size() - 1); i < simulationTimeHistory.size(); i++) {
            avgPerformanceBeforeReorder += simulationTimeHistory[i];
            countBefore++;
        }
        
        // Get samples after last reorder
        for (int i = 0; i < std::min(3, (int)simulationTimeHistory.size()); i++) {
            avgPerformanceAfterReorder += simulationTimeHistory[i];
            countAfter++;
        }
        
        if (countBefore > 0) avgPerformanceBeforeReorder /= countBefore;
        if (countAfter > 0) avgPerformanceAfterReorder /= countAfter;
        
        // If no valid samples, assume reordering is beneficial
        if (countBefore == 0 || countAfter == 0) return true;
        
        // Calculate benefit ratio - if after reordering is faster, it's beneficial
        double benefitRatio = avgPerformanceBeforeReorder / avgPerformanceAfterReorder;
        
        // Update moving average for trend analysis
        movingAverageRatio = movingAverageRatio * 0.8 + benefitRatio * 0.2;
        
        // Check for performance plateau (convergence)
        if (std::abs(benefitRatio - 1.0) < 0.03) {
            stableIterations++;
            if (stableIterations > 10) {
                hasPerformancePlateau = true;
            }
        } else {
            stableIterations = std::max(0, stableIterations - 1);
            if (stableIterations < 5) {
                hasPerformancePlateau = false;
            }
        }
        
        // Update adaptive scaling based on benefit ratio
        if (benefitRatio > 1.05) {
            // Reordering is clearly beneficial
            reorderCostScale = std::max(0.6, reorderCostScale * 0.95);
            performanceGainScale = std::min(1.1, performanceGainScale * 1.05);
            return true;
        } else if (benefitRatio < 0.95) {
            // Reordering may be harmful
            reorderCostScale = std::min(1.5, reorderCostScale * 1.05);
            performanceGainScale = std::max(0.9, performanceGainScale * 0.95);
            return false;
        }
        
        // No clear signal, maintain current strategy
        return benefitRatio >= 1.0;
    }

public:
    SFCDynamicReorderingStrategy(int windowSize = 10)
        : reorderTime(0.0),
          postReorderSimTime(0.0),
          lastSimTime(0.0),
          updateTime(0.0),
          degradationRate(0.0005), // Start with conservative degradation assumption
          iterationsSinceReorder(0),
          currentOptimalFrequency(15), // Start with a conservative default
          iterationCounter(0),
          consecutiveSkips(0),
          metricsWindowSize(windowSize),
          isInitialStageDone(false),
          initialSampleSize(5),
          initialPerformanceGain(0.0),
          reorderCostScale(0.8),    // Initially assume reordering is slightly beneficial
          performanceGainScale(1.0),
          movingAverageRatio(1.0),
          stableIterations(0),
          hasPerformancePlateau(false)
    {
    }

    // Update metrics with new timing information
    void updateMetrics(double newReorderTime, double newSimTime)
    {
        // Update last simulation time and increment iteration counter
        lastSimTime = newSimTime;
        iterationCounter++;
        
        // Update reorder time if reordering was performed
        if (newReorderTime > 0) {
            reorderTime = reorderTime * 0.7 + newReorderTime * 0.3;  // Exponential moving average
            postReorderSimTime = newSimTime;
            iterationsSinceReorder = 0;
        }
        
        // Add new simulation time to history
        simulationTimeHistory.push_front(newSimTime);
        while (simulationTimeHistory.size() > metricsWindowSize)
            simulationTimeHistory.pop_back();
            
        // If this is first time or right after reordering
        if (newReorderTime > 0 && simulationTimeHistory.size() > 1) {
            // Update initial performance gain after collecting enough samples
            if (!isInitialStageDone && iterationCounter >= initialSampleSize*2) {
                double avgBefore = 0.0, avgAfter = 0.0;
                int beforeCount = 0, afterCount = 0;
                
                // Calculate average sim time before reordering
                for (int i = initialSampleSize; i < std::min(initialSampleSize*2, (int)simulationTimeHistory.size()); i++) {
                    avgBefore += simulationTimeHistory[i];
                    beforeCount++;
                }
                
                // Calculate average sim time after reordering
                for (int i = 0; i < initialSampleSize && i < simulationTimeHistory.size(); i++) {
                    avgAfter += simulationTimeHistory[i];
                    afterCount++;
                }
                
                if (beforeCount > 0 && afterCount > 0) {
                    avgBefore /= beforeCount;
                    avgAfter /= afterCount;
                    initialPerformanceGain = (avgBefore - avgAfter) / avgBefore;
                    
                    // Set initial degradation rate based on performance gain
                    if (initialPerformanceGain > 0.01) {
                        degradationRate = initialPerformanceGain / (initialSampleSize * 4.0);
                    }
                    
                    isInitialStageDone = true;
                }
            }
        }
        
        // Calculate degradation rate based on history if we have enough data
        if (simulationTimeHistory.size() > 3 && iterationsSinceReorder > 1) {
            double recent = simulationTimeHistory[0];
            double previous = simulationTimeHistory[1];
            double diff = recent - previous;
            
            // Only update if difference is statistically significant
            if (diff > 0.01 && diff < previous * 0.3) {
                double newRate = diff;
                // Use exponential moving average to smooth the degradation rate
                degradationRate = degradationRate * 0.9 + newRate * 0.1;
            }
        }
        
        iterationsSinceReorder++;
    }

    // Decide whether to reorder particles in this iteration
    bool shouldReorder(double lastSimTime, double predictedReorderTime)
    {
        iterationsSinceReorder++;

        // Update metrics with new timing information
        updateMetrics(0.0, lastSimTime);
        
        // During initial sampling, use fixed frequency
        if (!isInitialStageDone) {
            bool shouldReorder = (iterationsSinceReorder >= initialSampleSize);
            if (shouldReorder) iterationsSinceReorder = 0;
            return shouldReorder;
        }
        
        // Recalculate optimal frequency periodically
        if (iterationsSinceReorder % 5 == 0) {
            currentOptimalFrequency = computeOptimalFrequency(1000);
            currentOptimalFrequency = std::max(5, std::min(150, currentOptimalFrequency));
        }
        
        // Decide whether to reorder - balance between frequency and cost-benefit
        bool shouldReorder = false;
        
        // If enough iterations have passed since last reordering
        if (iterationsSinceReorder >= currentOptimalFrequency) {
            // Check if reordering is generally beneficial
            if (isReorderingBeneficial()) {
                shouldReorder = true;
                consecutiveSkips = 0;
            } else {
                // Even if not beneficial, occasionally reorder to reassess
                consecutiveSkips++;
                if (consecutiveSkips >= 3) {
                    shouldReorder = true;
                    consecutiveSkips = 0;
                }
            }
        }
        
        // In some cases, reorder early if performance is degrading fast
        if (!shouldReorder && iterationsSinceReorder > currentOptimalFrequency/2) {
            // Check if performance has degraded significantly
            if (simulationTimeHistory.size() >= 3) {
                double recent = simulationTimeHistory[0];
                double previous = simulationTimeHistory[std::min(2, (int)simulationTimeHistory.size()-1)];
                if (recent > previous * 1.3) { // 30% degradation
                    shouldReorder = true;
                }
            }
        }
        
        // Reset counter if reordering
        if (shouldReorder) {
            iterationsSinceReorder = 0;
        }
        
        return shouldReorder;
    }

    // Public method for updating metrics with just sort time
    void updateMetrics(double sortTime)
    {
        // Call the internal method with proper defaults
        updateMetrics(sortTime, lastSimTime);
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

    // Get the current degradation rate
    double getDegradationRate() const
    {
        return degradationRate;
    }
    
    // Get performance gain estimate
    double getPerformanceGain() const
    {
        return initialPerformanceGain;
    }
    
    // Get current performance ratio trend
    double getPerformanceRatio() const
    {
        return movingAverageRatio;
    }
    
    // Check if performance has plateaued
    bool isPerformancePlateau() const
    {
        return hasPerformancePlateau;
    }

    // Reset the strategy
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

// =============================================================================
// KERNEL FUNCTIONS
// =============================================================================

/**
 * @brief Reset octree nodes and mutex
 */
__global__ void ResetKernel(Node *nodes, int *mutex, int nNodes, int nBodies)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nNodes)
    {
        // Reset node data
        nodes[i].isLeaf = (i < nBodies);
        nodes[i].firstChildIndex = -1;
        nodes[i].bodyIndex = (i < nBodies) ? i : -1;
        nodes[i].bodyCount = (i < nBodies) ? 1 : 0;
        nodes[i].position = Vector(0, 0, 0);
        nodes[i].mass = 0.0;
        nodes[i].radius = 0.0;
        nodes[i].min = Vector(0, 0, 0);
        nodes[i].max = Vector(0, 0, 0);

        // Reset mutex
        if (i < nNodes)
            mutex[i] = 0;
    }
}

/**
 * @brief Compute bounding box for all bodies with SFC ordering support
 */
__global__ void ComputeBoundingBoxKernel(Node *nodes, Body *bodies, int *orderedIndices, bool useSFC, int *mutex, int nBodies)
{
    // Use shared memory for reduction
    extern __shared__ double sharedMem[];
    double *minX = &sharedMem[0];
    double *minY = &sharedMem[blockDim.x];
    double *minZ = &sharedMem[2 * blockDim.x];
    double *maxX = &sharedMem[3 * blockDim.x];
    double *maxY = &sharedMem[4 * blockDim.x];
    double *maxZ = &sharedMem[5 * blockDim.x];

    // Thread ID
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;

    // Get real body index when using SFC
    int realBodyIndex = (useSFC && orderedIndices != nullptr && i < nBodies) ? orderedIndices[i] : i;

    // Initialize shared memory
    minX[tx] = (i < nBodies) ? bodies[realBodyIndex].position.x : DBL_MAX;
    minY[tx] = (i < nBodies) ? bodies[realBodyIndex].position.y : DBL_MAX;
    minZ[tx] = (i < nBodies) ? bodies[realBodyIndex].position.z : DBL_MAX;
    maxX[tx] = (i < nBodies) ? bodies[realBodyIndex].position.x : -DBL_MAX;
    maxY[tx] = (i < nBodies) ? bodies[realBodyIndex].position.y : -DBL_MAX;
    maxZ[tx] = (i < nBodies) ? bodies[realBodyIndex].position.z : -DBL_MAX;

    // Make sure all threads have loaded shared memory
    __syncthreads();

    // Reduction in shared memory
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

    // Only thread 0 updates the bounding box
    if (tx == 0)
    {
        // Insert all bodies in the root node (index 0)
        nodes[0].isLeaf = false;
        nodes[0].bodyCount = nBodies;
        nodes[0].min = Vector(minX[0], minY[0], minZ[0]);
        nodes[0].max = Vector(maxX[0], maxY[0], maxZ[0]);

        // Add padding to the bounding box
        Vector padding = (nodes[0].max - nodes[0].min) * 0.01;
        nodes[0].min = nodes[0].min - padding;
        nodes[0].max = nodes[0].max + padding;

        // Calculate node radius (half the maximum dimension)
        Vector dimensions = nodes[0].max - nodes[0].min;
        nodes[0].radius = max(max(dimensions.x, dimensions.y), dimensions.z) * 0.5;

        // Set node center position
        nodes[0].position = (nodes[0].min + nodes[0].max) * 0.5;
    }
}

/**
 * @brief Insert a body into the octree with SFC ordering support
 * Versión simplificada que solo maneja el ordenamiento de partículas
 */
__device__ bool InsertBody(Node *nodes, Body *bodies, int bodyIdx, int nodeIdx, int nNodes, int leafLimit)
{
    // Recursively insert the body into the tree
    Node &node = nodes[nodeIdx];

    // If this is an empty leaf node, store the body here
    if (node.isLeaf && node.bodyCount == 0)
    {
        node.bodyIndex = bodyIdx;
        node.bodyCount = 1;
        node.position = bodies[bodyIdx].position;
        node.mass = bodies[bodyIdx].mass;
        return true;
    }

    // If this is a leaf with a body already, we need to subdivide
    if (node.isLeaf && node.bodyCount > 0)
    {
        // Only subdivide if we have nodes available
        if (nodeIdx >= leafLimit)
            return false;

        // Create 8 children
        node.isLeaf = false;
        int firstChildIdx = atomicAdd(&nodes[nNodes-1].bodyCount, 8); // Use the last node's bodyCount as a counter
        
        // Check if we have enough nodes
        if (firstChildIdx + 7 >= nNodes)
            return false;
            
        node.firstChildIndex = firstChildIdx;

        // Move the existing body to the appropriate child
        int existingBodyIdx = node.bodyIndex;
        node.bodyIndex = -1; // No longer a leaf with a body

        // Calculate center of node
        Vector center = node.position;
        
        // Optimize child index calculation using bit operations
        Vector pos = bodies[existingBodyIdx].position;
        int childIdx = ((pos.x >= center.x) ? 1 : 0) |
                      ((pos.y >= center.y) ? 2 : 0) |
                      ((pos.z >= center.z) ? 4 : 0);
        
        // Create the child nodes with appropriate bounds
        for (int i = 0; i < 8; i++)
        {
            Node &child = nodes[firstChildIdx + i];
            child.isLeaf = true;
            child.firstChildIndex = -1;
            child.bodyIndex = -1;
            child.bodyCount = 0;
            child.mass = 0.0;
            
            // Calculate min/max for this child - faster calculation using bitwise operations
            Vector min = node.min;
            Vector max = node.max;
            
            // Adjust bounds based on octant using bit operations
            if (i & 1) min.x = center.x; else max.x = center.x;
            if (i & 2) min.y = center.y; else max.y = center.y;
            if (i & 4) min.z = center.z; else max.z = center.z;
            
            child.min = min;
            child.max = max;
            child.position = (min + max) * 0.5;
            
            // Use fmaxf for CUDA compatibility (double version)
            double dx = max.x - min.x;
            double dy = max.y - min.y;
            double dz = max.z - min.z;
            double maxDim = dx > dy ? dx : dy;
            maxDim = maxDim > dz ? maxDim : dz;
            child.radius = maxDim * 0.5;
        }
        
        // Insert the existing body into the appropriate child
        InsertBody(nodes, bodies, existingBodyIdx, firstChildIdx + childIdx, nNodes, leafLimit);
    }
    
    // Now determine which child the new body belongs to
    Vector pos = bodies[bodyIdx].position;
    Vector center = node.position;
    
    // Optimize child index calculation using bit operations
    int childIdx = ((pos.x >= center.x) ? 1 : 0) |
                  ((pos.y >= center.y) ? 2 : 0) |
                  ((pos.z >= center.z) ? 4 : 0);
    
    // Insert body into the appropriate child
    if (node.firstChildIndex >= 0 && node.firstChildIndex + childIdx < nNodes)
    {
        InsertBody(nodes, bodies, bodyIdx, node.firstChildIndex + childIdx, nNodes, leafLimit);
    }
    
    // Update node's center of mass and total mass
    double totalMass = node.mass + bodies[bodyIdx].mass;
    Vector weightedPos = node.position * node.mass + bodies[bodyIdx].position * bodies[bodyIdx].mass;
    
    if (totalMass > 0.0)
    {
        node.position = weightedPos * (1.0 / totalMass);
        node.mass = totalMass;
    }
    
    // Increment body count
    node.bodyCount++;
    
    return true;
}

/**
 * @brief Construct octree from bodies with SFC ordering support
 * Versión simplificada que solo maneja el ordenamiento de partículas
 */
__global__ void ConstructOctTreeKernel(Node *nodes, Body *bodies, Body *bodyBuffer, int *orderedIndices, bool useSFC,
                                      int rootIdx, int nNodes, int nBodies, int leafLimit)
{
    // Use shared memory to track total mass and center of mass
    extern __shared__ double sharedMem[];
    double *totalMass = &sharedMem[0];
    double3 *centerMass = (double3*)(totalMass + blockDim.x);
    
    // Get thread ID
    int i = threadIdx.x;
    
    // Each thread processes multiple bodies
    for (int bodyIdx = i; bodyIdx < nBodies; bodyIdx += blockDim.x)
    {
        // Get real body index when using SFC
        int realBodyIdx = (useSFC && orderedIndices != nullptr) ? orderedIndices[bodyIdx] : bodyIdx;
        
        // Copy body to buffer (optional, helps with memory access patterns)
        bodyBuffer[bodyIdx] = bodies[realBodyIdx];
        
        // Insert body into the octree
        InsertBody(nodes, bodies, realBodyIdx, rootIdx, nNodes, leafLimit);
    }
}

/**
 * @brief Compute forces using Barnes-Hut algorithm with SFC ordering support
 * Improved version that leverages spatial locality from SFC ordering
 */
__global__ void ComputeForceKernel(Node *nodes, Body *bodies, int *orderedIndices, bool useSFC,
                                  int nNodes, int nBodies, int leafLimit, double theta)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Get real body index when using SFC
    int realBodyIdx = (useSFC && orderedIndices != nullptr && i < nBodies) ? orderedIndices[i] : i;
    
    if (realBodyIdx >= nBodies || !bodies[realBodyIdx].isDynamic)
        return;

    // Use shared memory to cache node data for reuse across threads in the same warp
    __shared__ Node nodeCache[32];  // Cache for the most commonly accessed nodes
    __shared__ int cacheIndices[32];
    __shared__ int cacheSize;
    
    if (threadIdx.x == 0) {
        cacheSize = 0;
        // Pre-load root node
        nodeCache[0] = nodes[0];
        cacheIndices[0] = 0;
        cacheSize = 1;
    }
    __syncthreads();
    
    Vector acc(0.0, 0.0, 0.0);
    Vector pos = bodies[realBodyIdx].position;
    double bodyMass = bodies[realBodyIdx].mass;
    
    // Use shared memory for local stack
    extern __shared__ int sharedData[];
    const int MAX_STACK_SIZE = 32;
    int* localStack = &sharedData[threadIdx.x * MAX_STACK_SIZE];
    int stackSize = 0;
    
    // Load root node into stack if we have space
    if (stackSize < MAX_STACK_SIZE) {
        localStack[stackSize++] = 0;
    }
    
    // Process nodes with adaptive theta based on position in SFC
    // This improves accuracy where needed while allowing faster approximation elsewhere
    double adaptiveTheta = theta;
    if (useSFC) {
        // Adjust theta based on position in SFC for better performance
        // Bodies closer in SFC are likely to be spatially closer too
        int sectionSize = nBodies / 4;
        int section = i / sectionSize;
        
        // Larger theta (less accurate) for most bodies, smaller theta (more accurate) for important regions
        if (section == 0 || section == 3) {
            // More precise for the first and last parts of SFC (where there's higher concentration)
            adaptiveTheta = theta * 0.85;
        } else {
            // Less precise for the middle parts
            adaptiveTheta = theta * 1.15;
        }
        
        // Clamp theta to reasonable values
        adaptiveTheta = max(0.3, min(0.9, adaptiveTheta));
    }
    
    while (stackSize > 0)
    {
        int nodeIdx = localStack[--stackSize];
        
        // Try to find node in cache
        Node node;
        bool foundInCache = false;
        
        #pragma unroll
        for (int c = 0; c < 32; c++) {
            if (c < cacheSize && cacheIndices[c] == nodeIdx) {
                node = nodeCache[c];
                foundInCache = true;
                break;
            }
        }
        
        if (!foundInCache) {
            node = nodes[nodeIdx];
            
            // Try to add to cache if there's space
            if (cacheSize < 32) {
                int cacheIdx = atomicAdd(&cacheSize, 1);
                if (cacheIdx < 32) {
                    nodeCache[cacheIdx] = node;
                    cacheIndices[cacheIdx] = nodeIdx;
                }
            }
        }
        
        Vector nodePos = node.position;
        Vector dir = nodePos - pos;
        double distSqr = dir.lengthSquared();
        
        // Skip self-interaction
        if (distSqr < 1e-10) continue;
        
        // Check if this is an internal node
        if (!node.isLeaf)
        {
            double nodeSizeSqr = node.radius * node.radius * 4.0;
            
            // Apply Barnes-Hut criterion with adaptive theta
            if (nodeSizeSqr < distSqr * adaptiveTheta * adaptiveTheta)
            {
                // Node is far enough to be approximated
                double dist = sqrt(distSqr);
                double invDist3 = 1.0 / (dist * distSqr);
                acc = acc + dir * (node.mass * invDist3);
            }
            else
            {
                // Need to explore node's children
                int firstChildIdx = node.firstChildIndex;
                
                // Add children to the stack
                if (useSFC) {
                    // SFC-optimized traversal order
                    static const int childOrder[8] = {0, 1, 3, 2, 6, 7, 5, 4};
                    
                    // SIMD-friendly unrolled loop for better performance
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
                else {
                    // Standard traversal
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
            // Leaf node: directly calculate forces from all contained bodies
            double dist = sqrt(distSqr);
            double invDist3 = 1.0 / (dist * distSqr);
            acc = acc + dir * (node.mass * invDist3);
        }
    }
    
    // Update acceleration
    bodies[realBodyIdx].acceleration = acc;
    
    // Update velocity
    bodies[realBodyIdx].velocity = bodies[realBodyIdx].velocity + acc * DT;
    
    // Update position
    bodies[realBodyIdx].position = bodies[realBodyIdx].position + bodies[realBodyIdx].velocity * DT;
}

// Kernel to calculate energy values
__global__ void CalculateEnergiesKernel(Body *bodies, int nBodies, double *d_potentialEnergy, double *d_kineticEnergy)
{
    // Shared memory to store partial energy sums for each thread
    extern __shared__ double sharedEnergy[];
    double *sharedPotential = sharedEnergy;
    double *sharedKinetic = &sharedEnergy[blockDim.x];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;

    // Initialize shared memory
    sharedPotential[tx] = 0.0;
    sharedKinetic[tx] = 0.0;

    // Only compute energies if this thread corresponds to a valid body
    if (i < nBodies)
    {
        // Calculate kinetic energy for this body
        if (bodies[i].isDynamic)
        {
            double vSquared = bodies[i].velocity.lengthSquared();
            sharedKinetic[tx] = 0.5 * bodies[i].mass * vSquared;
        }

        // Calculate potential energy contribution for this body (direct method)
        // We only need to compute interactions with bodies that have higher indices
        // to avoid double-counting
        for (int j = i + 1; j < nBodies; j++)
        {
            // Vector from body i to body j
            Vector r = bodies[j].position - bodies[i].position;
            
            // Distance calculation with softening
            double distSqr = r.lengthSquared() + (E * E);
            double dist = sqrt(distSqr);
            
            // Skip if bodies are too close (collision)
            if (dist < COLLISION_TH)
                continue;
            
            // Gravitational potential energy: -G * m1 * m2 / r
            sharedPotential[tx] -= GRAVITY * bodies[i].mass * bodies[j].mass / dist;
        }
    }

    // Synchronize threads in the block
    __syncthreads();

    // Reduce within block using shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tx < s)
        {
            sharedPotential[tx] += sharedPotential[tx + s];
            sharedKinetic[tx] += sharedKinetic[tx + s];
        }
        __syncthreads();
    }

    // Let the first thread of each block write its result to global memory
    if (tx == 0)
    {
        atomicAdd(d_potentialEnergy, sharedPotential[0]);
        atomicAdd(d_kineticEnergy, sharedKinetic[0]);
    }
}

// Kernel para extraer posiciones
__global__ void extractPositionsKernel(Body* bodies, Vector* positions, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        positions[idx] = bodies[idx].position;
    }
}

// Kernel para calcular las claves SFC
__global__ void calculateSFCKeysKernel(Vector *positions, uint64_t *keys, int nBodies,
                                     Vector minBound, Vector maxBound, bool isHilbert)
{
    // Usar memoria compartida para tablas lookup
    __shared__ uint8_t hilbertMap[8][8];
    __shared__ uint8_t nextState[8][8];
    
    // Solo el primer hilo por bloque inicializa las tablas
    if (threadIdx.x == 0) {
        // Inicializar Hilbert Map
        hilbertMap[0][0] = 0; hilbertMap[0][1] = 1; hilbertMap[0][2] = 3; hilbertMap[0][3] = 2;
        hilbertMap[0][4] = 7; hilbertMap[0][5] = 6; hilbertMap[0][6] = 4; hilbertMap[0][7] = 5;
        
        hilbertMap[1][0] = 4; hilbertMap[1][1] = 5; hilbertMap[1][2] = 7; hilbertMap[1][3] = 6;
        hilbertMap[1][4] = 0; hilbertMap[1][5] = 1; hilbertMap[1][6] = 3; hilbertMap[1][7] = 2;
        
        hilbertMap[2][0] = 6; hilbertMap[2][1] = 7; hilbertMap[2][2] = 5; hilbertMap[2][3] = 4;
        hilbertMap[2][4] = 2; hilbertMap[2][5] = 3; hilbertMap[2][6] = 1; hilbertMap[2][7] = 0;
        
        hilbertMap[3][0] = 2; hilbertMap[3][1] = 3; hilbertMap[3][2] = 1; hilbertMap[3][3] = 0;
        hilbertMap[3][4] = 6; hilbertMap[3][5] = 7; hilbertMap[3][6] = 5; hilbertMap[3][7] = 4;
        
        hilbertMap[4][0] = 0; hilbertMap[4][1] = 7; hilbertMap[4][2] = 1; hilbertMap[4][3] = 6;
        hilbertMap[4][4] = 3; hilbertMap[4][5] = 4; hilbertMap[4][6] = 2; hilbertMap[4][7] = 5;
        
        hilbertMap[5][0] = 6; hilbertMap[5][1] = 1; hilbertMap[5][2] = 7; hilbertMap[5][3] = 0;
        hilbertMap[5][4] = 5; hilbertMap[5][5] = 2; hilbertMap[5][6] = 4; hilbertMap[5][7] = 3;
        
        hilbertMap[6][0] = 2; hilbertMap[6][1] = 5; hilbertMap[6][2] = 3; hilbertMap[6][3] = 4;
        hilbertMap[6][4] = 1; hilbertMap[6][5] = 6; hilbertMap[6][6] = 0; hilbertMap[6][7] = 7;
        
        hilbertMap[7][0] = 4; hilbertMap[7][1] = 3; hilbertMap[7][2] = 5; hilbertMap[7][3] = 2;
        hilbertMap[7][4] = 7; hilbertMap[7][5] = 0; hilbertMap[7][6] = 6; hilbertMap[7][7] = 1;
        
        // Inicializar Next State
        nextState[0][0] = 0; nextState[0][1] = 1; nextState[0][2] = 3; nextState[0][3] = 2;
        nextState[0][4] = 7; nextState[0][5] = 6; nextState[0][6] = 4; nextState[0][7] = 5;
        
        nextState[1][0] = 1; nextState[1][1] = 0; nextState[1][2] = 2; nextState[1][3] = 3;
        nextState[1][4] = 4; nextState[1][5] = 5; nextState[1][6] = 7; nextState[1][7] = 6;
        
        nextState[2][0] = 2; nextState[2][1] = 3; nextState[2][2] = 1; nextState[2][3] = 0;
        nextState[2][4] = 5; nextState[2][5] = 4; nextState[2][6] = 6; nextState[2][7] = 7;
        
        nextState[3][0] = 3; nextState[3][1] = 2; nextState[3][2] = 0; nextState[3][3] = 1;
        nextState[3][4] = 6; nextState[3][5] = 7; nextState[3][6] = 5; nextState[3][7] = 4;
        
        nextState[4][0] = 4; nextState[4][1] = 5; nextState[4][2] = 7; nextState[4][3] = 6;
        nextState[4][4] = 0; nextState[4][5] = 1; nextState[4][6] = 3; nextState[4][7] = 2;
        
        nextState[5][0] = 5; nextState[5][1] = 4; nextState[5][2] = 6; nextState[5][3] = 7;
        nextState[5][4] = 1; nextState[5][5] = 0; nextState[5][6] = 2; nextState[5][7] = 3;
        
        nextState[6][0] = 6; nextState[6][1] = 7; nextState[6][2] = 5; nextState[6][3] = 4;
        nextState[6][4] = 2; nextState[6][5] = 3; nextState[6][6] = 1; nextState[6][7] = 0;
        
        nextState[7][0] = 7; nextState[7][1] = 6; nextState[7][2] = 4; nextState[7][3] = 5;
        nextState[7][4] = 3; nextState[7][5] = 2; nextState[7][6] = 0; nextState[7][7] = 1;
    }
    
    // Asegurarse de que las tablas estén inicializadas
    __syncthreads();
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nBodies) return;
    
    Vector pos = positions[i];
    
    // Calcular factores de normalización directamente para reducir cálculos
    double invRangeX = 1.0 / (maxBound.x - minBound.x);
    double invRangeY = 1.0 / (maxBound.y - minBound.y);
    double invRangeZ = 1.0 / (maxBound.z - minBound.z);
    
    // Normalize position to [0,1] range with precomputed factors
    double normalizedX = (pos.x - minBound.x) * invRangeX;
    double normalizedY = (pos.y - minBound.y) * invRangeY;
    double normalizedZ = (pos.z - minBound.z) * invRangeZ;
    
    // Clamp to [0,1]
    normalizedX = max(0.0, min(0.999999, normalizedX));
    normalizedY = max(0.0, min(0.999999, normalizedY));
    normalizedZ = max(0.0, min(0.999999, normalizedZ));
    
    // Constant for conversion to integer - reduce shifts
    const uint32_t MAX_COORD = 0xFFFF;
    
    // Use 16 bits for faster calculation
    uint32_t x = static_cast<uint32_t>(normalizedX * MAX_COORD);
    uint32_t y = static_cast<uint32_t>(normalizedY * MAX_COORD);
    uint32_t z = static_cast<uint32_t>(normalizedZ * MAX_COORD);
    
    uint64_t key = 0;
    
    // Calculate Morton key (simpler, faster, but less cache-coherent)
    if (!isHilbert) {
        // Optimized bit-spreading using parallel operations
        // Use mask operations instead of loops for better performance
        // Unroll the calculations
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
    // Simplified Hilbert curve calculation (better locality but more compute)
    else {
        uint8_t state = 0;
        
        // Versión desenrollada para los primeros 4 bits para mejorar rendimiento
        // Bits 15-12
        {
            uint8_t octant = 0;
            if (x & 0x8000) octant |= 1;
            if (y & 0x8000) octant |= 2;
            if (z & 0x8000) octant |= 4;
            
            uint8_t position = hilbertMap[state][octant];
            key = (key << 3) | position;
            state = nextState[state][octant];
        }
        
        {
            uint8_t octant = 0;
            if (x & 0x4000) octant |= 1;
            if (y & 0x4000) octant |= 2;
            if (z & 0x4000) octant |= 4;
            
            uint8_t position = hilbertMap[state][octant];
            key = (key << 3) | position;
            state = nextState[state][octant];
        }
        
        {
            uint8_t octant = 0;
            if (x & 0x2000) octant |= 1;
            if (y & 0x2000) octant |= 2;
            if (z & 0x2000) octant |= 4;
            
            uint8_t position = hilbertMap[state][octant];
            key = (key << 3) | position;
            state = nextState[state][octant];
        }
        
        {
            uint8_t octant = 0;
            if (x & 0x1000) octant |= 1;
            if (y & 0x1000) octant |= 2;
            if (z & 0x1000) octant |= 4;
            
            uint8_t position = hilbertMap[state][octant];
            key = (key << 3) | position;
            state = nextState[state][octant];
        }
        
        // Bits 11-0
        // Procesar en bloques para mejorar rendimiento
        for (int j = 0; j < 3; j++) {
            for (int i = 3; i >= 0; i--) {
                uint8_t octant = 0;
                uint32_t mask = 1 << (i + j*4);
                if (x & mask) octant |= 1;
                if (y & mask) octant |= 2;
                if (z & mask) octant |= 4;
                
                uint8_t position = hilbertMap[state][octant];
                key = (key << 3) | position;
                state = nextState[state][octant];
            }
        }
    }
    
    keys[i] = key;
}

// =============================================================================
// SIMULATION CLASS
// =============================================================================

class SFCBarnesHutGPU
{
private:
    Body *h_bodies = nullptr;        // Host bodies
    Body *d_bodies = nullptr;        // Device bodies
    Body *d_bodiesBuffer = nullptr;  // Device buffer for bodies during tree construction
    Node *h_nodes = nullptr;         // Host nodes
    Node *d_nodes = nullptr;         // Device nodes
    int *d_mutex = nullptr;          // Device mutex for tree construction
    int nBodies;                     // Number of bodies
    int nNodes;                      // Maximum number of octree nodes
    int leafLimit;                   // Index limit for internal nodes
    bool useSFC;                     // Whether to use SFC ordering
    sfc::BodySorter *sorter;         // SFC sorter
    int *d_orderedIndices;           // Device ordered indices
    sfc::CurveType curveType;        // Type of SFC curve
    int reorderFrequency;            // How often to reorder bodies
    int iterationCounter;            // Counter for reordering
    Vector minBound;                 // Minimum bounds for SFC calculation
    Vector maxBound;                 // Maximum bounds for SFC calculation
    SimulationMetrics metrics;       // Performance metrics
    
    // Add dynamic reordering members
    bool useDynamicReordering;
    SFCDynamicReorderingStrategy reorderingStrategy;

    double potentialEnergy;
    double kineticEnergy;
    double totalEnergyAvg;
    double potentialEnergyAvg;
    double kineticEnergyAvg;
    
    // Performance stats
    float minTimeMs;                 // Minimum time per step
    float maxTimeMs;                 // Maximum time per step
    
    // Device memory for energy calculations
    double *d_potentialEnergy;
    double *d_kineticEnergy;
    double *h_potentialEnergy;
    double *h_kineticEnergy;

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

    void updateBoundingBox()
    {
        // Simplificamos esta función volviendo a la implementación original pero mejorando su eficiencia
        
        // Solo copiar si estamos usando SFC, si no es necesario
        if (useSFC) {
            // Solo necesitamos 2*sizeof(Vector) bytes por partícula (solo posiciones)
            Vector* h_positions = new Vector[nBodies];
            
            // Copiar solo las posiciones desde la GPU a la CPU
            for (int i = 0; i < nBodies; i++) {
                size_t offset = i * sizeof(Body) + offsetof(Body, position);
                cudaMemcpy(&h_positions[i], (char*)d_bodies + offset, sizeof(Vector), cudaMemcpyDeviceToHost);
            }
            
            // Calcular los límites
            minBound = Vector(INFINITY, INFINITY, INFINITY);
            maxBound = Vector(-INFINITY, -INFINITY, -INFINITY);
            
            // Calcular límites en paralelo si tenemos muchas partículas
            #pragma omp parallel for reduction(min:minBound.x,minBound.y,minBound.z) reduction(max:maxBound.x,maxBound.y,maxBound.z) if(nBodies > 50000)
            for (int i = 0; i < nBodies; i++) {
                Vector pos = h_positions[i];
                
                // Update bounds with atomic operations when in parallel
                minBound.x = std::min(minBound.x, pos.x);
                minBound.y = std::min(minBound.y, pos.y);
                minBound.z = std::min(minBound.z, pos.z);
                
                maxBound.x = std::max(maxBound.x, pos.x);
                maxBound.y = std::max(maxBound.y, pos.y);
                maxBound.z = std::max(maxBound.z, pos.z);
            }
            
            // Liberar memoria
            delete[] h_positions;
            
            // Add padding to avoid edge issues
            double padding = std::max(1.0e10, (maxBound.x - minBound.x) * 0.01);
            minBound.x -= padding;
            minBound.y -= padding;
            minBound.z -= padding;
            maxBound.x += padding;
            maxBound.y += padding;
            maxBound.z += padding;
        }
    }

    // Método para obtener los límites desde el nodo raíz del octree
    void updateBoundsFromRoot() 
    {
        // Si los nodos del octree ya están inicializados, podemos obtener los límites del nodo raíz
        if (d_nodes) {
            // Copiar solo el nodo raíz
            Node rootNode;
            cudaMemcpy(&rootNode, d_nodes, sizeof(Node), cudaMemcpyDeviceToHost);
            
            // Si el nodo raíz tiene límites válidos (no es el primer frame), usarlos
            if (rootNode.bodyCount > 0 && 
                !std::isinf(rootNode.min.x) && !std::isinf(rootNode.max.x)) {
                
                minBound = rootNode.min;
                maxBound = rootNode.max;
                
                // Añadir padding adicional
                Vector padding = (maxBound - minBound) * 0.05;
                minBound = minBound - padding;
                maxBound = maxBound + padding;
                
                return;
            }
        }
        
        // Si no tenemos un nodo raíz válido, usar el método estándar
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
        
        // Actualizar límites desde el octree si es posible
        updateBoundsFromRoot();
        
        // Instead of copying all bodies to host, just pass device pointers to sorter
        // This eliminates a major bottleneck in the SFC implementation
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
        // Verify initialization
        if (d_bodies == nullptr) {
            std::cerr << "Error: Device bodies not initialized in calculateEnergies" << std::endl;
            return;
        }
        
        // Skip if there are no bodies to calculate energy for
        if (nBodies <= 0) {
            potentialEnergy = 0.0;
            kineticEnergy = 0.0;
            return;
        }
        
        // Synchronize device before measuring time
        cudaDeviceSynchronize();
        
        // Measure execution time
        CudaTimer timer(metrics.energyCalculationTimeMs);
        
        // Reset energy values to zero
        CHECK_CUDA_ERROR(cudaMemset(d_potentialEnergy, 0, sizeof(double)));
        CHECK_CUDA_ERROR(cudaMemset(d_kineticEnergy, 0, sizeof(double)));
        
        // Configure block size
        int blockSize = g_blockSize;
        
        // Ensure blockSize is not larger than the number of bodies
        // This is important for small datasets to avoid wasting threads
        blockSize = std::min(blockSize, nBodies);
        
        // Ensure blockSize is a multiple of 32 (warp size)
        blockSize = (blockSize / 32) * 32;
        if (blockSize < 32) blockSize = 32;
        if (blockSize > 1024) blockSize = 1024;
        
        // Calculate grid size based on number of bodies
        int gridSize = (nBodies + blockSize - 1) / blockSize;
        if (gridSize < 1) gridSize = 1;
        
        // Calculate required shared memory size
        size_t sharedMemSize = 2 * blockSize * sizeof(double); // For potential and kinetic energy
        
        // Check if shared memory size exceeds device limits
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, device);
        
        if (sharedMemSize > props.sharedMemPerBlock) {
            // If shared memory requirements exceed device limits, use a smaller block size
            blockSize = (props.sharedMemPerBlock / (2 * sizeof(double))) & ~0x1F; // Round down to multiple of 32
            if (blockSize < 32) {
                std::cerr << "Error: Insufficient shared memory for energy calculation" << std::endl;
                return;
            }
            
            // Recalculate grid size and shared memory size
            gridSize = (nBodies + blockSize - 1) / blockSize;
            sharedMemSize = 2 * blockSize * sizeof(double);
        }
        
        // Launch kernel with error checking
        CUDA_KERNEL_CALL(CalculateEnergiesKernel, gridSize, blockSize, sharedMemSize, 0, 
                         d_bodies, nBodies, d_potentialEnergy, d_kineticEnergy);
        
        // Copy results back to host
        CHECK_CUDA_ERROR(cudaMemcpy(h_potentialEnergy, d_potentialEnergy, sizeof(double), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(h_kineticEnergy, d_kineticEnergy, sizeof(double), cudaMemcpyDeviceToHost));
        
        // Update class members
        potentialEnergy = *h_potentialEnergy;
        kineticEnergy = *h_kineticEnergy;
    }
    
    void initializeEnergyData()
    {
        // Allocate host memory for energy values
        h_potentialEnergy = new double[1];
        h_kineticEnergy = new double[1];
        
        // Initialize host memory
        *h_potentialEnergy = 0.0;
        *h_kineticEnergy = 0.0;
        
        // Allocate device memory for energy calculations
        CHECK_CUDA_ERROR(cudaMalloc(&d_potentialEnergy, sizeof(double)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_kineticEnergy, sizeof(double)));
        
        // Initialize device memory to zero
        CHECK_CUDA_ERROR(cudaMemset(d_potentialEnergy, 0, sizeof(double)));
        CHECK_CUDA_ERROR(cudaMemset(d_kineticEnergy, 0, sizeof(double)));
    }
    
    void cleanupEnergyData()
    {
        // Free device memory
        if (d_potentialEnergy) {
            cudaFree(d_potentialEnergy);
            d_potentialEnergy = nullptr;
        }
        
        if (d_kineticEnergy) {
            cudaFree(d_kineticEnergy);
            d_kineticEnergy = nullptr;
        }
        
        // Free host memory
        if (h_potentialEnergy) {
            delete[] h_potentialEnergy;
            h_potentialEnergy = nullptr;
        }
        
        if (h_kineticEnergy) {
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
          reorderingStrategy(10), // Start with window size of 10
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
        
        // Get GPU device information
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

        // Allocate host memory
        h_bodies = new Body[nBodies];
        h_nodes = new Node[nNodes];

        // Allocate device memory
        CHECK_CUDA_ERROR(cudaMalloc(&d_bodies, nBodies * sizeof(Body)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_bodiesBuffer, nBodies * sizeof(Body)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_nodes, nNodes * sizeof(Node)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_mutex, nNodes * sizeof(int)));

        // Initialize sorter if using SFC
        if (useSFC)
        {
            sorter = new sfc::BodySorter(numBodies, curveType);
        }

        // Initialize bodies
        initializeDistribution(bodyDist, massDist, seed);

        // Copy to device
        CHECK_CUDA_ERROR(cudaMemcpy(d_bodies, h_bodies, nBodies * sizeof(Body), cudaMemcpyHostToDevice));

        // Initialize bounds to invalid values to force update
        minBound = Vector(INFINITY, INFINITY, INFINITY);
        maxBound = Vector(-INFINITY, -INFINITY, -INFINITY);

        // Initialize energy calculation data
        initializeEnergyData();
    }

    ~SFCBarnesHutGPU()
    {
        // Free resources
        delete[] h_bodies;
        delete[] h_nodes;
        if (sorter) delete sorter;
        
        if (d_bodies) CHECK_CUDA_ERROR(cudaFree(d_bodies));
        if (d_bodiesBuffer) CHECK_CUDA_ERROR(cudaFree(d_bodiesBuffer));
        if (d_nodes) CHECK_CUDA_ERROR(cudaFree(d_nodes));
        if (d_mutex) CHECK_CUDA_ERROR(cudaFree(d_mutex));

        // Clean up energy calculation resources
        cleanupEnergyData();
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
        
        // Calculate shared memory size (6 arrays of doubles, each of size blockSize)
        size_t sharedMemSize = 6 * blockSize * sizeof(double);
        ComputeBoundingBoxKernel<<<gridSize, blockSize, sharedMemSize>>>(d_nodes, d_bodies, d_orderedIndices, useSFC, d_mutex, nBodies);
        CHECK_LAST_CUDA_ERROR();
    }

    void constructOctree()
    {
        CudaTimer timer(metrics.buildTimeMs);
        
        int blockSize = g_blockSize;
        
        // Calculate shared memory size for the octree kernel
        size_t sharedMemSize = blockSize * sizeof(double) +  // totalMass array
                              blockSize * sizeof(double3);  // centerMass array
                              
        ConstructOctTreeKernel<<<1, blockSize, sharedMemSize>>>(d_nodes, d_bodies, d_bodiesBuffer, d_orderedIndices, useSFC, 0, nNodes, nBodies, leafLimit);
        CHECK_LAST_CUDA_ERROR();
    }

    void computeForces()
    {
        CudaTimer timer(metrics.forceTimeMs);

        int blockSize = g_blockSize;
        int gridSize = (nBodies + blockSize - 1) / blockSize;
        
        // Define a maximum stack size constant for the kernel (must match kernel's MAX_STACK_SIZE)
        constexpr int MAX_STACK = 32; // Conservative value that should work on most GPUs
        
        // Calculate shared memory size for the force kernel
        // Each thread needs an integer stack of size MAX_STACK
        size_t sharedMemSize = blockSize * MAX_STACK * sizeof(int);
        
        // Check if shared memory requirements exceed device limits
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, device);
        
        if (sharedMemSize > props.sharedMemPerBlock) {
            // If shared memory is too large, adjust block size
            int maxThreads = props.sharedMemPerBlock / (MAX_STACK * sizeof(int));
            maxThreads = (maxThreads / 32) * 32; // Round down to multiple of warp size
            if (maxThreads < 32) {
                // If we can't have even 32 threads, reduce block size to minimum
                blockSize = 32;
                // Use as much shared memory as we can
                sharedMemSize = props.sharedMemPerBlock / 32 * 32; // Align to 32 bytes
            } else {
                blockSize = maxThreads;
                sharedMemSize = blockSize * MAX_STACK * sizeof(int);
            }
            // Recalculate grid size with new block size
            gridSize = (nBodies + blockSize - 1) / blockSize;
        }
        
        // Launch kernel with the validated parameters
        ComputeForceKernel<<<gridSize, blockSize, sharedMemSize>>>(
            d_nodes, d_bodies, d_orderedIndices, useSFC, nNodes, nBodies, leafLimit, g_theta);
        CHECK_LAST_CUDA_ERROR();
    }

    void update()
    {
        CudaTimer timer(metrics.totalTimeMs);
        
        // Ensure initialization
        checkInitialization();
        
        // Check if reordering is needed
        bool shouldReorder = false;
        if (useSFC) {
            // Use dynamic strategy to decide if reordering is needed
            shouldReorder = reorderingStrategy.shouldReorder(metrics.forceTimeMs, metrics.reorderTimeMs);
            
            if (shouldReorder) {
                orderBodiesBySFC();
                iterationCounter = 0;
            }
        }
        
        // Build octree
        resetOctree();
        // computeBoundingBox();
        constructOctree();
        
        // Just call the computeForces method which now has proper shared memory validation
        computeForces();
        
        // Calculate energies
        calculateEnergies();
        
        iterationCounter++;
        
        // Update dynamic strategy with the latest timing information
        if (useSFC) {
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
        if (useSFC) {
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
        // Initialize performance counters
        double totalTime = 0.0;
        minTimeMs = FLT_MAX;
        maxTimeMs = 0.0f;
        potentialEnergyAvg = 0.0;
        kineticEnergyAvg = 0.0;
        totalEnergyAvg = 0.0;
        
        // Print initial configuration
        std::cout << "Starting SFC Barnes-Hut GPU simulation..." << std::endl;
        std::cout << "Bodies: " << nBodies << ", Nodes: " << nNodes << std::endl;
        std::cout << "Theta parameter: " << g_theta << std::endl;
        if (useSFC)
            std::cout << "Using SFC ordering, curve type: " << (curveType == sfc::CurveType::MORTON ? "MORTON" : "HILBERT") << std::endl;
        else
            std::cout << "No SFC ordering used." << std::endl;
        
        // Run the simulation
        auto startTime = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < numIterations; i++)
        {
            // Update physics for one step
            update();
            
            // Update energy averages
            potentialEnergyAvg += potentialEnergy;
            kineticEnergyAvg += kineticEnergy;
            totalEnergyAvg += (potentialEnergy + kineticEnergy);
            
            // Track min/max times in milliseconds (metrics.totalTimeMs is already in ms)
            minTimeMs = std::min(minTimeMs, metrics.totalTimeMs);
            maxTimeMs = std::max(maxTimeMs, metrics.totalTimeMs);
            
            // Update total time
            totalTime += metrics.totalTimeMs;
        
            // Ensure all CUDA operations are completed 
            cudaDeviceSynchronize();
        }
        
        auto endTime = std::chrono::high_resolution_clock::now();
        double simTimeMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
        
        // Calculate averages
        if (numIterations > 0)
        {
            potentialEnergyAvg /= numIterations;
            kineticEnergyAvg /= numIterations;
            totalEnergyAvg /= numIterations;
        }
        
        // Print final simulation results
        std::cout << "Simulation completed in " << simTimeMs << " ms." << std::endl;
        printSummary(numIterations);
    }

    void run(int steps)
    {
        std::cout << "Running SFC Barnes-Hut GPU simulation for " << steps << " steps..." << std::endl;

        // Variables for time measurement
        float totalTime = 0.0f;
        float totalForceTime = 0.0f;
        float totalBuildTime = 0.0f;
        float totalBboxTime = 0.0f;
        float totalReorderTime = 0.0f;
        float minTime = std::numeric_limits<float>::max();
        float maxTime = 0.0f;
        double totalPotentialEnergy = 0.0;
        double totalKineticEnergy = 0.0;

        // Run simulation
        for (int step = 0; step < steps; step++)
        {
            update();

            // Update statistics
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

        // Calculate average energies
        potentialEnergyAvg = totalPotentialEnergy / steps;
        kineticEnergyAvg = totalKineticEnergy / steps;
        totalEnergyAvg = potentialEnergyAvg + kineticEnergyAvg;

        // Show statistics
        std::cout << "Simulation complete." << std::endl;
        std::cout << "Performance Summary:" << std::endl;
        std::cout << "  Average time per step: " << totalTime / steps << " ms" << std::endl;
        std::cout << "  Min time: " << minTime << " ms" << std::endl;
        std::cout << "  Max time: " << maxTime << " ms" << std::endl;
        std::cout << "  Build tree: " << totalBuildTime / steps << " ms" << std::endl;
        std::cout << "  Bounding box: " << totalBboxTime / steps << " ms" << std::endl;
        std::cout << "  Compute forces: " << totalForceTime / steps << " ms" << std::endl;
        if (useSFC) {
            std::cout << "  Reordering: " << totalReorderTime / steps << " ms" << std::endl;
        }
        std::cout << "Average Energy Values:" << std::endl;
        std::cout << "  Potential Energy: " << std::scientific << std::setprecision(6) << potentialEnergyAvg << std::endl;
        std::cout << "  Kinetic Energy:   " << std::scientific << std::setprecision(6) << kineticEnergyAvg << std::endl;
        std::cout << "  Total Energy:     " << std::scientific << std::setprecision(6) << totalEnergyAvg << std::endl;
        
        // Print SFC configuration if using SFC
        if (useSFC && reorderingStrategy.getOptimalFrequency() > 0) {
            std::cout << "SFC Configuration:" << std::endl;
            std::cout << "  Optimal reorder freq: " << reorderingStrategy.getOptimalFrequency() << std::endl;
            std::cout << "  Degradation rate: " << std::fixed << std::setprecision(6) 
                      << reorderingStrategy.getDegradationRate() << " ms/iter" << std::endl;
        }
    }

    // Getters para las métricas
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
    const char* getSortType() const { return useSFC ? (curveType == sfc::CurveType::MORTON ? "MORTON" : "HILBERT") : "NONE"; }
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
        if (useSFC) {
            std::cout << "  Reordering: " << std::fixed << std::setprecision(2) << totalReorderTime / steps << " ms" << std::endl;
        }
        std::cout << "Average Energy Values:" << std::endl;
        std::cout << "  Potential Energy: " << std::scientific << std::setprecision(6) << potentialEnergyAvg << std::endl;
        std::cout << "  Kinetic Energy:   " << std::scientific << std::setprecision(6) << kineticEnergyAvg << std::endl;
        std::cout << "  Total Energy:     " << std::scientific << std::setprecision(6) << totalEnergyAvg << std::endl;
        
        // Print SFC configuration if using SFC
        if (useSFC) {
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
    double theta = DEFAULT_THETA;

    // Variables for metrics
    bool saveMetricsToFile = false;
    std::string metricsFile = "./SFCBarnesHutGPU_metrics.csv";

    // Dynamic reordering is always enabled by default
    bool useDynamicReordering = true;

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

    // Update global parameters
    g_blockSize = blockSize;
    g_theta = theta;

    // Create simulation
    SFCBarnesHutGPU simulation(
        nBodies,
        nBodies * 8,  // numNodes = nBodies * 8
        8,            // leafNodeLimit = 8
        bodyDist,
        massDist,
        useSFC,
        curveType,
        useDynamicReordering,
        seed);

    // Run simulation
    simulation.runSimulation(numIterations);

    // Save metrics if requested
    if (saveMetricsToFile) {
        // Check if file exists
        bool fileExists = false;
        std::ifstream checkFile(metricsFile);
        if (checkFile.good()) {
            fileExists = true;
        }
        checkFile.close();
        
        // Initialize CSV file and save metrics
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
            simulation.getTotalEnergy()
        );
        
        std::cout << "Métricas guardadas en: " << metricsFile << std::endl;
    }

    return 0;
}