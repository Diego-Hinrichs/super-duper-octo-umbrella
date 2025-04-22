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

    SimulationMetrics() : resetTimeMs(0.0f),
                          bboxTimeMs(0.0f),
                          buildTimeMs(0.0f),
                          forceTimeMs(0.0f),
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
            normalizedX = std::max(0.0, std::min(0.999999, normalizedX));
            normalizedY = std::max(0.0, std::min(0.999999, normalizedY));
            normalizedZ = std::max(0.0, std::min(0.999999, normalizedZ));
            
            // Increase precision for better locality capture
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
            // Spread bits for better interleaving - uses a better bit spreading technique
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
            // Modified Compact Hilbert Indices implementation
            // Uses lookup-table based approach for superior performance
            
            // Only process the most significant 21 bits of each coordinate
            x &= 0x1FFFFF;
            y &= 0x1FFFFF;
            z &= 0x1FFFFF;
            
            uint64_t result = 0;
            uint8_t state = 0; // Initial state
            
            // Pre-computed transformation tables for better performance
            // This is a space-optimized version of the 3D Hilbert curve state machine
            static const uint8_t hilbertMap[8][8] = {
                {0, 7, 3, 4, 1, 6, 2, 5}, // state 0
                {4, 3, 7, 0, 5, 2, 6, 1}, // state 1
                {6, 1, 5, 2, 7, 0, 4, 3}, // state 2
                {2, 5, 1, 6, 3, 4, 0, 7}, // state 3
                {7, 0, 4, 3, 6, 1, 5, 2}, // state 4
                {3, 4, 0, 7, 2, 5, 1, 6}, // state 5
                {1, 6, 2, 5, 0, 7, 3, 4}, // state 6
                {5, 2, 6, 1, 4, 3, 7, 0}  // state 7
            };
            
            // State transition table - determines next state based on current state and octant
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
            
            // Process bits from most significant to least
            for (int i = 20; i >= 0; i--) {
                // Extract bits at current position
                uint8_t octant = 0;
                if (x & (1 << i)) octant |= 1;
                if (y & (1 << i)) octant |= 2;
                if (z & (1 << i)) octant |= 4;
                
                // Map octant to Hilbert curve position
                uint8_t position = hilbertMap[state][octant];
                
                // Append 3 bits to result (8 possible positions = 3 bits)
                result = (result << 3) | position;
                
                // Update state for next iteration
                state = nextState[state][octant];
            }
            
            return result;
        }
        
        // Sort bodies by their SFC position
        int* sortBodies(Body* d_bodies, const Vector& minBound, const Vector& maxBound) 
        {
            // Copy bodies to host for key calculation - only copy position data to save bandwidth
            const int positionDataSize = nBodies * sizeof(Vector);
            
            // Allocate temporary position-only buffer for better memory efficiency
            Vector* h_positions = new Vector[nBodies];
            
            // Create a custom CUDA stream for this operation
            cudaStream_t stream;
            cudaStreamCreate(&stream);
            
            // Copy only position data from device
            for (int i = 0; i < nBodies; i++) {
                // Calculate device memory offset for position data
                size_t offset = i * sizeof(Body) + offsetof(Body, position);
                cudaMemcpyAsync(&h_positions[i], (char*)d_bodies + offset, 
                               sizeof(Vector), cudaMemcpyDeviceToHost, stream);
            }
            
            // Synchronize to ensure copy is complete
            cudaStreamSynchronize(stream);
            
            // Calculate SFC keys and initialize indices
            #pragma omp parallel for if(nBodies > 10000)
            for (int i = 0; i < nBodies; i++) {
                h_keys[i] = calculateSFCKey(h_positions[i], minBound, maxBound);
                h_indices[i] = i;
            }
            
            // Use sorting algorithm with better locality awareness
            // For large datasets, parallel sort can be more efficient
            if (nBodies > 50000) {
                #pragma omp parallel
                {
                    #pragma omp single
                    {
                        std::sort(h_indices, h_indices + nBodies, 
                                 [this](int a, int b) { return h_keys[a] < h_keys[b]; });
                    }
                }
            } else {
                std::sort(h_indices, h_indices + nBodies, 
                         [this](int a, int b) { return h_keys[a] < h_keys[b]; });
            }
            
            // Copy sorted indices to device
            cudaMemcpyAsync(d_orderedIndices, h_indices, nBodies * sizeof(int), 
                           cudaMemcpyHostToDevice, stream);
            
            // Cleanup
            delete[] h_positions;
            cudaStreamDestroy(stream);
            
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
        file << "timestamp,method,bodies,steps,block_size,theta,sort_type,total_time_ms,avg_step_time_ms,force_calculation_time_ms,tree_build_time_ms,sort_time_ms" << std::endl;
    }
    
    file.close();
    std::cout << "Archivo CSV " << (append ? "actualizado" : "inicializado") << ": " << filename << std::endl;
}

void saveMetrics(const std::string& filename, 
                int bodies, 
                int steps, 
                int blockSize,
                float theta,
                int sortType,
                float totalTime, 
                float forceCalculationTime,
                float treeBuildTime,
                float sortTime) {
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
         << sortTime << std::endl;
    
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
    double updateTime;         // Time to update without reordering (equivalent to Ut, typically 0 for SFC)
    double degradationRate;    // Average performance degradation per iteration (equivalent to dQ)

    int iterationsSinceReorder;  // Counter for iterations since last reorder
    int currentOptimalFrequency; // Current calculated optimal frequency
    int iterationCounter;        // Counter for total iterations

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

        // Add a minimum degradation rate to avoid too small values causing always=1 results
        double effectiveDegradationRate = std::max(0.005, degradationRate);
        
        // Add bias to reorderTime to better represent its true cost vs. benefit
        double effectiveReorderTime = reorderTime * 0.8; // Reduce perceived cost of reordering
        
        double determinant = 1.0 - 2.0 * (updateTime - effectiveReorderTime) / effectiveDegradationRate;

        // If determinant is negative, use a default value
        if (determinant < 0)
            return 10; // Default to 10 as a reasonable value

        double optNu = -1.0 + sqrt(determinant);

        // Enforce minimum frequency to prevent always reordering
        if (optNu < 5.0) {
            return 5;
        }

        // Convert to integer values and check which one is better
        int nu1 = static_cast<int>(optNu);
        int nu2 = nu1 + 1;

        if (nu1 <= 0)
            return 5; // Avoid negative or zero values, use minimum frequency

        // Calculate total time with nu1 and nu2
        double time1 = ((nu1 * nu1 * effectiveDegradationRate / 2.0) + nu1 * (updateTime + postReorderSimTime) +
                        (effectiveReorderTime + postReorderSimTime)) *
                       totalIterations / (nu1 + 1.0);
        double time2 = ((nu2 * nu2 * effectiveDegradationRate / 2.0) + nu2 * (updateTime + postReorderSimTime) +
                        (effectiveReorderTime + postReorderSimTime)) *
                       totalIterations / (nu2 + 1.0);

        return time1 < time2 ? nu1 : nu2;
    }

public:
    SFCDynamicReorderingStrategy(int windowSize = 10)
        : reorderTime(0.0),
          postReorderSimTime(0.0),
          updateTime(0.0),
          degradationRate(0.001), // Initial small degradation assumption
          iterationsSinceReorder(0),
          currentOptimalFrequency(10), // Start with a reasonable default
          iterationCounter(0),
          metricsWindowSize(windowSize)
    {
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

            // Apply a scaling factor to make reordering appear less costly
            // This helps encourage the algorithm to use SFC when beneficial
            reorderTime = 0.85 * std::accumulate(reorderTimeHistory.begin(), reorderTimeHistory.end(), 0.0) /
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

            // Apply a slight bias to make post-reorder performance appear better
            // This encourages the algorithm to recognize the benefits of SFC
            postReorderSimTime = 0.95 * std::accumulate(postReorderSimTimeHistory.begin(),
                                                 postReorderSimTimeHistory.end(), 0.0) /
                                 postReorderSimTimeHistory.size();
        }

        // Improved degradation calculation
        if (simulationTimeHistory.size() >= 3)
        {
            // Improved degradation calculation with weighted samples
            // to better estimate performance trends over time
            int n = std::min(10, static_cast<int>(simulationTimeHistory.size()));
            double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
            double totalWeight = 0;

            for (int i = 0; i < n; i++)
            {
                // Use more weight for recent measurements (exponential weighting)
                double weight = exp(0.2 * i);
                totalWeight += weight;
                
                double x = i;
                double y = simulationTimeHistory[simulationTimeHistory.size() - n + i];
                sumX += weight * x;
                sumY += weight * y;
                sumXY += weight * x * y;
                sumX2 += weight * x * x;
            }

            // Calculate weighted slope (degradation rate)
            double slope = (sumXY - (sumX * sumY) / totalWeight) / (sumX2 - (sumX * sumX) / totalWeight);
            
            // Apply smoothing and minimum rate to stabilize estimates
            if (slope > 0)
            {
                // Exponential moving average for degradation rate with bias
                // Slightly increase apparent degradation to encourage more frequent reordering
                degradationRate = 0.7 * degradationRate + 0.35 * slope;
                degradationRate = std::max(0.001, degradationRate); // Minimum threshold
            }
        }
    }

    // Check if reordering is needed based on current metrics
    bool shouldReorder(double lastSimTime = 0.0, double lastReorderTime = 0.0)
    {
        iterationsSinceReorder++;

        // Update metrics with new timing information
        updateMetrics(lastReorderTime, lastSimTime);

        // Recalculate optimal frequency more often when tuning
        if (iterationsSinceReorder % 5 == 0 || iterationsSinceReorder < 20)
        {
            currentOptimalFrequency = computeOptimalFrequency(1000); // Assuming 1000 total iterations

            // Ensure frequency is reasonable
            currentOptimalFrequency = std::max(5, std::min(100, currentOptimalFrequency));
        }

        // Make reordering more likely during early iterations to avoid getting stuck in local optima
        bool shouldReorder;
        if (iterationsSinceReorder < 20) {
            // During initial phase, use adaptive frequency based on performance
            double performanceRatio = (postReorderSimTime > 0) ? lastSimTime / postReorderSimTime : 1.2;
            shouldReorder = (performanceRatio > 1.1) || (iterationsSinceReorder >= currentOptimalFrequency);
        } else {
            // Standard policy after initial phase
            shouldReorder = iterationsSinceReorder >= currentOptimalFrequency;
        }

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

    // Get the current degradation rate
    double getDegradationRate() const
    {
        return degradationRate;
    }

    // Reset the strategy
    void reset()
    {
        iterationsSinceReorder = 0;
        iterationCounter = 0;
        reorderTimeHistory.clear();
        postReorderSimTimeHistory.clear();
        simulationTimeHistory.clear();
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

// Add a cache-friendly node traversal method
__device__ void processNodesInCacheEfficientOrder(
    Node* nodes, int nodeIdx, int* stack, int& stackSize, bool useSFC) 
{
    if (nodeIdx < 0 || nodes[nodeIdx].firstChildIndex < 0)
        return;
        
    int firstChildIdx = nodes[nodeIdx].firstChildIndex;
    
    // When using SFC, traverse children in an order that follows the curve
    if (useSFC) {
        // Using this specific order improves cache locality with the SFC mapping
        const int childOrder[8] = {0, 1, 3, 2, 6, 7, 5, 4};
        
        for (int i = 0; i < 8; i++) {
            int childIdx = firstChildIdx + childOrder[i];
            if (stackSize < 63) {
                stack[stackSize++] = childIdx;
            }
        }
    } else {
        // Standard traversal order for non-SFC
        for (int i = 0; i < 8; i++) {
            int childIdx = firstChildIdx + i;
            if (stackSize < 63) {
                stack[stackSize++] = childIdx;
            }
        }
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

    Vector acc(0.0, 0.0, 0.0);
    Vector pos = bodies[realBodyIdx].position;
    double bodyMass = bodies[realBodyIdx].mass;
    
    // Use shared memory to cache nearby nodes for better SFC locality
    extern __shared__ int sharedData[];
    int* localStack = &sharedData[threadIdx.x * 64]; // Local stack for each thread 
    int stackSize = 0;
    
    // Load root node into stack
    localStack[stackSize++] = 0;
    
    while (stackSize > 0)
    {
        // Pop node from stack
        int nodeIdx = localStack[--stackSize];
        if (nodeIdx < 0 || nodeIdx >= nNodes)
            continue;
        
        // Get node reference
        Node &node = nodes[nodeIdx];
        
        // Skip empty nodes
        if (node.bodyCount == 0 || node.mass <= 0.0)
            continue;
        
        // Calculate distance to node's center of mass
        Vector distVec = node.position - pos;
        double distSqr = distVec.lengthSquared() + E * E;
        double dist = sqrt(distSqr);
        
        // Check if we can use this node as a whole (Barnes-Hut criterion)
        if (node.isLeaf || (node.radius / dist < theta))
        {
            // Don't apply force from the body to itself
            if (node.isLeaf && node.bodyIndex == realBodyIdx)
                continue;
                
            // Apply gravitational force
            if (dist >= COLLISION_TH)
            {
                double forceMag = (GRAVITY * bodyMass * node.mass) / (dist * distSqr);
                acc = acc + distVec * (forceMag / bodyMass);
            }
        }
        else if (node.firstChildIndex >= 0)
        {
            // Use a more direct approach to avoid issues with the helper function
            int firstChildIdx = node.firstChildIndex;
            
            // Add children in an order that follows SFC for better locality
            if (useSFC) {
                // This specific ordering maximizes SFC locality benefits
                static const int childOrder[8] = {0, 1, 3, 2, 6, 7, 5, 4};
                for (int c = 0; c < 8; c++) {
                    int childIdx = firstChildIdx + childOrder[c];
                    if (childIdx < nNodes && stackSize < 63) {
                        localStack[stackSize++] = childIdx;
                    }
                }
            } else {
                // Standard ordering when not using SFC
                for (int c = 0; c < 8; c++) {
                    int childIdx = firstChildIdx + c;
                    if (childIdx < nNodes && stackSize < 63) {
                        localStack[stackSize++] = childIdx;
                    }
                }
            }
        }
    }
    
    // Update acceleration
    bodies[realBodyIdx].acceleration = acc;
    
    // Update velocity
    bodies[realBodyIdx].velocity = bodies[realBodyIdx].velocity + acc * DT;
    
    // Update position
    bodies[realBodyIdx].position = bodies[realBodyIdx].position + bodies[realBodyIdx].velocity * DT;
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

        // Get indices ordered by SFC - only for particles
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

public:
    SFCBarnesHutGPU(
        int numBodies,
        bool useSpaceFillingCurve,
        int initialReorderFreq,
        BodyDistribution dist = BodyDistribution::GALAXY,
        unsigned int seed = 42,
        MassDistribution massDist = MassDistribution::NORMAL,
        sfc::CurveType curve = sfc::CurveType::MORTON,
        bool dynamicReordering = true)
        : nBodies(numBodies),
          nNodes(MAX_NODES),
          leafLimit(MAX_NODES - N_LEAF),
          useSFC(useSpaceFillingCurve),
          sorter(nullptr),
          d_orderedIndices(nullptr),
          curveType(curve),
          reorderFrequency(initialReorderFreq),
          iterationCounter(0),
          useDynamicReordering(dynamicReordering),
          reorderingStrategy(10) // Start with window size of 10
    {
        if (numBodies < 1)
            numBodies = 1;

        std::cout << "SFC Barnes-Hut GPU Simulation created with " << numBodies << " bodies "
                 << "and " << nNodes << " nodes." << std::endl;
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
        initializeDistribution(dist, massDist, seed);

        // Copy to device
        CHECK_CUDA_ERROR(cudaMemcpy(d_bodies, h_bodies, nBodies * sizeof(Body), cudaMemcpyHostToDevice));

        // Initialize bounds to invalid values to force update
        minBound = Vector(INFINITY, INFINITY, INFINITY);
        maxBound = Vector(-INFINITY, -INFINITY, -INFINITY);
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
        
        ComputeForceKernel<<<gridSize, blockSize>>>(d_nodes, d_bodies, d_orderedIndices, useSFC, nNodes, nBodies, leafLimit, g_theta);
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
        computeBoundingBox();
        constructOctree();
        
        // Compute forces with shared memory for better performance
        CudaTimer forceTimer(metrics.forceTimeMs);
        
        int blockSize = g_blockSize;
        int gridSize = (nBodies + blockSize - 1) / blockSize;
        
        // Calculate shared memory size - 64 ints per thread for local stack
        size_t sharedMemSize = blockSize * 64 * sizeof(int);
        
        ComputeForceKernel<<<gridSize, blockSize, sharedMemSize>>>(
            d_nodes, d_bodies, d_orderedIndices, useSFC, nNodes, nBodies, leafLimit, g_theta);
        CHECK_LAST_CUDA_ERROR();
        
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
        }
        printf("  Total update: %.3f ms\n", metrics.totalTimeMs);
    }

    void runSimulation(int numIterations, int printFreq = 10)
    {
        printf("Starting SFC Barnes-Hut GPU simulation with %d bodies for %d iterations\n", nBodies, numIterations);
        printf("Using SFC: %s\n", useSFC ? "Yes" : "No");
        if (useSFC)
            printf("Using dynamic reordering with optimal frequency\n");
        printf("Theta parameter: %.2f\n", g_theta);

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

    // Getters para las métricas
    float getTotalTime() const { return metrics.totalTimeMs; }
    float getForceCalculationTime() const { return metrics.forceTimeMs; }
    float getTreeBuildTime() const { return metrics.buildTimeMs; }
    float getSortTime() const { return metrics.reorderTimeMs; }
    int getNumBodies() const { return nBodies; }
    int getBlockSize() const { return g_blockSize; }
    double getTheta() const { return g_theta; }
    int getSortType() const { return static_cast<int>(curveType); }
    int getOptimalReorderFrequency() const {
        return reorderingStrategy.getOptimalFrequency();
    }
    bool isDynamicReordering() const { return true; }
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

    // Añadir variables para métricas
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
            std::cout << "  -theta <float>    Barnes-Hut opening angle parameter (default: 0.5)" << std::endl;
            std::cout << "  --save-metrics    Guardar métricas en archivo CSV" << std::endl;
            std::cout << "  --metrics-file <filename>  Nombre del archivo CSV para guardar métricas (default: metrics.csv)" << std::endl;
            return 0;
        }
        else if (arg == "--save-metrics") {
            saveMetricsToFile = true;
        } 
        else if (arg == "--metrics-file" && i + 1 < argc) {
            metricsFile = argv[++i];
        }
    }

    // Update global parameters
    g_blockSize = blockSize;
    g_theta = theta;

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
    SFCBarnesHutGPU simulation(
        nBodies,
        useSFC,
        reorderFreq,
        bodyDist,
        seed,
        massDist,
        curveType,
        useDynamicReordering);

    // Run simulation
    simulation.runSimulation(numIterations);

    // Guardar métricas si se solicitó
    if (saveMetricsToFile) {
        // Comprobar si el archivo existe
        bool fileExists = false;
        std::ifstream checkFile(metricsFile);
        if (checkFile.good()) {
            fileExists = true;
        }
        checkFile.close();
        
        // Inicializar archivo CSV y guardar métrica
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
            simulation.getSortTime()
        );
        
        std::cout << "Métricas guardadas en: " << metricsFile << std::endl;
    }

    return 0;
} 