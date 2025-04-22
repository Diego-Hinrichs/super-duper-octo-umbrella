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
    float totalTimeMs;

    SimulationMetrics() : resetTimeMs(0.0f),
                          bboxTimeMs(0.0f),
                          buildTimeMs(0.0f),
                          forceTimeMs(0.0f),
                          totalTimeMs(0.0f) {}
};

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
 * @brief Compute bounding box for all bodies
 */
__global__ void ComputeBoundingBoxKernel(Node *nodes, Body *bodies, int *mutex, int nBodies)
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

    // Initialize shared memory
    minX[tx] = (i < nBodies) ? bodies[i].position.x : DBL_MAX;
    minY[tx] = (i < nBodies) ? bodies[i].position.y : DBL_MAX;
    minZ[tx] = (i < nBodies) ? bodies[i].position.z : DBL_MAX;
    maxX[tx] = (i < nBodies) ? bodies[i].position.x : -DBL_MAX;
    maxY[tx] = (i < nBodies) ? bodies[i].position.y : -DBL_MAX;
    maxZ[tx] = (i < nBodies) ? bodies[i].position.z : -DBL_MAX;

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
 * @brief Insert a body into the octree
 * @param nodes Octree nodes
 * @param bodies Array of bodies
 * @param bodyIdx Index of body to insert
 * @param nodeIdx Current node index
 * @param nNodes Total number of nodes
 * @param leafLimit Index limit for internal nodes
 * @return true if insertion was successful
 */
__device__ bool InsertBody(Node *nodes, Body *bodies, int bodyIdx, int nodeIdx, int nNodes, int leafLimit)
{
    // Recursively insert the body into the tree
    // This is a simplified version - a full implementation would use atomics for thread safety

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
        
        // Determine which octant the existing body belongs to
        Vector pos = bodies[existingBodyIdx].position;
        int childIdx = 0;
        if (pos.x >= center.x) childIdx |= 1;
        if (pos.y >= center.y) childIdx |= 2;
        if (pos.z >= center.z) childIdx |= 4;
        
        // Create the child nodes with appropriate bounds
        for (int i = 0; i < 8; i++)
        {
            Node &child = nodes[firstChildIdx + i];
            child.isLeaf = true;
            child.firstChildIndex = -1;
            child.bodyIndex = -1;
            child.bodyCount = 0;
            child.mass = 0.0;
            
            // Calculate min/max for this child
            Vector min = node.min;
            Vector max = node.max;
            
            // Adjust bounds based on octant
            if (i & 1) min.x = center.x; else max.x = center.x;
            if (i & 2) min.y = center.y; else max.y = center.y;
            if (i & 4) min.z = center.z; else max.z = center.z;
            
            child.min = min;
            child.max = max;
            child.position = (min + max) * 0.5;
            child.radius = ((max - min) * 0.5).length();
        }
        
        // Insert the existing body into the appropriate child
        InsertBody(nodes, bodies, existingBodyIdx, firstChildIdx + childIdx, nNodes, leafLimit);
    }
    
    // Now determine which child the new body belongs to
    Vector pos = bodies[bodyIdx].position;
    Vector center = node.position;
    
    int childIdx = 0;
    if (pos.x >= center.x) childIdx |= 1;
    if (pos.y >= center.y) childIdx |= 2;
    if (pos.z >= center.z) childIdx |= 4;
    
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
 * @brief Construct octree from bodies
 */
__global__ void ConstructOctTreeKernel(Node *nodes, Body *bodies, Body *bodyBuffer, int rootIdx, int nNodes, int nBodies, int leafLimit)
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
        // Copy body to buffer (optional, helps with memory access patterns)
        bodyBuffer[bodyIdx] = bodies[bodyIdx];
        
        // Insert body into the octree
        InsertBody(nodes, bodies, bodyIdx, rootIdx, nNodes, leafLimit);
    }
}

/**
 * @brief Compute forces using Barnes-Hut algorithm
 */
__global__ void ComputeForceKernel(Node *nodes, Body *bodies, int nNodes, int nBodies, int leafLimit, double theta)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nBodies || !bodies[i].isDynamic)
        return;

    Vector acc(0.0, 0.0, 0.0);
    Vector pos = bodies[i].position;
    double bodyMass = bodies[i].mass;
    
    // Process octree to compute forces
    // This uses a stack-based traversal to avoid recursion
    constexpr int MAX_STACK = 64;
    int stack[MAX_STACK];
    int stackSize = 0;
    
    // Start with root node
    stack[stackSize++] = 0;
    
    while (stackSize > 0)
    {
        // Pop node from stack
        int nodeIdx = stack[--stackSize];
        if (nodeIdx < 0 || nodeIdx >= nNodes)
            continue;
        
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
            if (node.isLeaf && node.bodyIndex == i)
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
            // Node is too close, need to process its children
            for (int c = 0; c < 8; c++)
            {
                int childIdx = node.firstChildIndex + c;
                if (childIdx < nNodes && stackSize < MAX_STACK)
                {
                    stack[stackSize++] = childIdx;
                }
            }
        }
    }
    
    // Update acceleration
    bodies[i].acceleration = acc;
    
    // Update velocity
    bodies[i].velocity = bodies[i].velocity + acc * DT;
    
    // Update position
    bodies[i].position = bodies[i].position + bodies[i].velocity * DT;
}

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
        file << "timestamp,method,bodies,steps,block_size,theta,total_time_ms,avg_step_time_ms,force_calculation_time_ms,tree_build_time_ms" << std::endl;
    }
    
    file.close();
    std::cout << "Archivo CSV " << (append ? "actualizado" : "inicializado") << ": " << filename << std::endl;
}

void saveMetrics(const std::string& filename, 
                int bodies, 
                int steps, 
                int blockSize,
                float theta,
                float totalTime, 
                float forceCalculationTime,
                float treeBuildTime) {
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
         << "GPU_Barnes_Hut" << ","
         << bodies << ","
         << steps << ","
         << blockSize << ","
         << theta << ","
         << totalTime << ","
         << avgTimePerStep << ","
         << forceCalculationTime << ","
         << treeBuildTime << std::endl;
    
    file.close();
    std::cout << "Métricas guardadas en: " << filename << std::endl;
}

// =============================================================================
// SIMULATION CLASS
// =============================================================================

class BarnesHutGPU
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
    SimulationMetrics metrics;       // Performance metrics

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

    void checkInitialization()
    {
        if (!d_bodies || !d_nodes || !d_mutex || !d_bodiesBuffer)
        {
            std::cerr << "Error: Device memory not allocated." << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

public:
    BarnesHutGPU(
        int numBodies,
        BodyDistribution dist = BodyDistribution::GALAXY,
        unsigned int seed = 42,
        MassDistribution massDist = MassDistribution::NORMAL)
        : nBodies(numBodies),
          nNodes(MAX_NODES),
          leafLimit(MAX_NODES - N_LEAF)
    {
        if (numBodies < 1)
            numBodies = 1;

        std::cout << "Barnes-Hut GPU Simulation created with " << numBodies << " bodies "
                 << "and " << nNodes << " nodes." << std::endl;

        // Allocate host memory
        h_bodies = new Body[nBodies];
        h_nodes = new Node[nNodes];

        // Allocate device memory
        CHECK_CUDA_ERROR(cudaMalloc(&d_bodies, nBodies * sizeof(Body)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_bodiesBuffer, nBodies * sizeof(Body)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_nodes, nNodes * sizeof(Node)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_mutex, nNodes * sizeof(int)));

        // Initialize bodies
        initializeDistribution(dist, massDist, seed);

        // Copy to device
        CHECK_CUDA_ERROR(cudaMemcpy(d_bodies, h_bodies, nBodies * sizeof(Body), cudaMemcpyHostToDevice));
    }

    ~BarnesHutGPU()
    {
        // Free resources
        delete[] h_bodies;
        delete[] h_nodes;
        
        if (d_bodies) CHECK_CUDA_ERROR(cudaFree(d_bodies));
        if (d_bodiesBuffer) CHECK_CUDA_ERROR(cudaFree(d_bodiesBuffer));
        if (d_nodes) CHECK_CUDA_ERROR(cudaFree(d_nodes));
        if (d_mutex) CHECK_CUDA_ERROR(cudaFree(d_mutex));
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
        ComputeBoundingBoxKernel<<<gridSize, blockSize, sharedMemSize>>>(d_nodes, d_bodies, d_mutex, nBodies);
        CHECK_LAST_CUDA_ERROR();
    }

    void constructOctree()
    {
        CudaTimer timer(metrics.buildTimeMs);
        
        int blockSize = g_blockSize;
        
        // Calculate shared memory size for the octree kernel
        size_t sharedMemSize = blockSize * sizeof(double) +  // totalMass array
                              blockSize * sizeof(double3);  // centerMass array
                              
        ConstructOctTreeKernel<<<1, blockSize, sharedMemSize>>>(d_nodes, d_bodies, d_bodiesBuffer, 0, nNodes, nBodies, leafLimit);
        CHECK_LAST_CUDA_ERROR();
    }

    void computeForces()
    {
        CudaTimer timer(metrics.forceTimeMs);

        int blockSize = g_blockSize;
        int gridSize = (nBodies + blockSize - 1) / blockSize;
        
        ComputeForceKernel<<<gridSize, blockSize>>>(d_nodes, d_bodies, nNodes, nBodies, leafLimit, g_theta);
        CHECK_LAST_CUDA_ERROR();
    }

    void update()
    {
        CudaTimer timer(metrics.totalTimeMs);
        
        // Ensure initialization
        checkInitialization();
        
        // Build octree
        resetOctree();
        computeBoundingBox();
        constructOctree();
        
        // Compute forces
        computeForces();
    }

    void printPerformance()
    {
        printf("Performance Metrics:\n");
        printf("  Reset octree: %.3f ms\n", metrics.resetTimeMs);
        printf("  Compute bounding box: %.3f ms\n", metrics.bboxTimeMs);
        printf("  Build octree: %.3f ms\n", metrics.buildTimeMs);
        printf("  Force calculation: %.3f ms\n", metrics.forceTimeMs);
        printf("  Total update: %.3f ms\n", metrics.totalTimeMs);
    }

    void runSimulation(int numIterations, int printFreq = 10)
    {
        printf("Starting Barnes-Hut GPU simulation with %d bodies for %d iterations\n", nBodies, numIterations);
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
    int getNumBodies() const { return nBodies; }
    int getBlockSize() const { return g_blockSize; }
    double getTheta() const { return g_theta; }
};

// =============================================================================
// MAIN FUNCTION
// =============================================================================

int main(int argc, char **argv)
{
    // Default parameters
    int nBodies = 10000;
    BodyDistribution bodyDist = BodyDistribution::GALAXY;
    MassDistribution massDist = MassDistribution::NORMAL;
    unsigned int seed = 42;
    int numIterations = 100;
    int blockSize = DEFAULT_BLOCK_SIZE;
    double theta = DEFAULT_THETA;

    // Añadir variables para métricas
    bool saveMetricsToFile = false;
    std::string metricsFile = "./BarnesHutGPU_metrics.csv";

    // Parse command line arguments
    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "-n" && i + 1 < argc)
            nBodies = std::stoi(argv[++i]);
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
            std::cout << "  -dist <type>      Body distribution: galaxy, solar, uniform, random (default: galaxy)" << std::endl;
            std::cout << "  -mass <type>      Mass distribution: uniform, normal (default: normal)" << std::endl;
            std::cout << "  -seed <num>       Random seed (default: 42)" << std::endl;
            std::cout << "  -iter <num>       Number of iterations (default: 100)" << std::endl;
            std::cout << "  -block <num>      CUDA block size (default: 256)" << std::endl;
            std::cout << "  -theta <float>    Barnes-Hut opening angle parameter (default: 0.5)" << std::endl;
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
    BarnesHutGPU simulation(
        nBodies,
        bodyDist,
        seed,
        massDist);

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
            simulation.getTotalTime(),
            simulation.getForceCalculationTime(),
            simulation.getTreeBuildTime()
        );
        
        std::cout << "Métricas guardadas en: " << metricsFile << std::endl;
    }

    return 0;
} 