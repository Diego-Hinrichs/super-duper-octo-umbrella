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
    float totalTimeMs;

    SimulationMetrics() : resetTimeMs(0.0f),
                          bboxTimeMs(0.0f),
                          buildTimeMs(0.0f),
                          forceTimeMs(0.0f),
                          totalTimeMs(0.0f) {}
};

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

__global__ void ComputeBoundingBoxKernel(Node *nodes, Body *bodies, int *mutex, int nBodies)
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

    minX[tx] = (i < nBodies) ? bodies[i].position.x : DBL_MAX;
    minY[tx] = (i < nBodies) ? bodies[i].position.y : DBL_MAX;
    minZ[tx] = (i < nBodies) ? bodies[i].position.z : DBL_MAX;
    maxX[tx] = (i < nBodies) ? bodies[i].position.x : -DBL_MAX;
    maxY[tx] = (i < nBodies) ? bodies[i].position.y : -DBL_MAX;
    maxZ[tx] = (i < nBodies) ? bodies[i].position.z : -DBL_MAX;

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
        int childIdx = 0;
        if (pos.x >= center.x)
            childIdx |= 1;
        if (pos.y >= center.y)
            childIdx |= 2;
        if (pos.z >= center.z)
            childIdx |= 4;

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
            child.radius = ((max - min) * 0.5).length();
        }

        InsertBody(nodes, bodies, existingBodyIdx, firstChildIdx + childIdx, nNodes, leafLimit);
    }

    Vector pos = bodies[bodyIdx].position;
    Vector center = node.position;

    int childIdx = 0;
    if (pos.x >= center.x)
        childIdx |= 1;
    if (pos.y >= center.y)
        childIdx |= 2;
    if (pos.z >= center.z)
        childIdx |= 4;

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

__global__ void ConstructOctTreeKernel(Node *nodes, Body *bodies, Body *bodyBuffer, int rootIdx, int nNodes, int nBodies, int leafLimit)
{

    extern __shared__ double sharedMem[];
    double *totalMass = &sharedMem[0];
    double3 *centerMass = (double3 *)(totalMass + blockDim.x);

    int i = threadIdx.x;

    for (int bodyIdx = i; bodyIdx < nBodies; bodyIdx += blockDim.x)
    {

        bodyBuffer[bodyIdx] = bodies[bodyIdx];

        InsertBody(nodes, bodies, bodyIdx, rootIdx, nNodes, leafLimit);
    }
}

__global__ void ComputeForceKernel(Node *nodes, Body *bodies, int nNodes, int nBodies, int leafLimit, double theta)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nBodies || !bodies[i].isDynamic)
        return;

    Vector acc(0.0, 0.0, 0.0);
    Vector pos = bodies[i].position;
    double bodyMass = bodies[i].mass;

    constexpr int MAX_STACK = 64;
    int stack[MAX_STACK];
    int stackSize = 0;

    stack[stackSize++] = 0;

    while (stackSize > 0)
    {

        int nodeIdx = stack[--stackSize];
        if (nodeIdx < 0 || nodeIdx >= nNodes)
            continue;

        Node &node = nodes[nodeIdx];

        if (node.bodyCount == 0 || node.mass <= 0.0)
            continue;

        Vector distVec = node.position - pos;
        double distSqr = distVec.lengthSquared() + E * E;
        double dist = sqrt(distSqr);

        if (node.isLeaf || (node.radius / dist < theta))
        {

            if (node.isLeaf && node.bodyIndex == i)
                continue;

            if (dist >= COLLISION_TH)
            {
                double forceMag = (GRAVITY * bodyMass * node.mass) / (dist * distSqr);
                acc = acc + distVec * (forceMag / bodyMass);
            }
        }
        else if (node.firstChildIndex >= 0)
        {

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

    bodies[i].acceleration = acc;

    bodies[i].velocity = bodies[i].velocity + acc * DT;

    bodies[i].position = bodies[i].position + bodies[i].velocity * DT;
}

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
        file << "timestamp,method,bodies,steps,block_size,theta,total_time_ms,avg_step_time_ms,force_calculation_time_ms,tree_build_time_ms" << std::endl;
    }

    file.close();
    std::cout << "Archivo CSV " << (append ? "actualizado" : "inicializado") << ": " << filename << std::endl;
}

void saveMetrics(const std::string &filename,
                 int bodies,
                 int steps,
                 int blockSize,
                 float theta,
                 float totalTime,
                 float forceCalculationTime,
                 float treeBuildTime)
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

class BarnesHutGPU
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
    SimulationMetrics metrics;

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

        h_bodies = new Body[nBodies];
        h_nodes = new Node[nNodes];

        CHECK_CUDA_ERROR(cudaMalloc(&d_bodies, nBodies * sizeof(Body)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_bodiesBuffer, nBodies * sizeof(Body)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_nodes, nNodes * sizeof(Node)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_mutex, nNodes * sizeof(int)));

        initializeDistribution(dist, massDist, seed);

        CHECK_CUDA_ERROR(cudaMemcpy(d_bodies, h_bodies, nBodies * sizeof(Body), cudaMemcpyHostToDevice));
    }

    ~BarnesHutGPU()
    {

        delete[] h_bodies;
        delete[] h_nodes;

        if (d_bodies)
            CHECK_CUDA_ERROR(cudaFree(d_bodies));
        if (d_bodiesBuffer)
            CHECK_CUDA_ERROR(cudaFree(d_bodiesBuffer));
        if (d_nodes)
            CHECK_CUDA_ERROR(cudaFree(d_nodes));
        if (d_mutex)
            CHECK_CUDA_ERROR(cudaFree(d_mutex));
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
        ComputeBoundingBoxKernel<<<gridSize, blockSize, sharedMemSize>>>(d_nodes, d_bodies, d_mutex, nBodies);
        CHECK_LAST_CUDA_ERROR();
    }

    void constructOctree()
    {
        CudaTimer timer(metrics.buildTimeMs);

        int blockSize = g_blockSize;

        size_t sharedMemSize = blockSize * sizeof(double) +
                               blockSize * sizeof(double3);

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

        checkInitialization();

        resetOctree();
        computeBoundingBox();
        constructOctree();

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
        }

        printf("Simulation completed in %.3f ms (avg %.3f ms per iteration)\n",
               totalTime, totalTime / numIterations);
    }

    float getTotalTime() const { return metrics.totalTimeMs; }
    float getForceCalculationTime() const { return metrics.forceTimeMs; }
    float getTreeBuildTime() const { return metrics.buildTimeMs; }
    int getNumBodies() const { return nBodies; }
    int getBlockSize() const { return g_blockSize; }
    double getTheta() const { return g_theta; }
};

int main(int argc, char **argv)
{

    int nBodies = 10000;
    BodyDistribution bodyDist = BodyDistribution::GALAXY;
    MassDistribution massDist = MassDistribution::NORMAL;
    unsigned int seed = 42;
    int numIterations = 100;
    int blockSize = DEFAULT_BLOCK_SIZE;
    double theta = DEFAULT_THETA;

    bool saveMetricsToFile = false;
    std::string metricsFile = "./BarnesHutGPU_metrics.csv";

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
        else if (arg == "--save-metrics")
        {
            saveMetricsToFile = true;
        }
        else if (arg == "--metrics-file" && i + 1 < argc)
        {
            metricsFile = argv[++i];
        }
    }

    g_blockSize = blockSize;
    g_theta = theta;

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

    BarnesHutGPU simulation(
        nBodies,
        bodyDist,
        seed,
        massDist);

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
            simulation.getTotalTime(),
            simulation.getForceCalculationTime(),
            simulation.getTreeBuildTime());

        std::cout << "Métricas guardadas en: " << metricsFile << std::endl;
    }

    return 0;
}