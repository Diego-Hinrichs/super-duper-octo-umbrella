#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <string>
#include <random>
#include <limits>
#include <vector>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <sstream>
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
    bool isDynamic;      // Whether the body moves or is static
    double mass;         // Mass of the body
    double radius;       // Radius of the body
    Vector position;     // Position in 3D space
    Vector velocity;     // Velocity vector
    Vector acceleration; // Acceleration vector

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
    float totalTimeMs;
    float energyCalculationTimeMs; // Added time for energy calculation

    SimulationMetrics() : forceTimeMs(0.0f),
                          totalTimeMs(0.0f),
                          energyCalculationTimeMs(0.0f) {}
};

enum class BodyDistribution
{
    RANDOM,
};

enum class MassDistribution
{
    UNIFORM,
    NORMAL
};

constexpr double GRAVITY = 6.67430e-11;        // Gravitational constant
constexpr double SOFTENING_FACTOR = 0.5;       // Softening factor for avoiding div by 0
constexpr double TIME_STEP = 25000.0;          // Time step in seconds
constexpr double COLLISION_THRESHOLD = 1.0e10; // Collision threshold distance

constexpr double MAX_DIST = 5.0e11;     // Maximum distance for initial distribution
constexpr double EARTH_MASS = 5.974e24; // Mass of Earth in kg
constexpr double EARTH_DIA = 12756.0;   // Diameter of Earth in km

constexpr int DEFAULT_BLOCK_SIZE = 256; // Default CUDA block size

// Definir macros para simplificar el código
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

inline bool checkCudaAvailability()
{
    std::cout << "Verificando disponibilidad de CUDA..." << std::endl;

    // Intentar obtener información sobre la versión de CUDA
    int cudaRuntimeVersion = 0;
    cudaError_t verErr = cudaRuntimeGetVersion(&cudaRuntimeVersion);
    if (verErr == cudaSuccess)
    {
        int major = cudaRuntimeVersion / 1000;
        int minor = (cudaRuntimeVersion % 1000) / 10;
        std::cout << "CUDA Runtime: " << major << "." << minor << std::endl;
    }
    else
    {
        std::cerr << "No se pudo obtener versión CUDA: " << cudaGetErrorString(verErr) << std::endl;
    }

    // Verificar cuántos dispositivos CUDA están disponibles
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess)
    {
        std::cerr << "ERROR CUDA: " << cudaGetErrorString(error) << std::endl;
        std::cerr << "No se pudo obtener el número de dispositivos CUDA." << std::endl;
        return false;
    }

    if (deviceCount == 0)
    {
        std::cerr << "No se encontraron dispositivos compatibles con CUDA" << std::endl;
        return false;
    }

    // Imprimir información de las GPUs disponibles
    std::cout << "Se encontraron " << deviceCount << " dispositivos CUDA:" << std::endl;
    cudaDeviceProp deviceProp;
    for (int i = 0; i < deviceCount; i++)
    {
        cudaError_t propErr = cudaGetDeviceProperties(&deviceProp, i);
        if (propErr != cudaSuccess)
        {
            std::cerr << "Error al obtener propiedades del dispositivo " << i << ": " << cudaGetErrorString(propErr) << std::endl;
            continue;
        }

        std::cout << "CUDA Device " << i << ": " << deviceProp.name << std::endl;
        std::cout << "  Capacidad de computación: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "  Memoria global total: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Multiprocesadores: " << deviceProp.multiProcessorCount << std::endl;
        std::cout << "  Tamaño máximo de bloque: " << deviceProp.maxThreadsPerBlock << std::endl;
    }

    // Establecer dispositivo a utilizar (el primero por defecto)
    cudaError_t setErr = cudaSetDevice(0);
    if (setErr != cudaSuccess)
    {
        std::cerr << "Error al establecer el dispositivo CUDA 0: " << cudaGetErrorString(setErr) << std::endl;
        return false;
    }

    // Verificar memoria disponible
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    std::cout << "Memoria GPU disponible: " << free / (1024 * 1024) << " MB de " << total / (1024 * 1024) << " MB" << std::endl;

    std::cout << "Inicialización CUDA completada correctamente" << std::endl;
    return true;
}

// Clase para medir tiempo en CUDA
class CudaTimer
{
private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
    float &elapsed_time_;
    bool stopped_;

public:
    CudaTimer(float &elapsed_time) : elapsed_time_(elapsed_time), stopped_(false)
    {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
        cudaEventRecord(start_, 0);
    }

    ~CudaTimer()
    {
        if (!stopped_)
        {
            stop();
        }
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void stop()
    {
        cudaEventRecord(stop_, 0);
        cudaEventSynchronize(stop_);
        cudaEventElapsedTime(&elapsed_time_, start_, stop_);
        stopped_ = true;
    }
};

// Macros para verificación de errores
#define CHECK_CUDA_ERROR(val) checkCudaError((val), #val, __FILE__, __LINE__)
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

bool dirExists(const std::string &dirName)
{
    struct stat info;
    return stat(dirName.c_str(), &info) == 0 && (info.st_mode & S_IFDIR);
}

// Función para crear un directorio
bool createDir(const std::string &dirName)
{
#ifdef _WIN32
    int status = mkdir(dirName.c_str());
#else
    int status = mkdir(dirName.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
#endif
    return status == 0;
}

// Función para asegurar que el directorio existe
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
    // Extraer el directorio del nombre de archivo
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
        file << "timestamp,method,bodies,steps,block_size,total_time_ms,avg_step_time_ms,force_calculation_time_ms,memory_transfer_time_ms,potential_energy,kinetic_energy,total_energy" << std::endl;
    }

    file.close();
    std::cout << "Archivo CSV " << (append ? "actualizado" : "inicializado") << ": " << filename << std::endl;
}

void saveMetrics(const std::string &filename,
                 int bodies,
                 int steps,
                 int blockSize,
                 double totalTime,
                 double forceCalculationTime,
                 double memoryTransferTime,
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

    double avgTimePerStep = totalTime / steps;

    file << timestamp.str() << ","
         << "GPU_Direct_Sum" << ","
         << bodies << ","
         << steps << ","
         << blockSize << ","
         << totalTime << ","
         << avgTimePerStep << ","
         << forceCalculationTime << ","
         << memoryTransferTime << ","
         << potentialEnergy << ","
         << kineticEnergy << ","
         << totalEnergy << std::endl;

    file.close();
    std::cout << "Métricas guardadas en: " << filename << std::endl;
}

class BodySystem
{
private:
    int nBodies;
    Body *h_bodies;
    Body *d_bodies;
    bool _isInitialized;
    unsigned int randomSeed;

    void initRandomBodies(Vector centerPos, MassDistribution massDist);

public:
    // Constructor
    BodySystem(int numBodies,
               BodyDistribution dist = BodyDistribution::RANDOM,
               unsigned int seed = static_cast<unsigned int>(time(nullptr)),
               MassDistribution massDist = MassDistribution::UNIFORM);

    ~BodySystem();
    void setup();
    void copyBodiesToDevice();
    void copyBodiesFromDevice();
    Body *getHostBodies() const { return h_bodies; }
    Body *getDeviceBodies() const { return d_bodies; }
    int getNumBodies() const { return nBodies; }
    bool isInitialized() const { return _isInitialized; }
    void initBodies(BodyDistribution dist, unsigned int seed, MassDistribution massDist);
};

BodySystem::BodySystem(int numBodies, BodyDistribution dist, unsigned int seed, MassDistribution massDist)
    : nBodies(numBodies), h_bodies(nullptr), d_bodies(nullptr), _isInitialized(false), randomSeed(seed)
{
    std::cout << "Creating BodySystem with " << numBodies << " bodies." << std::endl;

    h_bodies = new Body[numBodies];
    initBodies(dist, seed, massDist);
}

BodySystem::~BodySystem()
{
    if (h_bodies)
    {
        delete[] h_bodies;
        h_bodies = nullptr;
    }

    if (d_bodies)
    {
        cudaFree(d_bodies);
        d_bodies = nullptr;
    }
}

void BodySystem::setup()
{
    if (_isInitialized)
    {
        std::cout << "BodySystem already initialized." << std::endl;
        return;
    }

    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_bodies, nBodies * sizeof(Body)));
    copyBodiesToDevice();
    _isInitialized = true;
}

// Copy bodies from host to device
void BodySystem::copyBodiesToDevice()
{
    if (!d_bodies)
    {
        std::cerr << "Device memory not allocated. Call setup() first." << std::endl;
        return;
    }

    CHECK_CUDA_ERROR(cudaMemcpy(d_bodies, h_bodies, nBodies * sizeof(Body), cudaMemcpyHostToDevice));
}

void BodySystem::copyBodiesFromDevice()
{
    if (!d_bodies)
    {
        std::cerr << "Device memory not allocated. Call setup() first." << std::endl;
        return;
    }

    CHECK_CUDA_ERROR(cudaMemcpy(h_bodies, d_bodies, nBodies * sizeof(Body), cudaMemcpyDeviceToHost));
}

void BodySystem::initBodies(BodyDistribution dist, unsigned int seed, MassDistribution massDist)
{
    Vector centerPos = Vector(0, 0, 0);
    initRandomBodies(centerPos, massDist);
}

void BodySystem::initRandomBodies(Vector centerPos, MassDistribution massDist)
{
    std::mt19937 gen(randomSeed);
    std::uniform_real_distribution<double> posDist(-100.0, 100.0);
    std::uniform_real_distribution<double> velDist(-5.0, 5.0);
    std::normal_distribution<double> normalPosDist(0.0, 5.0);
    std::normal_distribution<double> normalVelDist(0.0, 2.5);

    for (int i = 0; i < nBodies; i++)
    {
        // Position and velocity based on distribution
        if (massDist == MassDistribution::UNIFORM)
        {
            // Position
            h_bodies[i].position.x = centerPos.x + posDist(gen);
            h_bodies[i].position.y = centerPos.y + posDist(gen);
            h_bodies[i].position.z = centerPos.z + posDist(gen);

            // Velocity
            h_bodies[i].velocity.x = velDist(gen);
            h_bodies[i].velocity.y = velDist(gen);
            h_bodies[i].velocity.z = velDist(gen);
        }
        else
        { // NORMAL distribution
            // Position
            h_bodies[i].position.x = centerPos.x + normalPosDist(gen);
            h_bodies[i].position.y = centerPos.y + normalPosDist(gen);
            h_bodies[i].position.z = centerPos.z + normalPosDist(gen);

            // Velocity
            h_bodies[i].velocity.x = normalVelDist(gen);
            h_bodies[i].velocity.y = normalVelDist(gen);
            h_bodies[i].velocity.z = normalVelDist(gen);
        }

        h_bodies[i].mass = 1.0;
        h_bodies[i].radius = pow(h_bodies[i].mass / EARTH_MASS, 1.0 / 3.0) * (EARTH_DIA / 2.0);
        h_bodies[i].isDynamic = true;
        h_bodies[i].acceleration = Vector(0, 0, 0);
    }
}

__global__ void DirectSumForceKernel(Body *bodies, int nBodies);
__global__ void CalculateEnergiesKernel(Body *bodies, int nBodies, double *d_potentialEnergy, double *d_kineticEnergy);

class DirectSumGPU
{
private:
    BodySystem *bodySystem;    // Pointer to the body system
    SimulationMetrics metrics; // Performance metrics
    bool firstKernelLaunch;    // Control first kernel launch info
    double potentialEnergy;
    double kineticEnergy;
    double totalEnergyAvg;
    double potentialEnergyAvg;
    double kineticEnergyAvg;

    // Device memory for energy calculations
    double *d_potentialEnergy;
    double *d_kineticEnergy;
    double *h_potentialEnergy;
    double *h_kineticEnergy;

    // Compute forces and update positions
    void computeForces();
    void calculateEnergies();
    void initializeEnergyData();
    void cleanupEnergyData();

public:
    DirectSumGPU(BodySystem *system);
    ~DirectSumGPU();

    void update();
    const SimulationMetrics &getMetrics() const { return metrics; }
    void run(int steps);
    double getTotalTime() const { return metrics.totalTimeMs; }
    double getForceCalculationTime() const { return metrics.forceTimeMs; }
    double getEnergyCalculationTime() const { return metrics.energyCalculationTimeMs; }
    double getPotentialEnergy() const { return potentialEnergy; }
    double getKineticEnergy() const { return kineticEnergy; }
    double getTotalEnergy() const { return potentialEnergy + kineticEnergy; }
    double getPotentialEnergyAvg() const { return potentialEnergyAvg; }
    double getKineticEnergyAvg() const { return kineticEnergyAvg; }
    double getTotalEnergyAvg() const { return totalEnergyAvg; }
    int getBlockSize() const { return g_blockSize; }
    int getNumBodies() const { return bodySystem->getNumBodies(); }
};

__global__ void DirectSumForceKernel(Body *bodies, int nBodies)
{
    extern __shared__ char sharedMemory[];
    Vector *sharedPos = (Vector *)sharedMemory;
    double *sharedMass = (double *)(sharedPos + blockDim.x);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;

    Vector myPos = Vector(0, 0, 0);
    Vector myVel = Vector(0, 0, 0);
    Vector myAcc = Vector(0, 0, 0);
    double myMass = 0.0;
    bool isDynamic = false;
    bool isValid = false;

    if (i < nBodies && bodies != nullptr)
    {
        myPos = bodies[i].position;
        myVel = bodies[i].velocity;
        myMass = bodies[i].mass;
        isDynamic = bodies[i].isDynamic;
        isValid = true;
    }

    const int tileSize = blockDim.x;

    for (int tile = 0; tile < (nBodies + tileSize - 1) / tileSize; ++tile)
    {
        int idx = tile * tileSize + tx;

        sharedPos[tx] = Vector(0, 0, 0);
        sharedMass[tx] = 0.0;
        if (idx < nBodies && bodies != nullptr)
        {
            sharedPos[tx] = bodies[idx].position;
            sharedMass[tx] = bodies[idx].mass;
        }

        __syncthreads();
        if (isValid && isDynamic)
        {
            int tileLimit = min(tileSize, nBodies - tile * tileSize);
            for (int j = 0; j < tileLimit; ++j)
            {
                int jBody = tile * tileSize + j;
                if (jBody != i && sharedMass[j] > 0.0)
                {
                    double rx = sharedPos[j].x - myPos.x;
                    double ry = sharedPos[j].y - myPos.y;
                    double rz = sharedPos[j].z - myPos.z;
                    double distSqr = rx * rx + ry * ry + rz * rz + E * E;
                    if (distSqr >= COLLISION_TH * COLLISION_TH)
                    {
                        double dist = sqrt(distSqr);
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

    if (isValid && isDynamic && bodies != nullptr)
    {
        bodies[i].acceleration = myAcc;

        myVel.x += myAcc.x * DT;
        myVel.y += myAcc.y * DT;
        myVel.z += myAcc.z * DT;
        bodies[i].velocity = myVel;

        myPos.x += myVel.x * DT;
        myPos.y += myVel.y * DT;
        myPos.z += myVel.z * DT;
        bodies[i].position = myPos;
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

DirectSumGPU::DirectSumGPU(BodySystem *system)
    : bodySystem(system), firstKernelLaunch(true),
      potentialEnergy(0.0), kineticEnergy(0.0),
      totalEnergyAvg(0.0), potentialEnergyAvg(0.0), kineticEnergyAvg(0.0),
      d_potentialEnergy(nullptr), d_kineticEnergy(nullptr),
      h_potentialEnergy(nullptr), h_kineticEnergy(nullptr)
{
    if (!bodySystem->isInitialized())
    {
        std::cout << "Initializing body system..." << std::endl;
        bodySystem->setup();
    }

    initializeEnergyData();
}

DirectSumGPU::~DirectSumGPU()
{
    cudaDeviceSynchronize();
    cleanupEnergyData();
}

void DirectSumGPU::initializeEnergyData()
{
    h_potentialEnergy = new double[1];
    h_kineticEnergy = new double[1];

    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_potentialEnergy, sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_kineticEnergy, sizeof(double)));
}

void DirectSumGPU::cleanupEnergyData()
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

void DirectSumGPU::calculateEnergies()
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

    size_t sharedMemSize = 2 * blockSize * sizeof(double); // For potential and kinetic energy

    CUDA_KERNEL_CALL(CalculateEnergiesKernel, gridSize, blockSize, sharedMemSize, 0,
                     d_bodies, nBodies, d_potentialEnergy, d_kineticEnergy);
    CHECK_CUDA_ERROR(cudaMemcpy(h_potentialEnergy, d_potentialEnergy, sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_kineticEnergy, d_kineticEnergy, sizeof(double), cudaMemcpyDeviceToHost));

    potentialEnergy = *h_potentialEnergy;
    kineticEnergy = *h_kineticEnergy;
}

void DirectSumGPU::computeForces()
{
    Body *d_bodies = bodySystem->getDeviceBodies();
    int nBodies = bodySystem->getNumBodies();

    if (d_bodies == nullptr)
    {
        std::cerr << "Error: Device bodies not initialized in computeForces" << std::endl;
        return;
    }

    cudaDeviceSynchronize();

    CudaTimer timer(metrics.forceTimeMs);

    int blockSize = g_blockSize;
    if (blockSize < 32)
        blockSize = 32;
    if (blockSize > 1024)
        blockSize = 1024;

    int gridSize = (nBodies + blockSize - 1) / blockSize;
    if (gridSize < 1)
        gridSize = 1;

    size_t sharedMemSize = blockSize * sizeof(Vector) + blockSize * sizeof(double);

    if (firstKernelLaunch)
    {
        std::cout << "Primera ejecución del kernel DirectSum GPU:" << std::endl;
        std::cout << "- Grid size: " << gridSize << std::endl;
        std::cout << "- Block size: " << blockSize << std::endl;
        std::cout << "- Shared memory: " << sharedMemSize << " bytes" << std::endl;
        std::cout << "- Cuerpos: " << nBodies << std::endl;

        size_t free, total;
        cudaMemGetInfo(&free, &total);
        std::cout << "- Memoria GPU disponible: " << free / (1024 * 1024) << " MB de "
                  << total / (1024 * 1024) << " MB" << std::endl;

        firstKernelLaunch = false;
    }

    CUDA_KERNEL_CALL(DirectSumForceKernel, gridSize, blockSize, sharedMemSize, 0, d_bodies, nBodies);
}

void DirectSumGPU::update()
{
    CudaTimer timer(metrics.totalTimeMs);
    computeForces();
    calculateEnergies();
    cudaDeviceSynchronize();
}

void DirectSumGPU::run(int steps)
{
    std::cout << "Running DirectSum GPU simulation for " << steps << " steps..." << std::endl;

    // Variables para medir tiempo
    float totalTime = 0.0f;
    float minTime = std::numeric_limits<float>::max();
    float maxTime = 0.0f;

    // Variables para calcular energía promedio
    double totalPotentialEnergy = 0.0;
    double totalKineticEnergy = 0.0;

    // Ejecutar simulación
    for (int step = 0; step < steps; step++)
    {
        update();

        // Actualizar estadísticas
        totalTime += metrics.totalTimeMs;
        minTime = std::min(minTime, metrics.totalTimeMs);
        maxTime = std::max(maxTime, metrics.totalTimeMs);

        // Acumular energías
        totalPotentialEnergy += potentialEnergy;
        totalKineticEnergy += kineticEnergy;
    }

    // Calcular promedios de energía
    potentialEnergyAvg = totalPotentialEnergy / steps;
    kineticEnergyAvg = totalKineticEnergy / steps;
    totalEnergyAvg = potentialEnergyAvg + kineticEnergyAvg;

    // Copiar resultados al host para verificación
    bodySystem->copyBodiesFromDevice();

    // Mostrar estadísticas
    std::cout << "Simulation complete." << std::endl;
    std::cout << "Average time per step: " << totalTime / steps << " ms" << std::endl;
    std::cout << "Min time: " << minTime << " ms" << std::endl;
    std::cout << "Max time: " << maxTime << " ms" << std::endl;
    std::cout << "Average Energy Values:" << std::endl;
    std::cout << "  Potential Energy: " << std::scientific << std::setprecision(6) << potentialEnergyAvg << std::endl;
    std::cout << "  Kinetic Energy:   " << std::scientific << std::setprecision(6) << kineticEnergyAvg << std::endl;
    std::cout << "  Total Energy:     " << std::scientific << std::setprecision(6) << totalEnergyAvg << std::endl;
}

void printUsage()
{
    std::cout << "Usage: directsum_gpu [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -n <num>       Number of bodies (default: 10000)" << std::endl;
    std::cout << "  -s <num>       Number of simulation steps (default: 100)" << std::endl;
    std::cout << "  -b <num>       Block size for CUDA kernels (default: 256)" << std::endl;
    std::cout << "  -d <dist>      Body distribution: random, solar, galaxy, sphere (default: random)" << std::endl;
    std::cout << "  -m <dist>      Mass distribution: uniform, normal (default: uniform)" << std::endl;
    std::cout << "  -seed <num>    Random seed (default: time-based)" << std::endl;
    std::cout << "  -h, --help     Show this help message" << std::endl;
}

int main(int argc, char **argv)
{
    int numBodies = 10000;
    int numSteps = 100;
    BodyDistribution bodyDist = BodyDistribution::RANDOM;
    MassDistribution massDist = MassDistribution::UNIFORM;
    unsigned int seed = static_cast<unsigned int>(time(nullptr));

    bool saveMetricsToFile = false;
    std::string metricsFile = "./DirectSumGPU_metrics.csv";

    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help")
        {
            printUsage();
            return 0;
        }
        else if (arg == "-n" && i + 1 < argc)
        {
            numBodies = std::atoi(argv[++i]);
            if (numBodies <= 0)
            {
                std::cerr << "Error: Number of bodies must be positive" << std::endl;
                return 1;
            }
        }
        else if (arg == "-s" && i + 1 < argc)
        {
            numSteps = std::atoi(argv[++i]);
            if (numSteps <= 0)
            {
                std::cerr << "Error: Number of steps must be positive" << std::endl;
                return 1;
            }
        }
        else if (arg == "-b" && i + 1 < argc)
        {
            g_blockSize = std::atoi(argv[++i]);
            if (g_blockSize <= 0)
            {
                std::cerr << "Error: Block size must be positive" << std::endl;
                return 1;
            }
        }
        else if (arg == "-d" && i + 1 < argc)
        {
            std::string distStr = argv[++i];
            if (distStr == "random")
                bodyDist = BodyDistribution::RANDOM;
        }
        else if (arg == "-m" && i + 1 < argc)
        {
            std::string distStr = argv[++i];
            if (distStr == "normal")
                massDist = MassDistribution::NORMAL;
            else if (distStr == "uniform")
                massDist = MassDistribution::UNIFORM;
            else
            {
                std::cerr << "Error: Unknown mass distribution: " << distStr << std::endl;
                return 1;
            }
        }
        else if (arg == "-seed" && i + 1 < argc)
        {
            seed = static_cast<unsigned int>(std::atoi(argv[++i]));
        }
        else if (arg == "--save-metrics")
        {
            saveMetricsToFile = true;
        }
        else if (arg == "--metrics-file" && i + 1 < argc)
        {
            metricsFile = argv[++i];
        }
        else
        {
            std::cerr << "Error: Unknown option: " << arg << std::endl;
            printUsage();
            return 1;
        }
    }

    std::cout << "DirectSum GPU Algorithm" << std::endl;
    std::cout << "=======================" << std::endl;

    if (!checkCudaAvailability())
    {
        std::cerr << "CUDA is not available or initialization failed." << std::endl;
        return 1;
    }

    try
    {
        std::cout << "Creating body system with " << numBodies << " bodies..." << std::endl;
        BodySystem bodySystem(numBodies, bodyDist, seed, massDist);

        std::cout << "Initializing DirectSum GPU simulation..." << std::endl;
        DirectSumGPU simulation(&bodySystem);

        simulation.run(numSteps);

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
                numSteps,
                simulation.getBlockSize(),
                simulation.getTotalTime(),
                simulation.getForceCalculationTime(),
                simulation.getEnergyCalculationTime(),
                simulation.getPotentialEnergyAvg(),
                simulation.getKineticEnergyAvg(),
                simulation.getTotalEnergyAvg());

            std::cout << "Métricas guardadas en: " << metricsFile << std::endl;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}