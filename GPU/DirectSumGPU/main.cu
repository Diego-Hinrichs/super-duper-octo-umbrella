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
 * @brief Performance metrics for simulation timing
 */
struct SimulationMetrics
{
    float forceTimeMs;
    float totalTimeMs;

    SimulationMetrics() : forceTimeMs(0.0f),
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

/**
 * @brief Check if CUDA is available and print device information
 * @return True if CUDA is available, false otherwise
 */
inline bool checkCudaAvailability()
{
    std::cout << "Verificando disponibilidad de CUDA..." << std::endl;

    // Intentar obtener información sobre la versión de CUDA
    int cudaRuntimeVersion = 0;
    cudaError_t verErr = cudaRuntimeGetVersion(&cudaRuntimeVersion);
    if (verErr == cudaSuccess) {
        int major = cudaRuntimeVersion / 1000;
        int minor = (cudaRuntimeVersion % 1000) / 10;
        std::cout << "CUDA Runtime: " << major << "." << minor << std::endl;
    } else {
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
        if (propErr != cudaSuccess) {
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
    if (setErr != cudaSuccess) {
        std::cerr << "Error al establecer el dispositivo CUDA 0: " << cudaGetErrorString(setErr) << std::endl;
        return false;
    }
    
    // Verificar memoria disponible
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    std::cout << "Memoria GPU disponible: " << free / (1024*1024) << " MB de " << total / (1024*1024) << " MB" << std::endl;
    
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

// Macro para llamar a un kernel CUDA con verificación de errores
#define CUDA_KERNEL_CALL(kernel, gridSize, blockSize, sharedMem, stream, ...) \
    do                                                                        \
    {                                                                         \
        kernel<<<gridSize, blockSize, sharedMem, stream>>>(__VA_ARGS__);      \
        CHECK_LAST_CUDA_ERROR();                                              \
    } while (0) 

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
        file << "timestamp,method,bodies,steps,block_size,total_time_ms,avg_step_time_ms,force_calculation_time_ms,memory_transfer_time_ms" << std::endl;
    }
    
    file.close();
    std::cout << "Archivo CSV " << (append ? "actualizado" : "inicializado") << ": " << filename << std::endl;
}

void saveMetrics(const std::string& filename, 
                int bodies, 
                int steps, 
                int blockSize,
                double totalTime, 
                double forceCalculationTime,
                double memoryTransferTime) {
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
         << "GPU_Direct_Sum" << ","
         << bodies << ","
         << steps << ","
         << blockSize << ","
         << totalTime << ","
         << avgTimePerStep << ","
         << forceCalculationTime << ","
         << memoryTransferTime << std::endl;
    
    file.close();
    std::cout << "Métricas guardadas en: " << filename << std::endl;
}

// =============================================================================
// CLASE BODYSYSTEM PARA GESTIONAR LOS CUERPOS
// =============================================================================

class BodySystem
{
private:
    int nBodies;            // Number of bodies in the simulation
    Body *h_bodies;         // Host bodies array
    Body *d_bodies;         // Device bodies array
    bool _isInitialized;    // Flag to track initialization status
    unsigned int randomSeed;// Seed for random number generation
    
    // Initialize bodies with random distribution
    void initRandomBodies(Vector centerPos, MassDistribution massDist);
    
    // Initialize bodies simulating a solar system
    void initSolarSystem(Vector centerPos, MassDistribution massDist);
    
    // Initialize bodies simulating a galaxy
    void initGalaxy(Vector centerPos, MassDistribution massDist);
    
    // Initialize bodies in a uniform sphere
    void initUniformSphere(Vector centerPos, MassDistribution massDist);

public:
    // Constructor
    BodySystem(int numBodies, 
               BodyDistribution dist = BodyDistribution::RANDOM,
               unsigned int seed = static_cast<unsigned int>(time(nullptr)),
               MassDistribution massDist = MassDistribution::UNIFORM);
    
    // Destructor
    ~BodySystem();
    
    // Setup the system
    void setup();
    
    // Copy bodies between host and device
    void copyBodiesToDevice();
    void copyBodiesFromDevice();
    
    // Get pointers to the bodies
    Body* getHostBodies() const { return h_bodies; }
    Body* getDeviceBodies() const { return d_bodies; }
    
    // Get number of bodies
    int getNumBodies() const { return nBodies; }
    
    // Check if the system is initialized
    bool isInitialized() const { return _isInitialized; }
    
    // Initialize with specific distribution
    void initBodies(BodyDistribution dist, 
                    unsigned int seed, 
                    MassDistribution massDist);
};

// Constructor
BodySystem::BodySystem(int numBodies, BodyDistribution dist, unsigned int seed, MassDistribution massDist)
    : nBodies(numBodies), h_bodies(nullptr), d_bodies(nullptr), _isInitialized(false), randomSeed(seed)
{
    std::cout << "Creating BodySystem with " << numBodies << " bodies." << std::endl;
    
    // Allocate host memory
    h_bodies = new Body[numBodies];
    
    // Initialize bodies based on distribution
    initBodies(dist, seed, massDist);
}

// Destructor
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

// Setup the system
void BodySystem::setup()
{
    if (_isInitialized)
    {
        std::cout << "BodySystem already initialized." << std::endl;
        return;
    }
    
    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_bodies, nBodies * sizeof(Body)));
    
    // Copy bodies to device
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

// Copy bodies from device to host
void BodySystem::copyBodiesFromDevice()
{
    if (!d_bodies)
    {
        std::cerr << "Device memory not allocated. Call setup() first." << std::endl;
        return;
    }
    
    CHECK_CUDA_ERROR(cudaMemcpy(h_bodies, d_bodies, nBodies * sizeof(Body), cudaMemcpyDeviceToHost));
}

// Initialize bodies based on distribution
void BodySystem::initBodies(BodyDistribution dist, unsigned int seed, MassDistribution massDist)
{
    // Always use random bodies distribution regardless of requested distribution type
    // This simplified version only supports random initialization for now
    Vector centerPos = Vector(0, 0, 0);
    initRandomBodies(centerPos, massDist);
}

// Initialize random bodies
void BodySystem::initRandomBodies(Vector centerPos, MassDistribution massDist)
{
    std::mt19937 gen(randomSeed);
    std::uniform_real_distribution<double> posDist(-MAX_DIST, MAX_DIST);
    std::uniform_real_distribution<double> velDist(-1.0e3, 1.0e3);
    std::normal_distribution<double> normalPosDist(0.0, MAX_DIST/2.0);
    std::normal_distribution<double> normalVelDist(0.0, 5.0e2);
    
    for (int i = 0; i < nBodies; i++)
    {
        // Position and velocity based on distribution
        if (massDist == MassDistribution::UNIFORM) {
            // Position
            h_bodies[i].position.x = centerPos.x + posDist(gen);
            h_bodies[i].position.y = centerPos.y + posDist(gen);
            h_bodies[i].position.z = centerPos.z + posDist(gen);
            
            // Velocity
            h_bodies[i].velocity.x = velDist(gen);
            h_bodies[i].velocity.y = velDist(gen);
            h_bodies[i].velocity.z = velDist(gen);
        } else { // NORMAL distribution
            // Position
            h_bodies[i].position.x = centerPos.x + normalPosDist(gen);
            h_bodies[i].position.y = centerPos.y + normalPosDist(gen);
            h_bodies[i].position.z = centerPos.z + normalPosDist(gen);
            
            // Velocity
            h_bodies[i].velocity.x = normalVelDist(gen);
            h_bodies[i].velocity.y = normalVelDist(gen);
            h_bodies[i].velocity.z = normalVelDist(gen);
        }
        
        // Mass and radius always the same
        h_bodies[i].mass = 1.0;
        h_bodies[i].radius = pow(h_bodies[i].mass / EARTH_MASS, 1.0/3.0) * (EARTH_DIA/2.0);
        h_bodies[i].isDynamic = true;
        h_bodies[i].acceleration = Vector(0, 0, 0);
    }
}

// Initialize solar system-like distribution
void BodySystem::initSolarSystem(Vector centerPos, MassDistribution massDist)
{
    std::mt19937 gen(randomSeed);
    std::uniform_real_distribution<double> angleDist(0, 2.0 * M_PI);
    std::uniform_real_distribution<double> radiusDist(MIN_DIST, MAX_DIST);
    std::uniform_real_distribution<double> planetMassDist(0.05 * EARTH_MASS, 20.0 * EARTH_MASS);
    
    // Create a sun at the center
    h_bodies[0].position = centerPos;
    h_bodies[0].velocity = Vector(0, 0, 0);
    h_bodies[0].mass = SUN_MASS;
    h_bodies[0].radius = SUN_DIA / 2.0;
    h_bodies[0].isDynamic = false; // The sun doesn't move
    h_bodies[0].acceleration = Vector(0, 0, 0);
    
    // Create planets in orbit
    for (int i = 1; i < nBodies; i++)
    {
        double angle = angleDist(gen);
        double orbitRadius = radiusDist(gen);
        double mass = planetMassDist(gen);
        
        // Position in orbital plane (roughly xy-plane with some z variation)
        h_bodies[i].position.x = centerPos.x + orbitRadius * cos(angle);
        h_bodies[i].position.y = centerPos.y + orbitRadius * sin(angle);
        h_bodies[i].position.z = centerPos.z + orbitRadius * sin(angle) * 0.1; // Small z-variation
        
        // Calculate orbital velocity (perpendicular to radius)
        double orbitalSpeed = sqrt(GRAVITY * SUN_MASS / orbitRadius);
        h_bodies[i].velocity.x = -orbitalSpeed * sin(angle);
        h_bodies[i].velocity.y = orbitalSpeed * cos(angle);
        h_bodies[i].velocity.z = 0;
        
        h_bodies[i].mass = mass;
        h_bodies[i].radius = pow(mass / EARTH_MASS, 1.0/3.0) * (EARTH_DIA/2.0);
        h_bodies[i].isDynamic = true;
        h_bodies[i].acceleration = Vector(0, 0, 0);
    }
}

// Initialize galaxy-like distribution
void BodySystem::initGalaxy(Vector centerPos, MassDistribution massDist)
{
    std::mt19937 gen(randomSeed);
    std::uniform_real_distribution<double> angleDist(0, 2.0 * M_PI);
    std::uniform_real_distribution<double> radiusDist(0.1 * MAX_DIST, MAX_DIST);
    std::normal_distribution<double> heightDist(0, 0.1 * MAX_DIST);
    std::uniform_real_distribution<double> uniformMass(0.1 * EARTH_MASS, 10.0 * EARTH_MASS);
    
    // Create a central black hole
    h_bodies[0].position = centerPos;
    h_bodies[0].velocity = Vector(0, 0, 0);
    h_bodies[0].mass = 1000.0 * SUN_MASS; // Supermassive black hole
    h_bodies[0].radius = 10.0 * SUN_DIA / 2.0;
    h_bodies[0].isDynamic = false; // Black hole doesn't move
    h_bodies[0].acceleration = Vector(0, 0, 0);
    
    // Create stars in a spiral galaxy
    for (int i = 1; i < nBodies; i++)
    {
        double angle = angleDist(gen);
        double radius = radiusDist(gen);
        
        // Add spiral arms by modifying angle based on radius
        angle += 0.5 * radius / MAX_DIST * 2.0 * M_PI * 2.0; // 2 spiral arms
        
        // Position in galactic plane with height variation
        h_bodies[i].position.x = centerPos.x + radius * cos(angle);
        h_bodies[i].position.y = centerPos.y + radius * sin(angle);
        h_bodies[i].position.z = centerPos.z + heightDist(gen);
        
        // Calculate orbital velocity with rotational curve that flattens at large radii
        double vFactor = (radius < 0.1 * MAX_DIST) ? sqrt(radius / (0.1 * MAX_DIST)) : 1.0;
        double orbitalSpeed = vFactor * sqrt(GRAVITY * h_bodies[0].mass / (0.1 * MAX_DIST));
        
        h_bodies[i].velocity.x = -orbitalSpeed * sin(angle);
        h_bodies[i].velocity.y = orbitalSpeed * cos(angle);
        h_bodies[i].velocity.z = 0;
        
        h_bodies[i].mass = uniformMass(gen);
        h_bodies[i].radius = pow(h_bodies[i].mass / EARTH_MASS, 1.0/3.0) * (EARTH_DIA/2.0);
        h_bodies[i].isDynamic = true;
        h_bodies[i].acceleration = Vector(0, 0, 0);
    }
}

// Initialize uniform sphere distribution
void BodySystem::initUniformSphere(Vector centerPos, MassDistribution massDist)
{
    std::mt19937 gen(randomSeed);
    std::uniform_real_distribution<double> uniformDist(0.0, 1.0);
    std::uniform_real_distribution<double> uniformMass(0.1 * EARTH_MASS, 10.0 * EARTH_MASS);
    
    for (int i = 0; i < nBodies; i++)
    {
        // Generate points uniformly within a sphere
        double theta = 2.0 * M_PI * uniformDist(gen);
        double phi = acos(2.0 * uniformDist(gen) - 1.0);
        double r = MAX_DIST * pow(uniformDist(gen), 1.0/3.0); // Cube root for uniform volume
        
        h_bodies[i].position.x = centerPos.x + r * sin(phi) * cos(theta);
        h_bodies[i].position.y = centerPos.y + r * sin(phi) * sin(theta);
        h_bodies[i].position.z = centerPos.z + r * cos(phi);
        
        // Simple velocity model: orbit around center
        double orbitalSpeed = sqrt(GRAVITY * 1000.0 * SUN_MASS / r);
        h_bodies[i].velocity.x = orbitalSpeed * sin(phi) * sin(theta);
        h_bodies[i].velocity.y = -orbitalSpeed * sin(phi) * cos(theta);
        h_bodies[i].velocity.z = 0;
        
        h_bodies[i].mass = uniformMass(gen);
        h_bodies[i].radius = pow(h_bodies[i].mass / EARTH_MASS, 1.0/3.0) * (EARTH_DIA/2.0);
        h_bodies[i].isDynamic = true;
        h_bodies[i].acceleration = Vector(0, 0, 0);
    }
}

// =============================================================================
// IMPLEMENTACIÓN DEL ALGORITMO DIRECT SUM EN GPU
// =============================================================================

// Declaración adelantada del kernel
__global__ void DirectSumForceKernel(Body *bodies, int nBodies);

// Clase para gestionar la simulación DirectSum GPU
class DirectSumGPU
{
private:
    BodySystem *bodySystem;      // Pointer to the body system
    SimulationMetrics metrics;   // Performance metrics
    bool firstKernelLaunch;      // Control first kernel launch info
    
    // Compute forces and update positions
    void computeForces();

public:
    // Constructor
    DirectSumGPU(BodySystem *system);
    
    // Destructor
    ~DirectSumGPU();
    
    // Update simulation for one step
    void update();
    
    // Get performance metrics
    const SimulationMetrics& getMetrics() const { return metrics; }
    
    // Run simulation for n steps
    void run(int steps);

    // Añadir estos getters
    double getTotalTime() const { return metrics.totalTimeMs; }
    double getForceCalculationTime() const { return metrics.forceTimeMs; }
    int getBlockSize() const { return g_blockSize; }
    int getNumBodies() const { return bodySystem->getNumBodies(); }
};

// Kernel para calcular fuerzas y actualizar posiciones
__global__ void DirectSumForceKernel(Body *bodies, int nBodies)
{
  extern __shared__ char sharedMemory[];
  Vector *sharedPos = (Vector *)sharedMemory;
  double *sharedMass = (double *)(sharedPos + blockDim.x);

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int tx = threadIdx.x;

  // Cargar datos solo si es un índice válido para reducir divergencia
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

  // Procesar todos los tiles en orden para mejor localidad de memoria
  for (int tile = 0; tile < (nBodies + tileSize - 1) / tileSize; ++tile)
  {
    // Cargar este tile a memoria compartida
    int idx = tile * tileSize + tx;

    // Inicializar memoria compartida a valores por defecto
    sharedPos[tx] = Vector(0, 0, 0);
    sharedMass[tx] = 0.0;

    // Solo cargar datos válidos a memoria compartida
    if (idx < nBodies && bodies != nullptr)
    {
      sharedPos[tx] = bodies[idx].position;
      sharedMass[tx] = bodies[idx].mass;
    }

    __syncthreads();

    // Pre-comprobar si necesitamos calcular fuerzas para reducir divergencia
    if (isValid && isDynamic)
    {
      // Limitar el bucle al tamaño real del tile
      int tileLimit = min(tileSize, nBodies - tile * tileSize);

      for (int j = 0; j < tileLimit; ++j)
      {
        int jBody = tile * tileSize + j;

        // Evitar auto-interacción y solo considerar cuerpos con masa
        if (jBody != i && sharedMass[j] > 0.0)
        {
          // Vector de distancia
          double rx = sharedPos[j].x - myPos.x;
          double ry = sharedPos[j].y - myPos.y;
          double rz = sharedPos[j].z - myPos.z;

          // Distancia al cuadrado con suavizado
          double distSqr = rx * rx + ry * ry + rz * rz + E * E;

          // Optimización: solo calcular sqrt si es necesario
          if (distSqr >= COLLISION_TH * COLLISION_TH)
          {
            double dist = sqrt(distSqr);
            double forceMag = (GRAVITY * myMass * sharedMass[j]) / (dist * distSqr);

            // Acumular aceleración
            myAcc.x += rx * forceMag / myMass;
            myAcc.y += ry * forceMag / myMass;
            myAcc.z += rz * forceMag / myMass;
          }
        }
      }
    }

    __syncthreads();
  }

  // Actualizar el cuerpo solo si es válido y dinámico para reducir operaciones de memoria
  if (isValid && isDynamic && bodies != nullptr)
  {
    // Guardar aceleración
    bodies[i].acceleration = myAcc;

    // Actualizar velocidad
    myVel.x += myAcc.x * DT;
    myVel.y += myAcc.y * DT;
    myVel.z += myAcc.z * DT;
    bodies[i].velocity = myVel;

    // Actualizar posición
    myPos.x += myVel.x * DT;
    myPos.y += myVel.y * DT;
    myPos.z += myVel.z * DT;
    bodies[i].position = myPos;
  }
}

// Constructor
DirectSumGPU::DirectSumGPU(BodySystem *system)
    : bodySystem(system), firstKernelLaunch(true)
{
    if (!bodySystem->isInitialized())
    {
        std::cout << "Initializing body system..." << std::endl;
        bodySystem->setup();
    }
}

// Destructor
DirectSumGPU::~DirectSumGPU()
{
    // Asegurar que todos los recursos CUDA se liberan correctamente
    cudaDeviceSynchronize();
}

// Compute forces - implementación del método principal
void DirectSumGPU::computeForces()
{
    Body *d_bodies = bodySystem->getDeviceBodies();
    int nBodies = bodySystem->getNumBodies();
    
    // Verificar inicialización
    if (d_bodies == nullptr)
    {
        std::cerr << "Error: Device bodies not initialized in computeForces" << std::endl;
        return;
    }

    // Sincronizar el dispositivo antes de medir tiempos
    cudaDeviceSynchronize();

    // Medir tiempo de ejecución
    CudaTimer timer(metrics.forceTimeMs);

    // Configurar tamaño de bloque
    int blockSize = g_blockSize;
    // Asegurar que blockSize sea múltiplo de 32 (tamaño del warp)
    blockSize = (blockSize / 32) * 32;
    if (blockSize < 32) blockSize = 32;
    if (blockSize > 1024) blockSize = 1024;

    // Calcular gridSize según el número de cuerpos
    int gridSize = (nBodies + blockSize - 1) / blockSize;
    if (gridSize < 1) gridSize = 1;

    // Calcular memoria compartida requerida
    size_t sharedMemSize = blockSize * sizeof(Vector) + blockSize * sizeof(double);

    // Solo mostrar información detallada del kernel en la primera ejecución
    if (firstKernelLaunch)
    {
        std::cout << "Primera ejecución del kernel DirectSum GPU:" << std::endl;
        std::cout << "- Grid size: " << gridSize << std::endl;
        std::cout << "- Block size: " << blockSize << std::endl;
        std::cout << "- Shared memory: " << sharedMemSize << " bytes" << std::endl;
        std::cout << "- Cuerpos: " << nBodies << std::endl;
        
        // Verificar memoria disponible
        size_t free, total;
        cudaMemGetInfo(&free, &total);
        std::cout << "- Memoria GPU disponible: " << free / (1024*1024) << " MB de " 
                  << total / (1024*1024) << " MB" << std::endl;
        
        firstKernelLaunch = false;
    }
    
    // Lanzar kernel con verificación de errores
    CUDA_KERNEL_CALL(DirectSumForceKernel, gridSize, blockSize, sharedMemSize, 0, d_bodies, nBodies);
}

// Update - actualizar un paso de simulación
void DirectSumGPU::update()
{
    // Medir tiempo de ejecución total
    CudaTimer timer(metrics.totalTimeMs);

    // Compute forces and update positions in one kernel
    computeForces();
    
    // Sincronizar al final de la actualización
    cudaDeviceSynchronize();
}

// Run - ejecutar la simulación para n pasos
void DirectSumGPU::run(int steps)
{
    std::cout << "Running DirectSum GPU simulation for " << steps << " steps..." << std::endl;
    
    // Variables para medir tiempo
    float totalTime = 0.0f;
    float minTime = std::numeric_limits<float>::max();
    float maxTime = 0.0f;
    
    // Ejecutar simulación
    for (int step = 0; step < steps; step++)
    {
        update();
        
        // Actualizar estadísticas
        totalTime += metrics.totalTimeMs;
        minTime = std::min(minTime, metrics.totalTimeMs);
        maxTime = std::max(maxTime, metrics.totalTimeMs);
        
        // Mostrar progreso cada 10% o al menos cada 10 pasos
        if (step % std::max(steps / 10, 10) == 0 || step == steps - 1)
        {
            std::cout << "Step " << step + 1 << "/" << steps 
                      << " - Time: " << metrics.totalTimeMs << " ms" << std::endl;
        }
    }
    
    // Copiar resultados al host para verificación
    bodySystem->copyBodiesFromDevice();
    
    // Mostrar estadísticas
    std::cout << "Simulation complete." << std::endl;
    std::cout << "Average time per step: " << totalTime / steps << " ms" << std::endl;
    std::cout << "Min time: " << minTime << " ms" << std::endl;
    std::cout << "Max time: " << maxTime << " ms" << std::endl;
}

// =============================================================================
// FUNCIÓN PRINCIPAL
// =============================================================================

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
    // Parámetros por defecto
    int numBodies = 10000;
    int numSteps = 100;
    BodyDistribution bodyDist = BodyDistribution::RANDOM;
    MassDistribution massDist = MassDistribution::UNIFORM;
    unsigned int seed = static_cast<unsigned int>(time(nullptr));
    
    // Añadir nuevas variables para métricas
    bool saveMetricsToFile = false;
    std::string metricsFile = "./DirectSumGPU_metrics.csv";
    
    // Procesar argumentos de la línea de comandos
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
            if (distStr == "solar")
                bodyDist = BodyDistribution::SOLAR_SYSTEM;
            else if (distStr == "galaxy")
                bodyDist = BodyDistribution::GALAXY;
            else if (distStr == "sphere")
                bodyDist = BodyDistribution::UNIFORM_SPHERE;
            else if (distStr == "random")
                bodyDist = BodyDistribution::RANDOM;
            else
            {
                std::cerr << "Error: Unknown body distribution: " << distStr << std::endl;
                return 1;
            }
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
        else if (arg == "--save-metrics") {
            saveMetricsToFile = true;
        } else if (arg == "--metrics-file" && i + 1 < argc) {
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
    
    // Verificar disponibilidad de CUDA
    if (!checkCudaAvailability())
    {
        std::cerr << "CUDA is not available or initialization failed." << std::endl;
        return 1;
    }
    
    try
    {
        // Crear sistema de cuerpos
        std::cout << "Creating body system with " << numBodies << " bodies..." << std::endl;
        BodySystem bodySystem(numBodies, bodyDist, seed, massDist);
        
        // Inicializar simulación DirectSum GPU
        std::cout << "Initializing DirectSum GPU simulation..." << std::endl;
        DirectSumGPU simulation(&bodySystem);
        
        // Ejecutar simulación
        simulation.run(numSteps);
        
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
                numSteps,
                simulation.getBlockSize(),
                simulation.getTotalTime(),
                simulation.getForceCalculationTime(),
                simulation.getTotalTime() - simulation.getForceCalculationTime()
            );
            
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