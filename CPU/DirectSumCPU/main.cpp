#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <chrono>
#include <random>
#include <string>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <sys/stat.h>  // Para verificar/crear directorios

struct Vector {
    double x;
    double y;
    double z;

    Vector() : x(0.0), y(0.0), z(0.0) {}

    Vector(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    Vector operator+(const Vector &other) const { return Vector(x + other.x, y + other.y, z + other.z); }

    Vector operator-(const Vector &other) const { return Vector(x - other.x, y - other.y, z - other.z); }

    Vector operator*(double scalar) const { return Vector(x * scalar, y * scalar, z * scalar); }

    double dot(const Vector &other) const { return x * other.x + y * other.y + z * other.z; }

    double lengthSquared() const { return x * x + y * y + z * z; }

    double length() const { return sqrt(lengthSquared()); }
};

struct Body {
    Vector position;
    Vector velocity;
    Vector acceleration;
    double mass;
    bool isDynamic;

    Body() : mass(1.0), isDynamic(true) {}
    
    Body(const Vector& pos, const Vector& vel, double m, bool dynamic = true)
        : position(pos), velocity(vel), acceleration(), mass(m), isDynamic(dynamic) {}
};

constexpr double GRAVITY = 6.67430e-11;   // Gravitational constant
constexpr double DT = 0.005;              // Time step
constexpr double E = 0.01;                // Softening parameter
constexpr double COLLISION_TH = 0.01;     // Collision threshold

enum class BodyDistribution {
    RANDOM_UNIFORM,
    SOLAR_SYSTEM,
    GALAXY,
    COLLISION
};

enum class MassDistribution {
    UNIFORM,
    NORMAL
};

bool dirExists(const std::string& dirName) {
    struct stat info;
    return stat(dirName.c_str(), &info) == 0 && (info.st_mode & S_IFDIR);
}
bool createDir(const std::string& dirName) {
    #ifdef _WIN32
    int status = mkdir(dirName.c_str());
    #else
    int status = mkdir(dirName.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    #endif
    return status == 0;
}

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
        file << "timestamp,method,bodies,steps,threads,total_time_ms,avg_step_time_ms,force_calculation_time_ms" << std::endl;
    }
    
    file.close();
    std::cout << "Archivo CSV " << (append ? "actualizado" : "inicializado") << ": " << filename << std::endl;
}

void saveMetrics(const std::string& filename, 
                int bodies, 
                int steps, 
                int threads, 
                double totalTime, 
                double forceCalculationTime) {
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
         << "CPU_Direct_Sum" << ","
         << bodies << ","
         << steps << ","
         << threads << ","
         << totalTime << ","
         << avgTimePerStep << ","
         << forceCalculationTime << std::endl;
    
    file.close();
    std::cout << "Métricas guardadas en: " << filename << std::endl;
}

class CPUDirectSum {
private:
    std::vector<Body> bodies;
    int nBodies;
    bool useOpenMP;
    int numThreads;
    double totalTime;
    double forceCalculationTime;
    
public:
    CPUDirectSum(
        int numBodies,
        bool useParallelization = true,
        int threads = 0,
        BodyDistribution dist = BodyDistribution::RANDOM_UNIFORM,
        unsigned int seed = static_cast<unsigned int>(time(nullptr)),
        MassDistribution massDist = MassDistribution::UNIFORM
    ) : nBodies(numBodies), useOpenMP(useParallelization), totalTime(0.0), forceCalculationTime(0.0) {
        // Initialize thread count
        if (threads <= 0) {
            // Auto-detect number of available threads
            numThreads = omp_get_max_threads();
        } else {
            numThreads = threads;
        }
        
        // Initialize bodies based on the distribution
        bodies.resize(numBodies);
        initializeBodies(dist, seed, massDist);
        
        // Log configuration
        std::cout << "CPU Direct Sum Simulation created with " << numBodies << " bodies." << std::endl;
        if (useOpenMP) {
            std::cout << "OpenMP enabled with " << numThreads << " threads." << std::endl;
        } else {
            std::cout << "OpenMP disabled, using single-threaded mode." << std::endl;
        }
    }
    
    void initializeBodies(BodyDistribution dist, unsigned int seed, MassDistribution massDist) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> posDistrib(-100.0, 100.0);
        std::uniform_real_distribution<double> velDistrib(-5.0, 5.0);
        std::normal_distribution<double> normalPosDistrib(0.0, 50.0);
        std::normal_distribution<double> normalVelDistrib(0.0, 2.5);
        
        // Initialize with random uniform distribution regardless of requested distribution type
        // This simplified version only supports random initialization for now
        for (int i = 0; i < nBodies; i++) {
            if (massDist == MassDistribution::UNIFORM) {
                bodies[i].position = Vector(posDistrib(rng), posDistrib(rng), posDistrib(rng));
                bodies[i].velocity = Vector(velDistrib(rng), velDistrib(rng), velDistrib(rng));
            } else { // NORMAL distribution
                bodies[i].position = Vector(normalPosDistrib(rng), normalPosDistrib(rng), normalPosDistrib(rng));
                bodies[i].velocity = Vector(normalVelDistrib(rng), normalVelDistrib(rng), normalVelDistrib(rng));
            }
            
            // Establecer masa 1.0 para todos los cuerpos
            bodies[i].mass = 1.0;
            bodies[i].isDynamic = true;
        }
    }
    
    void computeForces() {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Compute forces using direct summation (O(n²) complexity)
        if (useOpenMP) {
            // Set the number of threads
            omp_set_num_threads(numThreads);

            #pragma omp parallel for
            for (int i = 0; i < nBodies; i++) {
                // Skip non-dynamic bodies
                if (!bodies[i].isDynamic)
                    continue;

                // Reset acceleration
                bodies[i].acceleration = Vector(0.0, 0.0, 0.0);

                // Compute force from all other bodies
                for (int j = 0; j < nBodies; j++) {
                    if (i == j)
                        continue; // Skip self-interaction

                    // Vector from body i to body j
                    Vector r = bodies[j].position - bodies[i].position;

                    // Distance calculation with softening
                    double distSqr = r.lengthSquared() + (E * E);
                    double dist = sqrt(distSqr);

                    // Skip if bodies are too close (collision)
                    if (dist < COLLISION_TH)
                        continue;

                    // Gravitational force: G * m1 * m2 / r^3 * r_vector
                    double forceMag = GRAVITY * bodies[i].mass * bodies[j].mass / (distSqr * dist);

                    // Update acceleration (F = ma, so a = F/m)
                    bodies[i].acceleration.x += (r.x * forceMag) / bodies[i].mass;
                    bodies[i].acceleration.y += (r.y * forceMag) / bodies[i].mass;
                    bodies[i].acceleration.z += (r.z * forceMag) / bodies[i].mass;
                }

                // Update velocity (Euler integration)
                bodies[i].velocity.x += bodies[i].acceleration.x * DT;
                bodies[i].velocity.y += bodies[i].acceleration.y * DT;
                bodies[i].velocity.z += bodies[i].acceleration.z * DT;

                // Update position
                bodies[i].position.x += bodies[i].velocity.x * DT;
                bodies[i].position.y += bodies[i].velocity.y * DT;
                bodies[i].position.z += bodies[i].velocity.z * DT;
            }
        } else {
            // Single-threaded computation
            for (int i = 0; i < nBodies; i++) {
                // Skip non-dynamic bodies
                if (!bodies[i].isDynamic)
                    continue;

                // Reset acceleration
                bodies[i].acceleration = Vector(0.0, 0.0, 0.0);

                // Compute force from all other bodies
                for (int j = 0; j < nBodies; j++) {
                    if (i == j)
                        continue; // Skip self-interaction

                    // Vector from body i to body j
                    Vector r = bodies[j].position - bodies[i].position;

                    // Distance calculation with softening
                    double distSqr = r.lengthSquared() + (E * E);
                    double dist = sqrt(distSqr);

                    // Skip if bodies are too close (collision)
                    if (dist < COLLISION_TH)
                        continue;

                    // Gravitational force: G * m1 * m2 / r^3 * r_vector
                    double forceMag = GRAVITY * bodies[i].mass * bodies[j].mass / (distSqr * dist);

                    // Update acceleration (F = ma, so a = F/m)
                    bodies[i].acceleration.x += (r.x * forceMag) / bodies[i].mass;
                    bodies[i].acceleration.y += (r.y * forceMag) / bodies[i].mass;
                    bodies[i].acceleration.z += (r.z * forceMag) / bodies[i].mass;
                }

                // Update velocity (Euler integration)
                bodies[i].velocity.x += bodies[i].acceleration.x * DT;
                bodies[i].velocity.y += bodies[i].acceleration.y * DT;
                bodies[i].velocity.z += bodies[i].velocity.z * DT;

                // Update position
                bodies[i].position.x += bodies[i].velocity.x * DT;
                bodies[i].position.y += bodies[i].velocity.y * DT;
                bodies[i].position.z += bodies[i].velocity.z * DT;
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        forceCalculationTime = std::chrono::duration<double, std::milli>(end - start).count();
    }

    void update() {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Simply compute forces and update positions
        computeForces();
        
        auto end = std::chrono::high_resolution_clock::now();
        totalTime = std::chrono::duration<double, std::milli>(end - start).count();
    }
    
    void printPerformanceMetrics() const {
        std::cout << "Performance Metrics (ms):" << std::endl;
        std::cout << "  Total time:           " << std::fixed << std::setprecision(3) << totalTime << std::endl;
        std::cout << "  Force calculation:     " << std::fixed << std::setprecision(3) << forceCalculationTime << std::endl;
    }
    
    double getTotalTime() const { return totalTime; }
    double getForceCalculationTime() const { return forceCalculationTime; }
    int getNumBodies() const { return nBodies; }
    int getNumThreads() const { return numThreads; }
    
    void run(int steps) {
        std::cout << "Running CPU Direct Sum simulation for " << steps << " steps..." << std::endl;
        
        double totalSim = 0.0;
        for (int step = 0; step < steps; step++) {
            update();
            totalSim += totalTime;
            
            if (step % 10 == 0) {
                std::cout << "Step " << step << " completed. ";
                printPerformanceMetrics();
            }
        }
        
        std::cout << "Simulation completed in " << totalSim << " ms." << std::endl;
        std::cout << "Average step time: " << totalSim / steps << " ms." << std::endl;
    }
};

// =============================================================================
// FUNCIÓN PRINCIPAL
// =============================================================================

int main(int argc, char* argv[]) {
    int nBodies = 1000;
    bool useOpenMP = true;
    int threads = 0; // Auto-detect
    BodyDistribution dist = BodyDistribution::RANDOM_UNIFORM;
    MassDistribution massDist = MassDistribution::UNIFORM;
    int steps = 100;
    bool saveMetricsToFile = false;
    std::string metricsFile = "./DirectSumCPU_metrics.csv";
    
    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-n" && i + 1 < argc) {
            nBodies = std::stoi(argv[++i]);
        } else if (arg == "--no-openmp") {
            useOpenMP = false;
        } else if (arg == "--threads" && i + 1 < argc) {
            threads = std::stoi(argv[++i]);
        } else if (arg == "--distribution" && i + 1 < argc) {
            std::string distStr = argv[++i];
            if (distStr == "random") {
                dist = BodyDistribution::RANDOM_UNIFORM;
            } else if (distStr == "solar") {
                dist = BodyDistribution::SOLAR_SYSTEM;
            } else if (distStr == "galaxy") {
                dist = BodyDistribution::GALAXY;
            } else if (distStr == "collision") {
                dist = BodyDistribution::COLLISION;
            }
        } else if (arg == "--mass" && i + 1 < argc) {
            std::string massStr = argv[++i];
            if (massStr == "uniform") {
                massDist = MassDistribution::UNIFORM;
            } else if (massStr == "normal") {
                massDist = MassDistribution::NORMAL;
            }
        } else if (arg == "--steps" && i + 1 < argc) {
            steps = std::stoi(argv[++i]);
        } else if (arg == "--save-metrics") {
            saveMetricsToFile = true;
        } else if (arg == "--metrics-file" && i + 1 < argc) {
            metricsFile = argv[++i];
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  -n N          Set number of bodies (default: 1000)" << std::endl;
            std::cout << "  --no-openmp         Disable OpenMP parallelization" << std::endl;
            std::cout << "  --threads N         Set number of threads (default: auto)" << std::endl;
            std::cout << "  --distribution TYPE Set body distribution (random, solar, galaxy, collision)" << std::endl;
            std::cout << "  --mass TYPE         Set mass distribution (uniform, normal)" << std::endl;
            std::cout << "  --steps N           Set simulation steps (default: 100)" << std::endl;
            std::cout << "  --save-metrics      Save metrics to CSV file" << std::endl;
            std::cout << "  --metrics-file PATH Set custom path for metrics file" << std::endl;
            std::cout << "  --help              Show this help message" << std::endl;
            return 0;
        }
    }
    
    // Create and run simulation
    CPUDirectSum simulation(nBodies, useOpenMP, threads, dist, time(nullptr), massDist);
    simulation.run(steps);
    
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
            steps,
            simulation.getNumThreads(),
            simulation.getTotalTime(),
            simulation.getForceCalculationTime()
        );
        
        std::cout << "Métricas guardadas en: " << metricsFile << std::endl;
    }
    
    return 0;
} 