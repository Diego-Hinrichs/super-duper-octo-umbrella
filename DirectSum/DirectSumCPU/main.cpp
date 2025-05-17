#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <chrono>
#include <random>
#include <string>
#include <iomanip>
#include <algorithm>
#include <limits>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <deque>
#include <numeric>

class DynamicReorderingStrategy
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

public:
    DynamicReorderingStrategy(int windowSize = 10)
        : reorderTime(0.0),
          postReorderSimTime(0.0),
          updateTime(0.0),
          degradationRate(0.001),
          iterationsSinceReorder(0),
          currentOptimalFrequency(10),
          metricsWindowSize(windowSize)
    {
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

    Vector() : x(0.0), y(0.0), z(0.0) {}
    Vector(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
    Vector operator+(const Vector &other) const { return Vector(x + other.x, y + other.y, z + other.z); }
    Vector operator-(const Vector &other) const { return Vector(x - other.x, y - other.y, z - other.z); }
    Vector operator*(double scalar) const { return Vector(x * scalar, y * scalar, z * scalar); }
    double dot(const Vector &other) const { return x * other.x + y * other.y + z * other.z; }
    double lengthSquared() const { return x * x + y * y + z * z; }
    double length() const { return sqrt(lengthSquared()); }
};

struct Body
{
    Vector position;
    Vector velocity;
    Vector acceleration;
    double mass;
    bool isDynamic;

    Body() : mass(1.0), isDynamic(true) {}
    Body(const Vector &pos, const Vector &vel, double m, bool dynamic = true)
        : position(pos), velocity(vel), acceleration(), mass(m), isDynamic(dynamic) {}
};

constexpr double GRAVITY = 6.67430e-11;
constexpr double DT = 0.005;
constexpr double E = 0.01;
constexpr double COLLISION_TH = 0.01;

enum class BodyDistribution
{
    RANDOM_UNIFORM,
    SOLAR_SYSTEM,
    GALAXY,
    COLLISION
};

enum class MassDistribution
{
    UNIFORM,
    NORMAL
};

enum class SFCType
{
    MORTON,
    HILBERT
};

inline uint64_t expandBits(uint64_t v)
{
    v = (v * 0x0001000001000001ULL) & 0xFFFF00000000FFFFULL;
    v = (v * 0x0000010000010001ULL) & 0x00FF0000FF0000FFULL;
    v = (v * 0x0000000100000001ULL) & 0xF00F00F00F00F00FULL;
    v = (v * 0x0000000000000010ULL) & 0x30C30C30C30C30C3ULL;
    v = (v * 0x0000000000000004ULL) & 0x9249249249249249ULL;
    return v;
}

uint64_t mortonEncode(double x, double y, double z, const Vector &min, const Vector &max)
{
    double x_norm = (x - min.x) / (max.x - min.x);
    double y_norm = (y - min.y) / (max.y - min.y);
    double z_norm = (z - min.z) / (max.z - min.z);

    uint64_t x_int = static_cast<uint64_t>(x_norm * 0x1FFFFF);
    uint64_t y_int = static_cast<uint64_t>(y_norm * 0x1FFFFF);
    uint64_t z_int = static_cast<uint64_t>(z_norm * 0x1FFFFF);

    return expandBits(x_int) | (expandBits(y_int) << 1) | (expandBits(z_int) << 2);
}

class CPUDirectSum
{
private:
    std::vector<Body> bodies;
    int nBodies;
    bool useOpenMP;
    int numThreads;
    double totalTime;
    double forceCalculationTime;
    double reorderTime;
    double bboxTime;
    double potentialEnergy;
    double kineticEnergy;
    double totalEnergyAvg;
    double potentialEnergyAvg;
    double kineticEnergyAvg;

    bool useSFC;
    SFCType sfcType;
    int reorderFrequency;
    int iterationCounter;

    std::vector<uint64_t> mortonCodes;
    std::vector<int> orderedIndices;

    Vector minBound;
    Vector maxBound;

    bool useDynamicReordering;
    DynamicReorderingStrategy reorderingStrategy;

public:
    CPUDirectSum(
        int numBodies,
        bool useParallelization = true,
        int threads = 0,
        bool enableSFC = true,
        SFCType sfc = SFCType::MORTON,
        int reorderFreq = 10,
        BodyDistribution dist = BodyDistribution::RANDOM_UNIFORM,
        unsigned int seed = static_cast<unsigned int>(time(nullptr)),
        MassDistribution massDist = MassDistribution::UNIFORM,
        bool dynamicReordering = false) : nBodies(numBodies),
                                          useOpenMP(useParallelization),
                                          totalTime(0.0),
                                          forceCalculationTime(0.0),
                                          reorderTime(0.0),
                                          bboxTime(0.0),
                                          potentialEnergy(0.0),
                                          kineticEnergy(0.0),
                                          totalEnergyAvg(0.0),
                                          potentialEnergyAvg(0.0),
                                          kineticEnergyAvg(0.0),
                                          useSFC(enableSFC),
                                          sfcType(sfc),
                                          reorderFrequency(reorderFreq),
                                          iterationCounter(0),
                                          useDynamicReordering(dynamicReordering),
                                          reorderingStrategy(10)
    {
        if (threads <= 0)
        {
            numThreads = omp_get_max_threads();
        }
        else
        {
            numThreads = threads;
        }

        if (useOpenMP)
        {
            omp_set_num_threads(numThreads);
        }

        bodies.resize(nBodies);
        mortonCodes.resize(nBodies);
        orderedIndices.resize(nBodies);

        initializeBodies(dist, seed, massDist);
    }

    void initializeBodies(BodyDistribution dist, unsigned int seed, MassDistribution massDist)
    {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> posDist(-1.0, 1.0);
        std::uniform_real_distribution<double> velDist(-0.1, 0.1);
        std::uniform_real_distribution<double> massDistUniform(0.1, 1.0);
        std::normal_distribution<double> massDistNormal(0.5, 0.2);

        for (int i = 0; i < nBodies; ++i)
        {
            double mass = massDist == MassDistribution::UNIFORM ? massDistUniform(rng) : massDistNormal(rng);
            mass = std::max(0.1, std::min(1.0, mass));

            Vector pos(posDist(rng), posDist(rng), posDist(rng));
            Vector vel(velDist(rng), velDist(rng), velDist(rng));

            bodies[i] = Body(pos, vel, mass);
        }
    }

    void computeBoundingBox()
    {
        auto start = std::chrono::high_resolution_clock::now();

        if (useOpenMP)
        {
            std::vector<Vector> localMin(numThreads, Vector(std::numeric_limits<double>::max(),
                                                          std::numeric_limits<double>::max(),
                                                          std::numeric_limits<double>::max()));
            std::vector<Vector> localMax(numThreads, Vector(std::numeric_limits<double>::lowest(),
                                                          std::numeric_limits<double>::lowest(),
                                                          std::numeric_limits<double>::lowest()));

#pragma omp parallel
            {
                int tid = omp_get_thread_num();
#pragma omp for
                for (int i = 0; i < nBodies; ++i)
                {
                    const Body &body = bodies[i];
                    localMin[tid].x = std::min(localMin[tid].x, body.position.x);
                    localMin[tid].y = std::min(localMin[tid].y, body.position.y);
                    localMin[tid].z = std::min(localMin[tid].z, body.position.z);
                    localMax[tid].x = std::max(localMax[tid].x, body.position.x);
                    localMax[tid].y = std::max(localMax[tid].y, body.position.y);
                    localMax[tid].z = std::max(localMax[tid].z, body.position.z);
                }
            }

            minBound = localMin[0];
            maxBound = localMax[0];

            for (int i = 1; i < numThreads; ++i)
            {
                minBound.x = std::min(minBound.x, localMin[i].x);
                minBound.y = std::min(minBound.y, localMin[i].y);
                minBound.z = std::min(minBound.z, localMin[i].z);
                maxBound.x = std::max(maxBound.x, localMax[i].x);
                maxBound.y = std::max(maxBound.y, localMax[i].y);
                maxBound.z = std::max(maxBound.z, localMax[i].z);
            }
        }
        else
        {
            minBound = Vector(std::numeric_limits<double>::max(),
                            std::numeric_limits<double>::max(),
                            std::numeric_limits<double>::max());
            maxBound = Vector(std::numeric_limits<double>::lowest(),
                            std::numeric_limits<double>::lowest(),
                            std::numeric_limits<double>::lowest());

            for (const Body &body : bodies)
            {
                minBound.x = std::min(minBound.x, body.position.x);
                minBound.y = std::min(minBound.y, body.position.y);
                minBound.z = std::min(minBound.z, body.position.z);
                maxBound.x = std::max(maxBound.x, body.position.x);
                maxBound.y = std::max(maxBound.y, body.position.y);
                maxBound.z = std::max(maxBound.z, body.position.z);
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        bboxTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    }

    void orderBodiesBySFC()
    {
        auto start = std::chrono::high_resolution_clock::now();

        computeBoundingBox();

        if (useOpenMP)
        {
#pragma omp parallel for
            for (int i = 0; i < nBodies; ++i)
            {
                mortonCodes[i] = mortonEncode(bodies[i].position.x,
                                            bodies[i].position.y,
                                            bodies[i].position.z,
                                            minBound,
                                            maxBound);
                orderedIndices[i] = i;
            }
        }
        else
        {
            for (int i = 0; i < nBodies; ++i)
            {
                mortonCodes[i] = mortonEncode(bodies[i].position.x,
                                            bodies[i].position.y,
                                            bodies[i].position.z,
                                            minBound,
                                            maxBound);
                orderedIndices[i] = i;
            }
        }

        std::sort(orderedIndices.begin(), orderedIndices.end(),
                  [this](int a, int b)
                  {
                      return mortonCodes[a] < mortonCodes[b];
                  });

        std::vector<Body> reorderedBodies(nBodies);
        for (int i = 0; i < nBodies; ++i)
        {
            reorderedBodies[i] = bodies[orderedIndices[i]];
        }
        bodies = std::move(reorderedBodies);

        auto end = std::chrono::high_resolution_clock::now();
        reorderTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    }

    void computeForces()
    {
        auto start = std::chrono::high_resolution_clock::now();

        if (useOpenMP)
        {
#pragma omp parallel for
            for (int i = 0; i < nBodies; ++i)
            {
                bodies[i].acceleration = Vector();
            }

#pragma omp parallel for
            for (int i = 0; i < nBodies; ++i)
            {
                for (int j = 0; j < nBodies; ++j)
                {
                    if (i == j)
                        continue;

                    Vector diff = bodies[j].position - bodies[i].position;
                    double distSquared = diff.lengthSquared();
                    double dist = sqrt(distSquared);

                    if (dist < COLLISION_TH)
                    {
                        Vector collisionForce = diff * (1.0 / dist);
                        bodies[i].acceleration = bodies[i].acceleration + collisionForce * 0.1;
                        continue;
                    }

                    double force = GRAVITY * bodies[j].mass / (distSquared + E * E);
                    Vector forceVector = diff * (force / dist);
                    bodies[i].acceleration = bodies[i].acceleration + forceVector;
                }
            }
        }
        else
        {
            for (int i = 0; i < nBodies; ++i)
            {
                bodies[i].acceleration = Vector();
            }

            for (int i = 0; i < nBodies; ++i)
            {
                for (int j = 0; j < nBodies; ++j)
                {
                    if (i == j)
                        continue;

                    Vector diff = bodies[j].position - bodies[i].position;
                    double distSquared = diff.lengthSquared();
                    double dist = sqrt(distSquared);

                    if (dist < COLLISION_TH)
                    {
                        Vector collisionForce = diff * (1.0 / dist);
                        bodies[i].acceleration = bodies[i].acceleration + collisionForce * 0.1;
                        continue;
                    }

                    double force = GRAVITY * bodies[j].mass / (distSquared + E * E);
                    Vector forceVector = diff * (force / dist);
                    bodies[i].acceleration = bodies[i].acceleration + forceVector;
                }
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        forceCalculationTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    }

    void calculateEnergies()
    {
        potentialEnergy = 0.0;
        kineticEnergy = 0.0;

        if (useOpenMP)
        {
            double localPotential = 0.0;
            double localKinetic = 0.0;

#pragma omp parallel reduction(+ : localPotential, localKinetic)
            {
#pragma omp for
                for (int i = 0; i < nBodies; ++i)
                {
                    for (int j = i + 1; j < nBodies; ++j)
                    {
                        Vector diff = bodies[j].position - bodies[i].position;
                        double dist = diff.length();
                        localPotential -= GRAVITY * bodies[i].mass * bodies[j].mass / dist;
                    }

                    localKinetic += 0.5 * bodies[i].mass * bodies[i].velocity.lengthSquared();
                }
            }

            potentialEnergy = localPotential;
            kineticEnergy = localKinetic;
        }
        else
        {
            for (int i = 0; i < nBodies; ++i)
            {
                for (int j = i + 1; j < nBodies; ++j)
                {
                    Vector diff = bodies[j].position - bodies[i].position;
                    double dist = diff.length();
                    potentialEnergy -= GRAVITY * bodies[i].mass * bodies[j].mass / dist;
                }

                kineticEnergy += 0.5 * bodies[i].mass * bodies[i].velocity.lengthSquared();
            }
        }

        totalEnergyAvg = (totalEnergyAvg * iterationCounter + potentialEnergy + kineticEnergy) / (iterationCounter + 1);
        potentialEnergyAvg = (potentialEnergyAvg * iterationCounter + potentialEnergy) / (iterationCounter + 1);
        kineticEnergyAvg = (kineticEnergyAvg * iterationCounter + kineticEnergy) / (iterationCounter + 1);
    }

    void update()
    {
        if (useSFC && (useDynamicReordering ? reorderingStrategy.shouldReorder() : iterationCounter % reorderFrequency == 0))
        {
            orderBodiesBySFC();
        }

        computeForces();

        if (useOpenMP)
        {
#pragma omp parallel for
            for (int i = 0; i < nBodies; ++i)
            {
                bodies[i].velocity = bodies[i].velocity + bodies[i].acceleration * DT;
                bodies[i].position = bodies[i].position + bodies[i].velocity * DT;
            }
        }
        else
        {
            for (int i = 0; i < nBodies; ++i)
            {
                bodies[i].velocity = bodies[i].velocity + bodies[i].acceleration * DT;
                bodies[i].position = bodies[i].position + bodies[i].velocity * DT;
            }
        }

        calculateEnergies();
        iterationCounter++;
    }

    void printPerformanceMetrics() const
    {
        std::cout << "Performance Metrics:" << std::endl;
        std::cout << "Total Time: " << std::fixed << std::setprecision(2) << totalTime << " ms" << std::endl;
        std::cout << "Force Calculation Time: " << std::fixed << std::setprecision(2) << forceCalculationTime << " ms" << std::endl;
        if (useSFC)
        {
            std::cout << "Reorder Time: " << std::fixed << std::setprecision(2) << reorderTime << " ms" << std::endl;
            std::cout << "Bounding Box Time: " << std::fixed << std::setprecision(2) << bboxTime << " ms" << std::endl;
        }
        std::cout << "Potential Energy: " << std::scientific << std::setprecision(6) << potentialEnergy << std::endl;
        std::cout << "Kinetic Energy: " << std::scientific << std::setprecision(6) << kineticEnergy << std::endl;
        std::cout << "Total Energy: " << std::scientific << std::setprecision(6) << (potentialEnergy + kineticEnergy) << std::endl;
    }

    void run(int steps)
    {
        auto start = std::chrono::high_resolution_clock::now();

        for (int step = 0; step < steps; ++step)
        {
            update();
        }

        auto end = std::chrono::high_resolution_clock::now();
        totalTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    }

    int getOptimalReorderFrequency() const
    {
        return reorderingStrategy.getOptimalFrequency();
    }

    double getTotalTime() const { return totalTime; }
    double getForceCalculationTime() const { return forceCalculationTime; }
    double getSortTime() const { return reorderTime; }
    int getNumBodies() const { return nBodies; }
    int getNumThreads() const { return numThreads; }
    int getSortType() const { return static_cast<int>(useSFC); }
    bool isDynamicReordering() const { return useDynamicReordering; }
    double getPotentialEnergy() const { return potentialEnergy; }
    double getKineticEnergy() const { return kineticEnergy; }
    double getTotalEnergy() const { return potentialEnergy + kineticEnergy; }
    double getPotentialEnergyAvg() const { return potentialEnergyAvg; }
    double getKineticEnergyAvg() const { return kineticEnergyAvg; }
    double getTotalEnergyAvg() const { return totalEnergyAvg; }
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
        file << "timestamp,method,bodies,steps,threads,sort_type,total_time_ms,avg_step_time_ms,force_calculation_time_ms,sort_time_ms,potential_energy,kinetic_energy,total_energy" << std::endl;
    }

    file.close();
    std::cout << "Archivo CSV " << (append ? "actualizado" : "inicializado") << ": " << filename << std::endl;
}

void saveMetrics(const std::string &filename,
                 int bodies,
                 int steps,
                 int threads,
                 int sortType,
                 double totalTime,
                 double forceCalculationTime,
                 double sortTime,
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
         << "CPU_Direct_Sum" << ","
         << bodies << ","
         << steps << ","
         << threads << ","
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

int main(int argc, char *argv[])
{
    int nBodies = 1000;
    int steps = 100;
    int threads = 0;
    bool useOpenMP = true;
    bool useSFC = true;
    SFCType sfcType = SFCType::MORTON;
    int reorderFreq = 10;
    bool useDynamicReordering = false;
    BodyDistribution dist = BodyDistribution::RANDOM_UNIFORM;
    MassDistribution massDist = MassDistribution::UNIFORM;
    std::string metricsFile = "metrics/direct_sum_metrics.csv";

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "-n" && i + 1 < argc)
        {
            nBodies = std::stoi(argv[++i]);
        }
        else if (arg == "-s" && i + 1 < argc)
        {
            steps = std::stoi(argv[++i]);
        }
        else if (arg == "-t" && i + 1 < argc)
        {
            threads = std::stoi(argv[++i]);
        }
        else if (arg == "-no-omp")
        {
            useOpenMP = false;
        }
        else if (arg == "-no-sfc")
        {
            useSFC = false;
        }
        else if (arg == "-hilbert")
        {
            sfcType = SFCType::HILBERT;
        }
        else if (arg == "-f" && i + 1 < argc)
        {
            reorderFreq = std::stoi(argv[++i]);
        }
        else if (arg == "-dynamic")
        {
            useDynamicReordering = true;
        }
        else if (arg == "-solar")
        {
            dist = BodyDistribution::SOLAR_SYSTEM;
        }
        else if (arg == "-galaxy")
        {
            dist = BodyDistribution::GALAXY;
        }
        else if (arg == "-collision")
        {
            dist = BodyDistribution::COLLISION;
        }
        else if (arg == "-normal-mass")
        {
            massDist = MassDistribution::NORMAL;
        }
        else if (arg == "-o" && i + 1 < argc)
        {
            metricsFile = argv[++i];
        }
    }

    std::ifstream checkFile(metricsFile);
    bool fileExists = checkFile.good();
    checkFile.close();

    if (!fileExists)
    {
        initializeCsv(metricsFile);
    }

    CPUDirectSum simulation(nBodies, useOpenMP, threads, useSFC, sfcType, reorderFreq, dist, time(nullptr), massDist, useDynamicReordering);
    simulation.run(steps);
    simulation.printPerformanceMetrics();

    saveMetrics(metricsFile,
                simulation.getNumBodies(),
                steps,
                simulation.getNumThreads(),
                simulation.getSortType(),
                simulation.getTotalTime(),
                simulation.getForceCalculationTime(),
                simulation.getSortTime(),
                simulation.getPotentialEnergy(),
                simulation.getKineticEnergy(),
                simulation.getTotalEnergy());

    return 0;
}
