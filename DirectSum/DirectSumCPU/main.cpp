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
#include "../../argparse.hpp"
#include "types.h"
#include "sfc_cpu.h"

// Constants
constexpr double GRAVITY = 6.67430e-11;
constexpr double DT = 0.005;
constexpr double E = 0.01;
constexpr double COLLISION_TH = 0.01;
constexpr double MAX_DIST = 1.0;
constexpr double CENTERX = 0.0;
constexpr double CENTERY = 0.0;
constexpr double CENTERZ = 0.0;
constexpr double EARTH_MASS = 5.972e24;
constexpr double EARTH_DIA = 12742000.0;

// Enums
enum class MassDistribution
{
    UNIFORM,
    NORMAL
};

enum class SFCCurveType
{
    MORTON = 0,
    HILBERT = 1
};

// DynamicReorderingStrategy class (renaming for consistency)
class SFCDynamicReorderingStrategy
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
        if (reorderTimeHistory.empty() || postReorderSimTimeHistory.empty())
            return 10;

        double avgReorderTime = std::accumulate(reorderTimeHistory.begin(), reorderTimeHistory.end(), 0.0) / reorderTimeHistory.size();
        double avgPostReorderTime = std::accumulate(postReorderSimTimeHistory.begin(), postReorderSimTimeHistory.end(), 0.0) / postReorderSimTimeHistory.size();

        if (avgReorderTime == 0 || avgPostReorderTime == 0)
            return 10;

        double ratio = avgReorderTime / avgPostReorderTime;
        int optimalFreq = static_cast<int>(std::sqrt(1.0 / ratio));

        return std::max(1, std::min(optimalFreq, 100));
    }

public:
    SFCDynamicReorderingStrategy(int windowSize = 10)
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
        reorderTime = newReorderTime;
        postReorderSimTime = newSimTime;

        reorderTimeHistory.push_back(newReorderTime);
        postReorderSimTimeHistory.push_back(newSimTime);

        if (reorderTimeHistory.size() > metricsWindowSize)
        {
            reorderTimeHistory.pop_front();
            postReorderSimTimeHistory.pop_front();
        }

        currentOptimalFrequency = computeOptimalFrequency(iterationsSinceReorder);
    }

    bool shouldReorder(double lastSimTime = 0.0, double lastReorderTime = 0.0)
    {
        iterationsSinceReorder++;
        if (iterationsSinceReorder >= currentOptimalFrequency)
        {
            iterationsSinceReorder = 0;
            return true;
        }
        return false;
    }

    void updateMetrics(double sortTime)
    {
        simulationTimeHistory.push_back(sortTime);
        if (simulationTimeHistory.size() > metricsWindowSize)
        {
            simulationTimeHistory.pop_front();
        }
    }

    void setWindowSize(int windowSize)
    {
        metricsWindowSize = windowSize;
        reorderTimeHistory.clear();
        postReorderSimTimeHistory.clear();
        simulationTimeHistory.clear();
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
        reorderTime = 0.0;
        postReorderSimTime = 0.0;
        updateTime = 0.0;
        iterationsSinceReorder = 0;
        currentOptimalFrequency = 10;
        reorderTimeHistory.clear();
        postReorderSimTimeHistory.clear();
        simulationTimeHistory.clear();
    }
};

enum class BodyDistribution
{
    RANDOM_UNIFORM,
    SOLAR_SYSTEM,
    GALAXY,
    COLLISION
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

class DirectSum
{
private:
    std::vector<Body> bodies;
    int nBodies;
    bool useOpenMP;
    int numThreads;
    double totalTime;
    double forceCalculationTime;
    double sfcTime;
    double bboxTime;
    double potentialEnergy;
    double kineticEnergy;
    double totalEnergyAvg;
    double potentialEnergyAvg;
    double kineticEnergyAvg;

    bool useSFC;
    SFCCurveType curveType;
    int iterationCounter;
    SFCDynamicReorderingStrategy reorderingStrategy;
    sfc::BodySorter<Body> *sorter;

    Vector minBound;
    Vector maxBound;

public:
    DirectSum(
        int numBodies,
        bool useParallelization = true,
        int threads = 0,
        unsigned int seed = static_cast<unsigned int>(time(nullptr)),
        MassDistribution massDist = MassDistribution::UNIFORM,
        bool useSFC_ = true,
        SFCCurveType sfcCurveType = SFCCurveType::MORTON) : nBodies(numBodies),
                                                           useOpenMP(useParallelization),
                                                           totalTime(0.0),
                                                           forceCalculationTime(0.0),
                                                           sfcTime(0.0),
                                                           bboxTime(0.0),
                                                           potentialEnergy(0.0),
                                                           kineticEnergy(0.0),
                                                           totalEnergyAvg(0.0),
                                                           potentialEnergyAvg(0.0),
                                                           kineticEnergyAvg(0.0),
                                                           useSFC(useSFC_),
                                                           curveType(sfcCurveType),
                                                           iterationCounter(0),
                                                           reorderingStrategy(10),
                                                           sorter(nullptr)
    {
        bodies.resize(numBodies);

        if (useOpenMP && threads > 0)
        {
            numThreads = threads;
            omp_set_num_threads(numThreads);
        }
        else if (useOpenMP)
        {
            numThreads = omp_get_max_threads();
        }
        else
        {
            numThreads = 1;
        }

        std::cout << "Direct Sum CPU simulation created with " << numBodies << " bodies." << std::endl;
        std::cout << "Using " << (useOpenMP ? numThreads : 1) << " thread(s)" << std::endl;
        if (useSFC)
        {
            std::string sfcTypeStr = (curveType == SFCCurveType::MORTON) ? "MORTON" : "HILBERT";
            std::cout << "SFC Ordering enabled with type " << sfcTypeStr << std::endl;
            
            sfc::CurveType sfcCurveType = (curveType == SFCCurveType::MORTON) ? 
                                      sfc::CurveType::MORTON : 
                                      sfc::CurveType::HILBERT;
            sorter = new sfc::BodySorter<Body>(numBodies, sfcCurveType);
        }

        initRandomBodies(seed, massDist);
    }

    ~DirectSum() {
        if (sorter) {
            delete sorter;
            sorter = nullptr;
        }
    }

    void initRandomBodies(unsigned int seed, MassDistribution massDist)
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
                bodies[i].position = Vector(
                    CENTERX + pos_dist(gen),
                    CENTERY + pos_dist(gen),
                    CENTERZ + pos_dist(gen));

                bodies[i].velocity = Vector(
                    vel_dist(gen),
                    vel_dist(gen),
                    vel_dist(gen));
            }
            else
            {
                bodies[i].position = Vector(
                    CENTERX + normal_pos_dist(gen),
                    CENTERY + normal_pos_dist(gen),
                    CENTERZ + normal_pos_dist(gen));

                bodies[i].velocity = Vector(
                    normal_vel_dist(gen),
                    normal_vel_dist(gen),
                    normal_vel_dist(gen));
            }

            bodies[i].mass = 1.0;
            bodies[i].radius = pow(bodies[i].mass / EARTH_MASS, 1.0 / 3.0) * (EARTH_DIA / 2.0);
            bodies[i].isDynamic = true;
            bodies[i].acceleration = Vector(0, 0, 0);
        }
    }

    void updateBoundingBox()
    {
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
                for (int i = 0; i < nBodies; i++)
                {
                    const Vector &pos = bodies[i].position;
                    localMin[tid].x = std::min(localMin[tid].x, pos.x);
                    localMin[tid].y = std::min(localMin[tid].y, pos.y);
                    localMin[tid].z = std::min(localMin[tid].z, pos.z);
                    localMax[tid].x = std::max(localMax[tid].x, pos.x);
                    localMax[tid].y = std::max(localMax[tid].y, pos.y);
                    localMax[tid].z = std::max(localMax[tid].z, pos.z);
                }
            }

            minBound = localMin[0];
            maxBound = localMax[0];

            for (int i = 1; i < numThreads; i++)
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

            for (int i = 0; i < nBodies; i++)
            {
                const Vector &pos = bodies[i].position;
                minBound.x = std::min(minBound.x, pos.x);
                minBound.y = std::min(minBound.y, pos.y);
                minBound.z = std::min(minBound.z, pos.z);
                maxBound.x = std::max(maxBound.x, pos.x);
                maxBound.y = std::max(maxBound.y, pos.y);
                maxBound.z = std::max(maxBound.z, pos.z);
            }
        }

        // Agregar un pequeño margen para evitar problemas numéricos
        double padding = std::max(1.0e-10, (maxBound.x - minBound.x) * 0.01);
        minBound.x -= padding;
        minBound.y -= padding;
        minBound.z -= padding;
        maxBound.x += padding;
        maxBound.y += padding;
        maxBound.z += padding;
    }

    void computeBoundingBox()
    {
        auto start = std::chrono::high_resolution_clock::now();

        // updateBoundingBox();

        auto end = std::chrono::high_resolution_clock::now();
        bboxTime = std::chrono::duration<double, std::milli>(end - start).count();
    }

    void sortBodiesBySFC()
    {
        if (!useSFC || !sorter) {
            return;
        }

        auto start = std::chrono::high_resolution_clock::now();

        updateBoundingBox();

        // Convert our Vector types to the SFC library's expected format
        const Vector& minB = minBound;
        const Vector& maxB = maxBound;
        
        sorter->sortBodies(bodies, minB, maxB);

        auto end = std::chrono::high_resolution_clock::now();
        sfcTime = std::chrono::duration<double, std::milli>(end - start).count();
    }

    void reorderBodies()
    {
        if (!useSFC) {
            return;
        }

        bool shouldReorder = reorderingStrategy.shouldReorder(forceCalculationTime, sfcTime);

        if (shouldReorder)
        {
            sortBodiesBySFC();
            iterationCounter = 0;
        }

        reorderingStrategy.updateMetrics(shouldReorder ? sfcTime : 0.0, forceCalculationTime);
        iterationCounter++;
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
        forceCalculationTime = std::chrono::duration<double, std::milli>(end - start).count();
    }

    void calculateEnergies()
    {
        potentialEnergy = 0.0;
        kineticEnergy = 0.0;

        if (useOpenMP)
        {
#pragma omp parallel
            {
                double localPotential = 0.0;
                double localKinetic = 0.0;

#pragma omp for
                for (int i = 0; i < nBodies; i++)
                {
                    for (int j = i + 1; j < nBodies; j++)
                    {
                        Vector diff = bodies[i].position - bodies[j].position;
                        double dist = diff.length();
                        if (dist > 0)
                        {
                            localPotential -= GRAVITY * bodies[i].mass * bodies[j].mass / dist;
                        }
                    }
                    localKinetic += 0.5 * bodies[i].mass * bodies[i].velocity.lengthSquared();
                }

#pragma omp critical
                {
                    potentialEnergy += localPotential;
                    kineticEnergy += localKinetic;
                }
            }
        }
        else
        {
            for (int i = 0; i < nBodies; i++)
            {
                for (int j = i + 1; j < nBodies; j++)
                {
                    Vector diff = bodies[i].position - bodies[j].position;
                    double dist = diff.length();
                    if (dist > 0)
                    {
                        potentialEnergy -= GRAVITY * bodies[i].mass * bodies[j].mass / dist;
                    }
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
        auto start = std::chrono::high_resolution_clock::now();

        reorderBodies();
        computeForces();

        // Actualizar posiciones y velocidades
        if (useOpenMP)
        {
#pragma omp parallel for
            for (int i = 0; i < nBodies; ++i)
            {
                if (bodies[i].isDynamic)
                {
                    bodies[i].velocity = bodies[i].velocity + bodies[i].acceleration * DT;
                    bodies[i].position = bodies[i].position + bodies[i].velocity * DT;
                }
            }
        }
        else
        {
            for (int i = 0; i < nBodies; ++i)
            {
                if (bodies[i].isDynamic)
                {
                    bodies[i].velocity = bodies[i].velocity + bodies[i].acceleration * DT;
                    bodies[i].position = bodies[i].position + bodies[i].velocity * DT;
                }
            }
        }

        calculateEnergies();

        auto end = std::chrono::high_resolution_clock::now();
        totalTime += std::chrono::duration<double, std::milli>(end - start).count();
    }

    void printPerformanceMetrics() const
    {
        std::cout << "Performance Metrics:" << std::endl;
        std::cout << "  Average time per step: " << std::fixed << std::setprecision(2) << totalTime / (iterationCounter > 0 ? iterationCounter : 1) << " ms" << std::endl;
        std::cout << "  Compute forces: " << std::fixed << std::setprecision(2) << forceCalculationTime << " ms" << std::endl;
        std::cout << "  Bounding box: " << std::fixed << std::setprecision(2) << bboxTime << " ms" << std::endl;
        if (useSFC)
        {
            std::cout << "  Reordering: " << std::fixed << std::setprecision(2) << sfcTime << " ms" << std::endl;
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
        }
    }

    void run(int steps)
    {
        std::cout << "Running Direct Sum CPU simulation for " << steps << " steps..." << std::endl;
        
        float totalRunTime = 0.0f;
        float totalForceTime = 0.0f;
        float totalReorderTime = 0.0f;
        float totalBboxTime = 0.0f;
        float minTime = std::numeric_limits<float>::max();
        float maxTime = 0.0f;
        double totalPotentialEnergy = 0.0;
        double totalKineticEnergy = 0.0;

        auto startTime = std::chrono::high_resolution_clock::now();

        for (int step = 0; step < steps; step++)
        {
            auto stepStart = std::chrono::high_resolution_clock::now();
            
            reorderBodies();
            computeForces();
            
            // Actualizar posiciones y velocidades
            if (useOpenMP)
            {
#pragma omp parallel for
                for (int i = 0; i < nBodies; ++i)
                {
                    if (bodies[i].isDynamic)
                    {
                        bodies[i].velocity = bodies[i].velocity + bodies[i].acceleration * DT;
                        bodies[i].position = bodies[i].position + bodies[i].velocity * DT;
                    }
                }
            }
            else
            {
                for (int i = 0; i < nBodies; ++i)
                {
                    if (bodies[i].isDynamic)
                    {
                        bodies[i].velocity = bodies[i].velocity + bodies[i].acceleration * DT;
                        bodies[i].position = bodies[i].position + bodies[i].velocity * DT;
                    }
                }
            }

            calculateEnergies();
            
            auto stepEnd = std::chrono::high_resolution_clock::now();
            double stepTime = std::chrono::duration<double, std::milli>(stepEnd - stepStart).count();
            
            totalRunTime += stepTime;
            totalForceTime += forceCalculationTime;
            totalReorderTime += sfcTime;
            totalBboxTime += bboxTime;
            minTime = std::min(minTime, (float)stepTime);
            maxTime = std::max(maxTime, (float)stepTime);
            totalPotentialEnergy += potentialEnergy;
            totalKineticEnergy += kineticEnergy;
            
            iterationCounter++;
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        totalTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();

        potentialEnergyAvg = totalPotentialEnergy / steps;
        kineticEnergyAvg = totalKineticEnergy / steps;
        totalEnergyAvg = potentialEnergyAvg + kineticEnergyAvg;

        printSummary(steps);
    }

    void printSummary(int steps)
    {
        std::cout << "Simulation complete." << std::endl;
        std::cout << "Performance Summary:" << std::endl;
        std::cout << "  Average time per step: " << std::fixed << std::setprecision(2) << totalTime / steps << " ms" << std::endl;
        std::cout << "  Compute forces: " << std::fixed << std::setprecision(2) << forceCalculationTime << " ms" << std::endl;
        std::cout << "  Bounding box: " << std::fixed << std::setprecision(2) << bboxTime << " ms" << std::endl;
        if (useSFC)
        {
            std::cout << "  Reordering: " << std::fixed << std::setprecision(2) << sfcTime << " ms" << std::endl;
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
        }
    }

    int getOptimalReorderFrequency() const
    {
        return reorderingStrategy.getOptimalFrequency();
    }

    double getTotalTime() const { return totalTime; }
    double getForceCalculationTime() const { return forceCalculationTime; }
    double getSortTime() const { return sfcTime; }
    int getNumBodies() const { return nBodies; }
    int getNumThreads() const { return numThreads; }
    int getSortType() const { return static_cast<int>(useSFC); }
    bool isDynamicReordering() const { return useSFC; }
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

int main(int argc, char *argv[])
{
    ArgumentParser parser("DirectSum CPU Simulation");
    
    // Add arguments with help messages and default values
    parser.add_argument("n", "Number of bodies", 1000);
    parser.add_flag("nosfc", "Disable Space-Filling Curve ordering");
    parser.add_argument("s", "Number of simulation steps", 100);
    parser.add_argument("t", "Number of threads (0 = auto)", 0);
    parser.add_argument("curve", "SFC curve type (morton, hilbert)", std::string("morton"));
    parser.add_argument("mass", "Mass distribution (uniform, normal)", std::string("normal"));
    parser.add_argument("seed", "Random seed", 42);
    
    // Parse command line arguments
    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        parser.print_help();
        return 1;
    }
    
    // Extract parsed arguments
    int nBodies = parser.get<int>("n");
    int steps = parser.get<int>("s");
    int threads = parser.get<int>("t");
    bool useSFC = !parser.get<bool>("nosfc");
    std::string curveStr = parser.get<std::string>("curve");
    SFCCurveType curveType = (curveStr == "hilbert") ? SFCCurveType::HILBERT : SFCCurveType::MORTON;
    
    // Parse mass distribution
    std::string massStr = parser.get<std::string>("mass");
    MassDistribution massDist = MassDistribution::NORMAL;
    if (massStr == "uniform") {
        massDist = MassDistribution::UNIFORM;
    }
    
    unsigned int seed = parser.get<int>("seed");
    
    DirectSum simulation(
        nBodies,
        threads > 0,
        threads,
        seed,
        massDist,
        useSFC,
        curveType);

    simulation.run(steps);

    return 0;
}


