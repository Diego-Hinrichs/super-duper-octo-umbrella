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
constexpr double DEFAULT_DOMAIN_SIZE = 1.0e6; // Default domain size for periodic boundaries

// Global domain size for periodic boundary conditions
double g_domainSize = DEFAULT_DOMAIN_SIZE;

// Periodic boundary condition functions
Vector applyPeriodicBoundary(Vector rij, double domainSize)
{
    Vector result = rij;
    double halfDomain = domainSize * 0.5;
    
    // Apply minimum image convention
    if (result.x > halfDomain) result.x -= domainSize;
    else if (result.x < -halfDomain) result.x += domainSize;
    
    if (result.y > halfDomain) result.y -= domainSize;
    else if (result.y < -halfDomain) result.y += domainSize;
    
    if (result.z > halfDomain) result.z -= domainSize;
    else if (result.z < -halfDomain) result.z += domainSize;
    
    return result;
}

Vector applyPeriodicPosition(Vector position, double domainSize)
{
    Vector result = position;
    
    // Wrap positions to stay within [0, L)
    result.x = fmod(result.x + domainSize, domainSize);
    if (result.x < 0) result.x += domainSize;
    
    result.y = fmod(result.y + domainSize, domainSize);
    if (result.y < 0) result.y += domainSize;
    
    result.z = fmod(result.z + domainSize, domainSize);
    if (result.z < 0) result.z += domainSize;
    
    return result;
}

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
    
    // New parameters for fixed reordering
    bool useFixedReordering;
    int fixedReorderFrequency;

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
    SFCDynamicReorderingStrategy(int windowSize = 10, bool fixedReordering = false, int reorderFreq = 10)
        : reorderTime(0.0),
          postReorderSimTime(0.0),
          updateTime(0.0),
          degradationRate(0.001),
          iterationsSinceReorder(0),
          currentOptimalFrequency(10),
          useFixedReordering(fixedReordering),
          fixedReorderFrequency(reorderFreq),
          metricsWindowSize(windowSize)
    {
        if (useFixedReordering) {
            currentOptimalFrequency = fixedReorderFrequency;
        }
    }

    void updateMetrics(double newReorderTime, double newSimTime)
    {
        // Only update metrics if we're not using fixed reordering
        if (!useFixedReordering) {
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
    }

    bool shouldReorder(double lastSimTime = 0.0, double lastReorderTime = 0.0)
    {
        iterationsSinceReorder++;
        
        // If using fixed reordering, just use the fixed frequency
        if (useFixedReordering) {
            if (iterationsSinceReorder >= fixedReorderFrequency) {
                iterationsSinceReorder = 0;
                return true;
            }
            return false;
        }
        
        // Otherwise use the dynamic approach
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

    void setFixedReordering(bool fixed, int frequency) {
        useFixedReordering = fixed;
        if (fixed && frequency > 0) {
            fixedReorderFrequency = frequency;
            currentOptimalFrequency = frequency;
        }
    }

    bool isFixedReordering() const {
        return useFixedReordering;
    }

    int getFixedReorderFrequency() const {
        return fixedReorderFrequency;
    }
    
    int getOptimalFrequency() const
    {
        return useFixedReordering ? fixedReorderFrequency : currentOptimalFrequency;
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
        if (!useFixedReordering) {
            currentOptimalFrequency = 10;
        }
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
        SFCCurveType sfcCurveType = SFCCurveType::MORTON,
        bool fixedReordering = false,
        int reorderFreq = 10) : nBodies(numBodies),
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
                                                           reorderingStrategy(10, fixedReordering, reorderFreq),
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
        // Use domain-relative distributions for periodic boundaries
        std::uniform_real_distribution<double> pos_dist(0.0, g_domainSize);
        std::uniform_real_distribution<double> vel_dist(-1.0e3, 1.0e3);
        std::normal_distribution<double> normal_pos_dist(g_domainSize * 0.5, g_domainSize * 0.25);
        std::normal_distribution<double> normal_vel_dist(0.0, 5.0e2);

        for (int i = 0; i < nBodies; i++)
        {
            if (massDist == MassDistribution::UNIFORM)
            {
                // Uniform distribution within the domain [0, L]
                bodies[i].position = Vector(
                    pos_dist(gen),
                    pos_dist(gen),
                    pos_dist(gen));

                bodies[i].velocity = Vector(
                    vel_dist(gen),
                    vel_dist(gen),
                    vel_dist(gen));
            }
            else
            {
                // Normal distribution centered in the domain
                bodies[i].position = Vector(
                    normal_pos_dist(gen),
                    normal_pos_dist(gen),
                    normal_pos_dist(gen));

                bodies[i].velocity = Vector(
                    normal_vel_dist(gen),
                    normal_vel_dist(gen),
                    normal_vel_dist(gen));
            }

            // Ensure positions are within domain bounds [0, L)
            bodies[i].position = applyPeriodicPosition(bodies[i].position, g_domainSize);

            bodies[i].mass = 1.0;
            bodies[i].radius = pow(bodies[i].mass / EARTH_MASS, 1.0 / 3.0) * (EARTH_DIA / 2.0);
            bodies[i].isDynamic = true;
            bodies[i].acceleration = Vector(0, 0, 0);
        }
    }

    void updateBoundingBox()
    {
        // Use fixed domain bounds for periodic boundary conditions
        minBound = Vector(0.0, 0.0, 0.0);
        maxBound = Vector(g_domainSize, g_domainSize, g_domainSize);
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
                    
                    // Apply periodic boundary conditions
                    diff = applyPeriodicBoundary(diff, g_domainSize);
                    
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
                    
                    // Apply periodic boundary conditions
                    diff = applyPeriodicBoundary(diff, g_domainSize);
                    
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
                        
                        // Apply periodic boundary conditions
                        diff = applyPeriodicBoundary(diff, g_domainSize);
                        
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
                    
                    // Apply periodic boundary conditions
                    diff = applyPeriodicBoundary(diff, g_domainSize);
                    
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
                    
                    // Apply periodic boundary conditions to positions
                    bodies[i].position = applyPeriodicPosition(bodies[i].position, g_domainSize);
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
                    
                    // Apply periodic boundary conditions to positions
                    bodies[i].position = applyPeriodicPosition(bodies[i].position, g_domainSize);
                }
            }
        }

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

    void run(int steps, bool calculateEnergy = true)
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
            
            update();

            auto stepEnd = std::chrono::high_resolution_clock::now();
            double stepTime = std::chrono::duration<double, std::milli>(stepEnd - stepStart).count();
            
            // Calcular energías por separado (no afecta el tiempo de simulación)
            if (calculateEnergy) {
                calculateEnergies();
                totalPotentialEnergy += potentialEnergy;
                totalKineticEnergy += kineticEnergy;
            }
            
            totalRunTime += stepTime;
            totalForceTime += forceCalculationTime;
            totalReorderTime += sfcTime;
            totalBboxTime += bboxTime;
            minTime = std::min(minTime, (float)stepTime);
            maxTime = std::max(maxTime, (float)stepTime);
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        totalTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();

        if (calculateEnergy) {
            potentialEnergyAvg = totalPotentialEnergy / steps;
            kineticEnergyAvg = totalKineticEnergy / steps;
            totalEnergyAvg = potentialEnergyAvg + kineticEnergyAvg;
        }

        printSummary(steps, calculateEnergy);
    }

    void printSummary(int steps, bool calculateEnergy = true)
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
        
        if (calculateEnergy) {
            std::cout << "Average Energy Values:" << std::endl;
            std::cout << "  Potential Energy: " << std::scientific << std::setprecision(6) << potentialEnergyAvg << std::endl;
            std::cout << "  Kinetic Energy:   " << std::scientific << std::setprecision(6) << kineticEnergyAvg << std::endl;
            std::cout << "  Total Energy:     " << std::scientific << std::setprecision(6) << totalEnergyAvg << std::endl;
        }

        if (useSFC)
        {
            std::cout << "SFC Configuration:" << std::endl;
            std::cout << "  Fixed reordering: " << (reorderingStrategy.isFixedReordering() ? "Yes" : "No") << std::endl;
            std::cout << "  Optimal reorder freq: " << reorderingStrategy.getOptimalFrequency() << std::endl;
            if (!reorderingStrategy.isFixedReordering()) {
                std::cout << "  Degradation rate: " << std::fixed << std::setprecision(6) 
                          << reorderingStrategy.getDegradationRate() << " ms/iter" << std::endl;
            }
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
    parser.add_argument("freq", "Reordering frequency for fixed mode", 10);
    parser.add_argument("fixreorder", "Use fixed reordering frequency (1=yes, 0=no)", 0);
    parser.add_argument("dist", "Body distribution (galaxy, solar, uniform, random)", std::string("galaxy"));
    parser.add_argument("mass", "Mass distribution (uniform, normal)", std::string("normal"));
    parser.add_argument("seed", "Random seed", 42);
    parser.add_argument("curve", "SFC curve type (morton, hilbert)", std::string("morton"));
    parser.add_argument("s", "Number of simulation steps", 100);
    parser.add_argument("t", "Number of threads (0 = auto)", 0);
    parser.add_argument("l", "Domain size L for periodic boundary conditions", DEFAULT_DOMAIN_SIZE);
    parser.add_argument("energy", "Calculate system energy (1=yes, 0=no)", 1);
    
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
    int reorderFreq = parser.get<int>("freq");
    bool useFixedReordering = parser.get<int>("fixreorder") != 0;
    bool calculateEnergy = parser.get<int>("energy") != 0;
    
    // Parse distribution type
    std::string distStr = parser.get<std::string>("dist");
    BodyDistribution bodyDist = BodyDistribution::GALAXY;
    if (distStr == "solar") {
        bodyDist = BodyDistribution::SOLAR_SYSTEM;
    } else if (distStr == "uniform") {
        bodyDist = BodyDistribution::RANDOM_UNIFORM;
    } else if (distStr == "random") {
        bodyDist = BodyDistribution::RANDOM_UNIFORM;
    }
    
    std::string curveStr = parser.get<std::string>("curve");
    SFCCurveType curveType = (curveStr == "hilbert") ? SFCCurveType::HILBERT : SFCCurveType::MORTON;
    
    // Parse mass distribution
    std::string massStr = parser.get<std::string>("mass");
    MassDistribution massDist = MassDistribution::NORMAL;
    if (massStr == "uniform") {
        massDist = MassDistribution::UNIFORM;
    }
    
    unsigned int seed = parser.get<int>("seed");
    double domainSize = parser.get<double>("l");
    
    // Set global domain size
    g_domainSize = domainSize;
    
    std::cout << "DirectSum CPU Simulation" << std::endl;
    std::cout << "Bodies: " << nBodies << std::endl;
    std::cout << "Steps: " << steps << std::endl;
    std::cout << "Threads: " << (threads > 0 ? threads : omp_get_max_threads()) << std::endl;
    std::cout << "Domain size (periodic boundaries): " << std::scientific << domainSize << std::endl;
    std::cout << "Calculate energy: " << (calculateEnergy ? "Yes" : "No") << std::endl;
    std::cout << "Using SFC: " << (useSFC ? "Yes" : "No") << std::endl;
    if (useSFC) {
        std::cout << "Curve type: " << (curveType == SFCCurveType::HILBERT ? "HILBERT" : "MORTON") << std::endl;
        std::cout << "Fixed reordering: " << (useFixedReordering ? "Yes" : "No") << std::endl;
        if (useFixedReordering) {
            std::cout << "Reordering frequency: " << reorderFreq << std::endl;
        }
    }
    
    DirectSum simulation(
        nBodies,
        threads > 0,
        threads,
        seed,
        massDist,
        useSFC,
        curveType,
        useFixedReordering,
        reorderFreq);

    simulation.run(steps, calculateEnergy);

    return 0;
}


