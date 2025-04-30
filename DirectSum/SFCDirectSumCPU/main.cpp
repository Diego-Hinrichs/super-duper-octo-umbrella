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

    Vector operator+(const Vector &other) const
    {
        return Vector(x + other.x, y + other.y, z + other.z);
    }

    Vector operator-(const Vector &other) const
    {
        return Vector(x - other.x, y - other.y, z - other.z);
    }

    Vector operator*(double scalar) const
    {
        return Vector(x * scalar, y * scalar, z * scalar);
    }

    double dot(const Vector &other) const
    {
        return x * other.x + y * other.y + z * other.z;
    }

    double lengthSquared() const
    {
        return x * x + y * y + z * z;
    }

    double length() const
    {
        return sqrt(lengthSquared());
    }
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

inline uint64_t expandBits(uint64_t v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

uint64_t mortonEncode(double x, double y, double z, const Vector &min, const Vector &max)
{

    double normalizedX = (x - min.x) / (max.x - min.x);
    double normalizedY = (y - min.y) / (max.y - min.y);
    double normalizedZ = (z - min.z) / (max.z - min.z);

    normalizedX = std::max(0.0, std::min(1.0, normalizedX));
    normalizedY = std::max(0.0, std::min(1.0, normalizedY));
    normalizedZ = std::max(0.0, std::min(1.0, normalizedZ));

    uint64_t intX = static_cast<uint64_t>(normalizedX * 1023.0);
    uint64_t intY = static_cast<uint64_t>(normalizedY * 1023.0);
    uint64_t intZ = static_cast<uint64_t>(normalizedZ * 1023.0);

    uint64_t mortonCode = expandBits(intX) | (expandBits(intY) << 1) | (expandBits(intZ) << 2);

    return mortonCode;
}

class SFCCPUDirectSum
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
    int reorderFrequency;
    int iterationCounter;

    std::vector<uint64_t> mortonCodes;
    std::vector<int> orderedIndices;

    Vector minBound;
    Vector maxBound;

    bool useDynamicReordering;
    SFCDynamicReorderingStrategy reorderingStrategy;

public:
    SFCCPUDirectSum(
        int numBodies,
        bool useParallelization = true,
        int threads = 0,
        bool enableSFC = true,
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

        bodies.resize(numBodies);
        initializeBodies(dist, seed, massDist);

        minBound = Vector(std::numeric_limits<double>::max(),
                          std::numeric_limits<double>::max(),
                          std::numeric_limits<double>::max());
        maxBound = Vector(std::numeric_limits<double>::lowest(),
                          std::numeric_limits<double>::lowest(),
                          std::numeric_limits<double>::lowest());

        if (useSFC)
        {
            mortonCodes.resize(numBodies);
            orderedIndices.resize(numBodies);

            for (int i = 0; i < numBodies; i++)
            {
                orderedIndices[i] = i;
            }
        }

        std::cout << "CPU SFC Direct Sum Simulation created with " << numBodies << " bodies." << std::endl;
        if (useOpenMP)
        {
            std::cout << "OpenMP enabled with " << numThreads << " threads." << std::endl;
        }
        else
        {
            std::cout << "OpenMP disabled, using single-threaded mode." << std::endl;
        }

        if (useSFC)
        {
            std::cout << "Space-Filling Curve ordering enabled with "
                      << (useDynamicReordering ? "dynamic" : "fixed") << " reorder frequency";
            if (!useDynamicReordering)
            {
                std::cout << " " << reorderFrequency;
            }
            std::cout << std::endl;
        }
    }

    void initializeBodies(BodyDistribution dist, unsigned int seed, MassDistribution massDist)
    {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> posDistrib(-100.0, 100.0);
        std::uniform_real_distribution<double> velDistrib(-5.0, 5.0);
        std::normal_distribution<double> normalPosDistrib(0.0, 50.0);
        std::normal_distribution<double> normalVelDistrib(0.0, 2.5);

        for (int i = 0; i < nBodies; i++)
        {
            if (massDist == MassDistribution::UNIFORM)
            {
                bodies[i].position = Vector(posDistrib(rng), posDistrib(rng), posDistrib(rng));
                bodies[i].velocity = Vector(velDistrib(rng), velDistrib(rng), velDistrib(rng));
            }
            else
            {
                bodies[i].position = Vector(normalPosDistrib(rng), normalPosDistrib(rng), normalPosDistrib(rng));
                bodies[i].velocity = Vector(normalVelDistrib(rng), normalVelDistrib(rng), normalVelDistrib(rng));
            }

            bodies[i].mass = 1.0;
            bodies[i].isDynamic = true;
        }
    }

    void computeBoundingBox()
    {
        auto start = std::chrono::high_resolution_clock::now();

        minBound = Vector(std::numeric_limits<double>::max(),
                          std::numeric_limits<double>::max(),
                          std::numeric_limits<double>::max());
        maxBound = Vector(std::numeric_limits<double>::lowest(),
                          std::numeric_limits<double>::lowest(),
                          std::numeric_limits<double>::lowest());

        if (useOpenMP)
        {

            omp_set_num_threads(numThreads);

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

                    localMin[tid].x = std::min(localMin[tid].x, bodies[i].position.x);
                    localMin[tid].y = std::min(localMin[tid].y, bodies[i].position.y);
                    localMin[tid].z = std::min(localMin[tid].z, bodies[i].position.z);

                    localMax[tid].x = std::max(localMax[tid].x, bodies[i].position.x);
                    localMax[tid].y = std::max(localMax[tid].y, bodies[i].position.y);
                    localMax[tid].z = std::max(localMax[tid].z, bodies[i].position.z);
                }
            }

            for (int i = 0; i < numThreads; i++)
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

            for (int i = 0; i < nBodies; i++)
            {

                minBound.x = std::min(minBound.x, bodies[i].position.x);
                minBound.y = std::min(minBound.y, bodies[i].position.y);
                minBound.z = std::min(minBound.z, bodies[i].position.z);

                maxBound.x = std::max(maxBound.x, bodies[i].position.x);
                maxBound.y = std::max(maxBound.y, bodies[i].position.y);
                maxBound.z = std::max(maxBound.z, bodies[i].position.z);
            }
        }

        double padding = std::max(1.0e-10, (maxBound.x - minBound.x) * 0.01);
        minBound.x -= padding;
        minBound.y -= padding;
        minBound.z -= padding;
        maxBound.x += padding;
        maxBound.y += padding;
        maxBound.z += padding;

        auto end = std::chrono::high_resolution_clock::now();
        bboxTime = std::chrono::duration<double, std::milli>(end - start).count();
    }

    void orderBodiesBySFC()
    {
        auto start = std::chrono::high_resolution_clock::now();

        if (!useSFC)
            return;

        if (minBound.x == std::numeric_limits<double>::max())
        {
            computeBoundingBox();
        }

        if (useOpenMP)
        {
            omp_set_num_threads(numThreads);

#pragma omp parallel for
            for (int i = 0; i < nBodies; i++)
            {
                mortonCodes[i] = mortonEncode(
                    bodies[i].position.x,
                    bodies[i].position.y,
                    bodies[i].position.z,
                    minBound,
                    maxBound);
            }
        }
        else
        {
            for (int i = 0; i < nBodies; i++)
            {
                mortonCodes[i] = mortonEncode(
                    bodies[i].position.x,
                    bodies[i].position.y,
                    bodies[i].position.z,
                    minBound,
                    maxBound);
            }
        }

        for (int i = 0; i < nBodies; i++)
        {
            orderedIndices[i] = i;
        }

        std::sort(orderedIndices.begin(), orderedIndices.end(),
                  [this](int a, int b)
                  {
                      return mortonCodes[a] < mortonCodes[b];
                  });

        std::vector<Body> reorderedBodies(nBodies);
        for (int i = 0; i < nBodies; i++)
        {
            reorderedBodies[i] = bodies[orderedIndices[i]];
        }

        bodies.swap(reorderedBodies);

        auto end = std::chrono::high_resolution_clock::now();
        reorderTime = std::chrono::duration<double, std::milli>(end - start).count();
    }

    void computeForces()
    {
        auto start = std::chrono::high_resolution_clock::now();

        if (useOpenMP)
        {

            omp_set_num_threads(numThreads);

#pragma omp parallel for
            for (int i = 0; i < nBodies; i++)
            {

                if (!bodies[i].isDynamic)
                    continue;

                bodies[i].acceleration = Vector(0.0, 0.0, 0.0);

                for (int j = 0; j < nBodies; j++)
                {
                    if (i == j)
                        continue;

                    Vector r = bodies[j].position - bodies[i].position;

                    double distSqr = r.lengthSquared() + (E * E);
                    double dist = sqrt(distSqr);

                    if (dist < COLLISION_TH)
                        continue;

                    double forceMag = GRAVITY * bodies[i].mass * bodies[j].mass / (distSqr * dist);

                    bodies[i].acceleration.x += (r.x * forceMag) / bodies[i].mass;
                    bodies[i].acceleration.y += (r.y * forceMag) / bodies[i].mass;
                    bodies[i].acceleration.z += (r.z * forceMag) / bodies[i].mass;
                }

                bodies[i].velocity.x += bodies[i].acceleration.x * DT;
                bodies[i].velocity.y += bodies[i].acceleration.y * DT;
                bodies[i].velocity.z += bodies[i].acceleration.z * DT;

                bodies[i].position.x += bodies[i].velocity.x * DT;
                bodies[i].position.y += bodies[i].velocity.y * DT;
                bodies[i].position.z += bodies[i].velocity.z * DT;
            }
        }
        else
        {

            for (int i = 0; i < nBodies; i++)
            {

                if (!bodies[i].isDynamic)
                    continue;

                bodies[i].acceleration = Vector(0.0, 0.0, 0.0);

                for (int j = 0; j < nBodies; j++)
                {
                    if (i == j)
                        continue;

                    Vector r = bodies[j].position - bodies[i].position;

                    double distSqr = r.lengthSquared() + (E * E);
                    double dist = sqrt(distSqr);

                    if (dist < COLLISION_TH)
                        continue;

                    double forceMag = GRAVITY * bodies[i].mass * bodies[j].mass / (distSqr * dist);

                    bodies[i].acceleration.x += (r.x * forceMag) / bodies[i].mass;
                    bodies[i].acceleration.y += (r.y * forceMag) / bodies[i].mass;
                    bodies[i].acceleration.z += (r.z * forceMag) / bodies[i].mass;
                }

                bodies[i].velocity.x += bodies[i].acceleration.x * DT;
                bodies[i].velocity.y += bodies[i].acceleration.y * DT;
                bodies[i].velocity.z += bodies[i].acceleration.z * DT;

                bodies[i].position.x += bodies[i].velocity.x * DT;
                bodies[i].position.y += bodies[i].velocity.y * DT;
                bodies[i].position.z += bodies[i].velocity.z * DT;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        forceCalculationTime = std::chrono::duration<double, std::milli>(end - start).count();
    }

    void calculateEnergies()
    {
        potentialEnergy = 0.0;
        kineticEnergy = 0.0;

        for (int i = 0; i < nBodies; i++)
        {
            for (int j = i + 1; j < nBodies; j++)
            {

                Vector r = bodies[j].position - bodies[i].position;

                double distSqr = r.lengthSquared() + (E * E);
                double dist = sqrt(distSqr);

                if (dist < COLLISION_TH)
                    continue;

                potentialEnergy -= GRAVITY * bodies[i].mass * bodies[j].mass / dist;
            }
        }

        for (int i = 0; i < nBodies; i++)
        {
            if (!bodies[i].isDynamic)
                continue;

            double vSquared = bodies[i].velocity.lengthSquared();
            kineticEnergy += 0.5 * bodies[i].mass * vSquared;
        }
    }

    void update()
    {
        auto start = std::chrono::high_resolution_clock::now();

        double lastSimTime = forceCalculationTime;
        double lastReorderTime = reorderTime;

        bool shouldReorder = false;
        if (useSFC)
        {
            if (useDynamicReordering)
            {

                shouldReorder = reorderingStrategy.shouldReorder(lastSimTime, lastReorderTime);
            }
            else
            {

                shouldReorder = (iterationCounter % reorderFrequency == 0);
            }

            if (shouldReorder)
            {

                computeBoundingBox();
                orderBodiesBySFC();
            }
        }

        iterationCounter++;

        computeForces();

        calculateEnergies();

        auto end = std::chrono::high_resolution_clock::now();
        totalTime = std::chrono::duration<double, std::milli>(end - start).count();

        if (useSFC && useDynamicReordering)
        {
            reorderingStrategy.updateMetrics(shouldReorder ? reorderTime : 0.0, forceCalculationTime);
        }
    }

    void printPerformanceMetrics() const
    {
        std::cout << "Performance Metrics (ms):" << std::endl;
        std::cout << "  Total time:           " << std::fixed << std::setprecision(3) << totalTime << std::endl;
        std::cout << "  Bounding box:         " << std::fixed << std::setprecision(3) << bboxTime << std::endl;
        std::cout << "  SFC reordering:       " << std::fixed << std::setprecision(3) << reorderTime << std::endl;
        std::cout << "  Force calculation:    " << std::fixed << std::setprecision(3) << forceCalculationTime << std::endl;

        if (useSFC && useDynamicReordering)
        {
            std::cout << "  Optimal reorder freq: " << reorderingStrategy.getOptimalFrequency() << std::endl;
            std::cout << "  Degradation rate:     " << std::fixed << std::setprecision(6)
                      << reorderingStrategy.getDegradationRate() << " ms/iter" << std::endl;
        }
    }

    void run(int steps)
    {
        std::cout << "Running CPU SFC Direct Sum simulation for " << steps << " steps..." << std::endl;

        double totalSim = 0.0;
        double totalPotentialEnergy = 0.0;
        double totalKineticEnergy = 0.0;

        for (int step = 0; step < steps; step++)
        {
            update();
            totalSim += totalTime;
            totalPotentialEnergy += potentialEnergy;
            totalKineticEnergy += kineticEnergy;
        }

        potentialEnergyAvg = totalPotentialEnergy / steps;
        kineticEnergyAvg = totalKineticEnergy / steps;
        totalEnergyAvg = potentialEnergyAvg + kineticEnergyAvg;

        std::cout << "Simulation completed in " << totalSim << " ms." << std::endl;
        std::cout << "Average step time: " << totalSim / steps << " ms." << std::endl;
        std::cout << "Average Energy Values:" << std::endl;
        std::cout << "  Potential Energy: " << std::scientific << std::setprecision(6) << potentialEnergyAvg << std::endl;
        std::cout << "  Kinetic Energy:   " << std::scientific << std::setprecision(6) << kineticEnergyAvg << std::endl;
        std::cout << "  Total Energy:     " << std::scientific << std::setprecision(6) << totalEnergyAvg << std::endl;

        if (useSFC && useDynamicReordering)
        {
            std::cout << "SFC Configuration:" << std::endl;
            std::cout << "  Optimal reorder freq: " << reorderingStrategy.getOptimalFrequency() << std::endl;
            std::cout << "  Degradation rate:     " << std::fixed << std::setprecision(6)
                      << reorderingStrategy.getDegradationRate() << " ms/iter" << std::endl;
        }
    }

    int getOptimalReorderFrequency() const
    {
        if (useDynamicReordering)
        {
            return reorderingStrategy.getOptimalFrequency();
        }
        return reorderFrequency;
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
         << "CPU_SFC_Direct_Sum" << ","
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
    bool useOpenMP = true;
    int threads = 0;
    BodyDistribution dist = BodyDistribution::RANDOM_UNIFORM;
    MassDistribution massDist = MassDistribution::UNIFORM;
    int steps = 100;
    bool useSFC = true;
    int reorderFreq = 10;
    bool useDynamicReordering = true;

    bool saveMetricsToFile = false;
    std::string metricsFile = "./SFCDirectSumCPU_metrics.csv";

    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];

        if (arg == "-n" && i + 1 < argc)
        {
            nBodies = std::stoi(argv[++i]);
        }
        else if (arg == "--no-openmp")
        {
            useOpenMP = false;
        }
        else if (arg == "--threads" && i + 1 < argc)
        {
            threads = std::stoi(argv[++i]);
        }
        else if (arg == "--distribution" && i + 1 < argc)
        {
            std::string distStr = argv[++i];
            if (distStr == "random")
            {
                dist = BodyDistribution::RANDOM_UNIFORM;
            }
            else if (distStr == "solar")
            {
                dist = BodyDistribution::SOLAR_SYSTEM;
            }
            else if (distStr == "galaxy")
            {
                dist = BodyDistribution::GALAXY;
            }
            else if (distStr == "collision")
            {
                dist = BodyDistribution::COLLISION;
            }
        }
        else if (arg == "--mass" && i + 1 < argc)
        {
            std::string massStr = argv[++i];
            if (massStr == "uniform")
            {
                massDist = MassDistribution::UNIFORM;
            }
            else if (massStr == "normal")
            {
                massDist = MassDistribution::NORMAL;
            }
        }
        else if (arg == "--steps" && i + 1 < argc)
        {
            steps = std::stoi(argv[++i]);
        }
        else if (arg == "--no-sfc")
        {
            useSFC = false;
        }
        else if (arg == "--reorder-freq" && i + 1 < argc)
        {

            i++;
        }
        else if (arg == "--dynamic-reordering")
        {
        }
        else if (arg == "--save-metrics")
        {
            saveMetricsToFile = true;
        }
        else if (arg == "--metrics-file" && i + 1 < argc)
        {
            metricsFile = argv[++i];
        }
        else if (arg == "--help")
        {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  -n N          Set number of bodies (default: 1000)" << std::endl;
            std::cout << "  --no-openmp         Disable OpenMP parallelization" << std::endl;
            std::cout << "  --threads N         Set number of threads (default: auto)" << std::endl;
            std::cout << "  --distribution TYPE Set body distribution (random, solar, galaxy, collision)" << std::endl;
            std::cout << "  --mass TYPE         Set mass distribution (uniform, normal)" << std::endl;
            std::cout << "  --steps N           Set simulation steps (default: 100)" << std::endl;
            std::cout << "  --no-sfc            Disable Space-Filling Curve ordering" << std::endl;
            std::cout << "  --save-metrics      Save metrics to CSV" << std::endl;
            std::cout << "  --metrics-file FILE Set metrics file (default: metrics.csv)" << std::endl;
            std::cout << "  --help              Show this help message" << std::endl;
            return 0;
        }
    }

    if (saveMetricsToFile)
    {
        std::ifstream checkFile(metricsFile);
        bool fileExists = checkFile.good();
        checkFile.close();

        initializeCsv(metricsFile, fileExists);
    }

    SFCCPUDirectSum simulation(
        nBodies,
        useOpenMP,
        threads,
        useSFC,
        reorderFreq,
        dist,
        time(nullptr),
        massDist,
        useDynamicReordering);

    simulation.run(steps);

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
            steps,
            simulation.getNumThreads(),
            simulation.getSortType(),
            simulation.getTotalTime(),
            simulation.getForceCalculationTime(),
            simulation.getSortTime(),
            simulation.getPotentialEnergy(),
            simulation.getKineticEnergy(),
            simulation.getTotalEnergy());

        std::cout << "Métricas guardadas en: " << metricsFile << std::endl;
    }

    return 0;
}