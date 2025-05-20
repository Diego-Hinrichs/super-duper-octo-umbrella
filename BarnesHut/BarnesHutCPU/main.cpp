#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <chrono>
#include <random>
#include <string>
#include <iomanip>
#include <memory>
#include <algorithm>
#include <limits>
#include <bitset>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <deque>
#include <numeric>
#include "../../argparse.hpp"

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
    uint64_t mortonCode;

    Body() : mass(1.0), isDynamic(true), mortonCode(0) {}

    Body(const Vector &pos, const Vector &vel, double m, bool dynamic = true)
        : position(pos), velocity(vel), acceleration(), mass(m), isDynamic(dynamic), mortonCode(0) {}
};

constexpr double GRAVITY = 6.67430e-11;
constexpr double DT = 0.005;
constexpr double E = 0.01;
constexpr double COLLISION_TH = 0.01;
constexpr double THETA = 0.5;

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

enum class SFCCurveType
{
    MORTON,
    HILBERT
};

struct CPUOctreeNode
{
    Vector center;
    double halfWidth;

    bool isLeaf;
    int bodyIndex;

    Vector centerOfMass;
    double totalMass;

    std::vector<int> bodies;
    CPUOctreeNode *children[8];

    CPUOctreeNode() : center(),
                      halfWidth(0.0),
                      isLeaf(true),
                      bodyIndex(-1),
                      centerOfMass(),
                      totalMass(0.0)
    {
        for (int i = 0; i < 8; i++)
        {
            children[i] = nullptr;
        }
    }

    ~CPUOctreeNode()
    {
        for (int i = 0; i < 8; i++)
        {
            if (children[i])
            {
                delete children[i];
                children[i] = nullptr;
            }
        }
    }

    int getOctant(const Vector &pos) const
    {
        int oct = 0;
        if (pos.x >= center.x)
            oct |= 1;
        if (pos.y >= center.y)
            oct |= 2;
        if (pos.z >= center.z)
            oct |= 4;
        return oct;
    }

    Vector getOctantCenter(int octant) const
    {
        Vector oct_center = center;
        double offset = halfWidth * 0.5;

        if (octant & 1)
            oct_center.x += offset;
        else
            oct_center.x -= offset;

        if (octant & 2)
            oct_center.y += offset;
        else
            oct_center.y -= offset;

        if (octant & 4)
            oct_center.z += offset;
        else
            oct_center.z -= offset;

        return oct_center;
    }
};

inline uint32_t expandBits(uint32_t v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

inline void rotateHilbert(uint32_t n, uint32_t *x, uint32_t *y, uint32_t *z, uint32_t rx, uint32_t ry, uint32_t rz)
{
    if (ry == 0)
    {
        if (rx == 1)
        {
            *x = n - 1 - *x;
            *y = n - 1 - *y;
        }
        std::swap(*x, *y);
    }
    if (rz == 1)
    {
        *x = n - 1 - *x;
        *z = n - 1 - *z;
    }
    std::swap(*x, *z);
}

inline uint64_t hilbertXYZToIndex(uint32_t n, uint32_t x, uint32_t y, uint32_t z)
{
    uint32_t rx, ry, rz, s, d = 0;
    for (s = n / 2; s > 0; s /= 2)
    {
        rx = (x & s) > 0;
        ry = (y & s) > 0;
        rz = (z & s) > 0;
        d += s * s * ((3 * rx) ^ ry);
        rotateHilbert(n, &x, &y, &z, rx, ry, rz);
    }
    return d;
}

inline uint64_t calculateHilbertCode(double x, double y, double z,
                                   double minX, double minY, double minZ,
                                   double maxX, double maxY, double maxZ)
{
    const uint32_t n = 1 << 10; // 10 bits per dimension
    uint32_t ix = static_cast<uint32_t>((x - minX) / (maxX - minX) * (n - 1));
    uint32_t iy = static_cast<uint32_t>((y - minY) / (maxY - minY) * (n - 1));
    uint32_t iz = static_cast<uint32_t>((z - minZ) / (maxZ - minZ) * (n - 1));
    return hilbertXYZToIndex(n, ix, iy, iz);
}

inline uint64_t calculateMortonCode(double x, double y, double z,
                                  double minX, double minY, double minZ,
                                  double maxX, double maxY, double maxZ)
{
    const uint32_t n = 1 << 10; // 10 bits per dimension
    uint32_t ix = static_cast<uint32_t>((x - minX) / (maxX - minX) * (n - 1));
    uint32_t iy = static_cast<uint32_t>((y - minY) / (maxY - minY) * (n - 1));
    uint32_t iz = static_cast<uint32_t>((z - minZ) / (maxZ - minZ) * (n - 1));

    ix = expandBits(ix);
    iy = expandBits(iy);
    iz = expandBits(iz);

    return (static_cast<uint64_t>(ix) << 2) | (static_cast<uint64_t>(iy) << 1) | static_cast<uint64_t>(iz);
}

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

class BarnesHut
{
private:
    std::vector<Body> bodies;
    int nBodies;
    bool useOpenMP;
    int numThreads;
    double totalTime;
    double forceCalculationTime;
    double octreeTime;
    double bboxTime;
    double sfcTime;
    double potentialEnergy;
    double kineticEnergy;
    double totalEnergyAvg;
    double potentialEnergyAvg;
    double kineticEnergyAvg;

    std::unique_ptr<CPUOctreeNode> root;

    Vector minBound;
    Vector maxBound;

    bool useSFC;
    SFCCurveType curveType;
    int iterationCounter;
    SFCDynamicReorderingStrategy reorderingStrategy;

public:
    BarnesHut(
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
                                                           octreeTime(0.0),
                                                           bboxTime(0.0),
                                                           sfcTime(0.0),
                                                           potentialEnergy(0.0),
                                                           kineticEnergy(0.0),
                                                           totalEnergyAvg(0.0),
                                                           potentialEnergyAvg(0.0),
                                                           kineticEnergyAvg(0.0),
                                                           useSFC(useSFC_),
                                                           curveType(sfcCurveType),
                                                           iterationCounter(0),
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

        initRandomBodies(seed, massDist);
    }

    void initRandomBodies(unsigned int seed, MassDistribution massDist)
    {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> posDist(-1.0, 1.0);
        std::uniform_real_distribution<double> velDist(-0.1, 0.1);

        bodies.resize(nBodies);
        for (int i = 0; i < nBodies; i++)
        {
            Vector pos(posDist(rng), posDist(rng), posDist(rng));
            Vector vel(velDist(rng), velDist(rng), velDist(rng));
            double mass = 1.0;

            if (massDist == MassDistribution::NORMAL)
            {
                std::normal_distribution<double> massDist(1.0, 0.2);
                mass = std::abs(massDist(rng));
            }

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

        auto end = std::chrono::high_resolution_clock::now();
        bboxTime = std::chrono::duration<double, std::milli>(end - start).count();
    }

    void computeMortonCodes()
    {
        if (!useSFC)
            return;

        auto start = std::chrono::high_resolution_clock::now();

        if (useOpenMP)
        {
#pragma omp parallel for
            for (int i = 0; i < nBodies; i++)
            {
                if (curveType == SFCCurveType::MORTON)
                {
                    bodies[i].mortonCode = calculateMortonCode(
                        bodies[i].position.x, bodies[i].position.y, bodies[i].position.z,
                        minBound.x, minBound.y, minBound.z,
                        maxBound.x, maxBound.y, maxBound.z);
                }
                else
                {
                    bodies[i].mortonCode = calculateHilbertCode(
                        bodies[i].position.x, bodies[i].position.y, bodies[i].position.z,
                        minBound.x, minBound.y, minBound.z,
                        maxBound.x, maxBound.y, maxBound.z);
                }
            }
        }
        else
        {
            for (int i = 0; i < nBodies; i++)
            {
                if (curveType == SFCCurveType::MORTON)
                {
                    bodies[i].mortonCode = calculateMortonCode(
                        bodies[i].position.x, bodies[i].position.y, bodies[i].position.z,
                        minBound.x, minBound.y, minBound.z,
                        maxBound.x, maxBound.y, maxBound.z);
                }
                else
                {
                    bodies[i].mortonCode = calculateHilbertCode(
                        bodies[i].position.x, bodies[i].position.y, bodies[i].position.z,
                        minBound.x, minBound.y, minBound.z,
                        maxBound.x, maxBound.y, maxBound.z);
                }
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        sfcTime = std::chrono::duration<double, std::milli>(end - start).count();
    }

    void sortBodiesBySFC()
    {
        if (!useSFC)
            return;

        auto start = std::chrono::high_resolution_clock::now();

        std::sort(bodies.begin(), bodies.end(),
                 [](const Body &a, const Body &b)
                 {
                     return a.mortonCode < b.mortonCode;
                 });

        auto end = std::chrono::high_resolution_clock::now();
        sfcTime += std::chrono::duration<double, std::milli>(end - start).count();
    }

    void reorderBodies()
    {
        if (!useSFC)
            return;

        computeBoundingBox();
        computeMortonCodes();
        sortBodiesBySFC();
    }

    void buildOctree()
    {
        auto start = std::chrono::high_resolution_clock::now();

        computeBoundingBox();
        if (useSFC)
        {
            computeMortonCodes();
            sortBodiesBySFC();
        }

        root = std::make_unique<CPUOctreeNode>();
        root->center = (minBound + maxBound) * 0.5;
        root->halfWidth = std::max({maxBound.x - minBound.x,
                                  maxBound.y - minBound.y,
                                  maxBound.z - minBound.z}) *
                         0.5;

        for (int i = 0; i < nBodies; i++)
        {
            CPUOctreeNode *current = root.get();
            while (!current->isLeaf)
            {
                int octant = current->getOctant(bodies[i].position);
                if (!current->children[octant])
                {
                    current->children[octant] = new CPUOctreeNode();
                    current->children[octant]->center = current->getOctantCenter(octant);
                    current->children[octant]->halfWidth = current->halfWidth * 0.5;
                }
                current = current->children[octant];
            }

            if (current->bodyIndex == -1)
            {
                current->bodyIndex = i;
            }
            else
            {
                current->isLeaf = false;
                int oldBodyIndex = current->bodyIndex;
                current->bodyIndex = -1;

                int octant = current->getOctant(bodies[oldBodyIndex].position);
                current->children[octant] = new CPUOctreeNode();
                current->children[octant]->center = current->getOctantCenter(octant);
                current->children[octant]->halfWidth = current->halfWidth * 0.5;
                current->children[octant]->bodyIndex = oldBodyIndex;

                octant = current->getOctant(bodies[i].position);
                if (!current->children[octant])
                {
                    current->children[octant] = new CPUOctreeNode();
                    current->children[octant]->center = current->getOctantCenter(octant);
                    current->children[octant]->halfWidth = current->halfWidth * 0.5;
                }
                current->children[octant]->bodyIndex = i;
            }
        }

        calculateCenterOfMass(root.get());

        auto end = std::chrono::high_resolution_clock::now();
        octreeTime = std::chrono::duration<double, std::milli>(end - start).count();
    }

    void calculateCenterOfMass(CPUOctreeNode *node)
    {
        if (!node)
            return;

        if (node->isLeaf)
        {
            if (node->bodyIndex != -1)
            {
                node->centerOfMass = bodies[node->bodyIndex].position;
                node->totalMass = bodies[node->bodyIndex].mass;
            }
            return;
        }

        Vector totalPos;
        double totalMass = 0.0;

        for (int i = 0; i < 8; i++)
        {
            if (node->children[i])
            {
                calculateCenterOfMass(node->children[i]);
                totalPos = totalPos + node->children[i]->centerOfMass * node->children[i]->totalMass;
                totalMass += node->children[i]->totalMass;
            }
        }

        if (totalMass > 0)
        {
            node->centerOfMass = totalPos * (1.0 / totalMass);
            node->totalMass = totalMass;
        }
    }

    void computeForceFromNode(Body &body, const CPUOctreeNode *node)
    {
        if (!node)
            return;

        Vector diff = node->centerOfMass - body.position;
        double distSquared = diff.lengthSquared();

        if (distSquared < 1e-10)
            return;

        double s = node->halfWidth * 2.0;
        double ratio = s * s / distSquared;

        if (node->isLeaf || ratio < THETA * THETA)
        {
            double force = GRAVITY * body.mass * node->totalMass / distSquared;
            Vector forceVec = diff * (force / sqrt(distSquared));
            body.acceleration = body.acceleration + forceVec * (1.0 / body.mass);
        }
        else
        {
            for (int i = 0; i < 8; i++)
            {
                if (node->children[i])
                {
                    computeForceFromNode(body, node->children[i]);
                }
            }
        }
    }

    void computeForces()
    {
        auto start = std::chrono::high_resolution_clock::now();

        if (useOpenMP)
        {
#pragma omp parallel for
            for (int i = 0; i < nBodies; i++)
            {
                bodies[i].acceleration = Vector();
                computeForceFromNode(bodies[i], root.get());
            }
        }
        else
        {
            for (int i = 0; i < nBodies; i++)
            {
                bodies[i].acceleration = Vector();
                computeForceFromNode(bodies[i], root.get());
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
        for (int i = 0; i < nBodies; i++)
        {
            bodies[i].velocity = bodies[i].velocity + bodies[i].acceleration * DT;
            bodies[i].position = bodies[i].position + bodies[i].velocity * DT;
        }

        if (useSFC && reorderingStrategy.shouldReorder())
        {
            reorderBodies();
        }

        iterationCounter++;
    }

    void printPerformanceMetrics() const
    {
        std::cout << "Performance Metrics:" << std::endl;
        std::cout << "Total Time: " << std::fixed << std::setprecision(2) << totalTime << " ms" << std::endl;
        std::cout << "Force Calculation Time: " << std::fixed << std::setprecision(2) << forceCalculationTime << " ms" << std::endl;
        std::cout << "Tree Build Time: " << std::fixed << std::setprecision(2) << octreeTime << " ms" << std::endl;
        if (useSFC)
        {
            std::cout << "SFC Time: " << std::fixed << std::setprecision(2) << sfcTime << " ms" << std::endl;
            std::cout << "Bounding Box Time: " << std::fixed << std::setprecision(2) << bboxTime << " ms" << std::endl;
        }
        std::cout << "Potential Energy: " << std::scientific << std::setprecision(6) << potentialEnergy << std::endl;
        std::cout << "Kinetic Energy: " << std::scientific << std::setprecision(6) << kineticEnergy << std::endl;
        std::cout << "Total Energy: " << std::scientific << std::setprecision(6) << (potentialEnergy + kineticEnergy) << std::endl;
    }

    void run(int steps)
    {
        auto start = std::chrono::high_resolution_clock::now();

        for (int step = 0; step < steps; step++)
        {
            buildOctree();
            computeForces();
            calculateEnergies();
            update();
        }

        auto end = std::chrono::high_resolution_clock::now();
        totalTime = std::chrono::duration<double, std::milli>(end - start).count();
    }

    double getTotalTime() const { return totalTime; }
    double getForceCalculationTime() const { return forceCalculationTime; }
    double getTreeBuildTime() const { return octreeTime; }
    double getSfcTime() const { return sfcTime; }
    double getPotentialEnergy() const { return potentialEnergy; }
    double getKineticEnergy() const { return kineticEnergy; }
    double getTotalEnergy() const { return potentialEnergy + kineticEnergy; }
    double getPotentialEnergyAvg() const { return potentialEnergyAvg; }
    double getKineticEnergyAvg() const { return kineticEnergyAvg; }
    double getTotalEnergyAvg() const { return totalEnergyAvg; }
    int getNumBodies() const { return nBodies; }
    int getNumThreads() const { return numThreads; }
    double getTheta() const { return THETA; }
    int getSortType() const { return useSFC ? 1 : 0; }
    bool isDynamicReordering() const { return useSFC; }
    int getOptimalReorderFrequency() const { return reorderingStrategy.getOptimalFrequency(); }
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
    ArgumentParser parser("BarnesHut CPU Simulation");
    
    // Add arguments with help messages and default values
    parser.add_argument("n", "Number of bodies", 1000);
    parser.add_argument("s", "Number of simulation steps", 100);
    parser.add_argument("t", "Number of threads (0 = auto)", 0);
    parser.add_flag("no-omp", "Disable OpenMP parallelization");
    parser.add_flag("no-sfc", "Disable Space-Filling Curve optimization");
    parser.add_flag("hilbert", "Use Hilbert curve instead of Morton curve");
    parser.add_flag("normal", "Use normal distribution for mass");
    
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
    bool useOpenMP = !parser.get<bool>("no-omp");
    bool useSFC = !parser.get<bool>("no-sfc");
    SFCCurveType curveType = parser.get<bool>("hilbert") ? SFCCurveType::HILBERT : SFCCurveType::MORTON;
    MassDistribution massDist = parser.get<bool>("normal") ? MassDistribution::NORMAL : MassDistribution::UNIFORM;
    
    BarnesHut simulation(
        nBodies,
        useOpenMP,
        threads,
        time(nullptr),
        massDist,
        useSFC,
        curveType);

    simulation.run(steps);
    simulation.printPerformanceMetrics();

    return 0;
}