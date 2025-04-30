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
#include <fstream>
#include <sstream>
#include <sys/stat.h>

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

class CPUBarnesHut
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
    double potentialEnergy;
    double kineticEnergy;
    double totalEnergyAvg;
    double potentialEnergyAvg;
    double kineticEnergyAvg;

    std::unique_ptr<CPUOctreeNode> root;

    Vector minBound;
    Vector maxBound;

public:
    CPUBarnesHut(
        int numBodies,
        bool useParallelization = true,
        int threads = 0,
        unsigned int seed = static_cast<unsigned int>(time(nullptr)),
        MassDistribution massDist = MassDistribution::UNIFORM) : nBodies(numBodies),
                                                                 useOpenMP(useParallelization),
                                                                 totalTime(0.0),
                                                                 forceCalculationTime(0.0),
                                                                 octreeTime(0.0),
                                                                 bboxTime(0.0),
                                                                 potentialEnergy(0.0),
                                                                 kineticEnergy(0.0),
                                                                 totalEnergyAvg(0.0),
                                                                 potentialEnergyAvg(0.0),
                                                                 kineticEnergyAvg(0.0)
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
        initRandomBodies(seed, massDist);

        minBound = Vector(std::numeric_limits<double>::max(),
                          std::numeric_limits<double>::max(),
                          std::numeric_limits<double>::max());
        maxBound = Vector(std::numeric_limits<double>::lowest(),
                          std::numeric_limits<double>::lowest(),
                          std::numeric_limits<double>::lowest());

        std::cout << "CPU Barnes-Hut Simulation created with " << numBodies << " bodies." << std::endl;
        if (useOpenMP)
        {
            std::cout << "OpenMP enabled with " << numThreads << " threads." << std::endl;
        }
        else
        {
            std::cout << "OpenMP disabled, using single-threaded mode." << std::endl;
        }
    }

    void initRandomBodies(unsigned int seed, MassDistribution massDist = MassDistribution::UNIFORM)
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

        double sizeX = maxBound.x - minBound.x;
        double sizeY = maxBound.y - minBound.y;
        double sizeZ = maxBound.z - minBound.z;
        double maxSize = std::max(std::max(sizeX, sizeY), sizeZ);

        double centerX = (minBound.x + maxBound.x) * 0.5;
        double centerY = (minBound.y + maxBound.y) * 0.5;
        double centerZ = (minBound.z + maxBound.z) * 0.5;

        minBound.x = centerX - maxSize * 0.5;
        minBound.y = centerY - maxSize * 0.5;
        minBound.z = centerZ - maxSize * 0.5;

        maxBound.x = centerX + maxSize * 0.5;
        maxBound.y = centerY + maxSize * 0.5;
        maxBound.z = centerZ + maxSize * 0.5;

        auto end = std::chrono::high_resolution_clock::now();
        bboxTime = std::chrono::duration<double, std::milli>(end - start).count();
    }

    void buildOctree()
    {
        auto start = std::chrono::high_resolution_clock::now();

        root.reset(new CPUOctreeNode());

        Vector center = Vector(
            (minBound.x + maxBound.x) * 0.5,
            (minBound.y + maxBound.y) * 0.5,
            (minBound.z + maxBound.z) * 0.5);
        double halfWidth = std::max(
            std::max(maxBound.x - center.x, maxBound.y - center.y),
            maxBound.z - center.z);

        root->center = center;
        root->halfWidth = halfWidth;

        for (int i = 0; i < nBodies; i++)
        {
            CPUOctreeNode *node = root.get();

            int level = 0;
            const int MAX_LEVEL = 20;

            while (!node->isLeaf && level < MAX_LEVEL)
            {
                int octant = node->getOctant(bodies[i].position);

                if (!node->children[octant])
                {
                    node->children[octant] = new CPUOctreeNode();
                    node->children[octant]->center = node->getOctantCenter(octant);
                    node->children[octant]->halfWidth = node->halfWidth * 0.5;
                }

                node = node->children[octant];
                level++;
            }

            if (node->bodyIndex == -1)
            {
                node->bodyIndex = i;
                node->centerOfMass = bodies[i].position;
                node->totalMass = bodies[i].mass;
            }
            else
            {
                if (level < MAX_LEVEL)
                {
                    int existingIdx = node->bodyIndex;

                    node->isLeaf = false;
                    node->bodyIndex = -1;

                    int octant = node->getOctant(bodies[existingIdx].position);
                    if (!node->children[octant])
                    {
                        node->children[octant] = new CPUOctreeNode();
                        node->children[octant]->center = node->getOctantCenter(octant);
                        node->children[octant]->halfWidth = node->halfWidth * 0.5;
                    }
                    node->children[octant]->bodyIndex = existingIdx;
                    node->children[octant]->centerOfMass = bodies[existingIdx].position;
                    node->children[octant]->totalMass = bodies[existingIdx].mass;

                    octant = node->getOctant(bodies[i].position);
                    if (!node->children[octant])
                    {
                        node->children[octant] = new CPUOctreeNode();
                        node->children[octant]->center = node->getOctantCenter(octant);
                        node->children[octant]->halfWidth = node->halfWidth * 0.5;
                    }
                    node->children[octant]->bodyIndex = i;
                    node->children[octant]->centerOfMass = bodies[i].position;
                    node->children[octant]->totalMass = bodies[i].mass;
                }
                else
                {
                    node->bodies.push_back(i);
                }
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
            return;
        }

        node->centerOfMass = Vector(0.0, 0.0, 0.0);
        node->totalMass = 0.0;

        for (int i = 0; i < 8; i++)
        {
            if (node->children[i])
            {
                calculateCenterOfMass(node->children[i]);

                if (node->children[i]->totalMass > 0.0)
                {
                    node->centerOfMass.x += node->children[i]->centerOfMass.x * node->children[i]->totalMass;
                    node->centerOfMass.y += node->children[i]->centerOfMass.y * node->children[i]->totalMass;
                    node->centerOfMass.z += node->children[i]->centerOfMass.z * node->children[i]->totalMass;
                    node->totalMass += node->children[i]->totalMass;
                }
            }
        }

        for (size_t i = 0; i < node->bodies.size(); i++)
        {
            int bodyIdx = node->bodies[i];
            node->centerOfMass.x += bodies[bodyIdx].position.x * bodies[bodyIdx].mass;
            node->centerOfMass.y += bodies[bodyIdx].position.y * bodies[bodyIdx].mass;
            node->centerOfMass.z += bodies[bodyIdx].position.z * bodies[bodyIdx].mass;
            node->totalMass += bodies[bodyIdx].mass;
        }

        if (node->totalMass > 0.0)
        {
            node->centerOfMass.x /= node->totalMass;
            node->centerOfMass.y /= node->totalMass;
            node->centerOfMass.z /= node->totalMass;
        }
    }

    void computeForceFromNode(Body &body, const CPUOctreeNode *node)
    {
        if (!node)
            return;

        if (node->totalMass <= 0.0)
            return;

        Vector r = node->centerOfMass - body.position;
        double distSqr = r.lengthSquared();

        if (node->isLeaf || (node->halfWidth * 2.0) / sqrt(distSqr) < THETA)
        {
            if (node->isLeaf && node->bodyIndex != -1 &&
                body.position.x == bodies[node->bodyIndex].position.x &&
                body.position.y == bodies[node->bodyIndex].position.y &&
                body.position.z == bodies[node->bodyIndex].position.z)
            {
                return;
            }

            double dist = sqrt(distSqr + (E * E));

            if (dist < COLLISION_TH)
                return;

            double forceMag = GRAVITY * body.mass * node->totalMass / (dist * dist * dist);

            body.acceleration.x += (r.x * forceMag) / body.mass;
            body.acceleration.y += (r.y * forceMag) / body.mass;
            body.acceleration.z += (r.z * forceMag) / body.mass;
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

        if (!root)
        {
            std::cerr << "Error: Octree not built before force computation" << std::endl;
            return;
        }

        if (useOpenMP)
        {
            omp_set_num_threads(numThreads);

#pragma omp parallel for
            for (int i = 0; i < nBodies; i++)
            {
                if (!bodies[i].isDynamic)
                    continue;

                bodies[i].acceleration = Vector(0.0, 0.0, 0.0);

                computeForceFromNode(bodies[i], root.get());

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

                computeForceFromNode(bodies[i], root.get());

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

        buildOctree();
        computeForces();

        calculateEnergies();

        auto end = std::chrono::high_resolution_clock::now();
        totalTime = std::chrono::duration<double, std::milli>(end - start).count();
    }

    void printPerformanceMetrics() const
    {
        std::cout << "Performance Metrics (ms):" << std::endl;
        std::cout << "  Total time:           " << std::fixed << std::setprecision(3) << totalTime << std::endl;
        std::cout << "  Bounding box:         " << std::fixed << std::setprecision(3) << bboxTime << std::endl;
        std::cout << "  Octree construction:  " << std::fixed << std::setprecision(3) << octreeTime << std::endl;
        std::cout << "  Force calculation:    " << std::fixed << std::setprecision(3) << forceCalculationTime << std::endl;
    }

    void run(int steps)
    {
        std::cout << "Running CPU Barnes-Hut simulation for " << steps << " steps..." << std::endl;

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
    }

    double getTotalTime() const { return totalTime; }
    double getForceCalculationTime() const { return forceCalculationTime; }
    double getTreeBuildTime() const { return octreeTime; }
    double getPotentialEnergy() const { return potentialEnergy; }
    double getKineticEnergy() const { return kineticEnergy; }
    double getTotalEnergy() const { return potentialEnergy + kineticEnergy; }
    double getPotentialEnergyAvg() const { return potentialEnergyAvg; }
    double getKineticEnergyAvg() const { return kineticEnergyAvg; }
    double getTotalEnergyAvg() const { return totalEnergyAvg; }
    int getNumBodies() const { return nBodies; }
    int getNumThreads() const { return numThreads; }
    double getTheta() const { return THETA; }
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
        file << "timestamp,method,bodies,steps,threads,theta,total_time_ms,avg_step_time_ms,force_calculation_time_ms,tree_build_time_ms,potential_energy,kinetic_energy,total_energy" << std::endl;
    }

    file.close();
    std::cout << "Archivo CSV " << (append ? "actualizado" : "inicializado") << ": " << filename << std::endl;
}

void saveMetrics(const std::string &filename,
                 int bodies,
                 int steps,
                 int threads,
                 double theta,
                 double totalTime,
                 double forceCalculationTime,
                 double treeBuildTime,
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
         << "CPU_Barnes_Hut" << ","
         << bodies << ","
         << steps << ","
         << threads << ","
         << theta << ","
         << totalTime << ","
         << avgTimePerStep << ","
         << forceCalculationTime << ","
         << treeBuildTime << ","
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
    bool saveMetricsToFile = false;
    std::string metricsFile = "./BarnesHutCPU_metrics.csv";

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
            std::cout << "  --save-metrics      Save performance metrics to a CSV file" << std::endl;
            std::cout << "  --metrics-file FILE Set the output CSV file (default: metrics.csv)" << std::endl;
            std::cout << "  --help              Show this help message" << std::endl;
            return 0;
        }
    }

    CPUBarnesHut simulation(
        nBodies,
        useOpenMP,
        threads,
        time(nullptr),
        massDist);

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
            simulation.getTheta(),
            simulation.getTotalTime(),
            simulation.getForceCalculationTime(),
            simulation.getTreeBuildTime(),
            simulation.getPotentialEnergyAvg(),
            simulation.getKineticEnergyAvg(),
            simulation.getTotalEnergyAvg());

        std::cout << "Métricas guardadas en: " << metricsFile << std::endl;
    }

    return 0;
}