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
#include <mutex>
#include <atomic>
#include "../../argparse.hpp"
#include "types.h"
#include "sfc_cpu.h"

constexpr double GRAVITY = 6.67430e-11;
constexpr double DT = 0.005;
constexpr double E = 0.01;
constexpr double COLLISION_TH = 0.01;
constexpr double THETA = 0.5;
constexpr double MAX_DIST = 1.0;
constexpr double CENTERX = 0.0;
constexpr double CENTERY = 0.0;
constexpr double CENTERZ = 0.0;
constexpr double EARTH_MASS = 5.972e24;
constexpr double EARTH_DIA = 12742000.0;

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

struct CPUOctreeNode
{
    Vector center;
    double halfWidth;

    bool isLeaf;
    int bodyIndex;

    Vector centerOfMass;
    double totalMass;

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

    int computeOptimalFrequency(int)
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

        if (static_cast<int>(reorderTimeHistory.size()) > metricsWindowSize)
        {
            reorderTimeHistory.pop_front();
            postReorderSimTimeHistory.pop_front();
        }

        currentOptimalFrequency = computeOptimalFrequency(iterationsSinceReorder);
    }

    bool shouldReorder(double, double)
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
        if (static_cast<int>(simulationTimeHistory.size()) > metricsWindowSize)
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
    int nodesCreated;

    Vector minBound;
    Vector maxBound;

    bool useSFC;
    SFCCurveType curveType;
    int iterationCounter;
    SFCDynamicReorderingStrategy reorderingStrategy;
    sfc::BodySorter<Body> *sorter;

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
                                                           nodesCreated(0),
                                                           useSFC(useSFC_),
                                                           curveType(sfcCurveType),
                                                           iterationCounter(0),
                                                           reorderingStrategy(10),
                                                           sorter(nullptr),
                                                           numThreads(1)
    {
        bodies.resize(numBodies);

        // Desactivar paralelización si hay muy pocos cuerpos
        // Un mínimo razonable sería al menos 8-16 cuerpos por thread
        if (useOpenMP && numBodies >= 32)
        {
            if (threads > 0)
            {
                // Si se especifica un número de threads, usamos ese valor
                numThreads = threads;
                omp_set_num_threads(numThreads);
            }
            else
            {
                // Si no se especifica, usamos 1 thread (secuencial)
                numThreads = 1;
                omp_set_num_threads(numThreads);
            }
        }
        else
        {
            // Con pocos cuerpos, usamos ejecución secuencial para evitar sobrecarga
            useOpenMP = false;
            numThreads = 1;
            omp_set_num_threads(1);
        }

        std::cout << "Barnes-Hut CPU simulation created with " << numBodies << " bodies." << std::endl;
        std::cout << "Using " << numThreads << " thread(s)" << std::endl;
        if (numBodies < 32 && threads > 1)
        {
            std::cout << "Note: Parallelization disabled due to small number of bodies (" << numBodies << "). " 
                      << "At least 32 bodies are required for parallel execution." << std::endl;
        }
        if (useSFC)
        {
            std::string sfcTypeStr = (curveType == SFCCurveType::MORTON) ? "MORTON" : "HILBERT";
            std::cout << "SFC Ordering enabled with type " << sfcTypeStr << std::endl;
            
            sfc::CurveType sfcType = (curveType == SFCCurveType::MORTON) ? 
                                      sfc::CurveType::MORTON : 
                                      sfc::CurveType::HILBERT;
            sorter = new sfc::BodySorter<Body>(numBodies, sfcType);
        }

        initRandomBodies(seed, massDist);
    }

    ~BarnesHut() {
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

    void computeBoundingBox()
    {
        auto start = std::chrono::high_resolution_clock::now();
        updateBoundingBox();
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

        sorter->sortBodies(bodies, minBound, maxBound);

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

    void buildOctree()
    {
        auto start = std::chrono::high_resolution_clock::now();

        updateBoundingBox();
        nodesCreated = 0;
        root.reset(new CPUOctreeNode());
        nodesCreated++;
        
        root->center = (minBound + maxBound) * 0.5;
        root->halfWidth = std::max({maxBound.x - minBound.x,
                                  maxBound.y - minBound.y,
                                  maxBound.z - minBound.z}) * 0.5;

        const int MAX_BODIES_PER_LEAF = 1;

        // Nueva implementación con particionamiento espacial para reducir contención
        if (useOpenMP && nBodies > 1000)
        {
            buildOctreeParallelOptimized(MAX_BODIES_PER_LEAF);
        }
        else
        {
            // Para tamaños pequeños o cuando no se usa OpenMP, usamos el enfoque secuencial
            for (int i = 0; i < nBodies; i++)
            {
                insertBodyIterative(i, std::numeric_limits<int>::max(), MAX_BODIES_PER_LEAF);
            }
        }

        calculateCenterOfMassBatch();

        auto end = std::chrono::high_resolution_clock::now();
        octreeTime = std::chrono::duration<double, std::milli>(end - start).count();
    }

    // Nueva función optimizada para construcción paralela con menos contención
    void buildOctreeParallelOptimized(int maxBodiesPerLeaf)
    {
        // Fase 1: Particionamiento espacial inicial para reducir contención
        const int NUM_SPATIAL_PARTITIONS = numThreads * 2; // Más particiones que threads
        std::vector<std::vector<int>> spatialPartitions(NUM_SPATIAL_PARTITIONS);
        
        // Dividir el espacio en regiones y asignar cuerpos a particiones
        double xRange = maxBound.x - minBound.x;
        double yRange = maxBound.y - minBound.y;
        double zRange = maxBound.z - minBound.z;
        
        int partitionsPerDim = static_cast<int>(std::cbrt(NUM_SPATIAL_PARTITIONS)) + 1;
        
        #pragma omp parallel for
        for (int i = 0; i < nBodies; i++)
        {
            const Vector& pos = bodies[i].position;
            
            int xIdx = static_cast<int>((pos.x - minBound.x) / xRange * partitionsPerDim);
            int yIdx = static_cast<int>((pos.y - minBound.y) / yRange * partitionsPerDim);
            int zIdx = static_cast<int>((pos.z - minBound.z) / zRange * partitionsPerDim);
            
            xIdx = std::min(xIdx, partitionsPerDim - 1);
            yIdx = std::min(yIdx, partitionsPerDim - 1);
            zIdx = std::min(zIdx, partitionsPerDim - 1);
            
            int partitionIdx = (xIdx * partitionsPerDim * partitionsPerDim + 
                               yIdx * partitionsPerDim + zIdx) % NUM_SPATIAL_PARTITIONS;
            
            #pragma omp critical(partition_assignment)
            {
                spatialPartitions[partitionIdx].push_back(i);
            }
        }
        
        // Fase 2: Construcción por lotes con menos contención
        std::atomic<int> atomicNodesCreated{1}; // Comenzamos con el nodo raíz
        std::vector<std::mutex> nodeMutexes(8); // Un mutex por octante del nodo raíz
        
        #pragma omp parallel
        {
            // Cada thread procesa múltiples particiones para balancear carga
            #pragma omp for schedule(dynamic, 1)
            for (int p = 0; p < NUM_SPATIAL_PARTITIONS; p++)
            {
                for (int bodyIdx : spatialPartitions[p])
                {
                    insertBodyLockFree(bodyIdx, maxBodiesPerLeaf, atomicNodesCreated, nodeMutexes);
                }
            }
        }
        
        nodesCreated = atomicNodesCreated.load();
    }
    
    // Versión con menos locks usando estrategia lock-free donde sea posible
    void insertBodyLockFree(int bodyIndex, int maxBodiesPerLeaf, 
                           std::atomic<int>& atomicNodesCreated,
                           std::vector<std::mutex>& nodeMutexes)
    {
        CPUOctreeNode* node = root.get();
        int depth = 0;
        int currentOctant = -1;
        
        while (node && depth < 20) // Limitar profundidad para evitar loops infinitos
        {
            // Determinar octante una sola vez por nivel
            int octant = node->getOctant(bodies[bodyIndex].position);
            
            // Usar diferentes estrategias según el nivel del árbol
            if (depth == 0)
            {
                // En el nivel raíz, usar mutex específico por octante
                std::lock_guard<std::mutex> lock(nodeMutexes[octant]);
                
                if (node->isLeaf)
                {
                    if (node->bodyIndex == -1)
                    {
                        node->bodyIndex = bodyIndex;
                        return;
                    }
                    else
                    {
                        // Subdividir nodo raíz
                        int oldBodyIndex = node->bodyIndex;
                        node->bodyIndex = -1;
                        node->isLeaf = false;
                        
                        // Crear nodos hijos si no existen
                        for (int i = 0; i < 8; i++)
                        {
                            if (!node->children[i])
                            {
                                node->children[i] = new CPUOctreeNode();
                                atomicNodesCreated.fetch_add(1, std::memory_order_relaxed);
                                node->children[i]->center = node->getOctantCenter(i);
                                node->children[i]->halfWidth = node->halfWidth * 0.5;
                            }
                        }
                        
                        // Insertar el cuerpo anterior
                        int oldOctant = node->getOctant(bodies[oldBodyIndex].position);
                        if (node->children[oldOctant]->isLeaf && node->children[oldOctant]->bodyIndex == -1)
                        {
                            node->children[oldOctant]->bodyIndex = oldBodyIndex;
                        }
                    }
                }
                
                // Crear nodo hijo si no existe
                if (!node->children[octant])
                {
                    node->children[octant] = new CPUOctreeNode();
                    atomicNodesCreated.fetch_add(1, std::memory_order_relaxed);
                    node->children[octant]->center = node->getOctantCenter(octant);
                    node->children[octant]->halfWidth = node->halfWidth * 0.5;
                }
                
                node = node->children[octant];
                depth++;
                continue;
            }
            
            // Para niveles más profundos, usar compare-and-swap cuando sea posible
            bool needsLock = false;
            
            // Verificación rápida sin lock
            if (node->isLeaf)
            {
                if (node->bodyIndex == -1)
                {
                    // Intentar inserción atómica
                    int expected = -1;
                    if (std::atomic_compare_exchange_weak(
                        reinterpret_cast<std::atomic<int>*>(&node->bodyIndex),
                        &expected, bodyIndex))
                    {
                        return; // Inserción exitosa
                    }
                    needsLock = true;
                }
                else
                {
                    needsLock = true;
                }
            }
            
            if (needsLock)
            {
                // Solo usar lock cuando sea absolutamente necesario
                static std::mutex deepNodeMutex;
                std::lock_guard<std::mutex> lock(deepNodeMutex);
                
                if (node->isLeaf)
                {
                    if (node->bodyIndex == -1)
                    {
                        node->bodyIndex = bodyIndex;
                        return;
                    }
                    else if (node->bodyIndex != bodyIndex)
                    {
                        // Subdividir
                        int oldBodyIndex = node->bodyIndex;
                        node->bodyIndex = -1;
                        node->isLeaf = false;
                        
                        // Crear nodos hijos
                        for (int i = 0; i < 8; i++)
                        {
                            if (!node->children[i])
                            {
                                node->children[i] = new CPUOctreeNode();
                                atomicNodesCreated.fetch_add(1, std::memory_order_relaxed);
                                node->children[i]->center = node->getOctantCenter(i);
                                node->children[i]->halfWidth = node->halfWidth * 0.5;
                            }
                        }
                        
                        // Insertar el cuerpo anterior
                        int oldOctant = node->getOctant(bodies[oldBodyIndex].position);
                        if (node->children[oldOctant]->isLeaf && node->children[oldOctant]->bodyIndex == -1)
                        {
                            node->children[oldOctant]->bodyIndex = oldBodyIndex;
                        }
                    }
                }
            }
            
            // Avanzar al siguiente nivel
            if (!node->children[octant])
            {
                static std::mutex childCreationMutex;
                std::lock_guard<std::mutex> lock(childCreationMutex);
                
                if (!node->children[octant])
                {
                    node->children[octant] = new CPUOctreeNode();
                    atomicNodesCreated.fetch_add(1, std::memory_order_relaxed);
                    node->children[octant]->center = node->getOctantCenter(octant);
                    node->children[octant]->halfWidth = node->halfWidth * 0.5;
                }
            }
            
            node = node->children[octant];
            depth++;
        }
    }

    // Versión iterativa (no recursiva) de insertBody para evitar stack overflow
    void insertBodyIterative(int bodyIndex, int maxDepth, int maxBodiesPerLeaf)
    {
        CPUOctreeNode* node = root.get();
        int depth = 0;
        
        while (node)
        {
            // Si alcanzamos la profundidad máxima o el número máximo de cuerpos por hoja, nos detenemos
            if (depth >= maxDepth)
            {
                if (useOpenMP)
                {
                    #pragma omp critical
                    {
                        if (node->isLeaf && node->bodyIndex == -1)
                        {
                            node->bodyIndex = bodyIndex;
                        }
                    }
                }
                else
                {
                    if (node->isLeaf && node->bodyIndex == -1)
                    {
                        node->bodyIndex = bodyIndex;
                    }
                }
                break;
            }
            
            bool isLeaf = false;
            int oldBodyIndex = -1;
            
            if (useOpenMP)
            {
                #pragma omp critical
                {
                    if (node->isLeaf)
                    {
                        if (node->bodyIndex == -1)
                        {
                            // Si es una hoja vacía, simplemente asignar el cuerpo
                            node->bodyIndex = bodyIndex;
                            isLeaf = true;
                        }
                        else
                        {
                            // Si es una hoja con un cuerpo, subdividir
                            oldBodyIndex = node->bodyIndex;
                            node->bodyIndex = -1;
                            node->isLeaf = false;
                        }
                    }
                }
            }
            else
            {
                if (node->isLeaf)
                {
                    if (node->bodyIndex == -1)
                    {
                        // Si es una hoja vacía, simplemente asignar el cuerpo
                        node->bodyIndex = bodyIndex;
                        isLeaf = true;
                    }
                    else
                    {
                        // Si es una hoja con un cuerpo, subdividir
                        oldBodyIndex = node->bodyIndex;
                        node->bodyIndex = -1;
                        node->isLeaf = false;
                    }
                }
            }
            
            if (isLeaf)
            {
                // Si era una hoja y asignamos el cuerpo, salimos
                break;
            }
            
            if (oldBodyIndex != -1)
            {
                // Tenemos que insertar el cuerpo antiguo primero
                int octantOld = node->getOctant(bodies[oldBodyIndex].position);
                
                if (useOpenMP)
                {
                    #pragma omp critical
                    {
                        if (!node->children[octantOld])
                        {
                            node->children[octantOld] = new CPUOctreeNode();
                            #pragma omp atomic
                            nodesCreated++;
                            node->children[octantOld]->center = node->getOctantCenter(octantOld);
                            node->children[octantOld]->halfWidth = node->halfWidth * 0.5;
                        }
                    }
                }
                else
                {
                    if (!node->children[octantOld])
                    {
                        node->children[octantOld] = new CPUOctreeNode();
                        nodesCreated++;
                        node->children[octantOld]->center = node->getOctantCenter(octantOld);
                        node->children[octantOld]->halfWidth = node->halfWidth * 0.5;
                    }
                }
                
                // Insertamos el cuerpo antiguo en el nodo hijo (iterativamente)
                CPUOctreeNode* childNode = node->children[octantOld];
                
                if (useOpenMP)
                {
                    #pragma omp critical
                    {
                        if (childNode->isLeaf && childNode->bodyIndex == -1)
                        {
                            childNode->bodyIndex = oldBodyIndex;
                        }
                        else
                        {
                            // Esto es raro pero puede ocurrir por condiciones de carrera
                            // Volvemos a insertar en el árbol de forma segura
                            #pragma omp taskwait
                        }
                    }
                    
                    // Si no pudimos insertar directamente, usamos recursión pero con otra llamada
                    if (childNode->bodyIndex != oldBodyIndex)
                    {
                        insertBodyIterative(oldBodyIndex, maxDepth - depth - 1, maxBodiesPerLeaf);
                    }
                }
                else
                {
                    if (childNode->isLeaf && childNode->bodyIndex == -1)
                    {
                        childNode->bodyIndex = oldBodyIndex;
                    }
                    else
                    {
                        // Volvemos a insertar en el árbol
                        insertBodyIterative(oldBodyIndex, maxDepth - depth - 1, maxBodiesPerLeaf);
                    }
                }
            }
            
            // Continuamos con el cuerpo actual
            int octant = node->getOctant(bodies[bodyIndex].position);
            
            if (useOpenMP)
            {
                #pragma omp critical
                {
                    if (!node->children[octant])
                    {
                        node->children[octant] = new CPUOctreeNode();
                        #pragma omp atomic
                        nodesCreated++;
                        node->children[octant]->center = node->getOctantCenter(octant);
                        node->children[octant]->halfWidth = node->halfWidth * 0.5;
                    }
                }
            }
            else
            {
                if (!node->children[octant])
                {
                    node->children[octant] = new CPUOctreeNode();
                    nodesCreated++;
                    node->children[octant]->center = node->getOctantCenter(octant);
                    node->children[octant]->halfWidth = node->halfWidth * 0.5;
                }
            }
            
            // Avanzamos al siguiente nodo en la iteración
            node = node->children[octant];
            depth++;
        }
    }

    // Versión por lotes del cálculo del centro de masa para mejor rendimiento
    void calculateCenterOfMassBatch()
    {
        std::vector<CPUOctreeNode*> nodes;
        std::vector<int> depths;
        std::vector<bool> processed;
        
        // Recorrer el árbol en anchura (BFS)
        if (root)
        {
            nodes.push_back(root.get());
            depths.push_back(0);
            processed.push_back(false);
        }
        
        int maxDepth = 0;
        
        // Primero, recorrer el árbol para encontrar la profundidad máxima
        while (!nodes.empty())
        {
            CPUOctreeNode* node = nodes.back();
            int depth = depths.back();
            bool isProcessed = processed.back();
            
            nodes.pop_back();
            depths.pop_back();
            processed.pop_back();
            
            if (isProcessed)
            {
                // Ya procesamos este nodo, ignorarlo
                continue;
            }
            
            maxDepth = std::max(maxDepth, depth);
            
            // Marcar como procesado
            processed.push_back(true);
            depths.push_back(depth);
            nodes.push_back(node);
            
            // Añadir los hijos
            for (int i = 0; i < 8; i++)
            {
                if (node->children[i])
                {
                    nodes.push_back(node->children[i]);
                    depths.push_back(depth + 1);
                    processed.push_back(false);
                }
            }
        }
        
        // Ahora calculamos los centros de masa por niveles, desde las hojas hacia arriba
        for (int level = maxDepth; level >= 0; level--)
        {
            std::vector<CPUOctreeNode*> levelNodes;
            
            // Recolectar todos los nodos de este nivel
            if (root)
            {
                nodes.clear();
                depths.clear();
                nodes.push_back(root.get());
                depths.push_back(0);
                
                while (!nodes.empty())
                {
                    CPUOctreeNode* node = nodes.back();
                    int depth = depths.back();
                    
                    nodes.pop_back();
                    depths.pop_back();
                    
                    if (depth == level)
                    {
                        levelNodes.push_back(node);
                        continue;
                    }
                    
                    if (depth < level)
                    {
                        // Añadir los hijos
                        for (int i = 0; i < 8; i++)
                        {
                            if (node->children[i])
                            {
                                nodes.push_back(node->children[i]);
                                depths.push_back(depth + 1);
                            }
                        }
                    }
                }
            }
            
            // Procesar todos los nodos de este nivel en paralelo
            if (useOpenMP)
            {
                #pragma omp parallel for
                for (size_t i = 0; i < levelNodes.size(); i++)
                {
                    CPUOctreeNode* node = levelNodes[i];
                    calculateCenterOfMass(node);
                }
            }
            else
            {
                for (CPUOctreeNode* node : levelNodes)
                {
                    calculateCenterOfMass(node);
                }
            }
        }
    }
    
    // Esta función ahora solo procesa un nodo dado, no recursivamente
    void calculateCenterOfMass(CPUOctreeNode* node)
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
                // Asumimos que los hijos ya están calculados por calculateCenterOfMassBatch
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

    void computeForceFromNode(Body &body, const CPUOctreeNode* node)
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
            // Mejorar localidad de caché y reducir contención de memoria
            const int CHUNK_SIZE = std::max(16, nBodies / (numThreads * 4));
            
            #pragma omp parallel for schedule(dynamic, CHUNK_SIZE)
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
        auto start = std::chrono::high_resolution_clock::now();

        reorderBodies();

        buildOctree();

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
        std::cout << "  Build tree: " << std::fixed << std::setprecision(2) << octreeTime << " ms" << std::endl;
        std::cout << "  Bounding box: " << std::fixed << std::setprecision(2) << bboxTime << " ms" << std::endl;
        std::cout << "  Compute forces: " << std::fixed << std::setprecision(2) << forceCalculationTime << " ms" << std::endl;
        std::cout << "  Tree nodes created: " << nodesCreated << std::endl;
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

    void printSummary(int steps)
    {
        std::cout << "Simulation complete." << std::endl;
        std::cout << "Performance Summary:" << std::endl;
        std::cout << "  Average time per step: " << std::fixed << std::setprecision(2) << totalTime / steps << " ms" << std::endl;
        std::cout << "  Build tree: " << std::fixed << std::setprecision(2) << octreeTime << " ms" << std::endl;
        std::cout << "  Bounding box: " << std::fixed << std::setprecision(2) << bboxTime << " ms" << std::endl;
        std::cout << "  Compute forces: " << std::fixed << std::setprecision(2) << forceCalculationTime << " ms" << std::endl;
        std::cout << "  Tree nodes created: " << nodesCreated << std::endl;
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
        std::cout << "Running Barnes-Hut CPU simulation for " << steps << " steps..." << std::endl;
        
        float totalRunTime = 0.0f;
        float totalForceTime = 0.0f;
        float totalBuildTime = 0.0f;
        float totalBboxTime = 0.0f;
        float totalReorderTime = 0.0f;
        float minTime = std::numeric_limits<float>::max();
        float maxTime = 0.0f;
        double totalPotentialEnergy = 0.0;
        double totalKineticEnergy = 0.0;

        auto startTime = std::chrono::high_resolution_clock::now();

        for (int step = 0; step < steps; step++)
        {
            auto stepStart = std::chrono::high_resolution_clock::now();
            
            reorderBodies();
            buildOctree();
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
            totalBuildTime += octreeTime;
            totalBboxTime += bboxTime;
            totalReorderTime += sfcTime;
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
};

int main(int argc, char *argv[])
{
    ArgumentParser parser("BarnesHut CPU Simulation");
    
    // Add arguments with help messages and default values
    parser.add_argument("n", "Number of bodies", 1000);
    parser.add_flag("nosfc", "Disable Space-Filling Curve ordering");
    parser.add_argument("s", "Number of simulation steps", 100);
    parser.add_argument("t", "Number of threads (0 or not specified = use 1 thread)", 0);
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
    
    // Determinar si se debe utilizar OpenMP basado en el número de threads
    bool useParallelization = true;  // Por defecto, usamos paralelización
    
    BarnesHut simulation(
        nBodies,
        useParallelization,  // siempre usamos OpenMP si está disponible
        threads,             // número de threads (0 = auto)
        seed,
        massDist,
        useSFC,
        curveType);

    simulation.run(steps);

    return 0;
}