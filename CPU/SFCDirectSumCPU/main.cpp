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
#include <sys/stat.h>  // Para verificar/crear directorios
#include <deque>
#include <numeric>

// =============================================================================
// DYNAMIC REORDERING STRATEGY
// =============================================================================

class SFCDynamicReorderingStrategy
{
private:
    // Parameters for the optimization formula
    double reorderTime;        // Time to reorder (equivalent to Rt)
    double postReorderSimTime; // Simulation time right after reordering (equivalent to Rq)
    double updateTime;         // Time to update without reordering (equivalent to Ut, typically 0 for SFC)
    double degradationRate;    // Average performance degradation per iteration (equivalent to dQ)

    int iterationsSinceReorder;  // Counter for iterations since last reorder
    int currentOptimalFrequency; // Current calculated optimal frequency

    // Tracking metrics for dynamic calculation
    int metricsWindowSize;
    std::deque<double> reorderTimeHistory;
    std::deque<double> postReorderSimTimeHistory;
    std::deque<double> simulationTimeHistory;

    // Calculate the optimal reordering frequency
    int computeOptimalFrequency(int totalIterations)
    {
        // Using the formula: ((nU*nU*dQ/2) + nU*(Ut+Rq) + (Rt+Rq)) * Nit/(nU+1)
        // The optimal frequency is where the derivative = 0

        double determinant = 1.0 - 2.0 * (updateTime - reorderTime) / degradationRate;

        // If determinant is negative, use a default value
        if (determinant < 0)
            return 10; // Default to 10 as a reasonable value

        double optNu = -1.0 + sqrt(determinant);

        // Convert to integer values and check which one is better
        int nu1 = static_cast<int>(optNu);
        int nu2 = nu1 + 1;

        if (nu1 <= 0)
            return 1; // Avoid negative or zero values

        // Calculate total time with nu1 and nu2
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
          degradationRate(0.001), // Initial small degradation assumption
          iterationsSinceReorder(0),
          currentOptimalFrequency(10), // Start with a reasonable default
          metricsWindowSize(windowSize)
    {
    }

    // Update metrics with new timing information
    void updateMetrics(double newReorderTime, double newSimTime)
    {
        // Update reorder time if available
        if (newReorderTime > 0)
        {
            reorderTimeHistory.push_back(newReorderTime);
            if (reorderTimeHistory.size() > metricsWindowSize)
            {
                reorderTimeHistory.pop_front();
            }

            // Recalculate average reorder time
            reorderTime = std::accumulate(reorderTimeHistory.begin(), reorderTimeHistory.end(), 0.0) /
                          reorderTimeHistory.size();
        }

        // Track simulation times to calculate degradation
        simulationTimeHistory.push_back(newSimTime);
        if (simulationTimeHistory.size() > metricsWindowSize)
        {
            simulationTimeHistory.pop_front();
        }

        // If first simulation after reorder, update postReorderSimTime
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

        // Calculate degradation rate if we have enough data
        if (simulationTimeHistory.size() >= 3)
        {
            // Simple linear regression on the most recent simulation times
            // to estimate the degradation rate
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

            // Calculate slope (degradation rate)
            double slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
            if (slope > 0)
            {
                degradationRate = slope;
            }
        }
    }

    // Check if reordering is needed based on current metrics
    bool shouldReorder(double lastSimTime = 0.0, double lastReorderTime = 0.0)
    {
        iterationsSinceReorder++;

        // Update metrics with new timing information
        updateMetrics(lastReorderTime, lastSimTime);

        // Recalculate optimal frequency periodically
        if (iterationsSinceReorder % 10 == 0)
        {
            currentOptimalFrequency = computeOptimalFrequency(1000); // Assuming 1000 total iterations

            // Ensure frequency is reasonable
            currentOptimalFrequency = std::max(1, std::min(100, currentOptimalFrequency));
        }

        // Decide if we should reorder based on current counter and optimal frequency
        bool shouldReorder = iterationsSinceReorder >= currentOptimalFrequency;

        // Reset counter if reordering
        if (shouldReorder)
        {
            iterationsSinceReorder = 0;
        }

        return shouldReorder;
    }

    // Public method for updating metrics with just sort time
    void updateMetrics(double sortTime)
    {
        // Call the internal method with proper defaults
        updateMetrics(sortTime, 0.0);
    }

    // Set the window size for metrics tracking
    void setWindowSize(int windowSize)
    {
        if (windowSize > 0) {
            metricsWindowSize = windowSize;
        }
    }

    // Get the current optimal frequency
    int getOptimalFrequency() const
    {
        return currentOptimalFrequency;
    }

    // Get the current degradation rate estimate
    double getDegradationRate() const
    {
        return degradationRate;
    }

    // Reset the strategy
    void reset()
    {
        iterationsSinceReorder = 0;
        reorderTimeHistory.clear();
        postReorderSimTimeHistory.clear();
        simulationTimeHistory.clear();
    }
};

// =============================================================================
// DEFINICIÓN DE TIPOS
// =============================================================================

struct Vector {
    double x;
    double y;
    double z;

    // Default constructor
    Vector() : x(0.0), y(0.0), z(0.0) {}

    // Constructor with initial values
    Vector(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    // Vector addition
    Vector operator+(const Vector &other) const {
        return Vector(x + other.x, y + other.y, z + other.z);
    }

    // Vector subtraction
    Vector operator-(const Vector &other) const {
        return Vector(x - other.x, y - other.y, z - other.z);
    }

    // Scalar multiplication
    Vector operator*(double scalar) const {
        return Vector(x * scalar, y * scalar, z * scalar);
    }

    // Dot product
    double dot(const Vector &other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    // Vector length squared
    double lengthSquared() const {
        return x * x + y * y + z * z;
    }

    // Vector length
    double length() const {
        return sqrt(lengthSquared());
    }
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

// =============================================================================
// CONSTANTES
// =============================================================================

constexpr double GRAVITY = 6.67430e-11;   // Gravitational constant
constexpr double DT = 0.005;              // Time step
constexpr double E = 0.01;                // Softening parameter
constexpr double COLLISION_TH = 0.01;     // Collision threshold

// =============================================================================
// DISTRIBUCIONES DE CUERPOS
// =============================================================================

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

// =============================================================================
// SPACE-FILLING CURVE
// =============================================================================

// Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit
inline uint64_t expandBits(uint64_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// Calculates Morton code for a 3D point
uint64_t mortonEncode(double x, double y, double z, const Vector& min, const Vector& max) {
    // Normalize coordinates to [0, 1]
    double normalizedX = (x - min.x) / (max.x - min.x);
    double normalizedY = (y - min.y) / (max.y - min.y);
    double normalizedZ = (z - min.z) / (max.z - min.z);
    
    // Clamp to ensure values are within [0, 1]
    normalizedX = std::max(0.0, std::min(1.0, normalizedX));
    normalizedY = std::max(0.0, std::min(1.0, normalizedY));
    normalizedZ = std::max(0.0, std::min(1.0, normalizedZ));
    
    // Convert to integers in range [0, 1023]
    uint64_t intX = static_cast<uint64_t>(normalizedX * 1023.0);
    uint64_t intY = static_cast<uint64_t>(normalizedY * 1023.0);
    uint64_t intZ = static_cast<uint64_t>(normalizedZ * 1023.0);
    
    // Interleave bits for x, y, z
    uint64_t mortonCode = expandBits(intX) | (expandBits(intY) << 1) | (expandBits(intZ) << 2);
    
    return mortonCode;
}

// =============================================================================
// SIMULACIÓN DIRECT SUM CPU CON SFC
// =============================================================================

class SFCCPUDirectSum {
private:
    std::vector<Body> bodies;
    int nBodies;
    bool useOpenMP;
    int numThreads;
    double totalTime;
    double forceCalculationTime;
    double reorderTime;
    double bboxTime;
    
    bool useSFC;
    int reorderFrequency;
    int iterationCounter;
    
    // Morton code ordering data
    std::vector<uint64_t> mortonCodes;
    std::vector<int> orderedIndices;
    
    // Bounding box data
    Vector minBound;
    Vector maxBound;
    
    // Dynamic reordering strategy
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
        bool dynamicReordering = false
    ) : nBodies(numBodies), 
        useOpenMP(useParallelization), 
        totalTime(0.0), 
        forceCalculationTime(0.0),
        reorderTime(0.0),
        bboxTime(0.0),
        useSFC(enableSFC),
        reorderFrequency(reorderFreq),
        iterationCounter(0),
        useDynamicReordering(dynamicReordering),
        reorderingStrategy(10) // Start with window size of 10
    {
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
        
        // Set initial bounds to invalid values to force computation
        minBound = Vector(std::numeric_limits<double>::max(),
                          std::numeric_limits<double>::max(),
                          std::numeric_limits<double>::max());
        maxBound = Vector(std::numeric_limits<double>::lowest(),
                          std::numeric_limits<double>::lowest(),
                          std::numeric_limits<double>::lowest());
        
        // Initialize Morton code and ordering structures
        if (useSFC) {
            mortonCodes.resize(numBodies);
            orderedIndices.resize(numBodies);
            
            // Initialize indices with identity mapping
            for (int i = 0; i < numBodies; i++) {
                orderedIndices[i] = i;
            }
        }
        
        // Log configuration
        std::cout << "CPU SFC Direct Sum Simulation created with " << numBodies << " bodies." << std::endl;
        if (useOpenMP) {
            std::cout << "OpenMP enabled with " << numThreads << " threads." << std::endl;
        } else {
            std::cout << "OpenMP disabled, using single-threaded mode." << std::endl;
        }
        
        if (useSFC) {
            std::cout << "Space-Filling Curve ordering enabled with "
                      << (useDynamicReordering ? "dynamic" : "fixed") << " reorder frequency"; 
            if (!useDynamicReordering) {
                std::cout << " " << reorderFrequency;
            }
            std::cout << std::endl;
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
    
    void computeBoundingBox() {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Reset bounds
        minBound = Vector(std::numeric_limits<double>::max(),
                          std::numeric_limits<double>::max(),
                          std::numeric_limits<double>::max());
        maxBound = Vector(std::numeric_limits<double>::lowest(),
                          std::numeric_limits<double>::lowest(),
                          std::numeric_limits<double>::lowest());
        
        // Find the minimum and maximum coordinates
        if (useOpenMP) {
            // Set the number of threads
            omp_set_num_threads(numThreads);
            
            // Local bounds for each thread
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
                for (int i = 0; i < nBodies; i++) {
                    // Update local min coords
                    localMin[tid].x = std::min(localMin[tid].x, bodies[i].position.x);
                    localMin[tid].y = std::min(localMin[tid].y, bodies[i].position.y);
                    localMin[tid].z = std::min(localMin[tid].z, bodies[i].position.z);
                    
                    // Update local max coords
                    localMax[tid].x = std::max(localMax[tid].x, bodies[i].position.x);
                    localMax[tid].y = std::max(localMax[tid].y, bodies[i].position.y);
                    localMax[tid].z = std::max(localMax[tid].z, bodies[i].position.z);
                }
            }
            
            // Combine results from all threads
            for (int i = 0; i < numThreads; i++) {
                minBound.x = std::min(minBound.x, localMin[i].x);
                minBound.y = std::min(minBound.y, localMin[i].y);
                minBound.z = std::min(minBound.z, localMin[i].z);
                
                maxBound.x = std::max(maxBound.x, localMax[i].x);
                maxBound.y = std::max(maxBound.y, localMax[i].y);
                maxBound.z = std::max(maxBound.z, localMax[i].z);
            }
        } else {
            // Single-threaded computation
            for (int i = 0; i < nBodies; i++) {
                // Update minimum bounds
                minBound.x = std::min(minBound.x, bodies[i].position.x);
                minBound.y = std::min(minBound.y, bodies[i].position.y);
                minBound.z = std::min(minBound.z, bodies[i].position.z);
                
                // Update maximum bounds
                maxBound.x = std::max(maxBound.x, bodies[i].position.x);
                maxBound.y = std::max(maxBound.y, bodies[i].position.y);
                maxBound.z = std::max(maxBound.z, bodies[i].position.z);
            }
        }
        
        // Add some padding to avoid edge cases
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
    
    void orderBodiesBySFC() {
        auto start = std::chrono::high_resolution_clock::now();
        
        if (!useSFC) return;
        
        // First, compute bounding box if needed
        if (minBound.x == std::numeric_limits<double>::max()) {
            computeBoundingBox();
        }
        
        // Calculate Morton codes for each body
        if (useOpenMP) {
            omp_set_num_threads(numThreads);
            
            #pragma omp parallel for
            for (int i = 0; i < nBodies; i++) {
                mortonCodes[i] = mortonEncode(
                    bodies[i].position.x,
                    bodies[i].position.y,
                    bodies[i].position.z,
                    minBound,
                    maxBound
                );
            }
        } else {
            for (int i = 0; i < nBodies; i++) {
                mortonCodes[i] = mortonEncode(
                    bodies[i].position.x,
                    bodies[i].position.y,
                    bodies[i].position.z,
                    minBound,
                    maxBound
                );
            }
        }
        
        // Set initial indices
        for (int i = 0; i < nBodies; i++) {
            orderedIndices[i] = i;
        }
        
        // Sort indices by Morton code
        std::sort(orderedIndices.begin(), orderedIndices.end(),
                  [this](int a, int b) {
                      return mortonCodes[a] < mortonCodes[b];
                  });
        
        // Create a reordered copy of the bodies
        std::vector<Body> reorderedBodies(nBodies);
        for (int i = 0; i < nBodies; i++) {
            reorderedBodies[i] = bodies[orderedIndices[i]];
        }
        
        // Swap with the original bodies
        bodies.swap(reorderedBodies);
        
        auto end = std::chrono::high_resolution_clock::now();
        reorderTime = std::chrono::duration<double, std::milli>(end - start).count();
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
                bodies[i].velocity.z += bodies[i].acceleration.z * DT;

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
        
        // Temporary variables for timing
        double lastSimTime = forceCalculationTime;
        double lastReorderTime = reorderTime;
        
        // Check if we need to reorder bodies
        bool shouldReorder = false;
        if (useSFC) {
            if (useDynamicReordering) {
                // Use dynamic strategy to decide if reordering is needed
                shouldReorder = reorderingStrategy.shouldReorder(lastSimTime, lastReorderTime);
            } else {
                // Use fixed frequency
                shouldReorder = (iterationCounter % reorderFrequency == 0);
            }
            
            if (shouldReorder) {
                // Recompute bounding box and reorder bodies
                computeBoundingBox();
                orderBodiesBySFC();
            }
        }
        
        // Increment iteration counter
        iterationCounter++;
        
        // Compute forces and update positions
        computeForces();
        
        auto end = std::chrono::high_resolution_clock::now();
        totalTime = std::chrono::duration<double, std::milli>(end - start).count();
        
        // If using dynamic strategy, update it with the latest timings
        if (useSFC && useDynamicReordering) {
            reorderingStrategy.updateMetrics(shouldReorder ? reorderTime : 0.0, forceCalculationTime);
        }
    }
    
    void printPerformanceMetrics() const {
        std::cout << "Performance Metrics (ms):" << std::endl;
        std::cout << "  Total time:           " << std::fixed << std::setprecision(3) << totalTime << std::endl;
        std::cout << "  Bounding box:         " << std::fixed << std::setprecision(3) << bboxTime << std::endl;
        std::cout << "  SFC reordering:       " << std::fixed << std::setprecision(3) << reorderTime << std::endl;
        std::cout << "  Force calculation:    " << std::fixed << std::setprecision(3) << forceCalculationTime << std::endl;
        
        if (useSFC && useDynamicReordering) {
            std::cout << "  Optimal reorder freq: " << reorderingStrategy.getOptimalFrequency() << std::endl;
            std::cout << "  Degradation rate:     " << std::fixed << std::setprecision(6) 
                      << reorderingStrategy.getDegradationRate() << " ms/iter" << std::endl;
        }
    }
    
    void run(int steps) {
        std::cout << "Running CPU SFC Direct Sum simulation for " << steps << " steps..." << std::endl;
        
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

    int getOptimalReorderFrequency() const {
        if (useDynamicReordering) {
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
};

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
        file << "timestamp,method,bodies,steps,threads,sort_type,total_time_ms,avg_step_time_ms,force_calculation_time_ms,sort_time_ms" << std::endl;
    }
    
    file.close();
    std::cout << "Archivo CSV " << (append ? "actualizado" : "inicializado") << ": " << filename << std::endl;
}

void saveMetrics(const std::string& filename, 
                int bodies, 
                int steps, 
                int threads,
                int sortType,
                double totalTime, 
                double forceCalculationTime,
                double sortTime) {
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
         << "CPU_SFC_Direct_Sum" << ","
         << bodies << ","
         << steps << ","
         << threads << ","
         << sortType << ","
         << totalTime << ","
         << avgTimePerStep << ","
         << forceCalculationTime << ","
         << sortTime << std::endl;
    
    file.close();
    std::cout << "Métricas guardadas en: " << filename << std::endl;
}

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
    bool useSFC = true; // Default enabled for this implementation
    int reorderFreq = 10;
    bool useDynamicReordering = true;
    
    // Añadir nuevas variables para métricas
    bool saveMetricsToFile = false;
    std::string metricsFile = "./SFCDirectSumCPU_metrics.csv";
    
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
        } else if (arg == "--no-sfc") {
            useSFC = false;
        } else if (arg == "--reorder-freq" && i + 1 < argc) {
            // Parameter removed, kept as no-op for backward compatibility
            i++;
        } else if (arg == "--dynamic-reordering") {
            // Parameter removed, kept as no-op for backward compatibility since dynamic reordering is enabled by default
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
            std::cout << "  --no-sfc            Disable Space-Filling Curve ordering" << std::endl;
            std::cout << "  --reorder-freq N    [DEPRECATED] SFC reordering frequency is now dynamic" << std::endl;
            std::cout << "  --dynamic-reordering [DEPRECATED] Dynamic reordering is enabled by default" << std::endl;
            std::cout << "  --save-metrics      Save metrics to CSV" << std::endl;
            std::cout << "  --metrics-file FILE Set metrics file (default: metrics.csv)" << std::endl;
            std::cout << "  --help              Show this help message" << std::endl;
            return 0;
        }
    }
    
    // Initialize metrics file if needed
    if (saveMetricsToFile) {
        std::ifstream checkFile(metricsFile);
        bool fileExists = checkFile.good();
        checkFile.close();
        
        initializeCsv(metricsFile, fileExists);
    }
    
    // Create and run simulation
    SFCCPUDirectSum simulation(
        nBodies, 
        useOpenMP, 
        threads, 
        useSFC, 
        reorderFreq, 
        dist, 
        time(nullptr), 
        massDist,
        useDynamicReordering
    );
    
    simulation.run(steps);
    
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
            steps,
            simulation.getNumThreads(),
            simulation.getSortType(),
            simulation.getTotalTime(),
            simulation.getForceCalculationTime(),
            simulation.getSortTime()
        );
        
        std::cout << "Métricas guardadas en: " << metricsFile << std::endl;
    }
    
    return 0;
} 