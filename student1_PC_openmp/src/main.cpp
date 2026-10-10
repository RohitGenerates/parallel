#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>
#include <chrono>
#include <cmath>
#include <fstream>
#include <string>
#include <unistd.h>

using namespace std;

// Represents a network packet with varying processing difficulty
struct Packet {
    int id;
    int payloadSize;
    long long complexityScore;
};

class PacketBuffer {
private:
    queue<Packet> q;
    int maxCapacity;
    omp_lock_t lock;
    bool doneProducing;

public:
    PacketBuffer(int capacity) : maxCapacity(capacity), doneProducing(false) {
        omp_init_lock(&lock);
    }

    ~PacketBuffer() {
        omp_destroy_lock(&lock);
    }

    void setDone() { doneProducing = true; }
    bool isDone() { return doneProducing && q.empty(); }

    bool push(const Packet& pkt) {
        bool success = false;
        omp_set_lock(&lock);
        if (q.size() < maxCapacity) {
            q.push(pkt);
            success = true;
        }
        omp_unset_lock(&lock);
        return success;
    }

    bool pop(Packet& pkt) {
        bool success = false;
        omp_set_lock(&lock);
        if (!q.empty()) {
            pkt = q.front();
            q.pop();
            success = true;
        }
        omp_unset_lock(&lock);
        return success;
    }
};

// Simulate complex processing
void processPacket(const Packet& p) {
    double result = 0.0;
    // The higher the complexity, the more loops (simulating variable processing times)
    for (long long i = 0; i < p.complexityScore; ++i) {
        result += sin(i) * cos(i);
    }
}

// ----- MODE 1: Queue based Producer-Consumer -----
double runQueueBasedPC(int numProducers, int numConsumers, int totalPackets) {
    PacketBuffer buffer(1000); // Max buffer size
    int packetsPerProducer = totalPackets / numProducers;
    int finishedProducers = 0;

    auto start = chrono::high_resolution_clock::now();

    #pragma omp parallel num_threads(numProducers + numConsumers)
    {
        int tid = omp_get_thread_num();

        if (tid < numProducers) {
            // Producer role
            for (int i = 0; i < packetsPerProducer; ++i) {
                Packet p;
                p.id = tid * packetsPerProducer + i;
                p.complexityScore = (rand() % 5000) + 1000; // Variable processing

                while (!buffer.push(p)) {
                    // Buffer full, wait a bit
                    // Yield is not great, but simulated by short loop
                }
            }

            // Allow last producer to mark done
            #pragma omp atomic
            finishedProducers++;
            if (finishedProducers == numProducers) {
                buffer.setDone();
            }

        } else {
            // Consumer role
            Packet p;
            while (true) {
                if (buffer.pop(p)) {
                    processPacket(p);
                } else if (buffer.isDone()) {
                    break;
                }
            }
        }
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;
    return duration.count();
}

// ----- MODE 2: Loop Scheduling Validation -----
double runLoopScheduling(int numThreads, const string& scheduleType, int totalPackets) {
    vector<Packet> allPackets(totalPackets);
    for(int i=0; i<totalPackets; i++) {
        allPackets[i].id = i;
        // Introduce imbalance: First half is light, second half is VERY heavy
        if(i < totalPackets / 2) allPackets[i].complexityScore = 100;
        else allPackets[i].complexityScore = 15000;
    }

    auto start = chrono::high_resolution_clock::now();

    omp_set_num_threads(numThreads);

    if (scheduleType == "static") {
        #pragma omp parallel for schedule(static, 100)
        for(int i=0; i<totalPackets; i++) {
            processPacket(allPackets[i]);
        }
    }
    else if (scheduleType == "dynamic") {
        #pragma omp parallel for schedule(dynamic, 100)
        for(int i=0; i<totalPackets; i++) {
            processPacket(allPackets[i]);
        }
    }
    else if (scheduleType == "guided") {
        #pragma omp parallel for schedule(guided)
        for(int i=0; i<totalPackets; i++) {
            processPacket(allPackets[i]);
        }
    }

    auto end = chrono::high_resolution_clock::now();
    return chrono::duration<double>(end - start).count();
}

int main(int argc, char** argv) {
    int totalPackets = 50000;
    cout << "=== Network Packet PC Simulation (Student A) ===" << endl;
    cout << "Opening CSV..." << endl;
    ofstream csv("results.csv");
    csv << "Type,Threads,Time(s)" << endl;

    // 1. Evaluate Queue PC with different Producer/Consumer balances
    cout << "\nEvaluating Queue-based P/C..." << endl;
    vector<pair<int, int>> configs = {{1, 1}, {2, 2}, {2, 4}, {4, 2}, {4, 4}, {4, 8}};
    for(auto& conf : configs) {
        cout << "Running config " << conf.first << "," << conf.second << " ..." << endl;
        double t = runQueueBasedPC(conf.first, conf.second, totalPackets);
        cout << "Producers: " << conf.first << ", Consumers: " << conf.second
             << " | Time: " << t << " s" << endl;
        csv << "Queue_P" << conf.first << "_C" << conf.second << ","
            << (conf.first + conf.second) << "," << t << endl;
    }

    // 2. Evaluate Loop Scheduling for uneven workloads
    cout << "\nEvaluating Loop Scheduling Strategies..." << endl;
    vector<int> threadCounts = {2, 4, 8, 16};
    vector<string> schedules = {"static", "dynamic", "guided"};

    for(int tc : threadCounts) {
        for(const string& sched : schedules) {
            double t = runLoopScheduling(tc, sched, totalPackets);
            cout << "Threads: " << tc << " | Schedule: " << sched
                 << " | Time: " << t << " s" << endl;
            csv << "Schedule_" << sched << "," << tc << "," << t << endl;
        }
    }

    csv.close();
    cout << "\nBenchmarking done! Results saved to results.csv." << endl;
    return 0;
}
