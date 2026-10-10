#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <math.h>

/* =====================================================================
 * STUDENT B - OPENMP PRODUCER/CONSUMER
 * ---------------------------------------------------------------------
 * Description:
 *   Simulates a system where producers generate large pseudo-random items
 *   and consumers process them to check for primality (computationally
 *   expensive). Demonstates scheduling and OpenMP critical sections.
 * =====================================================================
 */

#define MAX_BUFFER_SIZE 500
#define TOTAL_TASKS 40000

/* Global Shared Data */
long long task_buffer[MAX_BUFFER_SIZE];
int head = 0;
int tail = 0;
int count = 0;
int is_finished = 0;

/* Simple primality simulator (uses workload time) */
void process_item(long long item_value) {
    volatile double temp = 0.0;
    long long limit = item_value;
    for(long long i = 1; i < limit; i++) {
        temp += sqrt((double)i);
    }
}

/*
 * -----------------------------------------------------------
 * Phase 1: Pure Producer Consumer Ring Buffer using OpenMP
 * -----------------------------------------------------------
 */
double run_pc_ring_buffer(int num_producers, int num_consumers) {
    /* Reset state */
    head = 0; tail = 0; count = 0; is_finished = 0;
    int tasks_per_prod = TOTAL_TASKS / num_producers;

    double start_time = omp_get_wtime();
    int finished_producers = 0;

    #pragma omp parallel num_threads(num_producers + num_consumers)
    {
        int thread_id = omp_get_thread_num();

        if (thread_id < num_producers) {
            /* --- PRODUCER CODE --- */
            for (int i = 0; i < tasks_per_prod; i++) {
                int pushed = 0;
                long long new_item = (rand() % 1000) + 100; /* Simulated wait time */

                while(!pushed) {
                    #pragma omp critical
                    {
                        if (count < MAX_BUFFER_SIZE) {
                            task_buffer[tail] = new_item;
                            tail = (tail + 1) % MAX_BUFFER_SIZE;
                            count++;
                            pushed = 1;
                        }
                    }
                }
            }

            #pragma omp atomic
            finished_producers++;

            if (finished_producers == num_producers) {
                #pragma omp critical
                {
                    is_finished = 1;
                }
            }
        } else {
            /* --- CONSUMER CODE --- */
            while (1) {
                int popped = 0;
                long long item_to_process = 0;
                int check_finish = 0;

                #pragma omp critical
                {
                    if (count > 0) {
                        item_to_process = task_buffer[head];
                        head = (head + 1) % MAX_BUFFER_SIZE;
                        count--;
                        popped = 1;
                    } else if (is_finished) {
                        check_finish = 1;
                    }
                }

                if (popped) {
                    process_item(item_to_process);
                } else if (check_finish) {
                    break;
                }
            }
        }
    }

    double end_time = omp_get_wtime();
    return (end_time - start_time);
}

/*
 * -----------------------------------------------------------
 * Phase 2: OpenMP Scheduling Test
 * -----------------------------------------------------------
 */
double evaluate_scheduling(int thread_count, int sched_mode) {
    /* Generate arbitrary workload.
       Make 20% tasks very heavy, others light to test dynamic load balancing. */
    long long *workloads = (long long*)malloc(TOTAL_TASKS * sizeof(long long));
    for (int i = 0; i < TOTAL_TASKS; i++) {
        if (i % 5 == 0) workloads[i] = 2000;
        else workloads[i] = 100;
    }

    double start = omp_get_wtime();
    omp_set_num_threads(thread_count);

    if (sched_mode == 1) { // STATIC
        #pragma omp parallel for schedule(static, 50)
        for (int i = 0; i < TOTAL_TASKS; i++) {
            process_item(workloads[i]);
        }
    } else if (sched_mode == 2) { // DYNAMIC
        #pragma omp parallel for schedule(dynamic, 50)
        for (int i = 0; i < TOTAL_TASKS; i++) {
            process_item(workloads[i]);
        }
    } else if (sched_mode == 3) { // GUIDED
        #pragma omp parallel for schedule(guided)
        for (int i = 0; i < TOTAL_TASKS; i++) {
            process_item(workloads[i]);
        }
    }

    double end = omp_get_wtime();
    free(workloads);
    return end - start;
}

int main() {
    printf("**************************************************\n");
    printf("* OPENMP PRODUCER & CONSUMER (PRIME NUMBER GEN) *\n");
    printf("**************************************************\n\n");

    FILE *fp = fopen("benchmark_stats.tsv", "w");
    if(!fp) {
        printf("Error writing to file.\n");
        return 1;
    }
    fprintf(fp, "Category\tConfiguration\tTime_Sec\n");

    /* TEST 1: Ring Buffer P/C */
    printf("+------------------------------------------------+\n");
    printf("| RING BUFFER: (Producers vs Consumers)          |\n");
    printf("+-------+-------+--------------------------------+\n");
    printf("| PROD. | CONS. | TIME (s)                       |\n");
    printf("+-------+-------+--------------------------------+\n");

    int pc_tests[5][2] = {{1,1}, {2,2}, {1,4}, {4,1}, {4,4}};
    for (int i=0; i<5; i++) {
        int p = pc_tests[i][0];
        int c = pc_tests[i][1];
        double t = run_pc_ring_buffer(p, c);
        printf("| %5d | %5d | %30f |\n", p, c, t);
        fprintf(fp, "RingBuffer\t%dx%d\t%f\n", p, c, t);
    }
    printf("+-------+-------+--------------------------------+\n\n");

    /* TEST 2: Loop Scheduling */
    printf("+------------------------------------------------+\n");
    printf("| SCHEDULE EVALUATION (Static/Dynamic/Guided)    |\n");
    printf("+---------+-------------+------------------------+\n");
    printf("| THREADS | TYPE        | TIME (s)               |\n");
    printf("+---------+-------------+------------------------+\n");

    int t_counts[] = {2, 4, 8};
    const char *names[] = {"", "STATIC", "DYNAMIC", "GUIDED"};

    for (int t=0; t<3; t++) {
        for (int s=1; s<=3; s++) {
            double time_val = evaluate_scheduling(t_counts[t], s);
            printf("| %7d | %-11s | %22f |\n", t_counts[t], names[s], time_val);
            fprintf(fp, "Schedule_%s\t%d_threads\t%f\n", names[s], t_counts[t], time_val);
        }
    }
    printf("+---------+-------------+------------------------+\n");

    fclose(fp);
    printf("\nExecution completed. Data written to benchmark_stats.tsv\n");
    return 0;
}
