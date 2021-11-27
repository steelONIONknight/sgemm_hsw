#define _GNU_SOURCE

#include <arm_neon.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

// for apple m1
// there are 4 fmla ports
// so block is 4x4
// A [m, k]
// B [k, n]
// n = 16
void sgemm_kernel_arm(const float *a, const float *b, float *c, int m, int k,
                      int n) {
  int nh4 = m >> 2;
  int nw16 = n >> 4;
  for (int row = 0; row < nh4; row++) {
    const float *a_data = a + row * 4 * k;
    const float *b_data = b;
    float *c_data = c + row * 4 * n;
    float32x4_t v0 = vdupq_n_f32(0.f);
    float32x4_t v1 = vdupq_n_f32(0.f);
    float32x4_t v2 = vdupq_n_f32(0.f);
    float32x4_t v3 = vdupq_n_f32(0.f);
    float32x4_t v4 = vdupq_n_f32(0.f);
    float32x4_t v5 = vdupq_n_f32(0.f);
    float32x4_t v6 = vdupq_n_f32(0.f);
    float32x4_t v7 = vdupq_n_f32(0.f);
    float32x4_t v8 = vdupq_n_f32(0.f);
    float32x4_t v9 = vdupq_n_f32(0.f);
    float32x4_t v10 = vdupq_n_f32(0.f);
    float32x4_t v11 = vdupq_n_f32(0.f);
    float32x4_t v12 = vdupq_n_f32(0.f);
    float32x4_t v13 = vdupq_n_f32(0.f);
    float32x4_t v14 = vdupq_n_f32(0.f);
    float32x4_t v15 = vdupq_n_f32(0.f);
    for (int i = 0; i < k; i++) {
      const float *ptr_a = a_data + i;
      const float *ptr_b = b_data + i * n;
      float32x4_t v16 = vdupq_n_f32(ptr_a[0]);
      float32x4_t v17 = vdupq_n_f32(ptr_a[0 + k]);
      float32x4_t v18 = vdupq_n_f32(ptr_a[0 + k * 2]);
      float32x4_t v19 = vdupq_n_f32(ptr_a[0 + k * 3]);

      float32x4_t v20 = vld1q_f32(ptr_b);
      float32x4_t v21 = vld1q_f32(ptr_b + 4);
      float32x4_t v22 = vld1q_f32(ptr_b + 8);
      float32x4_t v23 = vld1q_f32(ptr_b + 12);

      v0 = vmlaq_f32(v0, v16, v20);
      v1 = vmlaq_f32(v1, v16, v21);
      v2 = vmlaq_f32(v2, v16, v22);
      v3 = vmlaq_f32(v3, v16, v23);

      v4 = vmlaq_f32(v4, v17, v20);
      v5 = vmlaq_f32(v5, v17, v21);
      v6 = vmlaq_f32(v6, v17, v22);
      v7 = vmlaq_f32(v7, v17, v23);

      v8 = vmlaq_f32(v8, v18, v20);
      v9 = vmlaq_f32(v9, v18, v21);
      v10 = vmlaq_f32(v10, v18, v22);
      v11 = vmlaq_f32(v11, v18, v23);

      v12 = vmlaq_f32(v12, v19, v20);
      v13 = vmlaq_f32(v13, v19, v21);
      v14 = vmlaq_f32(v14, v19, v22);
      v15 = vmlaq_f32(v15, v19, v23);
    }
    float *ptr_c = c_data;
    vst1q_f32(ptr_c, v0);
    vst1q_f32(ptr_c + 4, v1);
    vst1q_f32(ptr_c + 8, v2);
    vst1q_f32(ptr_c + 12, v3);

    ptr_c += n;
    vst1q_f32(ptr_c, v4);
    vst1q_f32(ptr_c + 4, v5);
    vst1q_f32(ptr_c + 8, v6);
    vst1q_f32(ptr_c + 12, v7);

    ptr_c += n;
    vst1q_f32(ptr_c, v8);
    vst1q_f32(ptr_c + 4, v9);
    vst1q_f32(ptr_c + 8, v10);
    vst1q_f32(ptr_c + 12, v11);

    ptr_c += n;
    vst1q_f32(ptr_c, v12);
    vst1q_f32(ptr_c + 4, v13);
    vst1q_f32(ptr_c + 8, v14);
    vst1q_f32(ptr_c + 12, v15);
  }
}

#ifdef __cplusplus
}
#endif

static double get_time(struct timespec *start, struct timespec *end) {
  return end->tv_sec - start->tv_sec + (end->tv_nsec - start->tv_nsec) * 1e-9;
}
#ifdef __x86_64__
static void thread_bind(int cpu) {
  cpu_set_t cpu_set;
  CPU_ZERO(&cpu_set);
  CPU_SET(cpu, &cpu_set);
  if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpu_set) !=
      0) {
    fprintf(stderr, "Error: cpu[%d] bind failed.\n", cpu);
    exit(0);
  }
}
#endif

static void *page_alloc(size_t size) {
  void *data = mmap(NULL, size, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, 0, 0);
  if (data == (void *)-1) {
    fprintf(stderr, "Error(MemData::Construction): mmap failed.\n");
    exit(0);
  }
  return data;
}

static void page_free(void *mem, size_t size) { munmap(mem, size); }

void save_bin(const char *file_name, float *rst, int num) {
  FILE *fp = fopen(file_name, "wb");
  fwrite(rst, num, sizeof(float), fp);
  fclose(fp);
}

void sgemm_naive(float *a, float *b, float *c, int m, int n, int k) {
  int i, j, kk;
  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      for (kk = 0; kk < k; kk++) {
        c[i * n + j] += a[i * k + kk] * b[kk * n + j];
      }
    }
  }
}

int check_res(float *basic, float *c, int m, int n, float eps) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      int idx = i * n + j;
      if (fabsf(basic[idx] - c[idx]) > eps) {
        fprintf(stderr, "row = %d, col = %d, true val = %f, res val = %f\n", i,
                j, basic[idx], c[idx]);
        return 0;
      }
    }
  }
  return 1;
}
int main(int argc, char *argv[]) {
  int i;

  if (argc != 3) {
    fprintf(stderr, "Usage: %s m k\n", argv[0]);
    return 0;
  }

  int m = atoi(argv[1]);
  int k = atoi(argv[2]);
  int n = 16;
  long comp = 2L * m * k * n;
  int loop_time = (int)(2e11 / comp);

  struct timespec start, end;
  double t, gflops;
#ifdef __x86_64__
  thread_bind(0);
#endif
  float *a = (float *)page_alloc(m * k * sizeof(float));
  float *b = (float *)page_alloc(k * n * sizeof(float));
  float *c1 = (float *)page_alloc(m * n * sizeof(float));
  float *c2 = (float *)page_alloc(m * n * sizeof(float));

  srand(time(NULL));

  for (i = 0; i < m * k; i++) {
    a[i] = (float)rand() / (float)RAND_MAX;
  }
  for (i = 0; i < k * n; i++) {
    b[i] = (float)rand() / (float)RAND_MAX;
  }

  // fma-tuned version
  // warm up
  for (i = 0; i < loop_time; i++) {
    sgemm_kernel_arm(a, b, c2, m, k, n);
  }
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  for (i = 0; i < loop_time; i++) {
    sgemm_kernel_arm(a, b, c2, m, k, n);
  }
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);

  t = get_time(&start, &end) / loop_time;
  gflops = (double)comp / t * 1e-9;

  printf("sgemm_kernel(%d, %d, %d): time = %lf us, perf = %lf GFLOPS.\n", m, n,
         k, t * 1e6, gflops);

  memset(c1, 0, m * n * sizeof(float));
  memset(c2, 0, m * n * sizeof(float));
  sgemm_naive(a, b, c1, m, n, k);
  sgemm_kernel_arm(a, b, c2, m, k, n);

  int res = check_res(c1, c2, m, n, 1e-5);
  if (res == 0) {
    fprintf(stderr, "check result failed\n");
  }

  page_free(a, m * k * sizeof(float));
  page_free(b, k * n * sizeof(float));
  page_free(c1, m * n * sizeof(float));
  page_free(c2, m * n * sizeof(float));

  return 0;
}
