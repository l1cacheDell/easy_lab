#include "multiply.h"

#include <emmintrin.h>
#include <immintrin.h>
#include <thread>
#include <vector>
#include <stdio.h>

#ifndef N
#define N 1024
#endif /*N*/

#ifndef M
#define M 1024
#endif /*M*/

#ifndef P
#define P 1024
#endif /*P*/

// a[N][M], b[M][P]

#define OFFSET(i, j, ld) ((i * ld) + j)
#define NUM_THREADS 32
#define LOOP_GAP 256 / 64   // 256-bit / 64-bit(double) 

inline void gemm_v1_kernel(double* a, double* b, double* c) {
    for(int row = 0; row < N; ++row) // N行
        for(int col = 0; col < P; ++col)    // P列
            for(int mid = 0; mid < M; ++mid)    // M是通用dim
                c[OFFSET(row, col, P)] += a[OFFSET(row, mid, M)] * b[OFFSET(mid, col, P)];
}

void gemm_avx_kernel(double* a, double* b, double* c, 
                    const int BM, const int BN, const int BK,
                    const int TM, const int TN) {
    for (int i = 0; i < TM; i += 1) {
        for (int j = 0; j < TN; j += LOOP_GAP * 2) {
            __asm__ volatile ("prefetcht1 (%0)" :: "r"(c + OFFSET(i, j + LOOP_GAP * 2, BN)));    // 64B = 8B x 8个数据, LOOP_GAP = 4，所以乘2

            __m256d vec_c = _mm256_loadu_pd(c + OFFSET(i, j, BN));  // c[i][j:j+4]
            __m256d vec_c2 = _mm256_loadu_pd(c + OFFSET(i, j + LOOP_GAP, BN));  // c[i+4][j:j+8]
            for (int m = 0; m < BK; m += 8) {   // 8个数据
                
                // cache blocking
                for (int tm = 0; tm < 8; tm++) {
                    const int m_idx = m + tm;

                    __m256d packed_a = _mm256_set1_pd(a[OFFSET(i, m_idx, BK)]);

                    __m256d vec_b = _mm256_loadu_pd(&b[OFFSET(m_idx, j, BN)]);
                    vec_c = _mm256_fmadd_pd(vec_b, packed_a, vec_c);

                    __m256d vec_b2 = _mm256_loadu_pd(&b[OFFSET(m_idx, j + LOOP_GAP, BN)]);
                    vec_c2 = _mm256_fmadd_pd(vec_b2, packed_a, vec_c2);

                    __asm__ volatile ("prefetchnta (%0)" :: "r"(b + OFFSET(m_idx + 1, j, BN)));
                }
                __asm__ volatile ("prefetchnta (%0)" :: "r"(&a[OFFSET(i, m + 8, BK)]));
            }

            // write back
            _mm256_storeu_pd(c + OFFSET(i, j, BN), vec_c);
            _mm256_storeu_pd(c + OFFSET(i, j + LOOP_GAP, BN), vec_c2);

        }
    }
}

void packing_a_4x256(double* src, double* dst) {
    // load data from src, store in dst.
    // 256 = 64 x 4个数据
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 256; j += 4) {
            _mm256_store_pd(dst + i * 256 + j, _mm256_loadu_pd(src + i * M + j));
        }
    }
}

void packing_b_256x64(double* src, double* dst) {
    for (int k = 0; k < 8; k++) {
        // 256 x 8
        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 8; j += 4) {
                _mm256_store_pd(dst + k * 256 * 8 + i * 8 + j, _mm256_loadu_pd(src + i * P + j + k * 8));
            }
        }
    }
}

void gemm_128x64_kernel(double* a, double* b, double* c, int tid) 
{
    /* The result matrix is 2. 128 x 64 each, and make sum.
        Split AB to 128 x 256 x 2, 256 x 64 x 2.
        Split subA to 16 x 256 x 8
        Split subA to 4 x 256 x 4, split subB to 256 x 8 x 8. */
    for (int times = 0; times < 2; times++) {
        for (int blk = 0; blk < 128; blk += 16) {
            // A: 16 x 256 x 8
            // B: 256 x 64

            // packing B once
            double* b_buffer = (double*)_mm_malloc(64 * 256 * sizeof(double), 64);
            packing_b_256x64(b, b_buffer);
            for (int j = 0; j < 64; j += 8) {
                // unroll 16 -> 4 x 4 manually
                // this loop computes 4 x 4 x 256 @ 256 x 8
                double* a_buffer = (double*)_mm_malloc(4 * 256 * sizeof(double), 64);
                {
                    // (1) 4 x 256, packing A. 256 x 8
                    const int row_index = blk + 0;
                    // printf("row index: %d, times: %d, tid: %d\n", row_index, times, tid);
                    packing_a_4x256(a + row_index * 512 + times * 256, a_buffer);

                    __m256d vec_c00 = _mm256_loadu_pd(c + row_index * 512 + j);
                    __m256d vec_c01 = _mm256_loadu_pd(c + row_index * 512 + j + 4);
                    __m256d vec_c10 = _mm256_loadu_pd(c + (row_index + 1) * 512 + j);
                    __m256d vec_c11 = _mm256_loadu_pd(c + (row_index + 1) * 512 + j + 4);
                    __m256d vec_c20 = _mm256_loadu_pd(c + (row_index + 2) * 512 + j);
                    __m256d vec_c21 = _mm256_loadu_pd(c + (row_index + 2) * 512 + j + 4);
                    __m256d vec_c30 = _mm256_loadu_pd(c + (row_index + 3) * 512 + j);
                    __m256d vec_c31 = _mm256_loadu_pd(c + (row_index + 3) * 512 + j + 4);

                    for (int m = 0; m < 256; m++) {
                        __m256d vec_a0 = _mm256_broadcast_sd(a_buffer + m);
                        __m256d vec_a1 = _mm256_broadcast_sd(a_buffer + m + 256);
                        __m256d vec_a2 = _mm256_broadcast_sd(a_buffer + m + 256 * 2);
                        __m256d vec_a3 = _mm256_broadcast_sd(a_buffer + m + 256 * 3);

                        __m256d vec_b0 = _mm256_load_pd(b_buffer + j * 256 + m * 8);
                        __m256d vec_b1 = _mm256_load_pd(b_buffer + j * 256 + m * 8 + 4);

                        vec_c00 = _mm256_fmadd_pd(vec_a0, vec_b0, vec_c00);
                        vec_c01 = _mm256_fmadd_pd(vec_a0, vec_b1, vec_c01);
                        vec_c10 = _mm256_fmadd_pd(vec_a1, vec_b0, vec_c10);
                        vec_c11 = _mm256_fmadd_pd(vec_a1, vec_b1, vec_c11);
                        vec_c20 = _mm256_fmadd_pd(vec_a2, vec_b0, vec_c20);
                        vec_c21 = _mm256_fmadd_pd(vec_a2, vec_b1, vec_c21);
                        vec_c30 = _mm256_fmadd_pd(vec_a3, vec_b0, vec_c30);
                        vec_c31 = _mm256_fmadd_pd(vec_a3, vec_b1, vec_c31);
                    }
                    _mm256_storeu_pd(c + row_index * 512 + j, vec_c00);
                    _mm256_storeu_pd(c + row_index * 512 + j + 4, vec_c01);
                    _mm256_storeu_pd(c + (row_index + 1) * 512 + j, vec_c10);
                    _mm256_storeu_pd(c + (row_index + 1) * 512 + j + 4, vec_c11);
                    _mm256_storeu_pd(c + (row_index + 2) * 512 + j, vec_c20);
                    _mm256_storeu_pd(c + (row_index + 2) * 512 + j + 4, vec_c21);
                    _mm256_storeu_pd(c + (row_index + 3) * 512 + j, vec_c30);
                    _mm256_storeu_pd(c + (row_index + 3) * 512 + j + 4, vec_c31);
                }

                {
                    // (2) 4 x 256
                    const int row_index = blk + 4;
                    packing_a_4x256(a + row_index * 512 + times * 256, a_buffer);

                    __m256d vec_c00 = _mm256_loadu_pd(c + row_index * 512 + j);
                    __m256d vec_c01 = _mm256_loadu_pd(c + row_index * 512 + j + 4);
                    __m256d vec_c10 = _mm256_loadu_pd(c + (row_index + 1) * 512 + j);
                    __m256d vec_c11 = _mm256_loadu_pd(c + (row_index + 1) * 512 + j + 4);
                    __m256d vec_c20 = _mm256_loadu_pd(c + (row_index + 2) * 512 + j);
                    __m256d vec_c21 = _mm256_loadu_pd(c + (row_index + 2) * 512 + j + 4);
                    __m256d vec_c30 = _mm256_loadu_pd(c + (row_index + 3) * 512 + j);
                    __m256d vec_c31 = _mm256_loadu_pd(c + (row_index + 3) * 512 + j + 4);

                    for (int m = 0; m < 256; m++) {
                        __m256d vec_a0 = _mm256_broadcast_sd(a_buffer + m);
                        __m256d vec_a1 = _mm256_broadcast_sd(a_buffer + m + 256);
                        __m256d vec_a2 = _mm256_broadcast_sd(a_buffer + m + 256 * 2);
                        __m256d vec_a3 = _mm256_broadcast_sd(a_buffer + m + 256 * 3);

                        __m256d vec_b0 = _mm256_load_pd(b_buffer + j * 256 + m * 8);
                        __m256d vec_b1 = _mm256_load_pd(b_buffer + j * 256 + m * 8 + 4);

                        vec_c00 = _mm256_fmadd_pd(vec_a0, vec_b0, vec_c00);
                        vec_c01 = _mm256_fmadd_pd(vec_a0, vec_b1, vec_c01);
                        vec_c10 = _mm256_fmadd_pd(vec_a1, vec_b0, vec_c10);
                        vec_c11 = _mm256_fmadd_pd(vec_a1, vec_b1, vec_c11);
                        vec_c20 = _mm256_fmadd_pd(vec_a2, vec_b0, vec_c20);
                        vec_c21 = _mm256_fmadd_pd(vec_a2, vec_b1, vec_c21);
                        vec_c30 = _mm256_fmadd_pd(vec_a3, vec_b0, vec_c30);
                        vec_c31 = _mm256_fmadd_pd(vec_a3, vec_b1, vec_c31);
                    }
                    _mm256_storeu_pd(c + row_index * 512 + j, vec_c00);
                    _mm256_storeu_pd(c + row_index * 512 + j + 4, vec_c01);
                    _mm256_storeu_pd(c + (row_index + 1) * 512 + j, vec_c10);
                    _mm256_storeu_pd(c + (row_index + 1) * 512 + j + 4, vec_c11);
                    _mm256_storeu_pd(c + (row_index + 2) * 512 + j, vec_c20);
                    _mm256_storeu_pd(c + (row_index + 2) * 512 + j + 4, vec_c21);
                    _mm256_storeu_pd(c + (row_index + 3) * 512 + j, vec_c30);
                    _mm256_storeu_pd(c + (row_index + 3) * 512 + j + 4, vec_c31);
                }

                {
                    // (3) 4 x 256
                    const int row_index = blk + 8;
                    packing_a_4x256(a + row_index * 512 + times * 256, a_buffer);

                    __m256d vec_c00 = _mm256_loadu_pd(c + row_index * 512 + j);
                    __m256d vec_c01 = _mm256_loadu_pd(c + row_index * 512 + j + 4);
                    __m256d vec_c10 = _mm256_loadu_pd(c + (row_index + 1) * 512 + j);
                    __m256d vec_c11 = _mm256_loadu_pd(c + (row_index + 1) * 512 + j + 4);
                    __m256d vec_c20 = _mm256_loadu_pd(c + (row_index + 2) * 512 + j);
                    __m256d vec_c21 = _mm256_loadu_pd(c + (row_index + 2) * 512 + j + 4);
                    __m256d vec_c30 = _mm256_loadu_pd(c + (row_index + 3) * 512 + j);
                    __m256d vec_c31 = _mm256_loadu_pd(c + (row_index + 3) * 512 + j + 4);

                    for (int m = 0; m < 256; m++) {
                        __m256d vec_a0 = _mm256_broadcast_sd(a_buffer + m);
                        __m256d vec_a1 = _mm256_broadcast_sd(a_buffer + m + 256);
                        __m256d vec_a2 = _mm256_broadcast_sd(a_buffer + m + 256 * 2);
                        __m256d vec_a3 = _mm256_broadcast_sd(a_buffer + m + 256 * 3);

                        __m256d vec_b0 = _mm256_load_pd(b_buffer + j * 256 + m * 8);
                        __m256d vec_b1 = _mm256_load_pd(b_buffer + j * 256 + m * 8 + 4);

                        vec_c00 = _mm256_fmadd_pd(vec_a0, vec_b0, vec_c00);
                        vec_c01 = _mm256_fmadd_pd(vec_a0, vec_b1, vec_c01);
                        vec_c10 = _mm256_fmadd_pd(vec_a1, vec_b0, vec_c10);
                        vec_c11 = _mm256_fmadd_pd(vec_a1, vec_b1, vec_c11);
                        vec_c20 = _mm256_fmadd_pd(vec_a2, vec_b0, vec_c20);
                        vec_c21 = _mm256_fmadd_pd(vec_a2, vec_b1, vec_c21);
                        vec_c30 = _mm256_fmadd_pd(vec_a3, vec_b0, vec_c30);
                        vec_c31 = _mm256_fmadd_pd(vec_a3, vec_b1, vec_c31);
                    }
                    _mm256_storeu_pd(c + row_index * 512 + j, vec_c00);
                    _mm256_storeu_pd(c + row_index * 512 + j + 4, vec_c01);
                    _mm256_storeu_pd(c + (row_index + 1) * 512 + j, vec_c10);
                    _mm256_storeu_pd(c + (row_index + 1) * 512 + j + 4, vec_c11);
                    _mm256_storeu_pd(c + (row_index + 2) * 512 + j, vec_c20);
                    _mm256_storeu_pd(c + (row_index + 2) * 512 + j + 4, vec_c21);
                    _mm256_storeu_pd(c + (row_index + 3) * 512 + j, vec_c30);
                    _mm256_storeu_pd(c + (row_index + 3) * 512 + j + 4, vec_c31);
                }

                {
                    // (4) 4 x 256
                    const int row_index = blk + 12;
                    packing_a_4x256(a + row_index * 512 + times * 256, a_buffer);

                    __m256d vec_c00 = _mm256_loadu_pd(c + row_index * 512 + j);
                    __m256d vec_c01 = _mm256_loadu_pd(c + row_index * 512 + j + 4);
                    __m256d vec_c10 = _mm256_loadu_pd(c + (row_index + 1) * 512 + j);
                    __m256d vec_c11 = _mm256_loadu_pd(c + (row_index + 1) * 512 + j + 4);
                    __m256d vec_c20 = _mm256_loadu_pd(c + (row_index + 2) * 512 + j);
                    __m256d vec_c21 = _mm256_loadu_pd(c + (row_index + 2) * 512 + j + 4);
                    __m256d vec_c30 = _mm256_loadu_pd(c + (row_index + 3) * 512 + j);
                    __m256d vec_c31 = _mm256_loadu_pd(c + (row_index + 3) * 512 + j + 4);

                    for (int m = 0; m < 256; m++) {
                        __m256d vec_a0 = _mm256_broadcast_sd(a_buffer + m);
                        __m256d vec_a1 = _mm256_broadcast_sd(a_buffer + m + 256);
                        __m256d vec_a2 = _mm256_broadcast_sd(a_buffer + m + 256 * 2);
                        __m256d vec_a3 = _mm256_broadcast_sd(a_buffer + m + 256 * 3);

                        __m256d vec_b0 = _mm256_load_pd(b_buffer + j * 256 + m * 8);
                        __m256d vec_b1 = _mm256_load_pd(b_buffer + j * 256 + m * 8 + 4);

                        vec_c00 = _mm256_fmadd_pd(vec_a0, vec_b0, vec_c00);
                        vec_c01 = _mm256_fmadd_pd(vec_a0, vec_b1, vec_c01);
                        vec_c10 = _mm256_fmadd_pd(vec_a1, vec_b0, vec_c10);
                        vec_c11 = _mm256_fmadd_pd(vec_a1, vec_b1, vec_c11);
                        vec_c20 = _mm256_fmadd_pd(vec_a2, vec_b0, vec_c20);
                        vec_c21 = _mm256_fmadd_pd(vec_a2, vec_b1, vec_c21);
                        vec_c30 = _mm256_fmadd_pd(vec_a3, vec_b0, vec_c30);
                        vec_c31 = _mm256_fmadd_pd(vec_a3, vec_b1, vec_c31);
                    }
                    _mm256_storeu_pd(c + row_index * 512 + j, vec_c00);
                    _mm256_storeu_pd(c + row_index * 512 + j + 4, vec_c01);
                    _mm256_storeu_pd(c + (row_index + 1) * 512 + j, vec_c10);
                    _mm256_storeu_pd(c + (row_index + 1) * 512 + j + 4, vec_c11);
                    _mm256_storeu_pd(c + (row_index + 2) * 512 + j, vec_c20);
                    _mm256_storeu_pd(c + (row_index + 2) * 512 + j + 4, vec_c21);
                    _mm256_storeu_pd(c + (row_index + 3) * 512 + j, vec_c30);
                    _mm256_storeu_pd(c + (row_index + 3) * 512 + j + 4, vec_c31);
                }
                _mm_free(a_buffer);
            }
            _mm_free(b_buffer);
        }
    }
}

void thread_gemm(double* a, double* b, double* c,
                const int BM, const int BN, const int BK) 
{
    // const int TM = (BM + num_threads - 1) / num_threads;    // 1024 / 32 = 32
    // const int TN = (BN + num_threads - 1) / num_threads;
    int TM, TN, num_threads;

    // 4行 8列布局
    switch (P) {
        case 128: {
            TM = 32;
            TN = 16;
            num_threads = 32;
            break;
        }
        case 512: {
            TM = 128;
            TN = 64;
            num_threads = 32;
            break;
        }
        case 1024: {
            TM = 256;
            TN = 128;
            num_threads = 32;
            break;
        }
        case 2048: {
            TM = 512;
            TN = 256;
            num_threads = 32;
            break;
        }
        case 2560: {
            TM = 640;
            TN = 320;
            num_threads = 32;
            break;
        }
        case 3072: {
            TM = 768;
            TN = 384;
            num_threads = 32;
            break;
        }
        default: {
            printf("Unsupported P.\n");
            exit(-1);
        }
    }
    std::vector<std::thread> gemm_threads;
    gemm_threads.reserve(num_threads);
    

    for (int tid = 0; tid < num_threads; tid++) {
        const int thread_block_row = tid / 8;
        const int thread_block_col = tid % 8;
        const int a_offset = thread_block_row * TM * BK;
        const int b_offset = thread_block_col * TN;
        const int c_offset = a_offset + b_offset;

        // printf("tid: %d, x: %d, y: %d\n", tid, a_i, b_j);

        // gemm_threads.emplace_back(gemm_avx_kernel, 
        //                         a + a_offset, b + b_offset, c + c_offset,
        //                         BM, BN, BK,
        //                         TM, TN);

        gemm_threads.emplace_back(gemm_128x64_kernel, 
                                a + a_offset, b + b_offset, c + c_offset, tid);
        // printf("A address: %p, offset: %d, c offset: %d\n", a + a_offset, a_offset, c_offset);
    }

    for (auto& t : gemm_threads) {
        t.join();
    }

    return;
}

void matrix_multiplication(double matrix1[N][M], double matrix2[M][P], double result_matrix[N][P])
{
    // compute gemm block size, tile size
    bool use_thread = (N % NUM_THREADS == 0 && M % NUM_THREADS == 0 && P % NUM_THREADS == 0 && N % 8 == 0 && M % 8 == 0 && P % 8 ==0 && N == M && M == P && P == N && P == 512); 
    if (!use_thread) {
        gemm_v1_kernel(&matrix1[0][0], &matrix2[0][0], &result_matrix[0][0]);
    } else {
        thread_gemm(&matrix1[0][0], &matrix2[0][0], &result_matrix[0][0], N, P, M);
    }
}
