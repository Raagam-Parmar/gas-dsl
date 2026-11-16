// Host harness: loads max_influence.ptx via the CUDA Driver API and verifies
// GPU max-influence propagation (f32 CAS max) against a CPU reference.
#include <cuda.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#define CHECK(call)                                                        \
    do {                                                                   \
        CUresult err = (call);                                             \
        if (err != CUDA_SUCCESS) {                                         \
            const char *msg = nullptr;                                     \
            cuGetErrorString(err, &msg);                                   \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
                    msg ? msg : "unknown");                                \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

static std::string readFile(const char *path) {
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

int main(int argc, char **argv) {
    const char *ptx_path = (argc > 1) ? argv[1] : "max_influence.ptx";

    const uint32_t V = 256;
    const unsigned ITERS = 10;
    std::vector<uint32_t> row_offsets(V + 1, 0);
    std::vector<uint32_t> src_indices;
    std::vector<uint32_t> dst_indices;
    std::vector<float>    weights;

    uint32_t seed = 42;
    auto next_rand = [&]() -> uint32_t {
        seed = seed * 1664525u + 1013904223u;
        return seed;
    };

    for (uint32_t u = 0; u < V; ++u) {
        row_offsets[u] = static_cast<uint32_t>(dst_indices.size());
        for (int k = 0; k < 4; ++k) {
            src_indices.push_back(u);
            dst_indices.push_back(next_rand() % V);
            // Attenuate by < 1.0 so scores converge without overflowing
            weights.push_back(0.5f + 0.04f * static_cast<float>(next_rand() % 10u));
        }
    }
    row_offsets[V] = static_cast<uint32_t>(dst_indices.size());
    const uint32_t E = static_cast<uint32_t>(dst_indices.size());

    std::vector<float> h_score(V, 0.0f);
    std::vector<float> cpu_score(V, 0.0f);
    h_score[0]   = 100.0f;
    cpu_score[0] = 100.0f;

    for (unsigned iter = 0; iter < ITERS; ++iter) {
        for (uint32_t e = 0; e < E; ++e) {
            uint32_t u = src_indices[e];
            uint32_t v = dst_indices[e];
            float cand = cpu_score[u] * weights[e];
            if (cand > cpu_score[v]) {
                cpu_score[v] = cand;
            }
        }
    }

    CHECK(cuInit(0));
    CUdevice dev;
    CHECK(cuDeviceGet(&dev, 0));

    char name[128];
    CHECK(cuDeviceGetName(name, sizeof(name), dev));
    printf("Running max_influence on: %s (V=%u, E=%u, iters=%u)\n", name, V, E, ITERS);

    CUcontext ctx;
    CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CHECK(cuCtxSetCurrent(ctx));

    std::string ptx = readFile(ptx_path);
    CUmodule mod;
    CHECK(cuModuleLoadData(&mod, ptx.c_str()));
    CUfunction kernel;
    CHECK(cuModuleGetFunction(&kernel, mod, "max_influence_scatter"));

    CUdeviceptr d_score, d_weight, d_row_offsets, d_src, d_dst;
    CHECK(cuMemAlloc(&d_score,       V * sizeof(float)));
    CHECK(cuMemAlloc(&d_weight,      E * sizeof(float)));
    CHECK(cuMemAlloc(&d_row_offsets, (V + 1) * sizeof(uint32_t)));
    CHECK(cuMemAlloc(&d_src,         E * sizeof(uint32_t)));
    CHECK(cuMemAlloc(&d_dst,         E * sizeof(uint32_t)));

    CHECK(cuMemcpyHtoD(d_score,       h_score.data(),     V * sizeof(float)));
    CHECK(cuMemcpyHtoD(d_weight,      weights.data(),     E * sizeof(float)));
    CHECK(cuMemcpyHtoD(d_row_offsets, row_offsets.data(), (V + 1) * sizeof(uint32_t)));
    CHECK(cuMemcpyHtoD(d_src,         src_indices.data(), E * sizeof(uint32_t)));
    CHECK(cuMemcpyHtoD(d_dst,         dst_indices.data(), E * sizeof(uint32_t)));

    uint32_t num_edges = E;
    uint32_t num_vertices = V;
    void *args[] = {
        &d_score, &d_weight, &d_row_offsets, &d_src, &d_dst,
        &num_edges, &num_vertices
    };

    const unsigned block = 256;
    const unsigned grid  = (E + block - 1) / block;
    for (unsigned iter = 0; iter < ITERS; ++iter) {
        CHECK(cuLaunchKernel(kernel, grid, 1, 1, block, 1, 1, 0, nullptr, args, nullptr));
    }
    CHECK(cuCtxSynchronize());

    CHECK(cuMemcpyDtoH(h_score.data(), d_score, V * sizeof(float)));
    int errors = 0;
    for (uint32_t i = 0; i < V; ++i) {
        if (std::fabs(h_score[i] - cpu_score[i]) > 1e-3f) {
            if (errors < 5)
                fprintf(stderr, "Mismatch at v=%u: gpu=%f cpu=%f\n", i, h_score[i], cpu_score[i]);
            ++errors;
        }
    }
    printf(errors ? "FAILED (%d errors)\n" : "PASSED\n", errors);

    cuMemFree(d_score); cuMemFree(d_weight); cuMemFree(d_row_offsets);
    cuMemFree(d_src);   cuMemFree(d_dst);
    cuModuleUnload(mod);
    cuDevicePrimaryCtxRelease(dev);
    return errors != 0;
}
