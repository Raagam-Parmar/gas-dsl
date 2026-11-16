// Host harness: loads pagerank.ptx via the CUDA Driver API and runs
// iterative pagerank_scatter + pagerank_apply against a CPU reference.
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
    const char *ptx_path = (argc > 1) ? argv[1] : "pagerank.ptx";

    // Build a deterministic directed graph with out_degree >= 1 for every vertex
    const uint32_t V = 256;
    const unsigned ITERS = 20;
    std::vector<uint32_t> row_offsets(V + 1, 0);
    std::vector<uint32_t> src_indices;
    std::vector<uint32_t> dst_indices;
    std::vector<float>    out_deg(V, 0.0f);

    uint32_t seed = 42;
    auto next_rand = [&]() -> uint32_t {
        seed = seed * 1664525u + 1013904223u;
        return seed;
    };

    for (uint32_t u = 0; u < V; ++u) {
        row_offsets[u] = static_cast<uint32_t>(dst_indices.size());
        const uint32_t deg = 4;
        for (uint32_t k = 0; k < deg; ++k) {
            uint32_t v = (u + k + 1 + (next_rand() % (V - 1))) % V;
            src_indices.push_back(u);
            dst_indices.push_back(v);
        }
        out_deg[u] = static_cast<float>(deg);
    }
    row_offsets[V] = static_cast<uint32_t>(dst_indices.size());
    const uint32_t E = static_cast<uint32_t>(dst_indices.size());

    std::vector<float> h_rank(V, 1.0f);
    std::vector<float> h_zero(V, 0.0f);

    // CPU golden reference matching pagerank_scatter + pagerank_apply
    std::vector<float> cpu_rank(V, 1.0f);
    std::vector<float> cpu_accum(V, 0.0f);
    for (unsigned iter = 0; iter < ITERS; ++iter) {
        std::fill(cpu_accum.begin(), cpu_accum.end(), 0.0f);
        for (uint32_t e = 0; e < E; ++e) {
            uint32_t u = src_indices[e];
            uint32_t v = dst_indices[e];
            cpu_accum[v] += cpu_rank[u] / out_deg[u];
        }
        for (uint32_t v = 0; v < V; ++v) {
            cpu_rank[v] = 0.15f + 0.85f * cpu_accum[v];
        }
    }

    // CUDA Driver API setup
    CHECK(cuInit(0));
    CUdevice dev;
    CHECK(cuDeviceGet(&dev, 0));

    char name[128];
    CHECK(cuDeviceGetName(name, sizeof(name), dev));
    printf("Running pagerank on: %s (V=%u, E=%u, iters=%u)\n", name, V, E, ITERS);

    CUcontext ctx;
    CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CHECK(cuCtxSetCurrent(ctx));

    std::string ptx = readFile(ptx_path);
    CUmodule mod;
    CHECK(cuModuleLoadData(&mod, ptx.c_str()));
    CUfunction scatter_fn, apply_fn;
    CHECK(cuModuleGetFunction(&scatter_fn, mod, "pagerank_scatter"));
    CHECK(cuModuleGetFunction(&apply_fn,   mod, "pagerank_apply"));

    CUdeviceptr d_rank, d_out_deg, d_rank_accum, d_row_offsets, d_src, d_dst;
    CHECK(cuMemAlloc(&d_rank,        V * sizeof(float)));
    CHECK(cuMemAlloc(&d_out_deg,     V * sizeof(float)));
    CHECK(cuMemAlloc(&d_rank_accum,  V * sizeof(float)));
    CHECK(cuMemAlloc(&d_row_offsets, (V + 1) * sizeof(uint32_t)));
    CHECK(cuMemAlloc(&d_src,         E * sizeof(uint32_t)));
    CHECK(cuMemAlloc(&d_dst,         E * sizeof(uint32_t)));

    CHECK(cuMemcpyHtoD(d_rank,        h_rank.data(),      V * sizeof(float)));
    CHECK(cuMemcpyHtoD(d_out_deg,     out_deg.data(),     V * sizeof(float)));
    CHECK(cuMemcpyHtoD(d_row_offsets, row_offsets.data(), (V + 1) * sizeof(uint32_t)));
    CHECK(cuMemcpyHtoD(d_src,         src_indices.data(), E * sizeof(uint32_t)));
    CHECK(cuMemcpyHtoD(d_dst,         dst_indices.data(), E * sizeof(uint32_t)));

    uint32_t num_edges = E;
    uint32_t num_vertices = V;
    void *scatter_args[] = {
        &d_rank, &d_out_deg, &d_rank_accum,
        &d_row_offsets, &d_src, &d_dst,
        &num_edges, &num_vertices
    };
    void *apply_args[] = {
        &d_rank, &d_out_deg, &d_rank_accum, &num_vertices
    };

    const unsigned block = 256;
    const unsigned grid_e = (E + block - 1) / block;
    const unsigned grid_v = (V + block - 1) / block;
    for (unsigned iter = 0; iter < ITERS; ++iter) {
        CHECK(cuMemcpyHtoD(d_rank_accum, h_zero.data(), V * sizeof(float)));
        CHECK(cuLaunchKernel(scatter_fn, grid_e, 1, 1, block, 1, 1, 0, nullptr, scatter_args, nullptr));
        CHECK(cuLaunchKernel(apply_fn,   grid_v, 1, 1, block, 1, 1, 0, nullptr, apply_args,   nullptr));
    }
    CHECK(cuCtxSynchronize());

    CHECK(cuMemcpyDtoH(h_rank.data(), d_rank, V * sizeof(float)));
    int errors = 0;
    for (uint32_t i = 0; i < V; ++i) {
        if (std::fabs(h_rank[i] - cpu_rank[i]) > 1e-3f) {
            if (errors < 5)
                fprintf(stderr, "Mismatch at v=%u: gpu=%f cpu=%f\n", i, h_rank[i], cpu_rank[i]);
            ++errors;
        }
    }
    printf(errors ? "FAILED (%d errors)\n" : "PASSED\n", errors);

    cuMemFree(d_rank); cuMemFree(d_out_deg); cuMemFree(d_rank_accum);
    cuMemFree(d_row_offsets); cuMemFree(d_src); cuMemFree(d_dst);
    cuModuleUnload(mod);
    cuDevicePrimaryCtxRelease(dev);
    return errors != 0;
}
