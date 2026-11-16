// Host harness: loads bfs.ptx via the CUDA Driver API and verifies
// GPU BFS (u32 min hop distance) against a CPU reference.
#include <cuda.h>
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
    const char *ptx_path = (argc > 1) ? argv[1] : "bfs.ptx";

    const uint32_t V = 256;
    std::vector<uint32_t> row_offsets(V + 1, 0);
    std::vector<uint32_t> src_indices;
    std::vector<uint32_t> dst_indices;

    uint32_t seed = 42;
    auto next_rand = [&]() -> uint32_t {
        seed = seed * 1664525u + 1013904223u;
        return seed;
    };

    for (uint32_t u = 0; u < V; ++u) {
        row_offsets[u] = static_cast<uint32_t>(dst_indices.size());
        if (u + 1 < V) {
            src_indices.push_back(u);
            dst_indices.push_back(u + 1);
        }
        for (int k = 0; k < 3; ++k) {
            src_indices.push_back(u);
            dst_indices.push_back(next_rand() % V);
        }
    }
    row_offsets[V] = static_cast<uint32_t>(dst_indices.size());
    const uint32_t E = static_cast<uint32_t>(dst_indices.size());

    const uint32_t INF = 0x3f3f3f3fu;
    std::vector<uint32_t> h_depth(V, INF);
    h_depth[0] = 0;

    std::vector<uint32_t> cpu_depth(V, INF);
    cpu_depth[0] = 0;
    for (uint32_t iter = 0; iter < V - 1; ++iter) {
        bool changed = false;
        for (uint32_t e = 0; e < E; ++e) {
            uint32_t u = src_indices[e];
            uint32_t v = dst_indices[e];
            if (cpu_depth[u] + 1u < cpu_depth[v]) {
                cpu_depth[v] = cpu_depth[u] + 1u;
                changed = true;
            }
        }
        if (!changed) break;
    }

    CHECK(cuInit(0));
    CUdevice dev;
    CHECK(cuDeviceGet(&dev, 0));

    char name[128];
    CHECK(cuDeviceGetName(name, sizeof(name), dev));
    printf("Running bfs on: %s (V=%u, E=%u)\n", name, V, E);

    CUcontext ctx;
    CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CHECK(cuCtxSetCurrent(ctx));

    std::string ptx = readFile(ptx_path);
    CUmodule mod;
    CHECK(cuModuleLoadData(&mod, ptx.c_str()));
    CUfunction kernel;
    CHECK(cuModuleGetFunction(&kernel, mod, "bfs_scatter"));

    CUdeviceptr d_depth, d_row_offsets, d_src, d_dst;
    CHECK(cuMemAlloc(&d_depth,       V * sizeof(uint32_t)));
    CHECK(cuMemAlloc(&d_row_offsets, (V + 1) * sizeof(uint32_t)));
    CHECK(cuMemAlloc(&d_src,         E * sizeof(uint32_t)));
    CHECK(cuMemAlloc(&d_dst,         E * sizeof(uint32_t)));

    CHECK(cuMemcpyHtoD(d_depth,       h_depth.data(),     V * sizeof(uint32_t)));
    CHECK(cuMemcpyHtoD(d_row_offsets, row_offsets.data(), (V + 1) * sizeof(uint32_t)));
    CHECK(cuMemcpyHtoD(d_src,         src_indices.data(), E * sizeof(uint32_t)));
    CHECK(cuMemcpyHtoD(d_dst,         dst_indices.data(), E * sizeof(uint32_t)));

    uint32_t num_edges = E;
    uint32_t num_vertices = V;
    void *args[] = {
        &d_depth, &d_row_offsets, &d_src, &d_dst,
        &num_edges, &num_vertices
    };

    const unsigned block = 256;
    const unsigned grid  = (E + block - 1) / block;
    for (uint32_t iter = 0; iter < V - 1; ++iter) {
        CHECK(cuLaunchKernel(kernel, grid, 1, 1, block, 1, 1, 0, nullptr, args, nullptr));
    }
    CHECK(cuCtxSynchronize());

    CHECK(cuMemcpyDtoH(h_depth.data(), d_depth, V * sizeof(uint32_t)));
    int errors = 0;
    for (uint32_t i = 0; i < V; ++i) {
        if (h_depth[i] != cpu_depth[i]) {
            if (errors < 5)
                fprintf(stderr, "Mismatch at v=%u: gpu=%u cpu=%u\n", i, h_depth[i], cpu_depth[i]);
            ++errors;
        }
    }
    printf(errors ? "FAILED (%d errors)\n" : "PASSED\n", errors);

    cuMemFree(d_depth); cuMemFree(d_row_offsets);
    cuMemFree(d_src);   cuMemFree(d_dst);
    cuModuleUnload(mod);
    cuDevicePrimaryCtxRelease(dev);
    return errors != 0;
}
