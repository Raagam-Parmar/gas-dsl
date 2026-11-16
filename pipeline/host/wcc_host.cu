// Host harness: loads wcc.ptx via the CUDA Driver API and verifies
// GPU Weakly Connected Components (u32 min label propagation) against CPU.
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
    const char *ptx_path = (argc > 1) ? argv[1] : "wcc.ptx";

    // Build an undirected graph (symmetrized edges) with 4 components
    const uint32_t V = 256;
    const uint32_t NUM_COMPS = 4;
    const uint32_t COMP_SIZE = V / NUM_COMPS;
    std::vector<uint32_t> row_offsets(V + 1, 0);
    std::vector<uint32_t> src_indices;
    std::vector<uint32_t> dst_indices;

    uint32_t seed = 42;
    auto next_rand = [&]() -> uint32_t {
        seed = seed * 1664525u + 1013904223u;
        return seed;
    };

    std::vector<std::vector<uint32_t>> adj(V);
    for (uint32_t c = 0; c < NUM_COMPS; ++c) {
        uint32_t base = c * COMP_SIZE;
        for (uint32_t i = 0; i < COMP_SIZE; ++i) {
            uint32_t u = base + i;
            if (i + 1 < COMP_SIZE) {
                uint32_t v = base + i + 1;
                adj[u].push_back(v);
                adj[v].push_back(u);
            }
            uint32_t v = base + (next_rand() % COMP_SIZE);
            adj[u].push_back(v);
            adj[v].push_back(u);
        }
    }

    for (uint32_t u = 0; u < V; ++u) {
        row_offsets[u] = static_cast<uint32_t>(dst_indices.size());
        for (uint32_t v : adj[u]) {
            src_indices.push_back(u);
            dst_indices.push_back(v);
        }
    }
    row_offsets[V] = static_cast<uint32_t>(dst_indices.size());
    const uint32_t E = static_cast<uint32_t>(dst_indices.size());

    std::vector<uint32_t> h_comp(V);
    std::vector<uint32_t> cpu_comp(V);
    for (uint32_t i = 0; i < V; ++i) {
        h_comp[i] = i;
        cpu_comp[i] = i;
    }

    for (uint32_t iter = 0; iter < COMP_SIZE; ++iter) {
        bool changed = false;
        for (uint32_t e = 0; e < E; ++e) {
            uint32_t u = src_indices[e];
            uint32_t v = dst_indices[e];
            if (cpu_comp[u] < cpu_comp[v]) {
                cpu_comp[v] = cpu_comp[u];
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
    printf("Running wcc on: %s (V=%u, E=%u)\n", name, V, E);

    CUcontext ctx;
    CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CHECK(cuCtxSetCurrent(ctx));

    std::string ptx = readFile(ptx_path);
    CUmodule mod;
    CHECK(cuModuleLoadData(&mod, ptx.c_str()));
    CUfunction kernel;
    CHECK(cuModuleGetFunction(&kernel, mod, "wcc_scatter"));

    CUdeviceptr d_comp, d_row_offsets, d_src, d_dst;
    CHECK(cuMemAlloc(&d_comp,        V * sizeof(uint32_t)));
    CHECK(cuMemAlloc(&d_row_offsets, (V + 1) * sizeof(uint32_t)));
    CHECK(cuMemAlloc(&d_src,         E * sizeof(uint32_t)));
    CHECK(cuMemAlloc(&d_dst,         E * sizeof(uint32_t)));

    CHECK(cuMemcpyHtoD(d_comp,        h_comp.data(),      V * sizeof(uint32_t)));
    CHECK(cuMemcpyHtoD(d_row_offsets, row_offsets.data(), (V + 1) * sizeof(uint32_t)));
    CHECK(cuMemcpyHtoD(d_src,         src_indices.data(), E * sizeof(uint32_t)));
    CHECK(cuMemcpyHtoD(d_dst,         dst_indices.data(), E * sizeof(uint32_t)));

    uint32_t num_edges = E;
    uint32_t num_vertices = V;
    void *args[] = {
        &d_comp, &d_row_offsets, &d_src, &d_dst,
        &num_edges, &num_vertices
    };

    const unsigned block = 256;
    const unsigned grid  = (E + block - 1) / block;
    for (uint32_t iter = 0; iter < COMP_SIZE; ++iter) {
        CHECK(cuLaunchKernel(kernel, grid, 1, 1, block, 1, 1, 0, nullptr, args, nullptr));
    }
    CHECK(cuCtxSynchronize());

    CHECK(cuMemcpyDtoH(h_comp.data(), d_comp, V * sizeof(uint32_t)));
    int errors = 0;
    for (uint32_t i = 0; i < V; ++i) {
        if (h_comp[i] != cpu_comp[i]) {
            if (errors < 5)
                fprintf(stderr, "Mismatch at v=%u: gpu=%u cpu=%u\n", i, h_comp[i], cpu_comp[i]);
            ++errors;
        }
    }
    printf(errors ? "FAILED (%d errors)\n" : "PASSED\n", errors);

    cuMemFree(d_comp); cuMemFree(d_row_offsets);
    cuMemFree(d_src);  cuMemFree(d_dst);
    cuModuleUnload(mod);
    cuDevicePrimaryCtxRelease(dev);
    return errors != 0;
}
