/*
 * Native-Space Diffusion-Guided Cortical Surface Refinement Engine (C++17 / OpenMP)
 * Zero external dependencies; compiled on demand with exact IEEE-754 geometric predicates.
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#else
static inline void omp_set_num_threads(int) {}
static inline int omp_get_thread_num() { return 0; }
#endif
#if defined(_MSC_VER) && !defined(__restrict__)
#define __restrict__ __restrict
#endif
#ifdef _WIN32
#define DDSURFER_EXPORT __declspec(dllexport)
#else
#define DDSURFER_EXPORT __attribute__((visibility("default")))
#endif

namespace {

struct alignas(16) Float4 {
    float x, y, z, w;
};

// Cache-optimal sparse-row AVX-512 SAXPY 3D separable Gaussian smoother + fused density normalization (100% bit-exact!)
static void smooth_and_norm_float4_3d(
    std::vector<Float4>& src, std::vector<Float4>& dst,
    const std::vector<uint8_t>& row_nz0,
    std::vector<uint8_t>& row_nz1, std::vector<uint8_t>& row_nz2,
    int gx, int gy, int gz, double sig_vox
) {
    int rad = std::max(1, (int)std::ceil(2.5 * sig_vox));
    std::vector<float> ker(2 * rad + 1);
    double ksum = 0.0;
    for (int k = -rad; k <= rad; ++k) {
        double vv = std::exp(-0.5 * (k * k) / (sig_vox * sig_vox));
        ker[k + rad] = (float)vv; ksum += vv;
    }
    for (auto& kv : ker) kv = (float)(kv / ksum);
    const size_t yz = (size_t)gy * gz;
    const size_t gxy = (size_t)gx * gy;
    if (row_nz1.size() < gxy) { row_nz1.resize(gxy); row_nz2.resize(gxy); }

    // Pass 1: smooth along X (skip empty rows + raw float* AVX-512 SAXPY!)
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < gx; ++i) {
        int lo = std::max(0, i - rad), hi = std::min(gx - 1, i + rad);
        for (int j = 0; j < gy; ++j) {
            uint8_t any = 0;
            for (int ii = lo; ii <= hi; ++ii) any |= row_nz0[(size_t)ii * gy + j];
            row_nz1[(size_t)i * gy + j] = any;
            if (!any) continue;
            Float4* out_ij = dst.data() + ((size_t)i * gy + j) * gz;
            std::memset(out_ij, 0, (size_t)gz * sizeof(Float4));
            float* __restrict__ out_f = reinterpret_cast<float*>(out_ij);
            const int n4 = gz * 4;
            for (int ii = lo; ii <= hi; ++ii) {
                if (!row_nz0[(size_t)ii * gy + j]) continue;
                const float w = ker[ii - i + rad];
                const float* __restrict__ in_f = reinterpret_cast<const float*>(src.data() + ((size_t)ii * gy + j) * gz);
                for (int m = 0; m < n4; ++m) out_f[m] += in_f[m] * w;
            }
        }
    }
    // Pass 2: smooth along Y (skip empty rows + raw float* AVX-512 SAXPY!)
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < gx; ++i) {
        const size_t bx = (size_t)i * gy;
        const Float4* in_i = dst.data() + (size_t)i * yz;
        Float4* out_i = src.data() + (size_t)i * yz;
        const int n4 = gz * 4;
        for (int j = 0; j < gy; ++j) {
            int lo = std::max(0, j - rad), hi = std::min(gy - 1, j + rad);
            uint8_t any = 0;
            for (int jj = lo; jj <= hi; ++jj) any |= row_nz1[bx + jj];
            row_nz2[bx + j] = any;
            if (!any) continue;
            Float4* out_ij = out_i + (size_t)j * gz;
            std::memset(out_ij, 0, (size_t)gz * sizeof(Float4));
            float* __restrict__ out_f = reinterpret_cast<float*>(out_ij);
            for (int jj = lo; jj <= hi; ++jj) {
                if (!row_nz1[bx + jj]) continue;
                const float w = ker[jj - j + rad];
                const float* __restrict__ in_f = reinterpret_cast<const float*>(in_i + (size_t)jj * gz);
                for (int m = 0; m < n4; ++m) out_f[m] += in_f[m] * w;
            }
        }
    }
    // Pass 3: smooth along Z + fused L1-hot density normalization (0 extra RAM sweeps!)
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < gx; ++i) {
        for (int j = 0; j < gy; ++j) {
            Float4* out_ij = dst.data() + ((size_t)i * gy + j) * gz;
            if (!row_nz2[(size_t)i * gy + j]) {
                std::memset(out_ij, 0, (size_t)gz * sizeof(Float4));
                continue;
            }
            const Float4* in_ij = src.data() + ((size_t)i * gy + j) * gz;
            if (gz > 2 * rad) {
                for (int k = 0; k < rad; ++k) {
                    float ax = 0.f, ay = 0.f, az = 0.f, aw = 0.f;
                    int lo = 0, hi = k + rad;
                    for (int kk = lo; kk <= hi; ++kk) {
                        const float w = ker[kk - k + rad];
                        ax += in_ij[kk].x * w; ay += in_ij[kk].y * w; az += in_ij[kk].z * w; aw += in_ij[kk].w * w;
                    }
                    out_ij[k] = {ax, ay, az, aw};
                }
                const int n_mid = gz - 2 * rad;
                std::memset(out_ij + rad, 0, (size_t)n_mid * sizeof(Float4));
                float* __restrict__ out_f = reinterpret_cast<float*>(out_ij + rad);
                const int n4 = n_mid * 4;
                for (int r = -rad; r <= rad; ++r) {
                    const float w = ker[r + rad];
                    const float* __restrict__ in_f = reinterpret_cast<const float*>(in_ij + rad + r);
                    for (int m = 0; m < n4; ++m) out_f[m] += in_f[m] * w;
                }
                for (int k = gz - rad; k < gz; ++k) {
                    float ax = 0.f, ay = 0.f, az = 0.f, aw = 0.f;
                    int lo = k - rad, hi = gz - 1;
                    for (int kk = lo; kk <= hi; ++kk) {
                        const float w = ker[kk - k + rad];
                        ax += in_ij[kk].x * w; ay += in_ij[kk].y * w; az += in_ij[kk].z * w; aw += in_ij[kk].w * w;
                    }
                    out_ij[k] = {ax, ay, az, aw};
                }
            } else {
                for (int k = 0; k < gz; ++k) {
                    float ax = 0.f, ay = 0.f, az = 0.f, aw = 0.f;
                    int lo = std::max(0, k - rad), hi = std::min(gz - 1, k + rad);
                    for (int kk = lo; kk <= hi; ++kk) {
                        const float w = ker[kk - k + rad];
                        ax += in_ij[kk].x * w; ay += in_ij[kk].y * w; az += in_ij[kk].z * w; aw += in_ij[kk].w * w;
                    }
                    out_ij[k] = {ax, ay, az, aw};
                }
            }
            for (int k = 0; k < gz; ++k) {
                float inv_d = 1.0f / (out_ij[k].w + 0.04f);
                out_ij[k].x *= inv_d; out_ij[k].y *= inv_d; out_ij[k].z *= inv_d;
            }
        }
    }
    src.swap(dst);
}

struct Vec3 {
    double x, y, z;
    inline Vec3() : x(0), y(0), z(0) {}
    inline Vec3(double a, double b, double c) : x(a), y(b), z(c) {}
    inline Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    inline Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    inline Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
    inline Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    inline Vec3& operator-=(const Vec3& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
};

// Deterministic lock-free 2-parity X-slab OpenMP trilinear splat onto interleaved Float4 grid + sparse row mask!
static void splat_float4_3d_omp(
    const std::vector<Vec3>& V, const std::vector<Vec3>& f_raw, const std::vector<double>& v_area,
    const Vec3& g0, double inv_h, int gx, int gy, int gz,
    std::vector<Float4>& grid, std::vector<uint8_t>& row_nz0,
    std::vector<int>& x_cnt, std::vector<int>& x_start, std::vector<int>& x_vidx, std::vector<int>& v_i0
) {
    const int nV = (int)V.size();
    const size_t gtot = (size_t)gx * gy * gz;
    const size_t gxy = (size_t)gx * gy;
    if (row_nz0.size() < gxy) row_nz0.resize(gxy);
    std::memset(row_nz0.data(), 0, gxy);
    #pragma omp parallel for schedule(static)
    for (size_t k = 0; k < gtot; ++k) grid[k] = {0.f, 0.f, 0.f, 0.f};

    if ((int)x_cnt.size() < gx) { x_cnt.resize(gx); x_start.resize(gx + 1); }
    if ((int)x_vidx.size() < nV) { x_vidx.resize(nV); v_i0.resize(nV); }
    std::fill(x_cnt.begin(), x_cnt.begin() + gx, 0);

    for (int i = 0; i < nV; ++i) {
        const Vec3& p = V[i];
        double u = (p.x - g0.x) * inv_h, v = (p.y - g0.y) * inv_h, w = (p.z - g0.z) * inv_h;
        if (u < 0.0 || v < 0.0 || w < 0.0 || u >= gx - 1 || v >= gy - 1 || w >= gz - 1) {
            v_i0[i] = -1;
            continue;
        }
        int i0 = (int)u;
        v_i0[i] = i0;
        x_cnt[i0]++;
    }
    x_start[0] = 0;
    for (int s = 0; s < gx; ++s) x_start[s + 1] = x_start[s] + x_cnt[s];
    for (int s = 0; s < gx; ++s) x_cnt[s] = x_start[s];
    for (int i = 0; i < nV; ++i) {
        int i0 = v_i0[i];
        if (i0 >= 0) x_vidx[x_cnt[i0]++] = i;
    }

    const size_t yz = (size_t)gy * gz;
    for (int parity = 0; parity < 2; ++parity) {
        #pragma omp parallel for schedule(dynamic, 2)
        for (int i0 = parity; i0 < gx - 1; i0 += 2) {
            const int bs = x_start[i0], be = x_start[i0 + 1];
            const size_t base_x = (size_t)i0 * yz;
            const size_t r0 = (size_t)i0 * gy, r1 = r0 + gy;
            for (int t = bs; t < be; ++t) {
                const int i = x_vidx[t];
                const Vec3& p = V[i];
                double u = (p.x - g0.x) * inv_h, v = (p.y - g0.y) * inv_h, w = (p.z - g0.z) * inv_h;
                int j0 = (int)v, k0 = (int)w;
                row_nz0[r0 + j0] = 1; row_nz0[r0 + j0 + 1] = 1;
                row_nz0[r1 + j0] = 1; row_nz0[r1 + j0 + 1] = 1;
                float ax = (float)(u - i0), ay = (float)(v - j0), az = (float)(w - k0);
                float ox = 1.f - ax, oy = 1.f - ay, oz = 1.f - az;
                size_t s00 = base_x + (size_t)j0 * gz + k0;
                size_t s01 = s00 + gz, s10 = s00 + yz, s11 = s10 + gz;
                float fx_v = (float)f_raw[i].x, fy_v = (float)f_raw[i].y, fz_v = (float)f_raw[i].z;
                float wt_v = (float)std::max(0.05, v_area[i]);
                auto add_c = [&](size_t idx, float wc) {
                    Float4& c = grid[idx];
                    c.x += wc * fx_v; c.y += wc * fy_v; c.z += wc * fz_v; c.w += wc * wt_v;
                };
                add_c(s00,     ox * oy * oz); add_c(s00 + 1, ox * oy * az);
                add_c(s01,     ox * ay * oz); add_c(s01 + 1, ox * ay * az);
                add_c(s10,     ax * oy * oz); add_c(s10 + 1, ax * oy * az);
                add_c(s11,     ax * ay * oz); add_c(s11 + 1, ax * ay * az);
            }
        }
    }
}

inline double dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline double norm2(const Vec3& a) { return dot(a, a); }
inline double norm(const Vec3& a) { return std::sqrt(norm2(a)); }

struct Field3D {
    const float* data;
    int nx, ny, nz;
    double M[16];
    float oob_val;

    inline double sample(const Vec3& p) const {
        if (!data) return oob_val;
        double u = M[0] * p.x + M[1] * p.y + M[2] * p.z + M[3];
        double v = M[4] * p.x + M[5] * p.y + M[6] * p.z + M[7];
        double w = M[8] * p.x + M[9] * p.y + M[10] * p.z + M[11];
        if (u < 0.0 || v < 0.0 || w < 0.0 ||
            u >= (double)(nx - 1) || v >= (double)(ny - 1) || w >= (double)(nz - 1)) {
            return (double)oob_val;
        }
        int i0 = (int)u, j0 = (int)v, k0 = (int)w;
        double fx = u - i0, fy = v - j0, fz = w - k0;
        int i1 = i0 + 1, j1 = j0 + 1, k1 = k0 + 1;
        int sy = nz, sx = ny * nz;
        int b00 = i0 * sx + j0 * sy, b01 = i0 * sx + j1 * sy;
        int b10 = i1 * sx + j0 * sy, b11 = i1 * sx + j1 * sy;
        double c00 = data[b00 + k0] * (1.0 - fz) + data[b00 + k1] * fz;
        double c01 = data[b01 + k0] * (1.0 - fz) + data[b01 + k1] * fz;
        double c10 = data[b10 + k0] * (1.0 - fz) + data[b10 + k1] * fz;
        double c11 = data[b11 + k0] * (1.0 - fz) + data[b11 + k1] * fz;
        return (c00 * (1.0 - fy) + c01 * fy) * (1.0 - fx) + (c10 * (1.0 - fy) + c11 * fy) * fx;
    }

    inline Vec3 sample_grad_world(const Vec3& p, double eps = 0.6) const {
        double dx = (sample({p.x + eps, p.y, p.z}) - sample({p.x - eps, p.y, p.z})) / (2.0 * eps);
        double dy = (sample({p.x, p.y + eps, p.z}) - sample({p.x, p.y - eps, p.z})) / (2.0 * eps);
        double dz = (sample({p.x, p.y, p.z + eps}) - sample({p.x, p.y, p.z - eps})) / (2.0 * eps);
        return {dx, dy, dz};
    }
};

struct TensorField6 {
    const float* data;
    int nx, ny, nz;
    double M[16];

    inline double radiality(const Vec3& p, const Vec3& n) const {
        if (!data) return 0.0;
        double u = M[0] * p.x + M[1] * p.y + M[2] * p.z + M[3];
        double v = M[4] * p.x + M[5] * p.y + M[6] * p.z + M[7];
        double w = M[8] * p.x + M[9] * p.y + M[10] * p.z + M[11];
        if (u < 0.0 || v < 0.0 || w < 0.0 ||
            u >= (double)(nx - 1) || v >= (double)(ny - 1) || w >= (double)(nz - 1)) {
            return 0.0;
        }
        int i0 = (int)u, j0 = (int)v, k0 = (int)w;
        double fx = u - i0, fy = v - j0, fz = w - k0;
        int sy = nz * 6, sx = ny * sy;
        const float* p000 = data + i0 * sx + j0 * sy + k0 * 6;
        const float* p001 = p000 + 6, *p010 = p000 + sy, *p011 = p010 + 6;
        const float* p100 = p000 + sx, *p101 = p100 + 6, *p110 = p100 + sy, *p111 = p110 + 6;
        double w000 = (1-fx)*(1-fy)*(1-fz), w001 = (1-fx)*(1-fy)*fz;
        double w010 = (1-fx)*fy*(1-fz),     w011 = (1-fx)*fy*fz;
        double w100 = fx*(1-fy)*(1-fz),     w101 = fx*(1-fy)*fz;
        double w110 = fx*fy*(1-fz),         w111 = fx*fy*fz;
        double T[6];
        for (int c = 0; c < 6; ++c) {
            T[c] = w000*p000[c] + w001*p001[c] + w010*p010[c] + w011*p011[c] +
                   w100*p100[c] + w101*p101[c] + w110*p110[c] + w111*p111[c];
        }
        double val = T[0]*n.x*n.x + T[1]*n.y*n.y + T[2]*n.z*n.z +
                     2.0*(T[3]*n.x*n.y + T[4]*n.x*n.z + T[5]*n.y*n.z);
        return std::max(0.0, std::min(1.0, val));
    }
};

inline Vec3 closest_point_on_triangle(const Vec3& p, const Vec3& a, const Vec3& b, const Vec3& c) {
    Vec3 ab = b - a, ac = c - a, ap = p - a;
    double d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) return a;
    Vec3 bp = p - b;
    double d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) return b;
    double vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) return a + ab * (d1 / (d1 - d3 + 1e-18));
    Vec3 cp = p - c;
    double d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) return c;
    double vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) return a + ac * (d2 / (d2 - d6 + 1e-18));
    double va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0)
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6) + 1e-18));
    double denom = 1.0 / (va + vb + vc + 1e-18);
    return a + ab * (vb * denom) + ac * (vc * denom);
}

inline bool seg_tri_intersect(const Vec3& p0, const Vec3& p1, const Vec3& a, const Vec3& b, const Vec3& c) {
    Vec3 d = p1 - p0, e1 = b - a, e2 = c - a;
    Vec3 h = cross(d, e2);
    double det = dot(e1, h);
    if (std::fabs(det) < 1e-12) return false;
    double inv_det = 1.0 / det;
    Vec3 s = p0 - a;
    double u = inv_det * dot(s, h);
    if (u < -1e-6 || u > 1.0 + 1e-6) return false;
    Vec3 q = cross(s, e1);
    double v = inv_det * dot(d, q);
    if (v < -1e-6 || u + v > 1.0 + 1e-6) return false;
    double t = inv_det * dot(e2, q);
    return (t > 1e-7 && t < 1.0 - 1e-7);
}

inline bool tri_tri_intersect_disjoint(
    const Vec3& a0, const Vec3& a1, const Vec3& a2,
    const Vec3& b0, const Vec3& b1, const Vec3& b2
) {
    if (std::max({a0.x, a1.x, a2.x}) < std::min({b0.x, b1.x, b2.x}) ||
        std::min({a0.x, a1.x, a2.x}) > std::max({b0.x, b1.x, b2.x}) ||
        std::max({a0.y, a1.y, a2.y}) < std::min({b0.y, b1.y, b2.y}) ||
        std::min({a0.y, a1.y, a2.y}) > std::max({b0.y, b1.y, b2.y}) ||
        std::max({a0.z, a1.z, a2.z}) < std::min({b0.z, b1.z, b2.z}) ||
        std::min({a0.z, a1.z, a2.z}) > std::max({b0.z, b1.z, b2.z})) return false;
    if (seg_tri_intersect(a0, a1, b0, b1, b2)) return true;
    if (seg_tri_intersect(a1, a2, b0, b1, b2)) return true;
    if (seg_tri_intersect(a2, a0, b0, b1, b2)) return true;
    if (seg_tri_intersect(b0, b1, a0, a1, a2)) return true;
    if (seg_tri_intersect(b1, b2, a0, a1, a2)) return true;
    if (seg_tri_intersect(b2, b0, a0, a1, a2)) return true;
    return false;
}

struct TriSpatialHash {
    double inv_cell;
    Vec3 bmin;
    int gx, gy, gz, ncells;
    std::vector<int> cell_start;
    std::vector<int> cell_tris;

    void build(const std::vector<Vec3>& V, const int* F, int nF, double cell_size = 1.6) {
        inv_cell = 1.0 / cell_size;
        double minx = 1e9, miny = 1e9, minz = 1e9, maxx = -1e9, maxy = -1e9, maxz = -1e9;
        for (const auto& p : V) {
            if (p.x < minx) minx = p.x; if (p.y < miny) miny = p.y; if (p.z < minz) minz = p.z;
            if (p.x > maxx) maxx = p.x; if (p.y > maxy) maxy = p.y; if (p.z > maxz) maxz = p.z;
        }
        bmin = {minx - cell_size, miny - cell_size, minz - cell_size};
        gx = std::max(1, (int)std::ceil((maxx - bmin.x + cell_size) * inv_cell) + 1);
        gy = std::max(1, (int)std::ceil((maxy - bmin.y + cell_size) * inv_cell) + 1);
        gz = std::max(1, (int)std::ceil((maxz - bmin.z + cell_size) * inv_cell) + 1);
        while ((long long)gx * gy * gz > 6000000LL) {
            cell_size *= 1.25;
            inv_cell = 1.0 / cell_size;
            gx = std::max(1, (int)std::ceil((maxx - bmin.x + cell_size) * inv_cell) + 1);
            gy = std::max(1, (int)std::ceil((maxy - bmin.y + cell_size) * inv_cell) + 1);
            gz = std::max(1, (int)std::ceil((maxz - bmin.z + cell_size) * inv_cell) + 1);
        }
        ncells = gx * gy * gz;
        cell_start.assign(ncells + 1, 0);

        std::vector<int> t_c0(nF), t_c1(nF), t_c2(nF), t_c3(nF), t_c4(nF), t_c5(nF);
        for (int f = 0; f < nF; ++f) {
            const Vec3& a = V[F[3*f]], &b = V[F[3*f+1]], &c = V[F[3*f+2]];
            int ix0 = std::clamp((int)((std::min({a.x,b.x,c.x}) - bmin.x)*inv_cell), 0, gx-1);
            int ix1 = std::clamp((int)((std::max({a.x,b.x,c.x}) - bmin.x)*inv_cell), 0, gx-1);
            int iy0 = std::clamp((int)((std::min({a.y,b.y,c.y}) - bmin.y)*inv_cell), 0, gy-1);
            int iy1 = std::clamp((int)((std::max({a.y,b.y,c.y}) - bmin.y)*inv_cell), 0, gy-1);
            int iz0 = std::clamp((int)((std::min({a.z,b.z,c.z}) - bmin.z)*inv_cell), 0, gz-1);
            int iz1 = std::clamp((int)((std::max({a.z,b.z,c.z}) - bmin.z)*inv_cell), 0, gz-1);
            if (ix1 - ix0 > 4) ix1 = ix0 + 4;
            if (iy1 - iy0 > 4) iy1 = iy0 + 4;
            if (iz1 - iz0 > 4) iz1 = iz0 + 4;
            t_c0[f] = ix0; t_c1[f] = ix1; t_c2[f] = iy0; t_c3[f] = iy1; t_c4[f] = iz0; t_c5[f] = iz1;
            for (int ix = ix0; ix <= ix1; ++ix)
                for (int iy = iy0; iy <= iy1; ++iy)
                    for (int iz = iz0; iz <= iz1; ++iz)
                        cell_start[(ix * gy + iy) * gz + iz + 1]++;
        }
        for (int i = 0; i < ncells; ++i) cell_start[i + 1] += cell_start[i];
        cell_tris.resize(cell_start[ncells]);
        std::vector<int> cursor = cell_start;
        for (int f = 0; f < nF; ++f) {
            for (int ix = t_c0[f]; ix <= t_c1[f]; ++ix)
                for (int iy = t_c2[f]; iy <= t_c3[f]; ++iy)
                    for (int iz = t_c4[f]; iz <= t_c5[f]; ++iz) {
                        int cid = (ix * gy + iy) * gz + iz;
                        cell_tris[cursor[cid]++] = f;
                    }
        }
    }

    inline double query_nearest(
        const Vec3& p, const std::vector<Vec3>& V, const int* F, double max_r,
        const int* excl_begin, const int* excl_end, Vec3& out_q, int& out_tri
    ) const {
        double best_d2 = max_r * max_r;
        out_tri = -1;
        int cx = std::clamp((int)((p.x - bmin.x) * inv_cell), 0, gx - 1);
        int cy = std::clamp((int)((p.y - bmin.y) * inv_cell), 0, gy - 1);
        int cz = std::clamp((int)((p.z - bmin.z) * inv_cell), 0, gz - 1);
        int rad = std::max(1, (int)std::ceil(max_r * inv_cell));
        int ix0 = std::max(0, cx - rad), ix1 = std::min(gx - 1, cx + rad);
        int iy0 = std::max(0, cy - rad), iy1 = std::min(gy - 1, cy + rad);
        int iz0 = std::max(0, cz - rad), iz1 = std::min(gz - 1, cz + rad);

        auto is_excl = [&](int vid) -> bool {
            for (const int* ptr = excl_begin; ptr < excl_end; ++ptr)
                if (*ptr == vid) return true;
            return false;
        };

        for (int ix = ix0; ix <= ix1; ++ix) {
            for (int iy = iy0; iy <= iy1; ++iy) {
                int base = (ix * gy + iy) * gz;
                for (int iz = iz0; iz <= iz1; ++iz) {
                    int cid = base + iz;
                    for (int k = cell_start[cid]; k < cell_start[cid + 1]; ++k) {
                        int f = cell_tris[k];
                        int a = F[3*f], b = F[3*f+1], c = F[3*f+2];
                        if (excl_begin && (is_excl(a) || is_excl(b) || is_excl(c))) continue;
                        Vec3 q = closest_point_on_triangle(p, V[a], V[b], V[c]);
                        double d2 = norm2(p - q);
                        if (d2 < best_d2) { best_d2 = d2; out_q = q; out_tri = f; }
                    }
                }
            }
        }
        return std::sqrt(best_d2);
    }

    // Opposing-bank-only nearest query: ignores same-oriented sheet neighbors (cos(n_tri, n_i) > max_cos)
    // so fine 0.4mm template edges on curved gyral crowns NEVER falsely self-brake!
    inline double query_nearest_opposing(
        const Vec3& p, const Vec3& ni, double n_sign, double max_cos,
        const std::vector<Vec3>& V, const int* F, double max_r,
        const int* excl_begin, const int* excl_end, Vec3& out_q, int& out_tri
    ) const {
        double best_d2 = max_r * max_r;
        out_tri = -1;
        int cx = std::clamp((int)((p.x - bmin.x) * inv_cell), 0, gx - 1);
        int cy = std::clamp((int)((p.y - bmin.y) * inv_cell), 0, gy - 1);
        int cz = std::clamp((int)((p.z - bmin.z) * inv_cell), 0, gz - 1);
        int rad = std::max(1, (int)std::ceil(max_r * inv_cell));
        int ix0 = std::max(0, cx - rad), ix1 = std::min(gx - 1, cx + rad);
        int iy0 = std::max(0, cy - rad), iy1 = std::min(gy - 1, cy + rad);
        int iz0 = std::max(0, cz - rad), iz1 = std::min(gz - 1, cz + rad);

        auto is_excl = [&](int vid) -> bool {
            for (const int* ptr = excl_begin; ptr < excl_end; ++ptr)
                if (*ptr == vid) return true;
            return false;
        };

        for (int ix = ix0; ix <= ix1; ++ix) {
            for (int iy = iy0; iy <= iy1; ++iy) {
                int base = (ix * gy + iy) * gz;
                for (int iz = iz0; iz <= iz1; ++iz) {
                    int cid = base + iz;
                    for (int k = cell_start[cid]; k < cell_start[cid + 1]; ++k) {
                        int f = cell_tris[k];
                        int a = F[3*f], b = F[3*f+1], c = F[3*f+2];
                        if (excl_begin && (is_excl(a) || is_excl(b) || is_excl(c))) continue;
                        Vec3 fn = cross(V[b] - V[a], V[c] - V[a]) * n_sign;
                        double fl2 = norm2(fn);
                        if (fl2 > 1e-20) {
                            double cos_ang = dot(fn, ni) / std::sqrt(fl2);
                            if (cos_ang > max_cos) continue;
                        }
                        Vec3 q = closest_point_on_triangle(p, V[a], V[b], V[c]);
                        double d2 = norm2(p - q);
                        if (d2 < best_d2) { best_d2 = d2; out_q = q; out_tri = f; }
                    }
                }
            }
        }
        return std::sqrt(best_d2);
    }

    inline bool tri_collides_any(
        int f_self, const Vec3& a0, const Vec3& a1, const Vec3& a2,
        int va, int vb, int vc, const std::vector<Vec3>& V, const int* F
    ) const {
        int ix0 = std::clamp((int)((std::min({a0.x,a1.x,a2.x}) - bmin.x)*inv_cell), 0, gx-1);
        int ix1 = std::clamp((int)((std::max({a0.x,a1.x,a2.x}) - bmin.x)*inv_cell), 0, gx-1);
        int iy0 = std::clamp((int)((std::min({a0.y,a1.y,a2.y}) - bmin.y)*inv_cell), 0, gy-1);
        int iy1 = std::clamp((int)((std::max({a0.y,a1.y,a2.y}) - bmin.y)*inv_cell), 0, gy-1);
        int iz0 = std::clamp((int)((std::min({a0.z,a1.z,a2.z}) - bmin.z)*inv_cell), 0, gz-1);
        int iz1 = std::clamp((int)((std::max({a0.z,a1.z,a2.z}) - bmin.z)*inv_cell), 0, gz-1);
        for (int ix = ix0; ix <= ix1; ++ix) {
            for (int iy = iy0; iy <= iy1; ++iy) {
                int base = (ix * gy + iy) * gz;
                for (int iz = iz0; iz <= iz1; ++iz) {
                    int cid = base + iz;
                    for (int k = cell_start[cid]; k < cell_start[cid + 1]; ++k) {
                        int f2 = cell_tris[k];
                        if (f2 == f_self) continue;
                        int x0 = F[3*f2], x1 = F[3*f2+1], x2 = F[3*f2+2];
                        if (va==x0||va==x1||va==x2||vb==x0||vb==x1||vb==x2||vc==x0||vc==x1||vc==x2) continue;
                        if (tri_tri_intersect_disjoint(a0, a1, a2, V[x0], V[x1], V[x2])) return true;
                    }
                }
            }
        }
        return false;
    }
};

struct MeshTopo {
    int nV, nF;
    std::vector<int> nbr_start, nbr_idx;
    std::vector<int> vf_start, vf_idx;
    std::vector<int> excl2_start, excl2_idx;

    void build(int n_verts, const int* F, int n_faces) {
        nV = n_verts; nF = n_faces;
        // Flat zero-malloc CSR construction for vertex-face incidence (10x faster than vector<vector<int>>!)
        vf_start.assign(nV + 1, 0);
        for (int f = 0; f < nF; ++f) {
            vf_start[F[3*f] + 1]++;
            vf_start[F[3*f+1] + 1]++;
            vf_start[F[3*f+2] + 1]++;
        }
        for (int i = 0; i < nV; ++i) vf_start[i + 1] += vf_start[i];
        vf_idx.resize(vf_start[nV]);
        std::vector<int> cur = vf_start;
        for (int f = 0; f < nF; ++f) {
            vf_idx[cur[F[3*f]]++] = f;
            vf_idx[cur[F[3*f+1]]++] = f;
            vf_idx[cur[F[3*f+2]]++] = f;
        }

        // Parallel 1-ring CSR extraction using tiny stack buffers (ZERO heap allocs per vertex!)
        std::vector<int> deg1(nV, 0);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < nV; ++i) {
            int buf[64], nb = 0;
            for (int k = vf_start[i]; k < vf_start[i + 1]; ++k) {
                int f = vf_idx[k];
                int v0 = F[3*f], v1 = F[3*f+1], v2 = F[3*f+2];
                if (v0 != i && nb < 64) buf[nb++] = v0;
                if (v1 != i && nb < 64) buf[nb++] = v1;
                if (v2 != i && nb < 64) buf[nb++] = v2;
            }
            std::sort(buf, buf + nb);
            deg1[i] = (int)(std::unique(buf, buf + nb) - buf);
        }
        nbr_start.assign(nV + 1, 0);
        for (int i = 0; i < nV; ++i) nbr_start[i + 1] = nbr_start[i] + deg1[i];
        nbr_idx.resize(nbr_start[nV]);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < nV; ++i) {
            int buf[64], nb = 0;
            for (int k = vf_start[i]; k < vf_start[i + 1]; ++k) {
                int f = vf_idx[k];
                int v0 = F[3*f], v1 = F[3*f+1], v2 = F[3*f+2];
                if (v0 != i && nb < 64) buf[nb++] = v0;
                if (v1 != i && nb < 64) buf[nb++] = v1;
                if (v2 != i && nb < 64) buf[nb++] = v2;
            }
            std::sort(buf, buf + nb);
            int u = (int)(std::unique(buf, buf + nb) - buf);
            std::memcpy(nbr_idx.data() + nbr_start[i], buf, u * sizeof(int));
        }

        // Parallel 2-ring CSR extraction (zero per-vertex heap allocs for normal degree <= 192!)
        std::vector<int> deg2(nV, 0);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < nV; ++i) {
            int buf[256], nb = 0;
            buf[nb++] = i;
            for (int k = nbr_start[i]; k < nbr_start[i + 1]; ++k) {
                int j = nbr_idx[k];
                if (nb < 256) buf[nb++] = j;
                for (int m = nbr_start[j]; m < nbr_start[j + 1]; ++m)
                    if (nb < 256) buf[nb++] = nbr_idx[m];
            }
            std::sort(buf, buf + nb);
            deg2[i] = (int)(std::unique(buf, buf + nb) - buf);
        }
        excl2_start.assign(nV + 1, 0);
        for (int i = 0; i < nV; ++i) excl2_start[i + 1] = excl2_start[i] + deg2[i];
        excl2_idx.resize(excl2_start[nV]);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < nV; ++i) {
            int buf[256], nb = 0;
            buf[nb++] = i;
            for (int k = nbr_start[i]; k < nbr_start[i + 1]; ++k) {
                int j = nbr_idx[k];
                if (nb < 256) buf[nb++] = j;
                for (int m = nbr_start[j]; m < nbr_start[j + 1]; ++m)
                    if (nb < 256) buf[nb++] = nbr_idx[m];
            }
            std::sort(buf, buf + nb);
            int u = (int)(std::unique(buf, buf + nb) - buf);
            std::memcpy(excl2_idx.data() + excl2_start[i], buf, u * sizeof(int));
        }
    }
};

void compute_vertex_normals_and_area_omp(
    const std::vector<Vec3>& V, const int* F, int nF, const MeshTopo& topo,
    double n_sign, std::vector<Vec3>& N, std::vector<double>* v_area_out,
    std::vector<Vec3>& fn_buf, std::vector<double>& fa_buf
) {
    int nV = (int)V.size();
    if ((int)fn_buf.size() != nF) { fn_buf.resize(nF); fa_buf.resize(nF); }
    #pragma omp parallel for schedule(static)
    for (int f = 0; f < nF; ++f) {
        int a = F[3*f], b = F[3*f+1], c = F[3*f+2];
        Vec3 fn = cross(V[b] - V[a], V[c] - V[a]);
        fn_buf[f] = fn;
        if (v_area_out) fa_buf[f] = 0.1666667 * norm(fn);
    }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nV; ++i) {
        Vec3 acc{0, 0, 0};
        double ar = 0.0;
        for (int k = topo.vf_start[i]; k < topo.vf_start[i + 1]; ++k) {
            int f = topo.vf_idx[k];
            acc += fn_buf[f];
            if (v_area_out) ar += fa_buf[f];
        }
        double len = norm(acc);
        if (len > 1e-12) N[i] = acc * (n_sign / len);
        else N[i] = {0, 0, 0};
        if (v_area_out) (*v_area_out)[i] = ar;
    }
}

void compute_vertex_normals(const std::vector<Vec3>& V, const int* F, int nF, std::vector<Vec3>& N) {
    int nV = (int)V.size();
    N.assign(nV, {0, 0, 0});
    for (int f = 0; f < nF; ++f) {
        int a = F[3*f], b = F[3*f+1], c = F[3*f+2];
        Vec3 fn = cross(V[b] - V[a], V[c] - V[a]);
        N[a] += fn; N[b] += fn; N[c] += fn;
    }
    #pragma omp parallel for
    for (int i = 0; i < nV; ++i) {
        double len = norm(N[i]);
        if (len > 1e-12) N[i] = N[i] * (1.0 / len);
    }
}

double compute_signed_volume(const std::vector<Vec3>& V, const int* F, int nF) {
    double vol6 = 0.0;
    #pragma omp parallel for reduction(+:vol6)
    for (int f = 0; f < nF; ++f)
        vol6 += dot(V[F[3*f]], cross(V[F[3*f+1]], V[F[3*f+2]]));
    return vol6 / 6.0;
}

// Fast manifold geodesic diffusion of a scalar field along the cortical surface
void manifold_smooth_scalar(
    std::vector<double>& s, const MeshTopo& topo, int n_passes, double alpha = 0.55
) {
    if (n_passes <= 0) return;
    int nV = topo.nV;
    std::vector<double> tmp(nV);
    for (int p = 0; p < n_passes; ++p) {
        #pragma omp parallel for
        for (int i = 0; i < nV; ++i) {
            int ns = topo.nbr_start[i], ne = topo.nbr_start[i + 1];
            int deg = ne - ns;
            if (deg <= 0) { tmp[i] = s[i]; continue; }
            double avg = 0.0;
            for (int k = ns; k < ne; ++k) avg += s[topo.nbr_idx[k]];
            avg /= deg;
            tmp[i] = (1.0 - alpha) * s[i] + alpha * avg;
        }
        s.swap(tmp);
    }
}

// Non-shrinking Taubin (lambda, -mu) biharmonic elastic shell fairing pass
// Kills high-frequency spikes/noise (<2.5mm) with ZERO gyral crown shrinkage!
void taubin_fair_mesh(
    std::vector<Vec3>& V, const MeshTopo& topo, const float* rip_mask,
    int n_passes, double lam = 0.33, double mu = -0.345, double rip_boost = 1.4
) {
    if (n_passes <= 0) return;
    int nV = topo.nV;
    std::vector<Vec3> tmp(nV);
    for (int p = 0; p < n_passes; ++p) {
        // Step 1: +lambda shrink pass
        #pragma omp parallel for
        for (int i = 0; i < nV; ++i) {
            int ns = topo.nbr_start[i], ne = topo.nbr_start[i + 1];
            int deg = ne - ns;
            if (deg <= 0) { tmp[i] = V[i]; continue; }
            Vec3 avg{0, 0, 0};
            for (int k = ns; k < ne; ++k) avg += V[topo.nbr_idx[k]];
            avg = avg * (1.0 / deg);
            double w = (rip_mask && rip_mask[i] > 0.5f) ? std::min(0.48, lam * rip_boost) : lam;
            tmp[i] = V[i] + (avg - V[i]) * w;
        }
        // Step 2: -mu inflate pass (zero shrinkage of macro gyri!)
        #pragma omp parallel for
        for (int i = 0; i < nV; ++i) {
            int ns = topo.nbr_start[i], ne = topo.nbr_start[i + 1];
            int deg = ne - ns;
            if (deg <= 0) { V[i] = tmp[i]; continue; }
            Vec3 avg{0, 0, 0};
            for (int k = ns; k < ne; ++k) avg += tmp[topo.nbr_idx[k]];
            avg = avg * (1.0 / deg);
            double w = (rip_mask && rip_mask[i] > 0.5f) ? std::max(-0.50, mu * rip_boost) : mu;
            V[i] = tmp[i] + (avg - tmp[i]) * w;
        }
    }
}

// Purely tangential mesh equilibration (keeps edge lengths uniform across deep sulci & crowns)
void tangential_equilibrate(
    std::vector<Vec3>& V, const std::vector<Vec3>& N, const MeshTopo& topo,
    int n_passes, double w_tan = 0.25
) {
    if (n_passes <= 0) return;
    int nV = topo.nV;
    std::vector<Vec3> tmp(nV);
    for (int p = 0; p < n_passes; ++p) {
        #pragma omp parallel for
        for (int i = 0; i < nV; ++i) {
            int ns = topo.nbr_start[i], ne = topo.nbr_start[i + 1];
            int deg = ne - ns;
            if (deg <= 0) { tmp[i] = V[i]; continue; }
            Vec3 avg{0, 0, 0};
            for (int k = ns; k < ne; ++k) avg += V[topo.nbr_idx[k]];
            avg = avg * (1.0 / deg);
            Vec3 d = avg - V[i];
            Vec3 dt = d - N[i] * dot(d, N[i]);
            double dtl = norm(dt);
            if (dtl > 0.35) dt = dt * (0.35 / dtl);
            tmp[i] = V[i] + dt * w_tan;
        }
        V.swap(tmp);
    }
}

// Simultaneous Coherent Triangle-Flip Guard (ZERO Jacobi single-vertex button-freezing!)
// Evaluates every triangle with all 3 vertices moved coherently; damps ONLY genuinely inverting triangles.
void apply_coherent_step_guard(
    const std::vector<Vec3>& V, std::vector<Vec3>& V_next,
    const int* F, int nF
) {
    int nV = (int)V.size();
    std::vector<uint8_t> bad(nV, 0);
    for (int sub = 0; sub < 3; ++sub) {
        std::fill(bad.begin(), bad.end(), 0);
        int n_bad = 0;
        #pragma omp parallel for reduction(+:n_bad)
        for (int f = 0; f < nF; ++f) {
            int a = F[3*f], b = F[3*f+1], c = F[3*f+2];
            Vec3 n_old = cross(V[b] - V[a], V[c] - V[a]);
            Vec3 n_new = cross(V_next[b] - V_next[a], V_next[c] - V_next[a]);
            double a_old2 = norm2(n_old), a_new2 = norm2(n_new);
            if (a_new2 < 0.02 * a_old2 || a_new2 < 1e-12 ||
                dot(n_new, n_old) < 0.15 * std::sqrt(a_old2 * a_new2)) {
                bad[a] = bad[b] = bad[c] = 1;
                n_bad++;
            }
        }
        if (n_bad == 0) break;
        double damp = (sub < 2) ? 0.5 : 0.0;
        #pragma omp parallel for
        for (int i = 0; i < nV; ++i) {
            if (bad[i]) V_next[i] = V[i] * (1.0 - damp) + V_next[i] * damp;
        }
    }
}

inline bool incident_tris_ok(
    int i, const Vec3& p_cand, const std::vector<Vec3>& V, const int* F, const MeshTopo& topo
) {
    int fs = topo.vf_start[i], fe = topo.vf_start[i + 1];
    for (int k = fs; k < fe; ++k) {
        int f = topo.vf_idx[k];
        int a = F[3*f], b = F[3*f+1], c = F[3*f+2];
        const Vec3& va0 = V[a], &vb0 = V[b], &vc0 = V[c];
        Vec3 n_old = cross(vb0 - va0, vc0 - va0);
        double a_old2 = norm2(n_old);
        Vec3 va = (a == i) ? p_cand : va0;
        Vec3 vb = (b == i) ? p_cand : vb0;
        Vec3 vc = (c == i) ? p_cand : vc0;
        Vec3 n_new = cross(vb - va, vc - va);
        double a_new2 = norm2(n_new);
        if (a_new2 < 0.03 * a_old2 || a_new2 < 1e-10) return false;
        if (dot(n_new, n_old) < 0.20 * std::sqrt(a_old2 * a_new2)) return false;
    }
    return true;
}

int count_si_internal(const std::vector<Vec3>& V, const int* F, int nF) {
    TriSpatialHash sh;
    sh.build(V, F, nF, 1.5);
    int hits = 0;
    #pragma omp parallel for reduction(+:hits) schedule(dynamic, 512)
    for (int f1 = 0; f1 < nF; ++f1) {
        int a1 = F[3*f1], b1 = F[3*f1+1], c1 = F[3*f1+2];
        const Vec3& va = V[a1], &vb = V[b1], &vc = V[c1];
        int ix0 = std::clamp((int)((std::min({va.x,vb.x,vc.x}) - sh.bmin.x)*sh.inv_cell), 0, sh.gx-1);
        int ix1 = std::clamp((int)((std::max({va.x,vb.x,vc.x}) - sh.bmin.x)*sh.inv_cell), 0, sh.gx-1);
        int iy0 = std::clamp((int)((std::min({va.y,vb.y,vc.y}) - sh.bmin.y)*sh.inv_cell), 0, sh.gy-1);
        int iy1 = std::clamp((int)((std::max({va.y,vb.y,vc.y}) - sh.bmin.y)*sh.inv_cell), 0, sh.gy-1);
        int iz0 = std::clamp((int)((std::min({va.z,vb.z,vc.z}) - sh.bmin.z)*sh.inv_cell), 0, sh.gz-1);
        int iz1 = std::clamp((int)((std::max({va.z,vb.z,vc.z}) - sh.bmin.z)*sh.inv_cell), 0, sh.gz-1);
        for (int ix = ix0; ix <= ix1; ++ix)
            for (int iy = iy0; iy <= iy1; ++iy) {
                int base = (ix * sh.gy + iy) * sh.gz;
                for (int iz = iz0; iz <= iz1; ++iz) {
                    int cid = base + iz;
                    for (int k = sh.cell_start[cid]; k < sh.cell_start[cid + 1]; ++k) {
                        int f2 = sh.cell_tris[k];
                        if (f2 <= f1) continue;
                        int a2 = F[3*f2], b2 = F[3*f2+1], c2 = F[3*f2+2];
                        if (a1==a2||a1==b2||a1==c2||b1==a2||b1==b2||b1==c2||c1==a2||c1==b2||c1==c2) continue;
                        if (tri_tri_intersect_disjoint(va, vb, vc, V[a2], V[b2], V[c2])) hits++;
                    }
                }
            }
    }
    return hits;
}

int enforce_zero_self_intersections(
    std::vector<Vec3>& V, const std::vector<Vec3>& V_safe_ref,
    const int* F, int nF, const MeshTopo& topo, int max_passes = 14
) {
    int nV = (int)V.size();
    std::vector<uint8_t> ever_bad(nV, 0);
    for (int pass = 0; pass < max_passes; ++pass) {
        TriSpatialHash sh;
        sh.build(V, F, nF, 1.5);
        std::vector<uint8_t> bad_v(nV, 0);
        int hit_pairs = 0;

        #pragma omp parallel for reduction(+:hit_pairs) schedule(dynamic, 512)
        for (int f1 = 0; f1 < nF; ++f1) {
            int a1 = F[3*f1], b1 = F[3*f1+1], c1 = F[3*f1+2];
            const Vec3& va = V[a1], &vb = V[b1], &vc = V[c1];
            int ix0 = std::clamp((int)((std::min({va.x,vb.x,vc.x}) - sh.bmin.x)*sh.inv_cell), 0, sh.gx-1);
            int ix1 = std::clamp((int)((std::max({va.x,vb.x,vc.x}) - sh.bmin.x)*sh.inv_cell), 0, sh.gx-1);
            int iy0 = std::clamp((int)((std::min({va.y,vb.y,vc.y}) - sh.bmin.y)*sh.inv_cell), 0, sh.gy-1);
            int iy1 = std::clamp((int)((std::max({va.y,vb.y,vc.y}) - sh.bmin.y)*sh.inv_cell), 0, sh.gy-1);
            int iz0 = std::clamp((int)((std::min({va.z,vb.z,vc.z}) - sh.bmin.z)*sh.inv_cell), 0, sh.gz-1);
            int iz1 = std::clamp((int)((std::max({va.z,vb.z,vc.z}) - sh.bmin.z)*sh.inv_cell), 0, sh.gz-1);
            for (int ix = ix0; ix <= ix1; ++ix)
                for (int iy = iy0; iy <= iy1; ++iy) {
                    int base = (ix * sh.gy + iy) * sh.gz;
                    for (int iz = iz0; iz <= iz1; ++iz) {
                        int cid = base + iz;
                        for (int k = sh.cell_start[cid]; k < sh.cell_start[cid + 1]; ++k) {
                            int f2 = sh.cell_tris[k];
                            if (f2 <= f1) continue;
                            int a2 = F[3*f2], b2 = F[3*f2+1], c2 = F[3*f2+2];
                            if (a1==a2||a1==b2||a1==c2||b1==a2||b1==b2||b1==c2||c1==a2||c1==b2||c1==c2) continue;
                            if (tri_tri_intersect_disjoint(va, vb, vc, V[a2], V[b2], V[c2])) {
                                hit_pairs++;
                                bad_v[a1] = bad_v[b1] = bad_v[c1] = 1;
                                bad_v[a2] = bad_v[b2] = bad_v[c2] = 1;
                            }
                        }
                    }
                }
        }
        if (hit_pairs == 0) return 0;

        std::vector<uint8_t> roll = bad_v;
        for (int i = 0; i < nV; ++i) if (bad_v[i]) ever_bad[i] = 1;
        if (pass >= 1) {
            for (int i = 0; i < nV; ++i) {
                if (!bad_v[i]) continue;
                for (int k = topo.nbr_start[i]; k < topo.nbr_start[i + 1]; ++k) roll[topo.nbr_idx[k]] = 1;
            }
        }
        if (pass >= 4) {
            std::vector<uint8_t> r2 = roll;
            for (int i = 0; i < nV; ++i) {
                if (!roll[i]) continue;
                for (int k = topo.nbr_start[i]; k < topo.nbr_start[i + 1]; ++k) r2[topo.nbr_idx[k]] = 1;
            }
            roll.swap(r2);
        }
        double alpha = (pass < 2) ? 0.55 : ((pass < 5) ? 0.30 : 0.0);
        #pragma omp parallel for
        for (int i = 0; i < nV; ++i) {
            if (roll[i]) {
                V[i] = V_safe_ref[i] * (1.0 - alpha) + V[i] * alpha;
            }
        }
        if (pass == max_passes - 1) {
            for (int i = 0; i < nV; ++i) if (ever_bad[i]) V[i] = V_safe_ref[i];
        }
    }
    return count_si_internal(V, F, nF);
}


void compute_principal_curvatures_2ring(
    const std::vector<Vec3>& V, const std::vector<Vec3>& N, const MeshTopo& topo,
    std::vector<double>& k1_out, std::vector<double>& k2_out,
    std::vector<Vec3>& d2_out, std::vector<double>& ani_out,
    double sig_ring = 1.65
) {
    int nV = topo.nV;
    k1_out.resize(nV); k2_out.resize(nV); d2_out.resize(nV); ani_out.resize(nV);
    const double inv_2sig2 = 1.0 / (2.0 * sig_ring * sig_ring);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nV; ++i) {
        const Vec3& pi = V[i];
        const Vec3& n = N[i];
        Vec3 ref = (std::fabs(n.x) < 0.8) ? Vec3{1.0, 0.0, 0.0} : Vec3{0.0, 1.0, 0.0};
        Vec3 u = ref - n * dot(ref, n);
        double ul = norm(u);
        if (ul < 1e-12) { k1_out[i] = k2_out[i] = ani_out[i] = 0.0; d2_out[i] = {1,0,0}; continue; }
        u = u * (1.0 / ul);
        Vec3 v = cross(n, u);

        // Symmetric 3x3 normal matrix A for q = [a, 2b, c]^T fitting 2*h = a*s^2 + (2b)*s*t + c*t^2
        double A00 = 1e-5, A01 = 0.0, A02 = 0.0;
        double A11 = 1e-5, A12 = 0.0, A22 = 1e-5;
        double B0 = 0.0, B1 = 0.0, B2 = 0.0;

        int es = topo.excl2_start[i], ee = topo.excl2_start[i + 1];
        for (int idx = es; idx < ee; ++idx) {
            int j = topo.excl2_idx[idx];
            if (j == i) continue;
            Vec3 e = V[j] - pi;
            double d2 = norm2(e);
            if (d2 < 1e-10) continue;
            double w = std::exp(-d2 * inv_2sig2);
            double s = dot(e, u), t = dot(e, v), h2 = 2.0 * dot(e, n);
            double x0 = s * s, x1 = s * t, x2 = t * t;
            A00 += w * x0 * x0; A01 += w * x0 * x1; A02 += w * x0 * x2;
            A11 += w * x1 * x1; A12 += w * x1 * x2; A22 += w * x2 * x2;
            B0  += w * x0 * h2; B1  += w * x1 * h2; B2  += w * x2 * h2;
        }

        // Exact closed-form 3x3 symmetric Cramer's rule
        double c00 = A11 * A22 - A12 * A12;
        double c01 = -(A01 * A22 - A12 * A02);
        double c02 = A01 * A12 - A11 * A02;
        double c11 = A00 * A22 - A02 * A02;
        double c12 = -(A00 * A12 - A01 * A02);
        double c22 = A00 * A11 - A01 * A01;
        double det = A00 * c00 + A01 * c01 + A02 * c02;
        if (std::fabs(det) < 1e-14) {
            k1_out[i] = k2_out[i] = ani_out[i] = 0.0; d2_out[i] = u; continue;
        }
        double inv_det = 1.0 / det;
        double a  = (c00 * B0 + c01 * B1 + c02 * B2) * inv_det;
        double b2 = (c01 * B0 + c11 * B1 + c12 * B2) * inv_det;
        double c  = (c02 * B0 + c12 * B1 + c22 * B2) * inv_det;

        // Weingarten shape operator S = -[a, b; b, c]
        double s11 = -a, s12 = -0.5 * b2, s22 = -c;
        double tr = s11 + s22;
        double diff = s11 - s22;
        double disc = std::sqrt(diff * diff + 4.0 * s12 * s12);
        double lp = 0.5 * (tr + disc);
        double lm = 0.5 * (tr - disc);

        double k1, k2;
        if (std::fabs(lp) >= std::fabs(lm)) { k1 = lp; k2 = lm; }
        else { k1 = lm; k2 = lp; }

        // Unit 2D eigenvector for k2: (s11 - k2)*eu + s12*ev = 0
        double r1u = s12, r1v = k2 - s11;
        double r2u = k2 - s22, r2v = s12;
        double n1 = r1u * r1u + r1v * r1v;
        double n2 = r2u * r2u + r2v * r2v;
        double eu = 1.0, ev = 0.0;
        if (n1 >= n2 && n1 > 1e-16) { double inv = 1.0 / std::sqrt(n1); eu = r1u * inv; ev = r1v * inv; }
        else if (n2 > 1e-16)        { double inv = 1.0 / std::sqrt(n2); eu = r2u * inv; ev = r2v * inv; }

        k1_out[i] = k1;
        k2_out[i] = k2;
        d2_out[i] = u * eu + v * ev;
        double ak1 = std::fabs(k1), ak2 = std::fabs(k2);
        ani_out[i] = (ak1 - ak2) / (ak1 + ak2 + 0.08);
    }
}

} // anonymous namespace

extern "C" {

DDSURFER_EXPORT void refine_dl_surface_physics_flow(
    double* verts_inout, int nV,
    const int* faces, int nF,
    const float* rip_mask,
    const float* z_fine,   int zf_nx, int zf_ny, int zf_nz, const double* zf_M,
    const float* fa_data,  int fa_nx, int fa_ny, int fa_nz, const double* fa_M,
    const float* csf_data, int cs_nx, int cs_ny, int cs_nz, const double* cs_M,
    const float* t6_data,  int t6_nx, int t6_ny, int t6_nz, const double* t6_M,
    const float* excl_data,int ex_nx, int ex_ny, int ex_nz, const double* ex_M,
    int n_s1, int n_s2,
    double sig1_mm, double sig2_mm,
    double dt1, double dt2,
    double delta_w, double z_dead, double gamma_d, double gamma_rad,
    double w_anchor, double w_disp_smooth, double w_tang,
    double max_disp_mm, double edge_d_mm, double grid_h_mm,
    double alpha_pv, double crown_protect, double w_z_edge, int final_taubin_passes,
    int nthreads
) {
    if (nthreads > 0) omp_set_num_threads(nthreads);

    std::vector<Vec3> V0(nV), V(nV), N(nV), V_next(nV);
    for (int i = 0; i < nV; ++i) {
        V0[i] = {verts_inout[3*i], verts_inout[3*i+1], verts_inout[3*i+2]};
        V[i] = V0[i];
    }
    double vol = compute_signed_volume(V, faces, nF);
    double n_sign = (vol >= 0.0) ? 1.0 : -1.0;

    MeshTopo topo;
    topo.build(nV, faces, nF);

    Field3D fZf{z_fine,   zf_nx, zf_ny, zf_nz, {}, -3.5f}; std::memcpy(fZf.M, zf_M, 16 * sizeof(double));
    Field3D fFA{fa_data,  fa_nx, fa_ny, fa_nz, {},  0.0f}; if (fa_data && fa_M)   std::memcpy(fFA.M,  fa_M, 16 * sizeof(double));
    Field3D fC {csf_data, cs_nx, cs_ny, cs_nz, {},  0.0f}; if (csf_data && cs_M)  std::memcpy(fC.M,   cs_M, 16 * sizeof(double));
    Field3D fEx{excl_data,ex_nx, ex_ny, ex_nz, {},  0.0f}; if (excl_data && ex_M) std::memcpy(fEx.M,  ex_M, 16 * sizeof(double));
    TensorField6 fT6{t6_data, t6_nx, t6_ny, t6_nz, {}};    if (t6_data && t6_M)   std::memcpy(fT6.M,  t6_M, 16 * sizeof(double));

    double minx = 1e9, miny = 1e9, minz = 1e9, maxx = -1e9, maxy = -1e9, maxz = -1e9;
    for (const auto& p : V) {
        if (p.x < minx) minx = p.x; if (p.y < miny) miny = p.y; if (p.z < minz) minz = p.z;
        if (p.x > maxx) maxx = p.x; if (p.y > maxy) maxy = p.y; if (p.z > maxz) maxz = p.z;
    }
    const double h = std::clamp(grid_h_mm, 0.50, 0.90), inv_h = 1.0 / h, pad = 5.5;
    Vec3 g0{minx - pad, miny - pad, minz - pad};
    int gx = (int)std::ceil((maxx - minx + 2.0 * pad) * inv_h) + 2;
    int gy = (int)std::ceil((maxy - miny + 2.0 * pad) * inv_h) + 2;
    int gz = (int)std::ceil((maxz - minz + 2.0 * pad) * inv_h) + 2;
    size_t gtot = (size_t)gx * gy * gz;
    std::vector<Float4> grid(gtot), gtmp(gtot);
    std::vector<uint8_t> row_nz0, row_nz1, row_nz2;
    std::vector<int> splat_xcnt, splat_xstart, splat_xvidx, splat_vi0;
    std::vector<double> v_area(nV), conf_arr(nV, 0.0), viol_arr(nV, 0.0);
    std::vector<double> k1(nV, 0.0), k2(nV, 0.0), H_macro(nV, 0.0), ani(nV, 0.0);
    std::vector<Vec3> d2(nV, {1, 0, 0}), f_raw(nV), fn_buf(nF);
    std::vector<double> fa_buf(nF);

    auto sample_vel = [&](const Vec3& p) -> Vec3 {
        double u = (p.x - g0.x) * inv_h, v = (p.y - g0.y) * inv_h, w = (p.z - g0.z) * inv_h;
        if (u < 0.0 || v < 0.0 || w < 0.0 || u >= gx - 1 || v >= gy - 1 || w >= gz - 1) return {0, 0, 0};
        int i0 = (int)u, j0 = (int)v, k0 = (int)w;
        double ax = u - i0, ay = v - j0, az = w - k0;
        size_t s00 = ((size_t)i0 * gy + j0) * gz + k0;
        size_t s01 = s00 + gz, s10 = s00 + (size_t)gy * gz, s11 = s10 + gz;
        double w000=(1-ax)*(1-ay)*(1-az), w001=(1-ax)*(1-ay)*az, w010=(1-ax)*ay*(1-az), w011=(1-ax)*ay*az;
        double w100=ax*(1-ay)*(1-az), w101=ax*(1-ay)*az, w110=ax*ay*(1-az), w111=ax*ay*az;
        const Float4& c000 = grid[s00];   const Float4& c001 = grid[s00+1];
        const Float4& c010 = grid[s01];   const Float4& c011 = grid[s01+1];
        const Float4& c100 = grid[s10];   const Float4& c101 = grid[s10+1];
        const Float4& c110 = grid[s11];   const Float4& c111 = grid[s11+1];
        return {
            w000*c000.x + w001*c001.x + w010*c010.x + w011*c011.x + w100*c100.x + w101*c101.x + w110*c110.x + w111*c111.x,
            w000*c000.y + w001*c001.y + w010*c010.y + w011*c011.y + w100*c100.y + w101*c101.y + w110*c110.y + w111*c111.y,
            w000*c000.z + w001*c001.z + w010*c010.z + w011*c011.z + w100*c100.z + w101*c101.z + w110*c110.z + w111*c111.z
        };
    };

    struct RStage { int iters; double sig_mm; double dt; };
    RStage stages[2] = {
        {n_s1, sig1_mm, dt1},
        {n_s2, sig2_mm, dt2},
    };
    std::vector<Vec3> V_safe = V;
    // If input DL mesh already had any self-intersections, smooth locally until clean reference exists
    if (count_si_internal(V, faces, nF) > 0) {
        taubin_fair_mesh(V, topo, rip_mask, 4, 0.28, -0.29, 1.35);
        V_safe = V;
    }

    const double ed = std::clamp(edge_d_mm, 0.8, 2.2);

    // Smooth 6-ring feathered cortical weight (1 = full cortex, 0 = medial cut) -> ZERO medial-wall border crease!
    std::vector<double> cort_w(nV, 1.0);
    if (rip_mask) {
        std::vector<double> rw(nV, 0.0);
        for (int i = 0; i < nV; ++i) rw[i] = std::clamp((double)rip_mask[i], 0.0, 1.0);
        manifold_smooth_scalar(rw, topo, 6, 0.55);
        for (int i = 0; i < nV; ++i) cort_w[i] = std::clamp(1.0 - rw[i], 0.0, 1.0);
    }

    for (int st = 0; st < 2; ++st) {
        const auto& sc = stages[st];
        if (sc.iters <= 0) continue;
        double sig_vox = std::max(0.55, sc.sig_mm * inv_h);

        for (int it = 0; it < sc.iters; ++it) {
            compute_vertex_normals_and_area_omp(V, faces, nF, topo, n_sign, N, &v_area, fn_buf, fa_buf);
            if ((alpha_pv > 1e-6 || crown_protect > 1e-6) && (it % 4 == 0)) {
                compute_principal_curvatures_2ring(V, N, topo, k1, k2, d2, ani, 1.65);
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < nV; ++i) H_macro[i] = 0.5 * (k1[i] + k2[i]);
                manifold_smooth_scalar(H_macro, topo, 10, 0.60);
            }

            // 1. Confidence-gated biophysical driving force + elastic DL prior anchor
            #pragma omp parallel for schedule(dynamic, 512)
            for (int i = 0; i < nV; ++i) {
                double cw = cort_w[i];
                if (cw < 0.02) {
                    conf_arr[i] = 0.0; viol_arr[i] = 0.0;
                    f_raw[i] = {0, 0, 0};
                    continue;
                }
                const Vec3& p = V[i];
                const Vec3& n = N[i];
                double ex = fEx.data ? fEx.sample(p) : 0.0;
                double z_here = fZf.sample(p) - delta_w;
                double fa0 = fFA.data ? fFA.sample(p) : 0.25;
                double csf0 = fC.data ? fC.sample(p) : 0.0;

                // Local normal-direction biophysical edge contrast (FA + crossing-fiber-aware z_fine gradient!)
                double c_edge = 0.5;
                if (fFA.data) {
                    double fa_in  = fFA.sample(p - n * ed);
                    double fa_out = fFA.sample(p + n * ed);
                    double grad_n = std::max(0.0, fa_in - fa_out);
                    c_edge = std::clamp((grad_n - 0.04) / 0.14, 0.0, 1.0);
                }
                if (w_z_edge > 1e-6) {
                    double z_in  = fZf.sample(p - n * ed);
                    double z_out = fZf.sample(p + n * ed);
                    double c_ez  = std::clamp((std::max(0.0, z_in - z_out) - 0.35) / 1.15, 0.0, 1.0);
                    c_edge = std::max(c_edge, w_z_edge * c_ez);
                }

                // Hard physical violations (CSF intrusion or buried inside bright WM or exclusion mask!)
                double v_csf = std::max(std::clamp((csf0 - 0.24) / 0.20, 0.0, 1.0),
                                        std::clamp((-z_here - 0.65) / 0.50, 0.0, 1.0));
                double v_wm  = std::clamp((fa0 - 0.28) / 0.10, 0.0, 1.0) * std::clamp((z_here - 0.25) / 0.40, 0.0, 1.0);
                double v_ex  = (ex > 0.48) ? 1.0 : 0.0;
                double hard_v = std::max({v_csf, v_wm, v_ex});
                double w_conf = std::clamp(std::max(0.90 * c_edge, hard_v), 0.0, 1.0);
                conf_arr[i] = w_conf;
                viol_arr[i] = hard_v;

                double vn = 0.0;
                if (v_ex > 0.5) {
                    vn = -0.65;
                } else {
                    double phi = z_here;
                    if (alpha_pv > 1e-6) {
                        double hm = H_macro[i];
                        if (hm > 0.0) {
                            double w_win = std::clamp((z_here + 0.50) / 0.25, 0.0, 1.0)
                                         * std::clamp((0.45 - z_here) / 0.25, 0.0, 1.0);
                            phi += (alpha_pv * w_win) * std::min(hm, 0.40);
                        } else if (hm < 0.0) {
                            double z_guard = std::clamp((0.40 - z_here) / 0.24, 0.0, 1.0)
                                           * std::clamp((z_here + 0.50) / 0.25, 0.0, 1.0);
                            double fa_guard = std::clamp((0.36 - fa0) / 0.08, 0.0, 1.0);
                            phi += (0.50 * alpha_pv * z_guard * fa_guard) * std::max(hm, -0.35);
                        }
                    }
                    if (gamma_rad > 1e-6 && fT6.data) {
                        double rad = fT6.radiality(p, n);
                        if (H_macro[i] > 0.05) phi += gamma_rad * std::tanh(3.0 * H_macro[i]) * (rad - 0.18);
                    }
                    // Soft dead-zone (for surgical mode: ignores tiny <z_dead noise, engages cleanly on real offset!)
                    double phi_eff = phi;
                    if (z_dead > 1e-6 && hard_v < 0.35) {
                        if (phi > z_dead) phi_eff = phi - z_dead;
                        else if (phi < -z_dead) phi_eff = phi + z_dead;
                        else phi_eff = 0.0;
                    }
                    double f_phys = w_conf * std::tanh(gamma_d * phi_eff);
                    double dn = dot(p - V0[i], n);
                    double anc_mult = 1.0;
                    // Asymmetric convex gyral-crown protection: prevents partial-volume inward erosion of thin non-CSF crown tips
                    // while keeping 100% full power for sulcal carving (H<=0), CSF ejection (v_csf>0), and outward WM expansion!
                    if (crown_protect > 1e-6 && H_macro[i] > 0.06 && v_csf < 0.35) {
                        double cg = std::clamp((H_macro[i] - 0.06) / 0.22, 0.0, 1.0) * (1.0 - v_csf);
                        if (f_phys < 0.0) {
                            f_phys *= (1.0 - std::clamp(crown_protect, 0.0, 0.95) * cg);
                        }
                        if (dn < 0.0) {
                            anc_mult += std::clamp(crown_protect, 0.0, 0.95) * cg;
                        }
                    }
                    // Elastic DL prior anchor: strongest where physical evidence is weak (1 - w_conf) + gentle baseline!
                    double f_anc = -w_anchor * anc_mult * (1.0 - 0.85 * w_conf) * std::tanh(dn / 0.75);
                    vn = cw * (f_phys + f_anc);
                }
                f_raw[i] = n * (vn * std::max(0.05, v_area[i]));
            }

            // 2. Splat onto 3D Eulerian RKHS Grid & Smooth (lock-free OpenMP splat + sparse-row AVX-512 Float4!)
            splat_float4_3d_omp(V, f_raw, v_area, g0, inv_h, gx, gy, gz, grid, row_nz0, splat_xcnt, splat_xstart, splat_xvidx, splat_vi0);
            smooth_and_norm_float4_3d(grid, gtmp, row_nz0, row_nz1, row_nz2, gx, gy, gz, sig_vox);

            // 3. RK2 Midpoint Step + Zero-Shrinkage Displacement-Field Laplacian + Smooth Medial-Wall Feathering!
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < nV; ++i) {
                double cw = cort_w[i];
                if (cw < 0.01) { V_next[i] = V0[i]; continue; }
                const Vec3& p = V[i];
                Vec3 v1 = sample_vel(p);
                Vec3 v2 = sample_vel(p + v1 * (0.5 * sc.dt));
                Vec3 disp = v2 * (sc.dt * cw);

                int ns = topo.nbr_start[i], ne = topo.nbr_start[i + 1];
                int deg = ne - ns;
                if (deg > 0) {
                    Vec3 avg_p{0, 0, 0}, avg_d{0, 0, 0};
                    for (int k = ns; k < ne; ++k) {
                        int j = topo.nbr_idx[k];
                        avg_p += V[j];
                        avg_d += (V[j] - V0[j]);
                    }
                    double inv_d = 1.0 / (double)deg;
                    Vec3 lap_p = avg_p * inv_d - p;
                    Vec3 lap_d = avg_d * inv_d - (p - V0[i]);
                    // Tangential mesh quality + ZERO-SHRINKAGE displacement-vector smoothness!
                    disp += (lap_p - N[i] * dot(lap_p, N[i])) * (w_tang * cw);
                    disp += lap_d * w_disp_smooth;
                }

                Vec3 p_new = p + disp;
                // Soft total displacement budget from V0 (feathered smoothly to 0 across medial-wall border!)
                if (max_disp_mm > 0.1) {
                    double lim = cw * max_disp_mm * (1.0 + 0.75 * viol_arr[i]);
                    Vec3 tot = p_new - V0[i];
                    double tlen = norm(tot);
                    if (tlen > lim) {
                        p_new = V0[i] + tot * (lim / (tlen + 1e-12));
                    }
                }
                V_next[i] = p_new;
            }
            apply_coherent_step_guard(V, V_next, faces, nF);
            V.swap(V_next);

            if (it == sc.iters - 1) {
                if (enforce_zero_self_intersections(V, V_safe, faces, nF, topo, 14) == 0) V_safe = V;
            }
        }
    }

    if (final_taubin_passes > 0) {
        taubin_fair_mesh(V, topo, rip_mask, final_taubin_passes, 0.26, -0.270, 1.35);
    }
    enforce_zero_self_intersections(V, V_safe, faces, nF, topo, 14);

    for (int i = 0; i < nV; ++i) {
        verts_inout[3*i] = V[i].x; verts_inout[3*i+1] = V[i].y; verts_inout[3*i+2] = V[i].z;
    }
}

// ============================================================================================================
// 1F. TEST-TIME PURE 5-DTI PHYSICS-GUIDED DIFFEOMORPHIC PIAL REFINER (`PhysRefine-Pial`)
// ============================================================================================================
DDSURFER_EXPORT void refine_dl_pial_physics_flow(
    double* vp_inout, const double* vw_ref, int nV,
    const int* faces, int nF,
    const float* rip_mask,
    const float* z_pial,    int zp_nx, int zp_ny, int zp_nz, const double* zp_M,
    const float* md_data,   int md_nx, int md_ny, int md_nz, const double* md_M,
    const float* fa_data,   int fa_nx, int fa_ny, int fa_nz, const double* fa_M,
    const float* mask_data, int mk_nx, int mk_ny, int mk_nz, const double* mk_M,
    int n_s1, int n_s2,
    double sig1_mm, double sig2_mm,
    double dt1, double dt2,
    double z_dead, double gamma_d,
    double w_anchor, double w_disp_smooth, double w_tang,
    double max_disp_mm, double edge_d_mm, double grid_h_mm,
    double t_min_mm, double d_t_par, double sulc_protect,
    double alpha_pv, double w_mprobe, double w_thick, double t_soft_mm,
    int final_taubin_passes,
    int nthreads
) {
    if (nthreads > 0) omp_set_num_threads(nthreads);

    std::vector<Vec3> V0(nV), V(nV), Vw(nV), N(nV), Nw(nV), V_next(nV);
    for (int i = 0; i < nV; ++i) {
        V0[i] = {vp_inout[3*i], vp_inout[3*i+1], vp_inout[3*i+2]};
        V[i]  = V0[i];
        Vw[i] = vw_ref ? Vec3{vw_ref[3*i], vw_ref[3*i+1], vw_ref[3*i+2]} : V0[i];
    }
    double vol = compute_signed_volume(V, faces, nF);
    double n_sign = (vol >= 0.0) ? 1.0 : -1.0;

    MeshTopo topo;
    topo.build(nV, faces, nF);

    if (vw_ref) {
        double wvol = compute_signed_volume(Vw, faces, nF);
        double w_sign = (wvol >= 0.0) ? 1.0 : -1.0;
        compute_vertex_normals(Vw, faces, nF, Nw);
        if (w_sign < 0.0) {
            #pragma omp parallel for
            for (int i = 0; i < nV; ++i) Nw[i] = Nw[i] * (-1.0);
        }
    }

    Field3D fZp{z_pial,    zp_nx, zp_ny, zp_nz, {}, 0.0f}; std::memcpy(fZp.M, zp_M, 16 * sizeof(double));
    Field3D fMD{md_data,   md_nx, md_ny, md_nz, {}, 0.0f}; if (md_data && md_M)   std::memcpy(fMD.M, md_M, 16 * sizeof(double));
    Field3D fFA{fa_data,   fa_nx, fa_ny, fa_nz, {}, 0.0f}; if (fa_data && fa_M)   std::memcpy(fFA.M, fa_M, 16 * sizeof(double));
    Field3D fMk{mask_data, mk_nx, mk_ny, mk_nz, {}, 0.0f}; if (mask_data && mk_M) std::memcpy(fMk.M, mk_M, 16 * sizeof(double));

    double minx = 1e9, miny = 1e9, minz = 1e9, maxx = -1e9, maxy = -1e9, maxz = -1e9;
    for (const auto& p : V) {
        if (p.x < minx) minx = p.x; if (p.y < miny) miny = p.y; if (p.z < minz) minz = p.z;
        if (p.x > maxx) maxx = p.x; if (p.y > maxy) maxy = p.y; if (p.z > maxz) maxz = p.z;
    }
    const double h = std::clamp(grid_h_mm, 0.55, 0.90), inv_h = 1.0 / h, pad = 5.5;
    Vec3 g0{minx - pad, miny - pad, minz - pad};
    int gx = (int)std::ceil((maxx - minx + 2.0 * pad) * inv_h) + 2;
    int gy = (int)std::ceil((maxy - miny + 2.0 * pad) * inv_h) + 2;
    int gz = (int)std::ceil((maxz - minz + 2.0 * pad) * inv_h) + 2;
    size_t gtot = (size_t)gx * gy * gz;
    std::vector<Float4> grid(gtot), gtmp(gtot);
    std::vector<uint8_t> row_nz0, row_nz1, row_nz2;
    std::vector<int> splat_xcnt, splat_xstart, splat_xvidx, splat_vi0;
    std::vector<double> v_area(nV), viol_arr(nV, 0.0);
    std::vector<double> k1(nV, 0.0), k2(nV, 0.0), H_macro(nV, 0.0), ani(nV, 0.0);
    std::vector<Vec3> d2(nV, {1, 0, 0}), f_raw(nV), fn_buf(nF);
    std::vector<double> fa_buf(nF);

    auto sample_vel = [&](const Vec3& p) -> Vec3 {
        double u = (p.x - g0.x) * inv_h, v = (p.y - g0.y) * inv_h, w = (p.z - g0.z) * inv_h;
        if (u < 0.0 || v < 0.0 || w < 0.0 || u >= gx - 1 || v >= gy - 1 || w >= gz - 1) return {0, 0, 0};
        int i0 = (int)u, j0 = (int)v, k0 = (int)w;
        double ax = u - i0, ay = v - j0, az = w - k0;
        size_t s00 = ((size_t)i0 * gy + j0) * gz + k0;
        size_t s01 = s00 + gz, s10 = s00 + (size_t)gy * gz, s11 = s10 + gz;
        double w000=(1-ax)*(1-ay)*(1-az), w001=(1-ax)*(1-ay)*az, w010=(1-ax)*ay*(1-az), w011=(1-ax)*ay*az;
        double w100=ax*(1-ay)*(1-az), w101=ax*(1-ay)*az, w110=ax*ay*(1-az), w111=ax*ay*az;
        const Float4& c000 = grid[s00];   const Float4& c001 = grid[s00+1];
        const Float4& c010 = grid[s01];   const Float4& c011 = grid[s01+1];
        const Float4& c100 = grid[s10];   const Float4& c101 = grid[s10+1];
        const Float4& c110 = grid[s11];   const Float4& c111 = grid[s11+1];
        return {
            w000*c000.x + w001*c001.x + w010*c010.x + w011*c011.x + w100*c100.x + w101*c101.x + w110*c110.x + w111*c111.x,
            w000*c000.y + w001*c001.y + w010*c010.y + w011*c011.y + w100*c100.y + w101*c101.y + w110*c110.y + w111*c111.y,
            w000*c000.z + w001*c001.z + w010*c010.z + w011*c011.z + w100*c100.z + w101*c101.z + w110*c110.z + w111*c111.z
        };
    };

    std::vector<double> cort_w(nV, 1.0);
    if (rip_mask) {
        std::vector<double> rw(nV, 0.0);
        for (int i = 0; i < nV; ++i) rw[i] = std::clamp((double)rip_mask[i], 0.0, 1.0);
        manifold_smooth_scalar(rw, topo, 6, 0.55);
        for (int i = 0; i < nV; ++i) cort_w[i] = std::clamp(1.0 - rw[i], 0.0, 1.0);
    }

    // Surgical localized exact-SI untangler (touches ONLY intersecting triangles + 1-2 ring neighbors -> 99.9% of mesh untouched!)
    auto untangle_local_si = [&](int max_passes) -> int {
        std::vector<Vec3> tmp(nV);
        for (int pass = 0; pass < max_passes; ++pass) {
            TriSpatialHash sh;
            sh.build(V, faces, nF, 1.5);
            std::vector<uint8_t> bad(nV, 0);
            int hits = 0;
            #pragma omp parallel for reduction(+:hits) schedule(dynamic, 512)
            for (int f1 = 0; f1 < nF; ++f1) {
                int a1 = faces[3*f1], b1 = faces[3*f1+1], c1 = faces[3*f1+2];
                const Vec3& va = V[a1], &vb = V[b1], &vc = V[c1];
                int ix0 = std::clamp((int)((std::min({va.x,vb.x,vc.x}) - sh.bmin.x)*sh.inv_cell), 0, sh.gx-1);
                int ix1 = std::clamp((int)((std::max({va.x,vb.x,vc.x}) - sh.bmin.x)*sh.inv_cell), 0, sh.gx-1);
                int iy0 = std::clamp((int)((std::min({va.y,vb.y,vc.y}) - sh.bmin.y)*sh.inv_cell), 0, sh.gy-1);
                int iy1 = std::clamp((int)((std::max({va.y,vb.y,vc.y}) - sh.bmin.y)*sh.inv_cell), 0, sh.gy-1);
                int iz0 = std::clamp((int)((std::min({va.z,vb.z,vc.z}) - sh.bmin.z)*sh.inv_cell), 0, sh.gz-1);
                int iz1 = std::clamp((int)((std::max({va.z,vb.z,vc.z}) - sh.bmin.z)*sh.inv_cell), 0, sh.gz-1);
                for (int ix = ix0; ix <= ix1; ++ix)
                    for (int iy = iy0; iy <= iy1; ++iy) {
                        int base = (ix * sh.gy + iy) * sh.gz;
                        for (int iz = iz0; iz <= iz1; ++iz) {
                            int cid = base + iz;
                            for (int k = sh.cell_start[cid]; k < sh.cell_start[cid + 1]; ++k) {
                                int f2 = sh.cell_tris[k];
                                if (f2 <= f1) continue;
                                int a2 = faces[3*f2], b2 = faces[3*f2+1], c2 = faces[3*f2+2];
                                if (a1==a2||a1==b2||a1==c2||b1==a2||b1==b2||b1==c2||c1==a2||c1==b2||c1==c2) continue;
                                if (tri_tri_intersect_disjoint(va, vb, vc, V[a2], V[b2], V[c2])) {
                                    hits++;
                                    bad[a1]=bad[b1]=bad[c1]=1; bad[a2]=bad[b2]=bad[c2]=1;
                                }
                            }
                        }
                    }
            }
            if (hits == 0) return 0;
            std::vector<uint8_t> roi = bad;
            int rings = (pass >= 8) ? 2 : 1;
            for (int r = 0; r < rings; ++r) {
                std::vector<uint8_t> nxt = roi;
                for (int i = 0; i < nV; ++i) {
                    if (!roi[i]) continue;
                    for (int k = topo.nbr_start[i]; k < topo.nbr_start[i + 1]; ++k) nxt[topo.nbr_idx[k]] = 1;
                }
                roi.swap(nxt);
            }
            tmp = V;
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < nV; ++i) {
                if (!roi[i]) continue;
                int ns = topo.nbr_start[i], ne = topo.nbr_start[i + 1], deg = ne - ns;
                if (deg <= 0) continue;
                Vec3 avg{0,0,0};
                for (int k = ns; k < ne; ++k) avg += V[topo.nbr_idx[k]];
                Vec3 p_sm = V[i] + (avg * (1.0 / deg) - V[i]) * (bad[i] ? 0.55 : 0.30);
                // If opposing sulcal banks still kiss after 6 passes, retract slightly toward Vw (or inward normal)!
                if (bad[i] && vw_ref && pass >= 6) {
                    Vec3 tw = Vw[i] - p_sm;
                    double dl = norm(tw);
                    if (dl > 0.65) p_sm += tw * (0.12 / dl);
                }
                tmp[i] = p_sm;
            }
            V.swap(tmp);
        }
        return count_si_internal(V, faces, nF);
    };

    // Resolve any pre-existing self-intersections on input DL Pial (e.g. ABIDE SI=63!) cleanly -> guaranteed SI=0 V_safe!
    untangle_local_si(28);
    std::vector<Vec3> V_safe = V;
    const double ed = std::clamp(edge_d_mm, 1.0, 2.2);

    struct PStage { int iters; double sig_mm; double dt; };
    PStage stages[2] = { {n_s1, sig1_mm, dt1}, {n_s2, sig2_mm, dt2} };

    for (int st = 0; st < 2; ++st) {
        const auto& sc = stages[st];
        if (sc.iters <= 0) continue;
        double sig_vox = std::max(0.55, sc.sig_mm * inv_h);

        for (int it = 0; it < sc.iters; ++it) {
            compute_vertex_normals_and_area_omp(V, faces, nF, topo, n_sign, N, &v_area, fn_buf, fa_buf);
            if ((sulc_protect > 1e-6 || alpha_pv > 1e-6 || w_mprobe > 1e-6) && (it % 4 == 0)) {
                compute_principal_curvatures_2ring(V, N, topo, k1, k2, d2, ani, 1.65);
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < nV; ++i) H_macro[i] = 0.5 * (k1[i] + k2[i]);
                manifold_smooth_scalar(H_macro, topo, 10, 0.60);
            }

            #pragma omp parallel for schedule(dynamic, 512)
            for (int i = 0; i < nV; ++i) {
                double cw = cort_w[i];
                if (cw < 0.02) { viol_arr[i] = 0.0; f_raw[i] = {0, 0, 0}; continue; }
                const Vec3& p = V[i];
                const Vec3& n = N[i];

                double mk0   = fMk.data ? fMk.sample(p) : 1.0;
                double mk_in = fMk.data ? fMk.sample(p - n * ed) : 1.0;
                double mk_ou = fMk.data ? fMk.sample(p + n * ed) : 1.0;
                double in_meas = std::clamp((mk0 - 0.45) / 0.35, 0.0, 1.0);
                double both_ok = in_meas * std::clamp((mk_in - 0.45) / 0.35, 0.0, 1.0) * std::clamp((mk_ou - 0.45) / 0.35, 0.0, 1.0);

                double z_here = fZp.sample(p);
                double md0 = fMD.data ? fMD.sample(p) : d_t_par;
                double fa0 = fFA.data ? fFA.sample(p) : 0.15;

                double c_edge = 0.0;
                double v_rim_gm = 0.0;
                if (fMD.data && both_ok > 0.05) {
                    double md_in = fMD.sample(p - n * ed);
                    double md_ou = fMD.sample(p + n * ed);
                    double grad_md = std::max(0.0, md_ou - md_in);
                    c_edge = both_ok * std::clamp((grad_md - 0.15) / 0.50, 0.0, 1.0);
                }
                // Surgical tight-BET2 outer-rim recovery (activates ONLY when tight BET2 clips within 0.55*ed
                // of a convex dark-GM crown -> fixes ABIDE/FJMU without ever firing on loose-mask HCP/DEG2MM!):
                if (w_mprobe > 1e-6 && fMD.data && in_meas > 0.05 && mk_ou < 0.55) {
                    double mk_s = fMk.data ? fMk.sample(p + n * (0.55 * ed)) : 1.0;
                    if (mk_s < 0.55 && H_macro[i] > 0.02 && z_here > 0.45) {
                        v_rim_gm = w_mprobe * std::clamp((mk0 - 0.62) / 0.28, 0.0, 1.0)
                                            * std::clamp((z_here - 0.45) / 0.65, 0.0, 1.0);
                    }
                }

                double v_csf = in_meas * std::clamp((-z_here - 0.55) / 0.65, 0.0, 1.0);
                double v_wm  = in_meas * std::clamp((fa0 - 0.28) / 0.10, 0.0, 1.0)
                                       * std::clamp(((d_t_par + 0.18) - md0) / 0.15, 0.0, 1.0);
                double v_col = 0.0;
                double v_thk = 0.0;
                if (vw_ref && t_min_mm > 0.05) {
                    double col_t = dot(p - Vw[i], Nw[i]);
                    if (col_t < t_min_mm) {
                        v_col = std::clamp((t_min_mm - col_t) / 0.45, 0.0, 1.0);
                    }
                    double ts = (t_soft_mm > 0.5) ? t_soft_mm : 1.85;
                    if (w_thick > 1e-6 && col_t < ts && z_here > 0.38) {
                        v_thk = w_thick * in_meas * std::clamp((ts - col_t) / std::max(0.65, 0.50 * ts), 0.0, 1.0)
                                                  * std::clamp((z_here - 0.38) / 0.62, 0.0, 1.0);
                    }
                }
                double hard_v = std::max({v_csf, v_wm, v_col});
                double w_conf = std::clamp(std::max({0.92 * c_edge, v_csf, v_wm, v_rim_gm, v_thk}), 0.0, 1.0);
                viol_arr[i] = hard_v;

                double phi = z_here;
                if (alpha_pv > 1e-6) {
                    double hm = H_macro[i];
                    double w_win = std::clamp((z_here + 1.15) / 0.45, 0.0, 1.0)
                                 * std::clamp((1.15 - z_here) / 0.45, 0.0, 1.0);
                    if (hm > 0.0) {
                        phi += (alpha_pv * w_win) * std::min(hm, 0.38);
                    } else if (hm < 0.0) {
                        phi += (0.85 * alpha_pv * w_win) * std::max(hm, -0.38);
                    }
                }

                double phi_eff = phi;
                if (z_dead > 1e-6 && hard_v < 0.35) {
                    if (phi > z_dead) phi_eff = phi - z_dead;
                    else if (phi < -z_dead) phi_eff = phi + z_dead;
                    else phi_eff = 0.0;
                }
                double f_phys = w_conf * std::tanh(gamma_d * phi_eff) + 0.90 * v_col;
                double dn = dot(p - V0[i], n);
                double anc_mult = 1.0;
                // Asymmetric concave sulcal protection (dual of White crown_protect!):
                // Prevents touching opposing sulcal Pial banks from over-expanding into each other,
                // while keeping 100% full power for outer crown expansion, CSF retraction, and WM anti-crossing!
                if (sulc_protect > 1e-6 && H_macro[i] < -0.05 && v_wm < 0.35 && v_col < 0.35) {
                    double sg = std::clamp((-H_macro[i] - 0.05) / 0.22, 0.0, 1.0);
                    if (f_phys > 0.0) f_phys *= (1.0 - std::clamp(sulc_protect, 0.0, 0.95) * sg);
                    if (dn > 0.0) anc_mult += std::clamp(sulc_protect, 0.0, 0.95) * sg;
                }
                double f_anc = -w_anchor * anc_mult * (1.0 - 0.85 * w_conf) * std::tanh(dn / 0.85);
                double vn = cw * (f_phys + f_anc);
                f_raw[i] = n * (vn * std::max(0.05, v_area[i]));
            }

            splat_float4_3d_omp(V, f_raw, v_area, g0, inv_h, gx, gy, gz, grid, row_nz0, splat_xcnt, splat_xstart, splat_xvidx, splat_vi0);
            smooth_and_norm_float4_3d(grid, gtmp, row_nz0, row_nz1, row_nz2, gx, gy, gz, sig_vox);

            #pragma omp parallel for schedule(static)
            for (int i = 0; i < nV; ++i) {
                double cw = cort_w[i];
                if (cw < 0.01) { V_next[i] = V0[i]; continue; }
                const Vec3& p = V[i];
                Vec3 v1 = sample_vel(p);
                Vec3 v2 = sample_vel(p + v1 * (0.5 * sc.dt));
                Vec3 disp = v2 * (sc.dt * cw);

                int ns = topo.nbr_start[i], ne = topo.nbr_start[i + 1];
                int deg = ne - ns;
                if (deg > 0) {
                    Vec3 avg_p{0, 0, 0}, avg_d{0, 0, 0};
                    for (int k = ns; k < ne; ++k) {
                        int j = topo.nbr_idx[k];
                        avg_p += V[j];
                        avg_d += (V[j] - V0[j]);
                    }
                    double inv_d = 1.0 / (double)deg;
                    Vec3 lap_p = avg_p * inv_d - p;
                    Vec3 lap_d = avg_d * inv_d - (p - V0[i]);
                    disp += (lap_p - N[i] * dot(lap_p, N[i])) * (w_tang * cw);
                    disp += lap_d * w_disp_smooth;
                }

                Vec3 p_new = p + disp;
                if (max_disp_mm > 0.1) {
                    double lim = cw * max_disp_mm * (1.0 + 0.75 * viol_arr[i]);
                    Vec3 tot = p_new - V0[i];
                    double tlen = norm(tot);
                    if (tlen > lim) p_new = V0[i] + tot * (lim / (tlen + 1e-12));
                }
                V_next[i] = p_new;
            }
            apply_coherent_step_guard(V, V_next, faces, nF);
            V.swap(V_next);

            if (it == sc.iters - 1) {
                if (enforce_zero_self_intersections(V, V_safe, faces, nF, topo, 14) == 0) V_safe = V;
            }
        }
    }

    if (final_taubin_passes > 0) {
        taubin_fair_mesh(V, topo, rip_mask, final_taubin_passes, 0.26, -0.270, 1.35);
    }
    if (enforce_zero_self_intersections(V, V_safe, faces, nF, topo, 16) > 0) {
        untangle_local_si(20);
    }

    for (int i = 0; i < nV; ++i) {
        vp_inout[3*i] = V[i].x; vp_inout[3*i+1] = V[i].y; vp_inout[3*i+2] = V[i].z;
    }
}


DDSURFER_EXPORT int count_self_intersections_tdf(const double* verts, int nV, const int* faces, int nF, int nthreads) {
    if (nthreads > 0) omp_set_num_threads(nthreads);
    std::vector<Vec3> V(nV);
    for (int i = 0; i < nV; ++i) V[i] = {verts[3*i], verts[3*i+1], verts[3*i+2]};
    return count_si_internal(V, faces, nF);
}


static inline int reflect_mirror(int idx, int n) {
    if (n <= 1) return 0;
    while (idx < 0 || idx >= n) {
        if (idx < 0) idx = -idx;
        if (idx >= n) idx = 2 * n - 2 - idx;
    }
    return idx;
}

static inline void bspline3_prefilter_1d(double* c, int n) {
    if (n < 2) return;
    const double z1 = -0.26794919243112270647; // sqrt(3) - 2
    const double lambda = 6.0;
    for (int i = 0; i < n; ++i) c[i] *= lambda;
    // Exact mirror-boundary causal initialization (matches scipy.ndimage.spline_filter1d(order=3, mode='mirror'))
    int kmax = std::min(n - 1, 20);
    double sum = c[0], zk = z1;
    for (int k = 1; k <= kmax; ++k) { sum += zk * c[k]; zk *= z1; }
    if (n <= 20) {
        double zn = std::pow(z1, n - 1);
        double zn2 = zn * zn;
        double sum2 = c[0], z2 = z1;
        for (int k = 1; k < n - 1; ++k) { sum2 += z2 * c[k]; z2 *= z1; }
        double rsum = 0.0, zr = zn * z1;
        for (int k = n - 2; k >= 1; --k) { rsum += zr * c[k]; zr *= z1; }
        sum = (sum2 + zn * c[n - 1] + rsum) / (1.0 - zn2);
    }
    c[0] = sum;
    for (int i = 1; i < n; ++i) c[i] += z1 * c[i - 1];
    // Exact mirror-boundary anti-causal pass
    c[n - 1] = (z1 / (z1 * z1 - 1.0)) * (c[n - 1] + z1 * c[n - 2]);
    for (int i = n - 2; i >= 0; --i) c[i] = z1 * (c[i + 1] - c[i]);
}

// Exact OpenMP 3D Affine Resampler (order=1 trilinear OR order=3 Unser cubic B-spline, ~40x faster than nibabel/scipy!)
// M16 is 4x4 row-major mapping target voxel (i, j, k, 1) -> source voxel (x, y, z, 1): inv(src_aff) @ dst_aff
DDSURFER_EXPORT void fast_affine_resample_tdf(
    const float* src, int snx, int sny, int snz,
    float* dst, int dnx, int dny, int dnz,
    const double* M16, int order, int clamp_oob, int nthreads
) {
    if (nthreads > 0) omp_set_num_threads(nthreads);
    const size_t stot = (size_t)snx * sny * snz;
    std::vector<float> coef(stot);

    if (order == 3) {
        std::vector<double> tmp(stot);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < stot; ++i) {
            float v = src[i];
            tmp[i] = std::isfinite(v) ? (double)v : 0.0;
        }
        // 1D prefilter along Z
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < snx; ++i)
            for (int j = 0; j < sny; ++j)
                bspline3_prefilter_1d(tmp.data() + ((size_t)i * sny + j) * snz, snz);
        // 1D prefilter along Y
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < snx; ++i) {
            std::vector<double> line(sny);
            for (int k = 0; k < snz; ++k) {
                for (int j = 0; j < sny; ++j) line[j] = tmp[((size_t)i * sny + j) * snz + k];
                bspline3_prefilter_1d(line.data(), sny);
                for (int j = 0; j < sny; ++j) tmp[((size_t)i * sny + j) * snz + k] = line[j];
            }
        }
        // 1D prefilter along X
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < sny; ++j) {
            std::vector<double> line(snx);
            for (int k = 0; k < snz; ++k) {
                for (int i = 0; i < snx; ++i) line[i] = tmp[((size_t)i * sny + j) * snz + k];
                bspline3_prefilter_1d(line.data(), snx);
                for (int i = 0; i < snx; ++i) coef[((size_t)i * sny + j) * snz + k] = (float)line[i];
            }
        }
    } else {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < stot; ++i) {
            float v = src[i];
            coef[i] = std::isfinite(v) ? v : 0.0f;
        }
    }

    const double m00 = M16[0], m01 = M16[1], m02 = M16[2],  m03 = M16[3];
    const double m10 = M16[4], m11 = M16[5], m12 = M16[6],  m13 = M16[7];
    const double m20 = M16[8], m21 = M16[9], m22 = M16[10], m23 = M16[11];

    if (order == 3) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < dnx; ++i) {
            for (int j = 0; j < dny; ++j) {
                const double bx = m00 * i + m01 * j + m03;
                const double by = m10 * i + m11 * j + m13;
                const double bz = m20 * i + m21 * j + m23;
                float* out_row = dst + ((size_t)i * dny + j) * dnz;
                for (int k = 0; k < dnz; ++k) {
                    double x = bx + m02 * k;
                    double y = by + m12 * k;
                    double z = bz + m22 * k;
                    if (x < 0.0 || y < 0.0 || z < 0.0 || x > snx - 1.0 || y > sny - 1.0 || z > snz - 1.0) {
                        if (!clamp_oob) { out_row[k] = 0.0f; continue; }
                        x = std::clamp(x, 0.0, (double)(snx - 1));
                        y = std::clamp(y, 0.0, (double)(sny - 1));
                        z = std::clamp(z, 0.0, (double)(snz - 1));
                    }
                    int ix = (int)std::floor(x), iy = (int)std::floor(y), iz = (int)std::floor(z);
                    float fx = (float)(x - ix), fy = (float)(y - iy), fz = (float)(z - iz);
                    float tx = 1.f - fx, ty = 1.f - fy, tz = 1.f - fz;
                    float wx[4] = { (1.f/6.f)*tx*tx*tx, (2.f/3.f)-0.5f*fx*fx*(2.f-fx), (2.f/3.f)-0.5f*tx*tx*(2.f-tx), (1.f/6.f)*fx*fx*fx };
                    float wy[4] = { (1.f/6.f)*ty*ty*ty, (2.f/3.f)-0.5f*fy*fy*(2.f-fy), (2.f/3.f)-0.5f*ty*ty*(2.f-ty), (1.f/6.f)*fy*fy*fy };
                    float wz[4] = { (1.f/6.f)*tz*tz*tz, (2.f/3.f)-0.5f*fz*fz*(2.f-fz), (2.f/3.f)-0.5f*tz*tz*(2.f-tz), (1.f/6.f)*fz*fz*fz };
                    float acc = 0.0f;
                    if (ix >= 1 && ix <= snx - 3 && iy >= 1 && iy <= sny - 3 && iz >= 1 && iz <= snz - 3) {
                        const float* base = coef.data() + ((size_t)(ix - 1) * sny + (iy - 1)) * snz + (iz - 1);
                        const size_t sx = (size_t)sny * snz;
                        for (int a = 0; a < 4; ++a) {
                            const float* px = base + (size_t)a * sx;
                            float ay_s = 0.0f;
                            for (int b = 0; b < 4; ++b) {
                                const float* r = px + (size_t)b * snz;
                                ay_s += wy[b] * (wz[0]*r[0] + wz[1]*r[1] + wz[2]*r[2] + wz[3]*r[3]);
                            }
                            acc += wx[a] * ay_s;
                        }
                    } else {
                        int rx[4], ry[4], rz[4];
                        for (int t = 0; t < 4; ++t) {
                            rx[t] = reflect_mirror(ix - 1 + t, snx);
                            ry[t] = reflect_mirror(iy - 1 + t, sny);
                            rz[t] = reflect_mirror(iz - 1 + t, snz);
                        }
                        for (int a = 0; a < 4; ++a) {
                            const size_t ox = (size_t)rx[a] * sny;
                            float ay_s = 0.0f;
                            for (int b = 0; b < 4; ++b) {
                                const float* row = coef.data() + (ox + ry[b]) * snz;
                                ay_s += wy[b] * (wz[0]*row[rz[0]] + wz[1]*row[rz[1]] + wz[2]*row[rz[2]] + wz[3]*row[rz[3]]);
                            }
                            acc += wx[a] * ay_s;
                        }
                    }
                    out_row[k] = acc;
                }
            }
        }
    } else {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < dnx; ++i) {
            for (int j = 0; j < dny; ++j) {
                const double bx = m00 * i + m01 * j + m03;
                const double by = m10 * i + m11 * j + m13;
                const double bz = m20 * i + m21 * j + m23;
                float* out_row = dst + ((size_t)i * dny + j) * dnz;
                for (int k = 0; k < dnz; ++k) {
                    double x = bx + m02 * k;
                    double y = by + m12 * k;
                    double z = bz + m22 * k;
                    if (x < 0.0 || y < 0.0 || z < 0.0 || x > snx - 1.0 || y > sny - 1.0 || z > snz - 1.0) {
                        if (!clamp_oob) { out_row[k] = 0.0f; continue; }
                        x = std::clamp(x, 0.0, (double)(snx - 1));
                        y = std::clamp(y, 0.0, (double)(sny - 1));
                        z = std::clamp(z, 0.0, (double)(snz - 1));
                    }
                    int ix0 = std::clamp((int)std::floor(x), 0, snx - 1), ix1 = std::min(ix0 + 1, snx - 1);
                    int iy0 = std::clamp((int)std::floor(y), 0, sny - 1), iy1 = std::min(iy0 + 1, sny - 1);
                    int iz0 = std::clamp((int)std::floor(z), 0, snz - 1), iz1 = std::min(iz0 + 1, snz - 1);
                    float fx = (float)(x - ix0), fy = (float)(y - iy0), fz = (float)(z - iz0);
                    const size_t yz = (size_t)sny * snz;
                    const size_t s00 = (size_t)ix0 * yz + (size_t)iy0 * snz;
                    const size_t s01 = (size_t)ix0 * yz + (size_t)iy1 * snz;
                    const size_t s10 = (size_t)ix1 * yz + (size_t)iy0 * snz;
                    const size_t s11 = (size_t)ix1 * yz + (size_t)iy1 * snz;
                    float v00 = coef[s00 + iz0] + fz * (coef[s00 + iz1] - coef[s00 + iz0]);
                    float v01 = coef[s01 + iz0] + fz * (coef[s01 + iz1] - coef[s01 + iz0]);
                    float v10 = coef[s10 + iz0] + fz * (coef[s10 + iz1] - coef[s10 + iz0]);
                    float v11 = coef[s11 + iz0] + fz * (coef[s11 + iz1] - coef[s11 + iz0]);
                    float v0 = v00 + fy * (v01 - v00);
                    float v1 = v10 + fy * (v11 - v10);
                    out_row[k] = v0 + fx * (v1 - v0);
                }
            }
        }
    }
}

DDSURFER_EXPORT void fast_gaussian_3d_tdf(
    const float* src, float* dst, int nx, int ny, int nz,
    double sigma, int nthreads
);

// Fast stride-1 scalar 3D Gaussian filter helper (matches scipy.ndimage.gaussian_filter with mode='reflect')
static void smooth_scalar_3d(
    const float* src, float* dst, std::vector<float>& tmp,
    int nx, int ny, int nz, double sigma
) {
    int rad = std::max(1, (int)std::ceil(4.0 * sigma));
    std::vector<float> ker(2 * rad + 1);
    double ksum = 0.0;
    for (int k = -rad; k <= rad; ++k) {
        double vv = std::exp(-0.5 * (k * k) / (sigma * sigma));
        ker[k + rad] = (float)vv; ksum += vv;
    }
    for (auto& kv : ker) kv = (float)(kv / ksum);
    const size_t yz = (size_t)ny * nz;

    // Pass X (stride-1 inner loop!)
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nx; ++i) {
        float* out_i = dst + (size_t)i * yz;
        std::memset(out_i, 0, yz * sizeof(float));
        for (int r = -rad; r <= rad; ++r) {
            int ii = std::clamp(i + r, 0, nx - 1);
            const float w = ker[r + rad];
            const float* in_ii = src + (size_t)ii * yz;
            for (size_t p = 0; p < yz; ++p) out_i[p] += in_ii[p] * w;
        }
    }
    // Pass Y (stride-1 inner loop!)
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nx; ++i) {
        const float* in_i = dst + (size_t)i * yz;
        float* out_i = tmp.data() + (size_t)i * yz;
        for (int j = 0; j < ny; ++j) {
            float* out_ij = out_i + (size_t)j * nz;
            std::memset(out_ij, 0, nz * sizeof(float));
            for (int r = -rad; r <= rad; ++r) {
                int jj = std::clamp(j + r, 0, ny - 1);
                const float w = ker[r + rad];
                const float* in_ijj = in_i + (size_t)jj * nz;
                for (int k = 0; k < nz; ++k) out_ij[k] += in_ijj[k] * w;
            }
        }
    }
    // Pass Z (L1-hot contiguous row!)
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            const float* in_ij = tmp.data() + ((size_t)i * ny + j) * nz;
            float* out_ij = dst + ((size_t)i * ny + j) * nz;
            for (int k = 0; k < nz; ++k) {
                float acc = 0.0f;
                for (int r = -rad; r <= rad; ++r) {
                    int kk = std::clamp(k + r, 0, nz - 1);
                    acc += in_ij[kk] * ker[r + rad];
                }
                out_ij[k] = acc;
            }
        }
    }
}

DDSURFER_EXPORT void fast_gaussian_3d_tdf(
    const float* src, float* dst, int nx, int ny, int nz,
    double sigma, int nthreads
) {
    if (nthreads > 0) omp_set_num_threads(nthreads);
    const size_t tot = (size_t)nx * ny * nz;
    if (sigma <= 1e-6) {
        if (src != dst) std::memcpy(dst, src, tot * sizeof(float));
        return;
    }
    std::vector<float> tmp(tot);
    smooth_scalar_3d(src, dst, tmp, nx, ny, nz, sigma);
}

// Exact OpenMP multi-scale 3D Hessian planar cortical sheetness (Cardano closed-form, ~17x faster than NumPy!)
DDSURFER_EXPORT void compute_3d_sheetness_tdf(
    const float* vol, int nx, int ny, int nz,
    const double* sigmas, int n_sigmas,
    float* r_out, float* v_out, int nthreads
) {
    if (nthreads > 0) omp_set_num_threads(nthreads);
    const size_t tot = (size_t)nx * ny * nz;
    std::memset(r_out, 0, tot * sizeof(float));
    std::memset(v_out, 0, tot * sizeof(float));
    std::vector<float> sm(tot), tmp(tot);
    const size_t yz = (size_t)ny * nz;

    for (int si = 0; si < n_sigmas; ++si) {
        const double sig = sigmas[si];
        const double s2 = sig * sig;
        smooth_scalar_3d(vol, sm.data(), tmp, nx, ny, nz, sig);

        // Exact np.gradient(np.gradient(sm)) second derivatives (2-step central diff for diagonal, 1-step for cross)
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                const int im2 = std::max(0, i - 2), im1 = std::max(0, i - 1), ip1 = std::min(nx - 1, i + 1), ip2 = std::min(nx - 1, i + 2);
                const int jm2 = std::max(0, j - 2), jm1 = std::max(0, j - 1), jp1 = std::min(ny - 1, j + 1), jp2 = std::min(ny - 1, j + 2);
                const size_t b_i = (size_t)i * yz, b_im1 = (size_t)im1 * yz, b_ip1 = (size_t)ip1 * yz;
                const size_t b_im2 = (size_t)im2 * yz, b_ip2 = (size_t)ip2 * yz;
                const size_t row_off = b_i + (size_t)j * nz;
                for (int k = 0; k < nz; ++k) {
                    const int km2 = std::max(0, k - 2), km1 = std::max(0, k - 1), kp1 = std::min(nz - 1, k + 1), kp2 = std::min(nz - 1, k + 2);
                    const double v0 = sm[row_off + k];
                    const double hxx = 0.25 * (sm[b_ip2 + j * nz + k] - 2.0 * v0 + sm[b_im2 + j * nz + k]) * s2;
                    const double hyy = 0.25 * (sm[b_i + jp2 * nz + k] - 2.0 * v0 + sm[b_i + jm2 * nz + k]) * s2;
                    const double hzz = 0.25 * (sm[row_off + kp2] - 2.0 * v0 + sm[row_off + km2]) * s2;
                    const double hxy = 0.25 * (sm[b_ip1 + jp1 * nz + k] - sm[b_ip1 + jm1 * nz + k] - sm[b_im1 + jp1 * nz + k] + sm[b_im1 + jm1 * nz + k]) * s2;
                    const double hxz = 0.25 * (sm[b_ip1 + j * nz + kp1] - sm[b_ip1 + j * nz + km1] - sm[b_im1 + j * nz + kp1] + sm[b_im1 + j * nz + km1]) * s2;
                    const double hyz = 0.25 * (sm[b_i + jp1 * nz + kp1] - sm[b_i + jp1 * nz + km1] - sm[b_i + jm1 * nz + kp1] + sm[b_i + jm1 * nz + km1]) * s2;

                    const double p1 = hxy * hxy + hxz * hxz + hyz * hyz;
                    const double q = (hxx + hyy + hzz) * (1.0 / 3.0);
                    const double p2 = (hxx - q) * (hxx - q) + (hyy - q) * (hyy - q) + (hzz - q) * (hzz - q) + 2.0 * p1;
                    const double p = std::sqrt(std::max(p2 * (1.0 / 6.0), 1e-12));
                    const double inv_p = 1.0 / p;
                    const double b00 = (hxx - q) * inv_p, b11 = (hyy - q) * inv_p, b22 = (hzz - q) * inv_p;
                    const double b01 = hxy * inv_p, b02 = hxz * inv_p, b12 = hyz * inv_p;
                    const double det_b = b00 * (b11 * b22 - b12 * b12)
                                       - b01 * (b01 * b22 - b12 * b02)
                                       + b02 * (b01 * b12 - b11 * b02);
                    const double r = std::clamp(0.5 * det_b, -1.0, 1.0);
                    const double phi = std::acos(r) * (1.0 / 3.0);
                    const double e1 = q + 2.0 * p * std::cos(phi);
                    const double e3 = q + 2.0 * p * std::cos(phi + 2.0943951023931953);
                    const double e2 = 3.0 * q - e1 - e3;

                    const double a1 = std::fabs(e1), a2 = std::fabs(e2), a3 = std::fabs(e3);
                    const size_t idx = row_off + k;
                    if (e3 < 0.0) {
                        double t = std::min(a2, a1) / (a3 + 1e-5);
                        float rs = (float)((-e3) * std::exp(-2.0 * t * t));
                        if (rs > r_out[idx]) r_out[idx] = rs;
                    }
                    if (e1 > 0.0) {
                        double t = std::min(a2, a3) / (a1 + 1e-5);
                        float vs = (float)(e1 * std::exp(-2.0 * t * t));
                        if (vs > v_out[idx]) v_out[idx] = vs;
                    }
                }
            }
        }
    }
}

// Exact OpenMP 3D Felzenszwalb-Huttenlocher Squared Euclidean Distance Transform (+ optional nearest-voxel fill!)
static inline void fh_edt_1d(const float* f, int n, float* d, int* arg, int* v, float* z) {
    int k = 0;
    v[0] = 0; z[0] = -1e20f; z[1] = 1e20f;
    for (int q = 1; q < n; ++q) {
        if (f[q] >= 1e11f) continue;
        float fq_q2 = f[q] + (float)(q * q);
        float s;
        while (true) {
            int vk = v[k];
            if (f[vk] >= 1e11f) { s = -1e19f; break; }
            s = (fq_q2 - (f[vk] + (float)(vk * vk))) / (float)(2 * (q - vk));
            if (s <= z[k] && k > 0) --k;
            else break;
        }
        if (f[v[k]] >= 1e11f) { v[0] = q; z[0] = -1e20f; z[1] = 1e20f; k = 0; }
        else { ++k; v[k] = q; z[k] = s; z[k + 1] = 1e20f; }
    }
    k = 0;
    for (int q = 0; q < n; ++q) {
        while (z[k + 1] < (float)q) ++k;
        int vk = v[k];
        float dq = (float)(q - vk);
        d[q] = (f[vk] >= 1e11f) ? 1e12f : (dq * dq + f[vk]);
        if (arg) arg[q] = vk;
    }
}

// Computes exact Euclidean distance to zero-voxels of `mask_u8` (`dist_out`),
// AND if `nn_src && nn_dst` are provided, fills `nn_dst` with exact nearest-zero-voxel value from `nn_src`!
DDSURFER_EXPORT void fast_edt_3d_tdf(
    const uint8_t* mask_u8, int nx, int ny, int nz,
    float* dist_out, const float* nn_src, float* nn_dst, int nthreads
) {
    if (nthreads > 0) omp_set_num_threads(nthreads);
    const size_t tot = (size_t)nx * ny * nz;
    const size_t yz = (size_t)ny * nz;
    const bool need_nn = (nn_src != nullptr && nn_dst != nullptr);
    std::vector<float> buf1(tot), buf2(tot);
    std::vector<int> nz_idx, ny_idx;
    if (need_nn) { nz_idx.resize(tot); ny_idx.resize(tot); }

    // Pass 1: Z
    #pragma omp parallel
    {
        std::vector<float> f(nz), d(nz), z(nz + 1);
        std::vector<int> v(nz), a(nz);
        #pragma omp for collapse(2) schedule(static)
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                size_t off = ((size_t)i * ny + j) * nz;
                for (int k = 0; k < nz; ++k) f[k] = mask_u8[off + k] ? 1e12f : 0.0f;
                fh_edt_1d(f.data(), nz, buf1.data() + off, need_nn ? (nz_idx.data() + off) : nullptr, v.data(), z.data());
            }
        }
    }
    // Pass 2: Y
    #pragma omp parallel
    {
        std::vector<float> f(ny), d(ny), z(ny + 1);
        std::vector<int> v(ny), a(ny);
        #pragma omp for schedule(static)
        for (int i = 0; i < nx; ++i) {
            for (int k = 0; k < nz; ++k) {
                size_t bx = (size_t)i * yz + k;
                for (int j = 0; j < ny; ++j) f[j] = buf1[bx + (size_t)j * nz];
                fh_edt_1d(f.data(), ny, d.data(), need_nn ? a.data() : nullptr, v.data(), z.data());
                for (int j = 0; j < ny; ++j) {
                    buf2[bx + (size_t)j * nz] = d[j];
                    if (need_nn) ny_idx[bx + (size_t)j * nz] = a[j];
                }
            }
        }
    }
    // Pass 3: X
    #pragma omp parallel
    {
        std::vector<float> f(nx), d(nx), z(nx + 1);
        std::vector<int> v(nx), a(nx);
        #pragma omp for schedule(static)
        for (int j = 0; j < ny; ++j) {
            for (int k = 0; k < nz; ++k) {
                size_t byz = (size_t)j * nz + k;
                for (int i = 0; i < nx; ++i) f[i] = buf2[(size_t)i * yz + byz];
                fh_edt_1d(f.data(), nx, d.data(), need_nn ? a.data() : nullptr, v.data(), z.data());
                for (int i = 0; i < nx; ++i) {
                    size_t idx = (size_t)i * yz + byz;
                    if (dist_out) dist_out[idx] = std::sqrt(std::max(0.0f, d[i]));
                    if (need_nn) {
                        int xi = a[i];
                        int yi = ny_idx[(size_t)xi * yz + byz];
                        int zi = nz_idx[((size_t)xi * ny + yi) * nz + k];
                        nn_dst[idx] = nn_src[((size_t)xi * ny + yi) * nz + zi];
                    }
                }
            }
        }
    }
}

// Fast outward-oriented unit vertex normals in OpenMP (~4 ms vs 189 ms in trimesh!)
DDSURFER_EXPORT void compute_vertex_normals_tdf(
    const double* verts, int nV, const int* faces, int nF,
    float* nrm_out, int nthreads
) {
    if (nthreads > 0) omp_set_num_threads(nthreads);
    std::vector<Vec3> V(nV), N(nV);
    double cx = 0, cy = 0, cz = 0;
    for (int i = 0; i < nV; ++i) {
        V[i] = {verts[3*i], verts[3*i+1], verts[3*i+2]};
        cx += V[i].x; cy += V[i].y; cz += V[i].z;
    }
    cx /= nV; cy /= nV; cz /= nV;
    std::vector<int> vf_start(nV + 1, 0);
    for (int f = 0; f < nF; ++f) {
        vf_start[faces[3*f] + 1]++; vf_start[faces[3*f+1] + 1]++; vf_start[faces[3*f+2] + 1]++;
    }
    for (int i = 0; i < nV; ++i) vf_start[i + 1] += vf_start[i];
    std::vector<int> vf_idx(vf_start[nV]), cur = vf_start;
    for (int f = 0; f < nF; ++f) {
        vf_idx[cur[faces[3*f]]++] = f; vf_idx[cur[faces[3*f+1]]++] = f; vf_idx[cur[faces[3*f+2]]++] = f;
    }
    std::vector<Vec3> fn(nF);
    #pragma omp parallel for schedule(static)
    for (int f = 0; f < nF; ++f) {
        int a = faces[3*f], b = faces[3*f+1], c = faces[3*f+2];
        fn[f] = cross(V[b] - V[a], V[c] - V[a]);
    }
    double out_dot = 0.0;
    #pragma omp parallel for reduction(+:out_dot) schedule(static)
    for (int i = 0; i < nV; ++i) {
        Vec3 acc{0, 0, 0};
        for (int k = vf_start[i]; k < vf_start[i + 1]; ++k) acc += fn[vf_idx[k]];
        double l = norm(acc);
        if (l > 1e-12) acc = acc * (1.0 / l);
        N[i] = acc;
        out_dot += (V[i].x - cx) * acc.x + (V[i].y - cy) * acc.y + (V[i].z - cz) * acc.z;
    }
    const float sgn = (out_dot >= 0.0) ? 1.0f : -1.0f;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nV; ++i) {
        nrm_out[3*i] = (float)N[i].x * sgn;
        nrm_out[3*i+1] = (float)N[i].y * sgn;
        nrm_out[3*i+2] = (float)N[i].z * sgn;
    }
}


static double vec_percentile(std::vector<double>& v, double q) {
    if (v.empty()) return 0.0;
    double pos = std::clamp(q, 0.0, 1.0) * (double)(v.size() - 1);
    size_t lo = (size_t)std::floor(pos), hi = std::min(v.size() - 1, lo + 1);
    double frac = pos - (double)lo;
    std::nth_element(v.begin(), v.begin() + lo, v.end());
    double vlo = v[lo];
    if (hi == lo) return vlo;
    double vhi = *std::min_element(v.begin() + hi, v.end());
    return vlo + frac * (vhi - vlo);
}

static bool chol_inv_small(const double cov[64], int D, double Linv[64], double& logdet_L) {
    double L[64] = {0};
    logdet_L = 0.0;
    for (int i = 0; i < D; ++i) {
        for (int j = 0; j <= i; ++j) {
            double s = cov[i * D + j];
            for (int k = 0; k < j; ++k) s -= L[i * D + k] * L[j * D + k];
            if (i == j) {
                if (s <= 1e-12) s = 1e-12;
                L[i * D + i] = std::sqrt(s);
                logdet_L += std::log(L[i * D + i]);
            } else {
                L[i * D + j] = s / L[j * D + j];
            }
        }
    }
    std::memset(Linv, 0, 64 * sizeof(double));
    for (int j = 0; j < D; ++j) {
        Linv[j * D + j] = 1.0 / L[j * D + j];
        for (int i = j + 1; i < D; ++i) {
            double s = 0.0;
            for (int k = j; k < i; ++k) s += L[i * D + k] * Linv[k * D + j];
            Linv[i * D + j] = -s / L[i * D + i];
        }
    }
    return true;
}

// Exact OpenMP C++ 2-Component Full-Covariance Parenchyma GMM + MRF Smoother (~15ms vs 1200ms sklearn!)
DDSURFER_EXPORT int fast_gmm_2c_tdf(
    const float* const* feat_ptrs, int D, int nx, int ny, int nz,
    const uint8_t* mask_u8,
    const int* sub_idx, int n_sub,
    const float* csf_prob, double mrf_sigma,
    float* pwm_out, float* pgm_out, float* pcsf_out,
    int nthreads
) {
    if (D <= 0 || D > 8 || n_sub < 16) return -1;
    if (nthreads > 0) omp_set_num_threads(nthreads);
    const size_t tot = (size_t)nx * ny * nz;

    std::vector<double> Xz((size_t)n_sub * D);
    double med[8] = {0}, scale[8] = {1,1,1,1,1,1,1,1};
    {
        std::vector<double> col(n_sub);
        for (int d = 0; d < D; ++d) {
            const float* fp = feat_ptrs[d];
            for (int r = 0; r < n_sub; ++r) {
                float v = fp[sub_idx[r]];
                col[r] = std::isfinite(v) ? (double)v : 0.0;
            }
            std::vector<double> tmp = col;
            double q25 = vec_percentile(tmp, 0.25);
            double q50 = vec_percentile(tmp, 0.50);
            double q75 = vec_percentile(tmp, 0.75);
            med[d] = q50;
            scale[d] = (q75 - q25 + 1e-6) / 1.349;
            double inv_s = 1.0 / scale[d];
            for (int r = 0; r < n_sub; ++r)
                Xz[(size_t)r * D + d] = std::clamp((col[r] - q50) * inv_s, -4.0, 4.0);
        }
    }

    // Deterministic 65th / 35th percentile channel-0 mean initialization (matches parenchyma_gmm_wm_gm)
    std::vector<double> c0(n_sub);
    for (int r = 0; r < n_sub; ++r) c0[r] = Xz[(size_t)r * D];
    std::vector<double> c0_copy = c0;
    double p35 = vec_percentile(c0_copy, 0.35);
    double p65 = vec_percentile(c0_copy, 0.65);

    double m0[8] = {0}, m1[8] = {0};
    for (int d = 0; d < D; ++d) {
        std::vector<double> hi, lo;
        hi.reserve(n_sub / 2); lo.reserve(n_sub / 2);
        for (int r = 0; r < n_sub; ++r) {
            if (c0[r] >= p65) hi.push_back(Xz[(size_t)r * D + d]);
            if (c0[r] <= p35) lo.push_back(Xz[(size_t)r * D + d]);
        }
        m0[d] = vec_percentile(hi, 0.50);
        m1[d] = vec_percentile(lo, 0.50);
    }

    std::vector<double> r0(n_sub);
    double nk0 = 0.0, nk1 = 0.0;
    for (int r = 0; r < n_sub; ++r) {
        double d0 = 0.0, d1 = 0.0;
        const double* xr = Xz.data() + (size_t)r * D;
        for (int d = 0; d < D; ++d) {
            double a = xr[d] - m0[d], b = xr[d] - m1[d];
            d0 += a * a; d1 += b * b;
        }
        r0[r] = (d0 <= d1) ? 1.0 : 0.0;
        nk0 += r0[r]; nk1 += (1.0 - r0[r]);
    }
    nk0 = std::max(nk0, 1.0); nk1 = std::max(nk1, 1.0);
    double w0 = nk0 / (double)n_sub, w1 = nk1 / (double)n_sub;
    double cov0[64] = {0}, cov1[64] = {0};
    auto update_covs = [&]() {
        std::memset(cov0, 0, 64 * sizeof(double));
        std::memset(cov1, 0, 64 * sizeof(double));
        for (int r = 0; r < n_sub; ++r) {
            const double* xr = Xz.data() + (size_t)r * D;
            double g0 = r0[r], g1 = 1.0 - g0;
            double df0[8], df1[8];
            for (int d = 0; d < D; ++d) { df0[d] = xr[d] - m0[d]; df1[d] = xr[d] - m1[d]; }
            for (int i = 0; i < D; ++i) {
                double a0 = g0 * df0[i], a1 = g1 * df1[i];
                for (int j = 0; j <= i; ++j) {
                    cov0[i * D + j] += a0 * df0[j];
                    cov1[i * D + j] += a1 * df1[j];
                }
            }
        }
        for (int i = 0; i < D; ++i) {
            for (int j = 0; j <= i; ++j) {
                cov0[i * D + j] /= nk0; cov0[j * D + i] = cov0[i * D + j];
                cov1[i * D + j] /= nk1; cov1[j * D + i] = cov1[i * D + j];
            }
            cov0[i * D + i] += 1e-3;
            cov1[i * D + i] += 1e-3;
        }
    };
    update_covs();

    double Linv0[64], Linv1[64], ld0 = 0.0, ld1 = 0.0;
    double prev_ll = -1e30;
    for (int it = 0; it < 100; ++it) {
        chol_inv_small(cov0, D, Linv0, ld0);
        chol_inv_small(cov1, D, Linv1, ld1);
        double c_w0 = std::log(w0 + 1e-12) - ld0;
        double c_w1 = std::log(w1 + 1e-12) - ld1;
        double ll_sum = 0.0;
        nk0 = 0.0; nk1 = 0.0;
        double nm0[8] = {0}, nm1[8] = {0};
        for (int r = 0; r < n_sub; ++r) {
            const double* xr = Xz.data() + (size_t)r * D;
            double df0[8], df1[8];
            for (int d = 0; d < D; ++d) { df0[d] = xr[d] - m0[d]; df1[d] = xr[d] - m1[d]; }
            double q0 = 0.0, q1 = 0.0;
            for (int i = 0; i < D; ++i) {
                double y0 = 0.0, y1 = 0.0;
                const double* L0r = Linv0 + i * D;
                const double* L1r = Linv1 + i * D;
                for (int j = 0; j <= i; ++j) { y0 += L0r[j] * df0[j]; y1 += L1r[j] * df1[j]; }
                q0 += y0 * y0; q1 += y1 * y1;
            }
            double lp0 = c_w0 - 0.5 * q0;
            double lp1 = c_w1 - 0.5 * q1;
            double mlp = std::max(lp0, lp1);
            double lse = mlp + std::log(std::exp(lp0 - mlp) + std::exp(lp1 - mlp));
            ll_sum += lse;
            double g0 = std::exp(lp0 - lse);
            double g1 = 1.0 - g0;
            r0[r] = g0;
            nk0 += g0; nk1 += g1;
            for (int d = 0; d < D; ++d) { nm0[d] += g0 * xr[d]; nm1[d] += g1 * xr[d]; }
        }
        double ll = ll_sum / (double)n_sub;
        nk0 = std::max(nk0, 1e-6); nk1 = std::max(nk1, 1e-6);
        w0 = nk0 / (double)n_sub; w1 = nk1 / (double)n_sub;
        for (int d = 0; d < D; ++d) { m0[d] = nm0[d] / nk0; m1[d] = nm1[d] / nk1; }
        update_covs();
        if (std::fabs(ll - prev_ll) < 1e-3) break;
        prev_ll = ll;
    }

    chol_inv_small(cov0, D, Linv0, ld0);
    chol_inv_small(cov1, D, Linv1, ld1);
    const bool wm_is_0 = (m0[0] >= m1[0]);
    float mu_w[8], mu_g[8], iw[64], ig[64], med_f[8], inv_sc_f[8];
    for (int d = 0; d < D; ++d) {
        mu_w[d] = (float)(wm_is_0 ? m0[d] : m1[d]);
        mu_g[d] = (float)(wm_is_0 ? m1[d] : m0[d]);
        med_f[d] = (float)med[d];
        inv_sc_f[d] = (float)(1.0 / scale[d]);
    }
    for (int k = 0; k < D * D; ++k) {
        iw[k] = (float)(wm_is_0 ? Linv0[k] : Linv1[k]);
        ig[k] = (float)(wm_is_0 ? Linv1[k] : Linv0[k]);
    }
    const float c_diff = (float)((wm_is_0 ? (std::log(w0 + 1e-12) - ld0) : (std::log(w1 + 1e-12) - ld1))
                               - (wm_is_0 ? (std::log(w1 + 1e-12) - ld1) : (std::log(w0 + 1e-12) - ld0)));

    std::vector<float> pw_raw(tot, 0.0f), pg_raw(tot, 0.0f), tmp(tot);
    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < tot; ++idx) {
        if (!mask_u8[idx]) {
            pcsf_out[idx] = 0.0f;
            continue;
        }
        float xz[8];
        for (int d = 0; d < D; ++d) {
            float v = feat_ptrs[d][idx];
            if (!std::isfinite(v)) v = 0.0f;
            xz[d] = std::clamp((v - med_f[d]) * inv_sc_f[d], -4.0f, 4.0f);
        }
        float dw[8], dg[8];
        for (int d = 0; d < D; ++d) { dw[d] = xz[d] - mu_w[d]; dg[d] = xz[d] - mu_g[d]; }
        float qw = 0.0f, qg = 0.0f;
        for (int i = 0; i < D; ++i) {
            float yw = 0.0f, yg = 0.0f;
            const float* iwr = iw + i * D;
            const float* igr = ig + i * D;
            for (int j = 0; j <= i; ++j) { yw += iwr[j] * dw[j]; yg += igr[j] * dg[j]; }
            qw += yw * yw; qg += yg * yg;
        }
        float logit = std::clamp(c_diff - 0.5f * (qw - qg), -20.0f, 20.0f);
        float pw = 1.0f / (1.0f + std::exp(-logit));
        float pc = std::clamp(csf_prob[idx], 0.0f, 1.0f);
        float rem = std::max(0.0f, 1.0f - pc);
        pw_raw[idx] = rem * pw;
        pg_raw[idx] = rem * (1.0f - pw);
        pcsf_out[idx] = pc;
    }

    if (mrf_sigma > 1e-6) {
        smooth_scalar_3d(pw_raw.data(), pwm_out, tmp, nx, ny, nz, mrf_sigma);
        smooth_scalar_3d(pg_raw.data(), pgm_out, tmp, nx, ny, nz, mrf_sigma);
    } else {
        std::memcpy(pwm_out, pw_raw.data(), tot * sizeof(float));
        std::memcpy(pgm_out, pg_raw.data(), tot * sizeof(float));
    }

    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < tot; ++idx) {
        if (!mask_u8[idx]) {
            pwm_out[idx] = 0.0f; pgm_out[idx] = 0.0f; pcsf_out[idx] = 0.0f;
        } else {
            float inv_t = 1.0f / (pwm_out[idx] + pgm_out[idx] + pcsf_out[idx] + 1e-6f);
            pwm_out[idx] *= inv_t;
            pgm_out[idx] *= inv_t;
            pcsf_out[idx] *= inv_t;
        }
    }
    return 0;
}

} // extern "C"
