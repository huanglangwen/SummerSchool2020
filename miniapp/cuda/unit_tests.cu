#include "linalg.h"
#include "operators.h"

using data::Field;

template <typename F>
bool run_test(F f, const char* name) {
    auto success = f();
    printf("%-25s : ", name);
    if(!success) {
        printf("\033[1;31mfailed\033[0m\n");
        return false;
    }
    printf("\033[1;32mpassed\033[0m\n");
    return true;
} 
template <typename T>
bool check_value(T value, T expected, T tol) {
    if(std::fabs(value-expected)>tol) {
        std::cout << "  expected " << expected << " got " << value << std::endl;
        return false;
    }
    return true;
}

bool test_scaled_diff() {
    auto n = 5;
    Field y(n,1);
    Field l(n,1);
    Field r(n,1);

    for(auto i=0; i<n; ++i) {
        l[i] = 7.0;
        r[i] = 2.0;
    }
    l.update_device();
    r.update_device();

    linalg::ss_scaled_diff(y, 2.0, l, r);
    y.update_host();

    bool status = true;
    for(auto i=0; i<n; ++i) {
        status = status && check_value(y[i], 10.0, 1.e-13);
    }
    return status;
}

bool test_fill() {
    auto n = 5;
    Field x(n,1);

    for(auto i=0; i<n; ++i) {
        x[i] = 3.0;
    }
    x.update_device();

    linalg::ss_fill(x, 2.0);
    x.update_host();

    bool status = true;
    for(auto i=0; i<n; ++i) {
        status = status && check_value(x[i], 2.0, 1.e-13);
    }
    return status;
}

bool test_axpy() {
    auto n = 5;
    Field x(n,1);
    Field y(n,1);

    for(auto i=0; i<n; ++i) {
        x[i] = 3.0;
        y[i] = 5.0;
    }
    x.update_device();
    y.update_device();

    linalg::ss_axpy(y, 0.5, x);
    y.update_host();

    bool status = true;
    for(auto i=0; i<n; ++i) {
        status = status && check_value(y[i], (0.5*3.0 + 5.0), 1.e-13);
    }
    return status;
}

bool test_add_scaled_diff() {
    auto n = 5;
    Field y(n,1);
    Field x(n,1);
    Field l(n,1);
    Field r(n,1);

    for(auto i=0; i<n; ++i) {
        x[i] = 3.0;
        l[i] = 7.0;
        r[i] = 2.0;
    }
    x.update_device();
    l.update_device();
    r.update_device();

    linalg::ss_add_scaled_diff(y, x, 1.5, l, r);
    y.update_host();

    bool status = true;
    for(auto i=0; i<n; ++i) {
        status = status && check_value(y[i], 3. + 1.5 * (7. - 2.), 1.e-13);
    }
    return status;
}

bool test_scale() {
    auto n = 5;
    Field x(n,1);
    Field y(n,1);

    for(auto i=0; i<n; ++i) {
        x[i] = 3.0;
    }
    x.update_device();

    linalg::ss_scale(y, 0.5, x);
    y.update_host();

    bool status = true;
    for(auto i=0; i<n; ++i) {
        status = status && check_value(y[i], 1.5, 1.e-13);
    }
    return status;
}

bool test_lcomb() {
    auto n = 5;
    Field x(n,1);
    Field y(n,1);
    Field z(n,1);

    for(auto i=0; i<n; ++i) {
        x[i] = 3.0;
        z[i] = 7.0;
    }
    x.update_device();
    z.update_device();

    linalg::ss_lcomb(y, 0.5, x, 2.0, z);
    y.update_host();

    bool status = true;
    for(auto i=0; i<n; ++i) {
        status = status && check_value(y[i], (0.5*3. + 2.*7.), 1.e-13);
    }
    return status;
}

bool test_copy() {
    auto n = 5;
    Field x(n,1);
    Field y(n,1);

    for(auto i=0; i<n; ++i) {
        x[i] = 3.0;
    }

    x.update_device();
    linalg::ss_copy(y, x);
    y.update_host();

    bool status = true;
    for(auto i=0; i<n; ++i) {
        status = status && check_value(y[i], x[i], 1.e-13);
    }

    return status;
}

bool test_dot() {
    auto n = 5;
    Field x(n,1);
    Field y(n,1);

    for(auto i=0; i<n; ++i) {
        x[i] = 3.0;
        y[i] = 7.0;
    }
    x.update_device();
    y.update_device();

    auto result = linalg::ss_dot(x, y);

    return check_value(result, n*3.*7., 1.e-13);
}

bool test_norm2() {
    auto n = 5;
    Field x(n,1);

    for(auto i=0; i<n; ++i) {
        x[i] = 2.0;
    }
    x.update_device();

    auto result = linalg::ss_norm2(x);

    return check_value(result, sqrt(2.0 * 2.0 * 5.0), 1.e-13);
}

void diffusion_cpu(const data::Field& U, data::Field& S, data::Discretization options)
{

    using data::bndE;
    using data::bndW;
    using data::bndN;
    using data::bndS;

    using data::x_old;

    double dxs = 1000. * (options.dx * options.dx);
    double alpha = options.alpha;
    int nx = options.nx;
    int ny = options.ny;
    int iend = nx - 1;
    int jend = ny - 1;

    // the interior grid points
#pragma omp parallel for
    for (int j = 1; j < jend; j++) {
        for (int i = 1; i < iend; i++) {
            S(i, j) = -(4. + alpha) * U(i, j)               // central point
                + U(i - 1, j) + U(i + 1, j) // east and west
                + U(i, j - 1) + U(i, j + 1) // north and south
                + alpha * x_old(i, j)
                + dxs * U(i, j) * (1.0 - U(i, j));
        }
    }

    // the east boundary
    {
        int i = nx - 1;
        for (int j = 1; j < jend; j++)
        {
            S(i, j) = -(4. + alpha) * U(i, j)
                + U(i - 1, j) + U(i, j - 1) + U(i, j + 1)
                + alpha * x_old(i, j) + bndE[j]
                + dxs * U(i, j) * (1.0 - U(i, j));
        }
    }

    // the west boundary
    {
        int i = 0;
        for (int j = 1; j < jend; j++)
        {
            S(i, j) = -(4. + alpha) * U(i, j)
                + U(i + 1, j) + U(i, j - 1) + U(i, j + 1)
                + alpha * x_old(i, j) + bndW[j]
                + dxs * U(i, j) * (1.0 - U(i, j));
        }
    }

    // the north boundary (plus NE and NW corners)
    {
        int j = ny - 1;

        {
            int i = 0; // NW corner
            S(i, j) = -(4. + alpha) * U(i, j)
                + U(i + 1, j) + U(i, j - 1)
                + alpha * x_old(i, j) + bndW[j] + bndN[i]
                + dxs * U(i, j) * (1.0 - U(i, j));
        }

        // north boundary
        for (int i = 1; i < iend; i++)
        {
            S(i, j) = -(4. + alpha) * U(i, j)
                + U(i - 1, j) + U(i + 1, j) + U(i, j - 1)
                + alpha * x_old(i, j) + bndN[i]
                + dxs * U(i, j) * (1.0 - U(i, j));
        }

        {
            int i = nx - 1; // NE corner
            S(i, j) = -(4. + alpha) * U(i, j)
                + U(i - 1, j) + U(i, j - 1)
                + alpha * x_old(i, j) + bndE[j] + bndN[i]
                + dxs * U(i, j) * (1.0 - U(i, j));
        }
    }

    // the south boundary
    {
        int j = 0;

        {
            int i = 0; // SW corner
            S(i, j) = -(4. + alpha) * U(i, j)
                + U(i + 1, j) + U(i, j + 1)
                + alpha * x_old(i, j) + bndW[j] + bndS[i]
                + dxs * U(i, j) * (1.0 - U(i, j));
        }

        // south boundary
        for (int i = 1; i < iend; i++)
        {
            S(i, j) = -(4. + alpha) * U(i, j)
                + U(i - 1, j) + U(i + 1, j) + U(i, j + 1)
                + alpha * x_old(i, j) + bndS[i]
                + dxs * U(i, j) * (1.0 - U(i, j));
        }

        {
            int i = nx - 1; // SE corner
            S(i, j) = -(4. + alpha) * U(i, j)
                + U(i - 1, j) + U(i, j + 1)
                + alpha * x_old(i, j) + bndE[j] + bndS[i]
                + dxs * U(i, j) * (1.0 - U(i, j));
        }
    }

}

bool test_diffusion() {
    using data::options;
    using namespace data;
    auto nx = 128;
    auto ny = 128;
    auto t = 0.01;
    options.nx = nx;
    options.ny = ny;
    options.N = nx * ny;
    options.nt = 100;
    options.dt = t / options.nt;
    options.dx = 1. / (options.nx - 1);
    options.alpha = (options.dx * options.dx) / (1. * options.dt);
    Field x_new(nx, ny), x_new2(nx, ny);
    Field b(nx, ny), b2(nx,ny);

    x_new.init(nx, ny);
    x_old.init(nx, ny);
    bndN.init(nx, 1);
    bndS.init(nx, 1);
    bndE.init(ny, 1);
    bndW.init(ny, 1);

    linalg::ss_fill(bndN, 0.);
    linalg::ss_fill(bndS, 0.);
    linalg::ss_fill(bndE, 0.);
    linalg::ss_fill(bndW, 0.);

    linalg::ss_fill(x_new, 0.); x_new.update_host();
    linalg::ss_fill(b, 0.); 
    linalg::ss_fill(b2, 0.); b2.update_host();
    double xc = 1.0 / 4.0;
    double yc = (ny - 1) * options.dx / 4;
    double radius = fmin(xc, yc) / 2.0;
    for (int j = 0; j < ny; j++)
    {
        double y = (j - 1) * options.dx;
        for (int i = 0; i < nx; i++)
        {
            double x = (i - 1) * options.dx;
            if ((x - xc) * (x - xc) + (y - yc) * (y - yc) < radius * radius)
                x_new[i + nx * j] = 0.1;
        }
    }
    x_new.update_device();
    linalg::ss_copy(x_new2, x_new);
    x_new2.update_host();
    operators::diffusion(x_new, b); cudaDeviceSynchronize();
    b.update_host();
    diffusion_cpu(x_new2, b2, options);

    double (*cmp1)[128][128], (*cmp2)[128][128];
    cmp1 = (decltype(cmp1))(b.host_data());
    cmp2 = (decltype(cmp2))(b2.host_data());
    
    bool status = true;
    for (auto i = 0; i < nx; ++i) {
        for (auto j = 0; j < ny; ++j) {
            status = status && check_value(*cmp1[i][j], *cmp2[i][j], 1.e-13);
        }   
    }

    return status;
}

////////////////////////////////////////////////////////////////////////////////
// main
////////////////////////////////////////////////////////////////////////////////
int main(void) {
    run_test(test_dot,          "ss_dot");
    run_test(test_norm2,        "ss_norm2");
    run_test(test_scaled_diff,  "ss_scaled_diff");
    run_test(test_fill,         "ss_fill");
    run_test(test_axpy,         "ss_axpy");
    run_test(test_add_scaled_diff, "ss_add_scaled_diff");
    run_test(test_scale,        "ss_scale");
    run_test(test_lcomb,        "ss_lcomb");
    run_test(test_copy,         "ss_copy");
    run_test(test_diffusion,    "diffusion");
}

