/*  ====================================================================
    2D TM FDTD – Textbook Single-Step Chebyshev (Collocated Grid)
    一次计算直接得到任意时刻 t 的场（无 dt、无累积误差）
    严格符合 Taflove 第3版 2.2.2 节
    ==================================================================== */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <fcntl.h>
#include <complex.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---------------------- 网格参数 ---------------------- */
const int    Nx = 360;                 // 必须和生成初始场时一致！
const int    Ny = 240;
const double Lx = 18.0, Ly = 12.0;
const double h  = Lx / Nx;              // 方形网格
const long long Ncell = (long long)Nx * Ny;
const long long Nvec  = 3 * Ncell;      // Ez + Hx + Hy





void write_fields_to_csv(const double complex *psi_t, double t_final)
{
    const double complex *Ez = psi_t;           // 前 Ncell 个
    const double complex *Hx = psi_t + Ncell;   // 中间 Ncell 个
    const double complex *Hy = psi_t + 2*Ncell; // 最后 Ncell 个
    char name[64];

    const int NNx = Nx;
    const int NNy = Ny;
    

    /* 你可以在这里选择输出实部、虚部还是模值
       这里默认输出实部（最常用） */
    #define OUTPUT_REAL(c)   creal(c)
    // #define OUTPUT_REAL(c)   cabs(c)      // 如果想输出幅度就把上面这行取消注释

    /* 1. E_z_t.csv */
    {
        snprintf(name, sizeof(name), "Ez_t_%.3f.csv", t_final);
        FILE *fp = fopen(name, "w");
        if (!fp) { perror("fopen E_z_t.csv"); exit(1); }

        for (int j = 0; j < NNx; j++) {           // x 方向（行）
            for (int i = 0; i < NNy; i++) {       // y 方向（列）
                long long k = (long long)i + (long long)NNy * j;
                if (i > 0) fprintf(fp, ",");
                fprintf(fp, "%.15e", OUTPUT_REAL(Ez[k]));
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
        printf("-> E_z_%.3f.csv written (%d x %d)\n", t_final, NNx, NNy);
    }

    /* 2. H_x_t.csv */
    {
        snprintf(name, sizeof(name), "Hx_t_%.3f.csv", t_final);
        FILE *fp = fopen(name, "w");
        if (!fp) { perror("fopen H_x_t.csv"); exit(1); }

        for (int j = 0; j < NNx; j++) {           // x 方向（行）
            for (int i = 0; i < NNy; i++) {       // y 方向（列）
                long long k = (long long)i + (long long)NNy * j;
                if (i > 0) fprintf(fp, ",");
                fprintf(fp, "%.15e", OUTPUT_REAL(Hx[k]));
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
        printf("-> H_x_%.3f.csv written (%d x %d)\n", t_final, NNx, NNy);
    }

    /* 3. H_y_t.csv */
    {
        snprintf(name, sizeof(name), "Hy_t_%.3f.csv", t_final);
        FILE *fp = fopen(name, "w");
        if (!fp) { perror("fopen H_y_t.csv"); exit(1); }

        for (int j = 0; j < NNx; j++) {           // x 方向（行）
            for (int i = 0; i < NNy; i++) {       // y 方向（列）
                long long k = (long long)i + (long long)NNy * j;
                if (i > 0) fprintf(fp, ",");
                fprintf(fp, "%.15e", OUTPUT_REAL(Hy[k]));
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
        printf("-> H_y_%.3f.csv written (%d x %d)\n", t_final, NNx, NNy);
    }

    #undef OUTPUT_REAL
}


/* ---------------------- 高精度贝塞尔函数（作业版） ---------------------- */
int BesselJ(double x, int m_max, double Jn[])
{
    if (x < 0 || m_max < 0) return -1;
    if (x == 0.0) {
        Jn[0] = 1.0;
        for (int n = 1; n <= m_max; n++) Jn[n] = 0.0;
        return 0;
    }

    const double eps = 1e-16;
    int x_int = (int)ceil(x);
    int M = 10 * x_int + m_max + 100;
    if (M < 2 * m_max) M = 2 * m_max;

    char tmpname[] = "/tmp/bessel_tmp_XXXXXX";
    int fd = mkstemp(tmpname);
    if (fd == -1) return -2;

    FILE *fp = fdopen(fd, "wb+");
    if (!fp) {
        close(fd);
        unlink(tmpname);
        return -2;
    }

    long long file_size = (long long)(M + 2) * sizeof(double);
    if (ftruncate(fd, file_size) == -1) {
        fclose(fp);
        unlink(tmpname);
        return -3;
    }

    double *f_low = NULL;   // 关键：显式初始化为 NULL

    // 安全的读写宏
    #define WRITE_F(idx, val) do { \
        double _v = (val); \
        if (fseek(fp, (long)(idx) * sizeof(double), SEEK_SET) != 0 || \
            fwrite(&_v, sizeof(double), 1, fp) != 1) \
            goto cleanup; \
    } while(0)

    #define READ_F(idx, var) do { \
        if (fseek(fp, (long)(idx) * sizeof(double), SEEK_SET) != 0 || \
            fread(&(var), sizeof(double), 1, fp) != 1) \
            goto cleanup; \
    } while(0)

    // 初始化边界
    WRITE_F(M,     0.0);
    WRITE_F(M - 1, 1.0);

    double fn, fn_plus_1, fn_minus_1;

    for (int n = M - 1; n >= 1; n--) {
        READ_F(n,     fn);
        READ_F(n + 1, fn_plus_1);

        fn_minus_1 = (2.0 * n / x) * fn - fn_plus_1;

        if (fabs(fn_minus_1) > 1.0 / eps) {
            double scale = eps;
            for (int k = n - 1; k <= M; k++) {
                double val;
                READ_F(k, val);
                val *= scale;
                WRITE_F(k, val);
            }
            fn_minus_1 *= scale;
        }

        WRITE_F(n - 1, fn_minus_1);
    }

    // 分配低阶数组
    f_low = calloc(m_max + 1, sizeof(double));
    if (!f_low) goto cleanup;   // 失败直接跳到清理

    // 读取需要的 J_n
    for (int n = 0; n <= m_max; n++) {
        READ_F(n, f_low[n]);
    }

    // 归一化
    double sum = f_low[0] * f_low[0];
    for (int n = 1; n <= m_max; n++) {
        sum += 2.0 * f_low[n] * f_low[n];
    }
    double scale = 1.0 / sqrt(sum);

    for (int n = 0; n <= m_max; n++) {
        Jn[n] = scale * f_low[n];
        if (fabs(Jn[n]) < eps && n > (int)(x + 30.0))
            Jn[n] = 0.0;
    }

    free(f_low);
    fclose(fp);
    unlink(tmpname);
    return 0;

    cleanup:
    if (f_low) free(f_low);   // 现在 f_low 一定是指向有效内存或 NULL
    fclose(fp);
    unlink(tmpname);
    return -10;
}


int find_min_n_double(double z, double kappa) {
    if (kappa <= 0.0) return 0;                    // 无效输入
    if (kappa >= 1.0) return 0;                    // 连 n=0 都满足

    double abs_z = fabs(z);
    double term = 1.0;                             // n=0 时的值：|z|^0 / (2^0 * 0!) = 1
    int n = 0;

    // 如果 n=0 就满足，直接返回
    if (term <= kappa) return 0;

    // 逐项计算下一项：term_{n+1} = term_n * |z| / (2 * (n+1))
    while (term > kappa) {
        n++;
        term *= abs_z / (2.0 * n);                  // 关键：边乘边除，避免溢出

        // 防止无限循环（虽然理论上不会，但数值下溢后 term 会变成 0）
        if (term == 0.0) break;

        // 可选：设置一个合理上限，防止极端情况死循环
        if (n > 10000) break;
    }

    return n;
}


/* ---------------------- H 算符（同位网格，反对称） ---------------------- */
void apply_H(const double complex *in, double complex *out, void *userdata)
{
    (void)userdata;
    const double inv_2h = 1.0 / (2.0 * h);

    const double complex *Ez = in;
    const double complex *Hx = in + Ncell;
    const double complex *Hy = in + 2*Ncell;

    double complex *dEz = out;
    double complex *dHx = out + Ncell;
    double complex *dHy = out + 2*Ncell;

    /*
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            long long k = i + Nx*j;

            int ip = (i+1 < Nx) ? i+1 : Nx-1;
            int im = (i-1 >= 0) ? i-1 : 0;
            int jp = (j+1 < Ny) ? j+1 : Ny-1;
            int jm = (j-1 >= 0) ? j-1 : 0;

            double complex dHy_dx = (Hy[ip + Nx*j] - Hy[im + Nx*j]) * inv_2h;
            double complex dHx_dy = (Hx[i + Nx*jp] - Hx[i + Nx*jm]) * inv_2h;
            dEz[k] = dHy_dx - dHx_dy;

            double complex dEz_dy = (Ez[i + Nx*jp] - Ez[i + Nx*jm]) * inv_2h;
            dHx[k] = -dEz_dy;

            double complex dEz_dx = (Ez[ip + Nx*j] - Ez[im + Nx*j]) * inv_2h;
            dHy[k] = dEz_dx;
        }
    }
       
    */

    /*
    for (int i = 0; i < Ny; i++) {
        for (int j = 0; j < Nx; j++) {
            long long k = i + Ny*j;

           int ip = (i+1 < Ny) ? i+1 : Ny-1;
            int im = (i-1 >= 0) ? i-1 : 0;
            int jp = (j+1 < Nx) ? j+1 : Nx-1;
            int jm = (j-1 >= 0) ? j-1 : 0;

            double complex dHx_dy = (Hx[ip + Ny*j] - Hx[im + Ny*j]) * inv_2h;
            double complex dHy_dx = (Hy[i + Ny*jp] - Hy[i + Ny*jm]) * inv_2h;
            //dEz[k] = dHx_dy - dHy_dx;
            dEz[k] = -1 * (dHx_dy - dHy_dx);

            double complex dEz_dx = (Ez[i + Ny*jp] - Ez[i + Ny*jm]) * inv_2h;
            //dHy[k] = -dEz_dx;
            dHy[k] = dEz_dx;

            double complex dEz_dy = (Ez[ip + Ny*j] - Ez[im + Ny*j]) * inv_2h;
            //dHx[k] = dEz_dy;
            
            //!!//dHx[k] = -1 * dEz_dy;
            dHx[k] = -1 * dEz_dy;
        }
    }
    */

    for (int i = 0; i < Ny; i++) {
        for (int j = 0; j < Nx; j++) {
            long long k = i + Ny*j;

           int ip = (i+1 < Ny) ? i+1 : Ny-1;
            int im = (i-1 >= 0) ? i-1 : 0;
            int jp = (j+1 < Nx) ? j+1 : Nx-1;
            int jm = (j-1 >= 0) ? j-1 : 0;

            double complex dHx_dy = (Hx[ip + Ny*j] - Hx[im + Ny*j]) * inv_2h;
            double complex dHy_dx = (Hy[i + Ny*jp] - Hy[i + Ny*jm]) * inv_2h;
            //dEz[k] = dHx_dy - dHy_dx;
            dEz[k] = -1*(dHx_dy - dHy_dx);

            double complex dEz_dx = (Ez[i + Ny*jp] - Ez[i + Ny*jm]) * inv_2h;
            //dHy[k] = -dEz_dx;
            dHy[k] =  dEz_dx;

            double complex dEz_dy = (Ez[ip + Ny*j] - Ez[im + Ny*j]) * inv_2h;
            //dHx[k] = dEz_dy;
            
            //!!//dHx[k] = -1 * dEz_dy;
            dHx[k] = - dEz_dy;
        }
    }



}

double estimate_A_norm_inf(void *userdata)
{
    (void)userdata;
    //return 4.0 / h;   // 同位网格中心差分最大特征值
   return 4.0 / h;
}

/* ---------------------- CSV 输出（只输出 Ez 部分） ---------------------- */
void write_Ez_csv(const double complex *psi, const char *filename)
{
    FILE *fp = fopen(filename, "w");
    if (!fp) { perror("fopen csv"); exit(1); }
    for (long long k = 0; k < Ncell; k++)
        fprintf(fp, "%.15e\n", creal(psi[k]));  // 只输出 Ez
    fclose(fp);
}

void write_Jn_csv(const double *J, const char *filename,int n0)
{
    FILE *fp = fopen(filename, "w");
    if (!fp) { perror("fopen csv"); exit(1); }
    for (long long k = 0; k < n0; k++)
        fprintf(fp, "%.15e\n", J[k]);  // 只输出 Ez
    fclose(fp);
}
/* ---------------------- 单步 Chebyshev ---------------------- */
void chebyshev_single_step(double t_final, double kappa)
{
    double complex *T_prev = calloc(Nvec, sizeof(double complex));
    double complex *T_curr = calloc(Nvec, sizeof(double complex));
    double complex *temp   = calloc(Nvec, sizeof(double complex));
    double complex *psi_t  = calloc(Nvec, sizeof(double complex));
    //double *psi_re  = calloc(Nvec, sizeof(double));

    /* ---------- 正确读取实数初始场 ---------- */
    printf("Loading real-valued initial fields (double)...\n");
    FILE *fEz = fopen("Ez_0_matlab.bin", "rb");
    FILE *fHx = fopen("Hx_0_matlab.bin", "rb");
    FILE *fHy = fopen("Hy_0_matlab.bin", "rb");
    if (!fEz || !fHx || !fHy) {
        perror("Open initial field file failed");
        fprintf(stderr, "Please make sure Ez_0_matlab.bin, Hx_0_matlab.bin, Hy_0_matlab.bin exist!\n");
        exit(1);
    }

    double *buf = malloc(Ncell * sizeof(double));
      // 替换原来的三行 fread
    size_t read1 = fread(buf, sizeof(double), Ncell, fEz);
    // 赋值到 T_curr（T0）
    for (long long k1 = 0; k1 < Nx; k1 ++) {
        for(long long k2 = 0; k2 < Ny; k2 ++){
        T_curr[k2 + k1 * Ny]  = -1 * buf[k1 + k2 * Nx] + 0.0*I;  // Ez
        //psi_re[k]           = buf[k];
        }
    }
    
    size_t read2 = fread(buf, sizeof(double), Ncell, fHx);
    // 赋值到 T_curr（T0）
    /*
    for (long long k = 0; k < Ncell; k++) {
        T_curr[k + Ncell]   = buf[k] + 0.0*I;  // Hx（你原来的 buf 重复用了，但逻辑错，但不影响崩溃）
        //T_curr[k + Ncell]   = 0.0 + 0.0*I; 
        //psi_re[k + Ncell]   = buf[k];
    }
    */

    for (long long k1 = 0; k1 < Nx; k1 ++) {
        for(long long k2 = 0; k2 < Ny; k2 ++){
        T_curr[k2 + k1 * Ny + Ncell]  = -1 * buf[k1 + k2 * Nx] + 0.0*I;  // Ez
        //psi_re[k]           = buf[k];
        }
    }
    
    size_t read3 = fread(buf, sizeof(double), Ncell, fHy);
    // 赋值到 T_curr（T0）
    /*
    for (long long k = 0; k < Ncell; k++) {
        T_curr[k + 2*Ncell] = buf[k] + 0.0*I;  // Hy
        //T_curr[k + 2*Ncell] = 0.0 + 0.0*I;
        //psi_re[k + 2*Ncell] = buf[k];
    }
    */
    for (long long k1 = 0; k1 < Nx; k1 ++) {
        for(long long k2 = 0; k2 < Ny; k2 ++){
        T_curr[k2 + k1 * Ny + 2*Ncell]  = -1 * buf[k1 + k2 * Nx] + 0.0*I;  // Ez
        //psi_re[k]           = buf[k];
        }
    }



    if (read1 != Ncell || read2 != Ncell || read3 != Ncell) {
        fprintf(stderr, "Error: Failed to read full data from initial field files!\n");
        fprintf(stderr, "Expected %lld doubles, but got %zu, %zu, %zu\n",
                Ncell, read1, read2, read3);
        free(buf);
        fclose(fEz); fclose(fHx); fclose(fHy);
        exit(1);
    }

    //write_real_to_csv(psi_re, 100);
    //free(psi_re);

    free(buf); fclose(fEz); fclose(fHx); fclose(fHy);
    printf("Initial fields loaded successfully.\n");

    write_fields_to_csv(T_curr, 0);
    
    double E0 = 0.0;
    for (long long k = 0; k < Ncell; k++) {
        double ezf = creal(T_curr[k]);
        double hxf = creal(T_curr[k+Ncell]);
        double hyf = creal(T_curr[k+2*Ncell]);
        E0 += ezf*ezf + hxf*hxf + hyf*hyf;
    }
    E0 *= h*h;

    /* ---------- 计算 z = ||A|| * t ---------- */
    double A_norm = estimate_A_norm_inf(NULL);
    printf("A_norm=%.3f\n",A_norm);
    double z = A_norm * t_final;
    printf("z=%.3f\n",z);
    double B_scale = 1.0 / A_norm;
    double complex iB_2 = 2.0 * B_scale;

    printf("Single-step to t = %.6f, z = ||A||*t = %.6f\n", t_final, z);

    /* ---------- 确定截断阶数 ---------- */
    const int n = find_min_n_double(z, kappa) + 20;
    printf("Using Chebyshev order = %d\n", n);

    double *Jn = calloc(n + 10, sizeof(double));
    BesselJ(z, n+2, Jn);
    printf("Bessel calculation done!\n");

    /* ---------- 初始化 T0 和 T1 ---------- */
    for (long long i = 0; i < Nvec; i++) T_prev[i] = T_curr[i];               // T0
    apply_H(T_curr, temp, NULL);
    for (long long i = 0; i < Nvec; i++) T_curr[i] = B_scale * temp[i];   // T1

    /* ---------- 主递推 + 加权求和 ---------- */
    for (long long i = 0; i < Nvec; i++) psi_t[i] = Jn[0] * T_prev[i];        // J0 T0

    for (int m = 1; m <= n; m++) {
        // 加权当前 T_m
        double coeff = 2.0 * Jn[m];
        for (long long i = 0; i < Nvec; i++) psi_t[i] += coeff * T_curr[i];

        // 递推 T_{m+1} = 2 i B T_m + T_{m-1}
        apply_H(T_curr, temp, NULL);
        for (long long i = 0; i < Nvec; i++)
            temp[i] = iB_2 * temp[i] + T_prev[i];

        double complex *swap = T_prev;
        T_prev = T_curr;
        T_curr = temp;
        temp = swap;

        if (fabs(Jn[m+1]) < kappa) {
            printf("Truncated at n = %d (|J_%d(z)| = %.3e < κ)\n", m, m+1, fabs(Jn[m+1]));
            break;
        }
    }

    /* ---------- 输出最终场 + 能量守恒检查 ---------- */

    
    char name[64];

    /*
    
    snprintf(name, sizeof(name), "Ez_t_%.3f.csv", t_final);
    write_Ez_csv(psi_t, name);
    
    */


    write_fields_to_csv(psi_t, t_final);

    
    snprintf(name, sizeof(name), "J_n.csv");
    write_Jn_csv(Jn, name, n);

    


    // 能量守恒

    double Ef = 0.0;
    for (long long k = 0; k < Ncell; k++) {         
        double ezf = creal(psi_t[k]);
        double hxf = creal(psi_t[k+Ncell]);
        double hyf = creal(psi_t[k+2*Ncell]);
        Ef += ezf*ezf + hxf*hxf + hyf*hyf;
    }
    Ef *= h*h;
    
    
/*
    double E0 = 0.0, Ef = 0.0;
    for (long long k = 0; k < Ncell; k++) {
        double ez0 = creal(T_prev[k]);           // T0 的 Ez
        double ezf = creal(psi_t[k]);
        double hxf = creal(psi_t[k+Ncell]);
        double hyf = creal(psi_t[k+2*Ncell]);
        E0 += ez0*ez0;
        Ef += ezf*ezf + hxf*hxf + hyf*hyf;
    }
    E0 *= h*h; Ef *= h*h;
*/

    printf("Initial energy E0 = %.15e\n", E0);
    printf("Final   energy Ef = %.15e\n", Ef);
    printf("Relative error     = %.6e\n", fabs(Ef - E0)/E0);
    


    free(T_prev); free(T_curr); free(temp); free(psi_t); free(Jn);
    printf("Done! Result saved to %s\n", name);
}

/* ---------------------- 主函数 ---------------------- */
int main()
{
    printf("=== 2D TM Single-Step Chebyshev (Textbook Correct) ===\n");

    double t;
    int ret;

    do {
        printf("t = ");
        fflush(stdout);  // 确保提示立即显示

        ret = scanf("%lf", &t);

        if (ret != 1) {
            // 输入不是数字（例如输入字母、符号等）
            printf("错误：请输入一个有效的数字！\n");
            // 清空输入缓冲区，防止无限循环
            while (getchar() != '\n');
        } else if (t < 0 || t > 20) {
            // 数字有效，但超出范围
            printf("错误：t 必须在 0 到 20 之间！\n");
        } else {
            // 输入正确，跳出循环
            break;
        }
    } while (1);

    printf("输入成功：t = %.6f\n", t);

    chebyshev_single_step(t, 1e-13);   // 直接跳到 t = 10.0（和书上图对比）
    return 0;
}