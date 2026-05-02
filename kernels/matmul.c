void foo(float *a, float *b, float *c, int m, int n, int k) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      c[i*n + j] = 0;
      for (int s = 0; s < k; ++s) {
        c[i*n + j] += a[i*k + s] * b[s*n + j];
      }
    }
  }
}
