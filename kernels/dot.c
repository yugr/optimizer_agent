double dot(double *a, double *b, int n) {
  double ans = 0;
  for (int i = 0; i < n; ++i) {
    ans += a[i] * b[i];
  }
  return ans;
}
