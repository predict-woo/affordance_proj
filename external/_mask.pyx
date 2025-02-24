cdef np.ndarray[np.double_t, ndim=1] np_poly
n = len(poly)
Rs = RLEs(n)
for i, p in enumerate(poly):
    np_poly = np.array(p, dtype=np.double, order='F')
    rleFrPoly( <RLE*>&Rs._R[i], <const double*> np_poly.data, <unsigned long>(len(np_poly)//2), h, w ) 