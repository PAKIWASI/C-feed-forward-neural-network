

// will i need these (from CPP project)

// (1 x m) * (m x n) => (1 x n)
void vector_xply_matrix(const std::vector<float>& vec,
                        const std::vector<std::vector<float>>& matrix,
                        std::vector<float>& output);


// dL/dW = dz/dW x dL/dz = xT x dL/dz (m x 1) x (1 x n)
// (1 x m)T * (1 x n) => (m x 1) * (1 x n) => (m x n)
void vector_transpose_xply_vector( const std::vector<float>& vec1,
                                   const std::vector<float>& vec2,
                                   std::vector<std::vector<float>>& matrix);

// (1 x n) * (m x n)T => (1 x n) * (n x m) => (1 x m)
void vector_xply_matrix_transpose(
                               const std::vector<float>& vec,
                               const std::vector<std::vector<float>>& matrix,
                               std::vector<float>& output     );
