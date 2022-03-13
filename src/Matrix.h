#ifndef MATRIX_MATRIX_H
#define MATRIX_MATRIX_H
#include <cstring>
#include <ostream>
#include <complex>
#include <iomanip>
#define MAXROWS  10
#define MAXCOLS  10
using namespace std;

struct Matrix_Shape_Not_Match_Exception : public std::exception {
    char will_return[300] = "\0";
    explicit Matrix_Shape_Not_Match_Exception(const char* message, const char* file_name, int32_t Line)
    {
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
        sprintf_s(will_return, "%s,Shape not Match in %d line , %s file", message, Line, file_name);
#else
        sprintf(will_return, "%s, in %d line", message, Line, file_name);
#endif
    }
    [[nodiscard]] const char* what() const noexcept override {
        return will_return;
    }
};
template<class T>
class Matrix {
public:
    T** data;
    int row{};
    int col{};

public:
    Matrix();
    Matrix(int row, int col);
    Matrix(int row, int col, T* array);
    void setMatrix(const T*);
    //friend std::ostream& operator<< (std::ostream& os,const Matrix<T> &matrix);
    Matrix<T> operator +(const Matrix<T>& matrix2)const;
    Matrix<T> operator -(const Matrix<T>& matrix2)const;
    Matrix<T> operator *(T other)const;
    Matrix<T> operator /(T other)const;
    bool operator==(Matrix<T> matrix)const;
    Matrix<T> element_wise_mul(Matrix<T>matrix2)const;//???????????
    Matrix<T> transposition()const;
    T trace()const;
    T max(int, int)const;
    T min(int, int)const;
    T determinant()const;
    T subDeterminant(int)const;
    T sum(int, int)const;
    T average(int, int)const;
    Matrix<T> reshape(int rowIn, int colIn) const;
    Matrix<T> conjugation() const;
    T dot(Matrix<T> matrix2) const;
    Matrix<T> operator*(Matrix<T> matrix2) const;
    Matrix<T> cross(Matrix<T> matrix2)const;
    Matrix<T> getAStart(int)const;
    Matrix<T> inverse(int) const;
    Matrix<T> slicing(int x, int y, int rowNum, int colNum) const;
    Matrix<T> convolution(int stride, int padding, Matrix<T> kernel) const;
    cv::Mat transfer()const;
};
template<class T>
Matrix<T>::Matrix() {
    row = MAXROWS;
    col = MAXCOLS;
}

template<class T>
Matrix<T>::Matrix(int row, int col) {
    if (row <= 0 || col <= 0) {
        throw Matrix_Shape_Not_Match_Exception(
            "Invalid row or column!", __FILE__, __LINE__);
    }
    this->row = row;
    this->col = col;
    data = new T * [row];
    for (int i = 0; i < row; i++) {
        data[i] = new T[col];
        //memset(data[i],0,col*sizeof(data));
    }
}

template<class T>
Matrix<T>::Matrix(int row, int col, T* array) {
    this->row = row;
    this->col = col;
    data = new T * [row];
    for (int i = 0; i < row; i++) {
        data[i] = new T[col];
    }
    this->setMatrix(array);
}
template<class T>
void Matrix<T>::setMatrix(const T* array) {
    for (int i = 0, temp = 0; i < row; i++) {
        for (int j = 0; j < col; j++, temp++) {
            data[i][j] = array[temp];
        }
    }
}

template<class T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& matrix) {//????
    std::cout.setf(std::ios_base::floatfield, std::ios_base::fixed);
    std::cout.precision(2);
    for (int i = 0; i < matrix.row; i++) {
        for (int j = 0; j < matrix.col; j++) {
            os << setw(5) << std::left << matrix.data[i][j] << " ";
        }
        os << std::endl;
    }
    return os;
}

template<typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T>& matrix2)const {
    Matrix<T> result(row, col);
    if (this->row == matrix2.row && this->col == matrix2.col) {
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                result.data[i][j] = data[i][j] + matrix2.data[i][j];
            }
        }
    }
    else {
        throw Matrix_Shape_Not_Match_Exception(
            "shape not match,matrix can not be added!", __FILE__, __LINE__);
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::operator-(const Matrix<T>& matrix2) const {
    Matrix<T>result(row, col);
    if (row == matrix2.row && col == matrix2.col) {
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                result.data[i][j] = data[i][j] - matrix2.data[i][j];
            }
        }
    }
    else {
        throw Matrix_Shape_Not_Match_Exception(
            "shape not match,matrix cannot subtracted!", __FILE__, __LINE__);
    }
    return result;
}
template<class T>
Matrix<T> Matrix<T>::operator*(T other)const {
    Matrix<T>result(row, col);
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            result.data[i][j] = data[i][j] * other;
        }
    }
    return result;
}
template<class T>
Matrix<T> Matrix<T>::operator/(T other)const {
    Matrix<T>result(row, col);
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            result.data[i][j] = data[i][j] / other;
        }
    }
    return result;
}

template<class T>
Matrix<T> Matrix<T>::transposition() const {
    Matrix<T> result(col, row);
    for (int i = 0; i < col; i++) {//in th order of the result (after transposition)
        for (int j = 0; j < row; j++) {
            result.data[i][j] = data[j][i];
        }
    }
    return result;
}

template<class T>
Matrix<T> Matrix<T>::element_wise_mul(Matrix<T> matrix2) const {
    if (this->row != matrix2.row || this->col != matrix2.col) {
        throw Matrix_Shape_Not_Match_Exception(
            "shape not match,can not element_wise_mul!", __FILE__, __LINE__);
    }
    Matrix<T>result(row, col);
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            result.data[i][j] = data[i][j] * matrix2.data[i][j];
        }
    }
    return result;
}

template<class T>
T Matrix<T>::trace() const {
    T result = 0;
    if (row == col) {
        for (int i = 0; i < row; i++) {
            result += data[i][i];
        }
    }
    else {
        throw Matrix_Shape_Not_Match_Exception(
            "The matrix is not square,cannot calculate trace!", __FILE__, __LINE__);
    }
    return result;
}

template<class T>
T Matrix<T>::determinant() const {
    if (row != col) {
        throw Matrix_Shape_Not_Match_Exception(
            "The matrix is not square,cannot calculate determinant!", __FILE__, __LINE__);
    }
    else {
        return this->subDeterminant(row);
    }

}


template<class T>
T Matrix<T>::subDeterminant(int dimension) const {
    T result = 0;
    if (dimension == 1) {
        return data[0][0];
    }
    else {
        // may brute_force by <algebraic cofactor> // need to copy every submatrix
        for (int k = 0, sign = 1; k < dimension; k++, sign *= -1) {
            // ?????????????????
            Matrix<T> sub(dimension - 1, dimension - 1);
            // ?????????????
            for (int i = 0; i < dimension - 1; i++)
                for (int j = 0; j < dimension - 1; j++)
                    sub.data[i][j] = data[i + 1][j < k ? j : j + 1];

            // ???????????????????????????????0?????????????????
            if (data[0][k])
                result += sign * data[0][k] * sub.subDeterminant(dimension - 1);
        }

    }
    return result;
}
template<class T>
T Matrix<T>::max(int rowIn, int colIn) const {
    if (row < -1 || row > this->row || row == 0) {
        throw std::invalid_argument("The input row out of matrix row range!");
    }
    if (colIn <-1 || colIn > this->col || col == 0) {
        throw std::invalid_argument("The input row out of matrix col range!");
    }
    if (rowIn != -1 && colIn != -1) {
        return data[rowIn - 1][colIn - 1];
    }
    else if (rowIn == -1 && colIn != -1) {
        T result = data[0][colIn - 1];
        for (int i = 1; i < row; i++) {
            result = std::max(data[i][colIn - 1], result);
        }
        return result;
    }
    else if (colIn == -1 && rowIn != -1) {
        T result = data[rowIn - 1][0];
        for (int i = 1; i < col; i++) {
            result = std::max(data[rowIn - 1][i], result);
        }
        return result;
    }
    else {
        T result = data[0][0];
        for (int i = 1; i < row; i++) {
            for (int j = 0; j < col; j++) {
                result = std::max(data[i][j], result);
            }
        }
        return result;
    }
}
template<class T>
T Matrix<T>::min(int rowIn, int colIn) const {
    if (row < -1 || row > this->row || row == 0) {
        throw std::invalid_argument("The input row out of matrix row range!");
    }
    if (colIn <-1 || colIn > this->col || col == 0) {
        throw std::invalid_argument("The input row out of matrix col range!");
    }
    if (rowIn != -1 && colIn != -1) {
        return data[rowIn - 1][colIn - 1];
    }
    else if (rowIn == -1 && colIn != -1) {
        T result = data[0][colIn - 1];
        for (int i = 1; i < row; i++) {
            result = std::min(data[i][colIn - 1], result);
        }
        return result;
    }
    else if (colIn == -1 && rowIn != -1) {
        T result = data[0][0];
        for (int i = 1; i < row; i++) {
            for (int j = 0; j < col; j++) {
                result = std::min(data[rowIn - 1][i], result);
            }
        }
        return result;
    }
    else {
        T result = data[0][0];
        for (int i = 1; i < row; i++) {
            for (int j = 0; j < col; j++) {
                result = std::min(data[i][j], result);
            }
        }
        return result;
    }

}

template<class T>
T Matrix<T>::sum(int rowIn, int colIn) const {
    if (row < -1 || row > this->row || row == 0) {
        throw std::invalid_argument("The input row out of matrix row range!");
    }
    if (colIn <-1 || colIn > this->col || col == 0) {
        throw std::invalid_argument("The input row out of matrix col range!");
    }
    if (rowIn != -1 && colIn != -1) {
        return data[rowIn - 1][colIn - 1];
    }
    else if (rowIn == -1 && colIn != -1) {
        T result = data[0][colIn - 1];
        for (int i = 1; i < row; i++) {
            result += data[i][colIn - 1];
        }
        return  result;
    }
    else if (colIn == -1 && rowIn != -1) {
        T result = data[rowIn - 1][0];
        for (int i = 1; i < col; i++) {
            result += data[rowIn - 1][i];
        }
        return result;
    }
    else {
        T result = 0;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                result += this->data[i][j];
            }
        }
        return result;
    }
}

template<class T>
T Matrix<T>::average(int rowIn, int colIn) const {
    if (row < -1 || row > this->row) {
        throw std::invalid_argument("The input row out of matrix row range!");
    }
    if (colIn <-1 || colIn > this->col) {
        throw std::invalid_argument("The input row out of matrix col range!");
    }
    if (rowIn != -1 && colIn != -1) {
        return data[rowIn - 1][colIn - 1];
    }
    else if (rowIn == -1 && colIn != -1) {
        T result = sum(rowIn, colIn) / row;
        return  result;
    }
    else if (colIn == -1 && rowIn != -1) {
        T result = sum(rowIn, colIn) / col;
        return result;
    }
    else {
        T result = this->sum(rowIn, colIn) / (row * col);
    }
}
template<class T>
Matrix<T> Matrix<T>::reshape(int rowIn, int colIn) const {
    int size = row * col;
    if ((rowIn == -1 && colIn == -1) || size % colIn != 0 || size % rowIn != 0 || (colIn * rowIn > 0 && colIn * rowIn != size)) {
        throw Matrix_Shape_Not_Match_Exception(
            "The input shape not match, can not reshape!", __FILE__, __LINE__);
    }
    else {
        if (rowIn == -1) {
            rowIn = size / col;
        }
        else if (colIn == -1) {
            colIn = size / row;
        }
        Matrix<T> result(rowIn, colIn);
        for (int i = 0, temp1 = 0, temp2 = 0; i < rowIn; i++) {
            for (int j = 0; j < colIn; j++, temp2++) {
                if (temp2 >= this->col) {
                    temp2 %= col;
                    temp1++;
                }
                result.data[i][j] = data[temp1][temp2];
            }
        }
        return result;
    }
}

template<class T>
Matrix<T> Matrix<T>::conjugation() const {
    Matrix<T> m = this->transposition();//?????
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            double x = m.data[i][j].imag();
            m.data[i][j].imag(-x);
        }
    }
    return m;
}

template<class T>
Matrix<T> Matrix<T>::operator*(Matrix<T> matrix2) const {
    Matrix<T> matrix(row, matrix2.col);
    if (col != matrix2.row) {
        throw Matrix_Shape_Not_Match_Exception(
            "Shape can not match, can not multiply!", __FILE__, __LINE__);
    }
    else {
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < matrix2.col; j++) {
                matrix.data[i][j] = 0;
                for (int k = 0; k < col; k++) {
                    matrix.data[i][j] += data[i][k] * matrix2.data[k][j];
                }
            }
        }
    }
    return matrix;
}

template<class T>
T Matrix<T>::dot(Matrix<T> matrix2) const {//????????????
    if (this->col == 1 && matrix2.col == 1 && this->row == matrix2.row) {//????????????
        Matrix<T> m = this->transposition();
        return (m * matrix2).data[0][0];
    }
    else {
        if (this->row != matrix2.row && this->col == 1 && matrix2.col == 1) {
            throw Matrix_Shape_Not_Match_Exception(
                "size do not equal bewtween two vectors!", __FILE__, __LINE__);
        }
        else {
            throw Matrix_Shape_Not_Match_Exception(
                "Not vectors, cannot do dot product!", __FILE__, __LINE__);
        }
    }
}

template<class T>
Matrix<T> Matrix<T>::cross(Matrix<T> matrix2) const {//???,????n
    if (this->row != 1 || matrix2.row != 1) {
        throw Matrix_Shape_Not_Match_Exception(
            "Not vectors, cannot do dot product!", __FILE__, __LINE__);
    }
    if (this->col > 3 || matrix2.col > 3) {
        throw Matrix_Shape_Not_Match_Exception("two vectors do not 3 dimension", __FILE__, __LINE__);
    }
    else {
        Matrix<T> m(1, 3);
        m.data[0][0] = this->data[0][1] * matrix2.data[0][2] - this->data[0][2] * matrix2.data[0][1];
        m.data[0][1] = this->data[0][2] * matrix2.data[0][0] - this->data[0][0] * matrix2.data[0][2];
        m.data[0][2] = this->data[0][0] * matrix2.data[0][1] - this->data[0][1] * matrix2.data[0][0];
        return m;
    }
}

//???????????§Ö????????????????????????A*
template<class T>
Matrix<T>  Matrix<T>::getAStart(int n)const
{
    Matrix<T> ans(n, n);
    if (n == 1)
    {
        ans.data[0][0] = 1;
        return ans;
    }
    int i, j, k, t;
    Matrix<T> temp(n - 1, n - 1);
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            for (k = 0; k < n - 1; k++) {
                for (t = 0; t < n - 1; t++) {
                    temp.data[k][t] = this->data[k >= i ? k + 1 : k][t >= j ? t + 1 : t];
                }
            }
            ans.data[j][i] = temp.determinant();
            if ((i + j) % 2 == 1)
            {
                ans.data[j][i] = -ans.data[j][i];
            }
        }
    }
    return ans;
}

template<class T>
Matrix<T> Matrix<T>::inverse(int n) const {
    T m = this->determinant();//???
    Matrix<T> t(n, n);
    Matrix<T> inverseMatrix(n, n);
    if (m == 0) {
        throw Matrix_Shape_Not_Match_Exception(
            "determinant equals 0, cannot find inverse of the matrix!", __FILE__, __LINE__);
    }
    else {
        t = this->getAStart(n);//??????????t?????
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                inverseMatrix.data[i][j] = t.data[i][j] / m;
            }
        }
    }
    return inverseMatrix;
}

template<class T>
Matrix<T> Matrix<T>::convolution(int stride, int padding, Matrix<T> kernel) const {
    kernel = kernel.transposition().transposition();
    if (padding<0 || stride <= 0 || stride>row || row - 3 + 2 * padding <= 0 || row != col) {  //the size of padding should be  0 or (kernel.row-1)/2  (namely 3)
        throw std::invalid_argument("input parameter is invalid");
    }
    else {
        if (padding != 1 && padding != 0) {
            padding = 0;
        }
        int size = (row - 3 + 2 * padding) / stride + 1;
        Matrix<T> output(size, size);
        if (padding == 0) {
            for (int i = 1, outRow = 0; i < row - 1; i += stride, outRow++) {
                for (int j = 1, outCol = 0; j < col - 1; j += stride, outCol++) {
                    output.data[outRow][outCol] = this->slicing(i, j, 3, 3).element_wise_mul(kernel).sum(-1, -1);
                }
            }
        }
        else {
            auto* temp = new Matrix<T>(row + 2, col + 2);
            for (int i = 0, tmpRow = 0, tmpCol = 0; i < temp->row; i++) {
                for (int j = 0; j < temp->col; j++) {
                    if (i == 0 || j == 0 || i == temp->row - 1 || j == temp->col - 1) {
                        temp->data[i][j] = 0;
                    }
                    else {
                        temp->data[i][j] = data[tmpRow][tmpCol];
                        tmpCol++;
                        if (tmpCol == col) {
                            tmpRow++;
                            tmpCol = 0;
                        }
                    }
                }
            }
          
            for (int i = 1, outRow = 0; i < temp->row - 1; i += stride, outRow++) {
                for (int j = 1, outCol = 0; j < temp->col - 1; j += stride, outCol++) {
                    output.data[outRow][outCol] = temp->slicing(i, j, 3, 3).element_wise_mul(kernel).sum(-1, -1);
                }
            }
            delete temp;
        }
        return output;
    }

}

template<class T>
Matrix<T> Matrix<T>::slicing(int x, int y, int rowNum, int colNum) const { //x-1,y-1 is the actual coordinate
    if (x <= 0 || y <= 0 || rowNum <= 0 || colNum <= 0 || x + rowNum - 1 > col || y + colNum - 1 > col || x > row || y > col) {
        throw std::invalid_argument("The row or column your input is out of range!");
    }
    Matrix<T> output(rowNum, colNum);
    T* array = new T[rowNum * colNum];
    for (int i = 0, temp = 0; i < rowNum; i++) {
        for (int j = 0; j < colNum; j++, temp++) {
            array[temp] = this->data[i + x - 1][j + y - 1];
        }
    }
    output.setMatrix(array);
    delete[]array;
    return output;
}

template<class T>
bool Matrix<T>::operator==(Matrix<T> matrix) const {
    if (this->row != matrix.row && this->col != matrix.col) {
        return false;
    }
    if (this->row == matrix.row && this->col == matrix.col) {
        for (int i = 0; i < row; ++i) {
            for (int j = 0; j < col; ++j) {
                if (this->data[i][j] != matrix.data[i][j]) {
                    return false;
                }
            }
        }
    }
    return true;
}

template <class T>
void Matrix_Hessenberg(Matrix<T>& A1, Matrix<T>& ret)
{
    if (A1.row != A1.col) {
        throw Matrix_Shape_Not_Match_Exception(
            "Matrix not square,can not get eigenValue!", __FILE__, __LINE__);
    }
    int i, j, k, MaxNumber;
    T temp;
    int n = A1.row;
    Matrix<T> A(n, n);
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            A.data[i][j] = A1.data[i][j];
        }
    }
    for (k = 1; k < n - 1; k++)
    {
        i = k - 1;
        MaxNumber = k;
        temp = abs(A.data[k][i]);
        for (j = k + 1; j < n; j++)
        {
            if (abs(A.data[j][i]) > temp)

            {
                temp = abs(A.data[j][i]);
                MaxNumber = j;
            }
        }
        ret.data[0][0] = A.data[MaxNumber][i];
        i = MaxNumber;
        if (ret.data[0][0] != 0)
        {
            if (i != k)
            {
                for (j = k - 1; j < n; j++)
                {
                    temp = A.data[i][j];
                    A.data[i][j] = A.data[k][j];
                    A.data[k][j] = temp;
                }
                for (j = 0; j < n; j++)
                {
                    temp = A.data[j][i];
                    A.data[j][i] = A.data[j][k];
                    A.data[j][k] = temp;
                }
            }
            for (i = k + 1; i < n; i++) {
                temp = A.data[i][k - 1] / ret.data[0][0];
                A.data[i][k - 1] = 0;
                for (j = k; j < n; j++) {
                    A.data[i][j] -= temp * A.data[k][j];
                }
                for (j = 0; j < n; j++) {
                    A.data[j][k] += temp * A.data[j][i];
                }
            }
        }
    }
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            ret.data[i][j] = A.data[i][j];
        }
    }
}

template <class T>
bool op(Matrix<T> B1, int temp)
{
    int n = B1.row;
    T NUM1, NUM2;//?????????????
    //?§Ø???????????
    if (B1.data[temp][temp] == 0)//???????????0
        for (int i = temp + 1; i < n; i++)//??????????????
        {
            T t;
            if (B1.data[i][temp] != 0)//?????????0????
            {
                for (int j = 0; j < n; j++)//????
                {
                    t = -B1.data[temp][j];
                    B1.data[temp][j] = B1.data[i][j];
                    B1.data[i][j] = t;
                }
                break;
            }
            if (i == n - 1)//?????0?§µ???0??????
            {
                return false;
            }
        }
    NUM1 = B1.data[temp][temp];//?????
    for (int i = temp + 1; i < n; i++)//????????????§Ù??????
    {
        NUM2 = B1.data[i][temp];//???????
        for (int j = temp; j < n; j++)
        {
            //?????????????????????0
            B1.data[i][j] = B1.data[i][j] - NUM2 * B1.data[temp][j] / NUM1;
        }
    }
    return true;
}

template <class T>
bool upSquare(Matrix<T> C1, Matrix<T> B1) {
    int n = C1.row;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            B1.data[i][j] = C1.data[i][j];
        }
    }
    for (int i = 0; i < n - 1; i++)//?????i??????????§Þ????
    {
        if (op(B1, i) == false)//????0???????????
        {
            return false;
        }
    }
    return true;
}


template <class T>
bool Math_Matrix_EigTor(Matrix<T>& K1, T* EigValue, Matrix<T>& V1) {
    int n = K1.row;
    Matrix<T> C1(n, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C1.data[i][j] = K1.data[i][j];
        }
    }
    T value = EigValue[0];
    for (int i = 0; i < n; i++) {
        C1.data[i][i] -= value;
    }
    Matrix<T> B1(n, n);
    upSquare(C1, B1);
    for (int i = 0; i < EigValue[1]; i++) {
        for (int j = 0; j < EigValue[1]; j++) {
            if (i == j) {
                V1.data[n - j - 1][i] = 1;
            }
            else {
                V1.data[n - j - 1][i] = 0;
            }
        }
        for (int p = n - EigValue[1] - 1; p >= 0; p--) {
            T total = 0;
            for (int j = p + 1; j < n; j++) {
                total -= B1.data[p][j] * V1.data[j][i];
            }
            V1.data[p][i] = total / B1.data[p][p];
        }
    }
    return true;
}
template <class T>
int Matrix_EigenValue(Matrix<T>& K1, int LoopNumber, double Error1, Matrix<T>& Back)
{
    if (K1.row != K1.col) {

    }
    int i, j, k, t, m, Loop1;
    T b, c, d, g, xy, p, q, r, x, s, e, f, z, y, temp;
    int n = K1.row;
    Matrix<T> A(n, n);
    Matrix<T>& a = A;
    Matrix_Hessenberg(K1, a);
    Matrix<T> Ret(Back.row, Back.col - 1);
    m = n;
    Loop1 = LoopNumber;
    while (m != 0)
    {
        t = m - 1;
        while (t > 0)
        {
            temp = abs(A.data[t - 1][t - 1]);
            temp += abs(A.data[t][t]);
            temp = temp * Error1;
            if (abs(A.data[t][t - 1]) > temp)
            {
                t--;
            }
            else
            {
                break;
            }
        }
        if (t == m - 1)
        {
            Ret.data[m - 1][0] = A.data[m - 1][m - 1];
            m -= 1;
            Loop1 = LoopNumber;
        }
        else if (t == m - 2)
        {
            b = -A.data[m - 1][m - 1] - A.data[m - 2][m - 2];
            c = A.data[m - 1][m - 1] * A.data[m - 2][m - 2] - A.data[m - 1][m - 2] * A.data[m - 2][m - 1];
            d = b * b - 4 * c;
            y = sqrt(abs(d));
            if (d > 0)
            {
                xy = 1;
                if (b < 0)
                {
                    xy = -1;
                }
                Ret.data[m - 1][0] = -(b + xy * y) / 2;
                Ret.data[m - 2][0] = c / Ret.data[m - 1][0];
            }
            else
            {
                throw Matrix_Shape_Not_Match_Exception(
                    "doesn't exist real eigenValue!", __FILE__, __LINE__);
                return -1;
            }
            m -= 2;
            Loop1 = LoopNumber;
        }
        else
        {
            if (Loop1 < 1)
            {
                return -1;
            }
            Loop1--;
            j = t + 2;
            while (j < m)
            {
                A.data[j][j - 2] = 0;
                j++;
            }
            j = t + 3;
            while (j < m)
            {
                A.data[j][j - 3] = 0;
                j++;
            }
            k = t;
            while (k < m - 1)
            {
                if (k != t)
                {
                    p = A.data[k][k - 1];
                    q = A.data[k + 1][k - 1];
                    if (k != m - 2)
                    {
                        r = A.data[k + 2][k - 1];
                    }
                    else
                    {
                        r = 0;
                    }
                }
                else
                {
                    b = A.data[m - 1][m - 1];
                    c = A.data[m - 2][m - 2];
                    x = b + c;
                    y = b * c - A.data[m - 2][m - 1] * A.data[m - 1][m - 2];
                    p = A.data[t][t] * (A.data[t][t] - x) + A.data[t][t + 1] * A.data[t + 1][t] + y;
                    q = A.data[t + 1][t] * (A.data[t][t] + A.data[t + 1][t + 1] - x);
                    r = A.data[t + 1][t] * A.data[t + 2][t + 1];
                }
                if (p != 0 || q != 0 || r != 0)
                {
                    if (p < 0)
                    {
                        xy = -1;
                    }
                    else
                    {
                        xy = 1;
                    }
                    s = xy * sqrt(p * p + q * q + r * r);
                    if (k != t)
                    {
                        A.data[k][k - 1] = -s;
                    }
                    e = -q / s;
                    f = -r / s;
                    x = -p / s;
                    y = -x - f * r / (p + s);
                    g = e * r / (p + s);
                    z = -x - e * q / (p + s);
                    for (j = k; j < m; j++)
                    {
                        b = A.data[k][j];
                        c = A.data[k + 1][j];
                        p = x * b + e * c;
                        q = e * b + y * c;
                        r = f * b + g * c;
                        if (k != m - 2)
                        {
                            b = A.data[k + 2][j];
                            p += f * b;
                            q += g * b;
                            r += z * b;
                            A.data[k + 2][j] = r;
                        }
                        A.data[k + 1][j] = q;
                        A.data[k][j] = p;
                    }
                    j = k + 3;
                    if (j > m - 2)
                    {
                        j = m - 1;
                    }
                    for (i = t; i < j + 1; i++)
                    {
                        b = A.data[i][k];
                        c = A.data[i][k + 1];
                        p = x * b + e * c;
                        q = e * b + y * c;
                        r = f * b + g * c;
                        if (k != m - 2)
                        {
                            b = A.data[i][k + 2];
                            p += f * b;
                            q += g * b;
                            r += z * b;
                            A.data[i][k + 2] = r;
                        }
                        A.data[i][k + 1] = q;
                        A.data[i][k] = p;
                    }
                }
                k++;
            }
        }
    }
    int count = 0;
    for (int index = 0; index < Back.row; index++) {
        Back.data[index][1] = 0;
    }
    Back.data[0][0] = Ret.data[0][0];
    Back.data[0][1] = 1;
    count++;
    for (int index = 1; index < Ret.row; index++) {
        bool flag = false;
        for (int jndex = 0; jndex < count; jndex++) {
            if (Ret.data[index][0] == Back.data[jndex][0]) {
                Back.data[jndex][1]++;
                flag = true;
                break;
            }
        }
        if (!flag) {
            Back.data[count][0] = Ret.data[index][0];
            Back.data[count][1]++;
            count++;
        }
    }
    return count;
}

cv::Mat Matrix<int16_t>::transfer()const {
    cv::Mat output(this->row, this->col, CV_16S);
    for (int i = 0; i < output.rows; i++) {
        for (int j = 0; j < output.cols; j++) {
            output.at<int16_t>(i, j) = data[i][j];
        }
    }
    return output;
}

cv::Mat Matrix<float>::transfer()const {
    cv::Mat output(this->row, this->col, CV_32F);
    for (int i = 0; i < output.rows; i++) {
        for (int j = 0; j < output.cols; j++) {
            output.at<float>(i, j) = data[i][j];
        }
    }
    return output;
}

cv::Mat Matrix<double>::transfer()const {
    cv::Mat output(this->row, this->col, CV_64F);
    for (int i = 0; i < output.rows; i++) {
        for (int j = 0; j < output.cols; j++) {
            output.at<double>(i, j) = data[i][j];
        }
    }
    return output;
}


Matrix<int16_t> transfer1(const cv::Mat m) {
    Matrix<int16_t> output(m.rows, m.cols);
    for (int i = 0; i < output.row; i++) {
        for (int j = 0; j < output.col; j++) {
            output.data[i][j] = m.at<int16_t>(i, j);
        }
    }
    return output;
}
Matrix<float> transfer2(const cv::Mat m) {
    Matrix<float> output(m.rows, m.cols);
    for (int i = 0; i < output.row; i++) {
        for (int j = 0; j < output.col; j++) {
            output.data[i][j] = m.at<float>(i, j);
        }
    }
    return output;
}
Matrix<double> transfer3(const cv::Mat m) {
    Matrix<double> output(m.rows, m.cols);
    for (int i = 0; i < output.row; i++) {
        for (int j = 0; j < output.col; j++) {
            output.data[i][j] = m.at<double>(i, j);
        }
    }
    return output;
}




#endif //MATRIX_MATRIX_H


