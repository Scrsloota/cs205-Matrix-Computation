#ifndef PROJECT_SPAREMATRIX_H
#define PROJECT_SPAREMATRIX_H
const int MAX_N = 100;
#include <iostream>
#include "Matrix.h"
template <class T>
class elem_node {
public:
    int elem_row;// 元素行
    int elem_col;//元素列
    T elem_value;//元素值
    elem_node  operator =(const elem_node<T>& x){
        elem_col =x.elem_col;
        elem_row =x.elem_row;
        elem_value = x.elem_value;
        return *this;
    }
};

template <class T>
class Node {
public:
    int total_row;//总行数
    int total_col;//总列数
    int total_num;//非0元素总数
    elem_node<T> data[MAX_N];//非0元素数组
};

template<class T>
class SparseMatrix {
public:
    Node<T> mat;
    SparseMatrix();
    ~SparseMatrix();
    SparseMatrix(const T *array, int rows, int cols);
    bool value(int elem,int i,int j);
    SparseMatrix<T> operator +(const SparseMatrix<T>&b);
    SparseMatrix<T> operator -(const SparseMatrix<T> &b);
    SparseMatrix<T> operator *(const SparseMatrix<T> &b);
    SparseMatrix<T> Trans();
    //SparseMatrix<T>* MultiMatrix(SparseMatrix<T> *MB);
};

template <class T>
SparseMatrix<T>::SparseMatrix(){
    mat.total_row=0;
    mat.total_col=0;
    mat.total_num=0;
}

template <class T>
SparseMatrix<T>::~SparseMatrix<T>() {
    delete &mat;
}


template <typename T>
SparseMatrix<T>::SparseMatrix(const T *array, int rows, int cols) {//构建稀疏矩阵三元组
    int i, j;
    T fdata;
    mat.total_num = 0;
    mat.total_col = cols;
    mat.total_row = rows;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fdata = array[i*cols+j];
            if(fdata!=0){
                mat.data[mat.total_num].elem_row = i;
                mat.data[mat.total_num].elem_col = j;
                mat.data[mat.total_num].elem_value = fdata;
                mat.total_num++;
            }
        }
    }
}

template <class T>
std::ostream &operator<<(std::ostream &os, const SparseMatrix<T> &matrix) {
    std::cout.setf(std::ios_base::floatfield, std::ios_base::fixed);
    std::cout.precision(2);
    for (int i = 0; i < matrix.mat.total_num; i++) {
        os<<matrix.mat.data[i].elem_row<<" "<<matrix.mat.data[i].elem_col<<" "<<matrix.mat.data[i].elem_value;
        os << std::endl;
    }
    return os;
}


template <class T>
bool SparseMatrix<T>::value(int elem, int i, int j){//将指定位置赋值 如超出界限则返回false
    i--;
    j--;
    if(i>=mat.total_row||j>= mat.total_col){
        return false;
    }else{
        int k = 0,k1;
        while (k<mat.total_num&&i>mat.data[k].elem_row){//lines not reach
            k++;
        }
        while (k<mat.total_num&&i==mat.data[k].elem_row&&j>mat.data[k].elem_col){//line reach, column not reach
            k++;
        }
        if(mat.data[k].elem_row == i&&mat.data[k].elem_col ==j){
            mat.data[k].elem_value = elem;
        }else{
            for(k1 = mat.total_num -1;k1>=k;k1--){
                mat.data[k1+1] = mat.data[k1];
            }
            mat.data[k].elem_row = i;
            mat.data[k].elem_col = j;
            mat.data[k].elem_value = elem;
            mat.total_num++;
        }
    }
    return true;
}

template<class T>
SparseMatrix<T> SparseMatrix<T>::operator+(const SparseMatrix<T> &b) {
    int number1 = mat.total_num;
    int number2 = b.mat.total_num;
    SparseMatrix<T> matrix;
    int p=0,q=0;
    int total = number2+number1;
    for(int i=0;i<number1+number2;i++){
        if(q>=number2||
           p<number1&&(mat.data[p].elem_row<b.mat.data[q].elem_row||
                       (mat.data[p].elem_row==b.mat.data[q].elem_row&&mat.data[p].elem_col<b.mat.data[q].elem_col))){
            matrix.mat.data[matrix.mat.total_num]=mat.data[p];
            matrix.mat.total_num++;
            p++;
        } else if(q<number2&&p<number1&&mat.data[p].elem_row==b.mat.data[q].elem_row&&mat.data[p].elem_col==b.mat.data[q].elem_col){
            matrix.mat.data[matrix.mat.total_num].elem_row=mat.data[p].elem_row;
            matrix.mat.data[matrix.mat.total_num].elem_col = mat.data[p].elem_col;
            matrix.mat.data[matrix.mat.total_num].elem_value = mat.data[p].elem_value+b.mat.data[p].elem_value;
            matrix.mat.total_num++;
            p++;
            q++;
            total--;
        }else{
            matrix.mat.data[matrix.mat.total_num]=b.mat.data[q];
            matrix.mat.total_num++;
            q++;
        }
    }
    return matrix;
}

template <class T>
SparseMatrix<T> SparseMatrix<T>::operator-(const SparseMatrix<T> &b){
    int number1 = mat.total_num;
    int number2 = b.mat.total_num;
    SparseMatrix<T> matrix;
    int p=0,q=0;
    int total = number2+number1;
    for(int i=0;i<number1+number2;i++){
        if(q>=number2||
           p<number1&&(mat.data[p].elem_row<b.mat.data[q].elem_row||
                       (mat.data[p].elem_row==b.mat.data[q].elem_row&&mat.data[p].elem_col<b.mat.data[q].elem_col))){
            matrix.mat.data[matrix.mat.total_num]=mat.data[p];
            matrix.mat.total_num++;
            p++;
        } else if(q<number2&&p<number1&&mat.data[p].elem_row==b.mat.data[q].elem_row&&mat.data[p].elem_col==b.mat.data[q].elem_col){
            matrix.mat.data[matrix.mat.total_num].elem_row=mat.data[p].elem_row;
            matrix.mat.data[matrix.mat.total_num].elem_col = mat.data[p].elem_col;
            matrix.mat.data[matrix.mat.total_num].elem_value = mat.data[p].elem_value-b.mat.data[p].elem_value;
            matrix.mat.total_num++;
            p++;
            q++;
            total--;
        }else{
            matrix.mat.data[matrix.mat.total_num].elem_row=b.mat.data[q].elem_row;
            matrix.mat.data[matrix.mat.total_num].elem_col = b.mat.data[q].elem_col;
            matrix.mat.data[matrix.mat.total_num].elem_value = -1*b.mat.data[1].elem_value;
            matrix.mat.total_num++;
            q++;
        }
    }
    return matrix;
}

template<class T>
SparseMatrix<T> SparseMatrix<T>::operator* (const SparseMatrix<T> &b){
    if(mat.total_col!=b.mat.total_row){
        throw Matrix_Shape_Not_Match_Exception(
                "Column of the first matrix and row of the second matrix not match, cannot multiply!", __FILE__, __LINE__);
    }
    if(mat.total_num==0||b.mat.total_num==0){
        throw Matrix_Shape_Not_Match_Exception(
                "Result a zero matrix!", __FILE__, __LINE__);
    }
    SparseMatrix<T> MulMatrix;
    MulMatrix.mat.total_row = mat.total_row;
    MulMatrix.mat.total_col = b.mat.total_col;
    MulMatrix.mat.total_num =0;

    T* row_num = new T[b.mat.total_row]{0};
    for(int i=0;i<b.mat.total_num;i++){
        row_num[b.mat.data[i].elem_row]++;
    }
    T* row_position = new T[b.mat.total_row]{0};
    for(int i=1;i<b.mat.total_row;i++){
        row_position[i] = row_position[i-1]+row_num[i-1];
    }
    T av = 0,bv = 0;
    T ar = 0,br = 0,ac =0 ,bc = 0;
    T nb = 0;
    T inb = 0;
    T *row_result = new T[b.mat.total_col]{0};
    for(int p =0;p<mat.total_num;p++){
        ar = mat.data[p].elem_row;
        ac = mat.data[p].elem_col;
        av = mat.data[p].elem_value;
        br = ac;
        inb = row_position[br];
        nb = row_num[br];
        for(int n=0;n< nb;n++){
            br = b.mat.data[inb+n].elem_row;
            bc = b.mat.data[inb+n].elem_col;
            bv = b.mat.data[inb+n].elem_value;
            row_result[bc]+=av*bv;
        }
        if( p ==mat.total_num -1||mat.data[p+1].elem_row!=ar){
            for(int j=0;j<b.mat.total_col;j++){
                if(row_result[j]!=0){
                    MulMatrix.mat.data[MulMatrix.mat.total_num].elem_row = ar;
                    MulMatrix.mat.data[MulMatrix.mat.total_num].elem_col = j;
                    MulMatrix.mat.data[MulMatrix.mat.total_num].elem_value = row_result[j];
                    MulMatrix.mat.total_num++;
                }
                row_result[j]=0;
            }
        }

    }
    return MulMatrix;
}

template<class T>
SparseMatrix<T> SparseMatrix<T>::Trans() {
    SparseMatrix<int> TM;
    TM.mat.total_num = this->mat.total_num;
    TM.mat.total_row = this->mat.total_col;
    TM.mat.total_col = this->mat.total_row;

    T * col_num = new T[this->mat.total_col]{0};
    for(int i=0;i<this->mat.total_num;i++){
        col_num[mat.data[i].elem_col]++;
    }
    T * col_position = new T[this->mat.total_col]{0};
    for(int i=1;i<this->mat.total_col;i++){
        col_position[i] = col_position[i-1]+col_num[i-1];
    }
    int p = 0;
    for(int i=0;i<this->mat.total_num;i++){
        p = col_position[this->mat.data[i].elem_col]++;
        TM.mat.data[p].elem_row = this->mat.data[i].elem_col;
        TM.mat.data[p].elem_col = this->mat.data[i].elem_row;
        TM.mat.data[p].elem_value = this->mat.data[i].elem_value;
    }
    return TM;
}

#endif //PROJECT_SPAREMATRIX_H
