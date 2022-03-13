#include <iostream>
#include <opencv2/opencv.hpp>
#include<iomanip>
#include "Matrix.h"
using namespace std;
int main() {

    //transfer the matrix (int) from this library to the matrix of OpenCV
    int16_t i[6] = { 1,2,3,4,5,6 };
    Matrix<int16_t> ma(2, 3, i);
    cout << "1.The matrix from this libary in int:" << endl << ma ;
    cv::Mat matrix1 = ma.transfer();
    cout << "The matrix from OpenCV transered from library:" << endl << matrix1 << endl ;
    cout<<"The transposition of this matrix:" <<endl<<matrix1.t()<<endl;
    cout << endl;

    //transfer the matrix (float) from this library to the matrix of OpenCV
    float f[6] = { 1.1f,2.2f,3.3f,4.0f,2.2f,2.3f };
    Matrix<float>mf(2, 3, f);
    cout << "2.The matrix from this libary in float:" << endl << mf ;
    cv::Mat matrix2 = mf.transfer();
    cout << "The matrix from OpenCV transered from library:" << endl << matrix2 << endl;
    cout << "matrix * matrix^T = " << endl << matrix2 * (matrix2.t()) << endl;
    cout << endl;

    //transfer the matrix (double) from this library to the matrix of OpenCV
    double d[9] = { 1.1,2.2,1.1,3.3,3.2,1.1,1.3,1.4,1.5 };
    //double a[] = { 1,-3,3,3,-5,3,6,-6,4 };
    Matrix<double>md(3, 3 , d);
    cout << "3.The matrix from this libary in double:" << endl << md ;
    cv::Mat matrix3 = md.transfer();
    cout << "The matrix from OpenCV transered from library:" << endl << matrix3 << endl;
    cv::Mat eValuesMat;
    cv::Mat eVectorsMat;
    eigen(matrix3, eValuesMat, eVectorsMat);
    cout << "Eigenvalues of the matrix : " << endl<< eValuesMat << endl;
    cout << "Eigenvector of the matrix : "<< endl << eVectorsMat << endl;
    cout << "Inverse of the matrix :" << endl << matrix3.inv() << endl;
    cout << endl;

    //transfer the matrix (int) from OpenCV to the matrix of this library
    cv::Mat m1(3, 2, CV_16S, i);
    cout << "1.The matrix from OpenCV in int:" << endl << m1 << endl;
    Matrix<int16_t> mm1 = transfer1(m1);
    cout << "The matrix from library transered from OpenCV:" << endl << mm1 <<endl;

    //transfer the matrix (float) from OpenCV to the matrix of this library
    cv::Mat m2(3, 2, CV_32F, f);
    cout << "2.The matrix from OpenCV in float:" << endl << m2 << endl;
    Matrix<float> mm2 = transfer2(m2);
    cout << "The matrix from library transered from OpenCV:" << endl << mm2<<endl;
    
    //transfer the matrix (double) from OpenCV to the matrix of this library
    cv::Mat m3(3, 3, CV_64F, d);
    cout << "3.The matrix from OpenCV in double:" << endl << m3 << endl;
    Matrix<double> mm3 = transfer3(m3);
    cout << "The matrix from library transered from OpenCV:" << endl << mm3<<endl;
    cout << mm3.inverse(3)<<endl;
  
    return 0; 
}
