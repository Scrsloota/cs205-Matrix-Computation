#include <iostream>
#include <random>
#include "Matrix.h"
#include "gtest/gtest.h"


TEST(TestCase,test1){//构造任意大小的矩阵
    random_device rd;
    default_random_engine gen = default_random_engine(rd());
    uniform_int_distribution<int> dis(-1000,1000);
    Matrix<int> matrix(100,100);
    int *array = new int[100*100];
    for(int i=0;i<100*100;i++){//生成随机int类型
        array[i] = dis(gen);
    }
    matrix.setMatrix(array);
    cout << matrix;
}
TEST(TestCase,test2){//测试有参构造方法
    int a[] = {1,2,3,4};
    Matrix<int> m1(2, 2);
    m1.setMatrix(a);
    std::cout << "create an matrix in type of integer:\n";
    cout << m1;
    double b[] = {1.03,2,3,4.999,5.67897,
                  1,2,5,6,8,
                  23,46.764,666,78,9.1246,
                  124.399,3,6,88.1543,0};
    Matrix<double> m2(5, 4);
    m2.setMatrix(b);
    std::cout << "create an matrix in type of double:\n";
    cout << m2;
    complex<double> c[] = {{23,123.3},{213,3.44},{14.22,123},
                           {1.43,23},{55,3.44},{9.22,12389},
                           {1.7,3.09},{6.545,3.44},{3.22,0.98}};
    Matrix<std::complex<double>> m3(3, 4);
    m3.setMatrix(c);
    std::cout << "create an matrix in type of complex:\n";
    cout << m3 << endl;
}

TEST(TestCase,test3){//out of range
    Matrix<double> m4(0, -1);
    cout << m4;
}

TEST(TestCase,test4){//测试转置
    double b[] = {1.03,2,3,4.999,5.67897,
                  1,2,5,6,8,
                  23,46.764,666,78,9.1246,
                  124.399,3,6,88.1543,0};
    Matrix<double> m2(5, 4);
    m2.setMatrix(b);
    std::cout << "A matrix as follow:\n";
    cout << m2;
    std::cout << "get transposition:\n";
    cout << m2.transposition()<<endl;
}
TEST(Testcase,test5){//共轭
    complex<double> c[] = {{23,123.3},{213,3.44},{14.22,123},
                           {1.43,23},{55,3.44},{9.22,12389},
                           {1.7,3.09},{6.545,3.44},{3.22,0.98}};
    Matrix<std::complex<double>> m3(3, 4);
    m3.setMatrix(c);
    std::cout << "A complex matrix as follow:\n";
    cout << m3;
    std::cout << "get conjugation:\n";
    cout << m3.conjugation()<<endl;
}

TEST(TestCase,tset6){//operator plus
    Matrix<int> m1(5,3);
    int a[] = {1,2,3,4,5,6,7,8,9,10,2,3,4,56,5};
    m1.setMatrix(a);
    Matrix<int> m4(5,3);
    int b[] = {1,10,3,4,5,6,7,8,9,10,7,8,9,10,2};
    m4.setMatrix(b);
    std::cout << "operator plus:\n";
    cout << m1 << " +\n" << m4 << " =\n" << m1+m4 << endl;
}

TEST(TsetCase,teat7){//operator minus
    Matrix<int> m1(5,2);
    int a[] = {1,2,3,4,5,6,7,8,9,10};
    m1.setMatrix(a);
    Matrix<int> m4(5,2);
    int b[] = {1,10,3,4,5,6,7,8,9,10};
    m4.setMatrix(b);
    std::cout << "operator minus:\n";
    cout << m1 << " -\n" << m4 << " =\n" << m1-m4 << endl;
    Matrix<int> mcorrect(5,2);
    int test[]={0,-8,0,0,0,0,0,0,0,0};
    mcorrect.setMatrix(test);
    GTEST_CHECK_(mcorrect == m1-m4);
}
TEST(Testcase,test8){//operator multiply_matrix_number
    Matrix<double> m1(2,3);
    double a[] = {1.01, 1.01, 4.04,5.03,1.01,4.00};
    m1.setMatrix(a);
    cout << "Matrix m1:\n" << m1;
    Matrix<double> m2(2,3);
    double b[] = {2.02,2.02, 8.08,10.06,2.02,8.00};
    m2.setMatrix(b);
    std::cout << "operator multiply_matrix_number:\n";
    cout <<"m1*2 is\n" << m1*2 << endl;
    GTEST_CHECK_(m2 == m1*2);
}

TEST(Testcase,test9){//matrix * matrix
    std::cout << "Matrix*Matrix:\n";
    Matrix<int> m1(2,3);
    int a[] = {1,2,3,4,5,6};
    m1.setMatrix(a);
    cout << "Matrix m1:\n";
    cout << m1;
    Matrix<int> m2(3,2);
    int b[] = {1,4,2,5,3,6};
    cout << "Matrix m2:\n";
    m2.setMatrix(b);
    cout << m2;
    std::cout << "m1*m2 = \n" << m1*m2;
    Matrix<int> result(2,2);
    int c[]={14,32,32,77};
    result.setMatrix(c);
    GTEST_CHECK_(result==m1*m2);
}
TEST(Testcase,test10){//matrix * matrix exception
    Matrix<int> m1(2,3);
    int a[] = {1,2,3,4,5,6};
    m1.setMatrix(a);
    cout << "Matrix m1:\n";
    cout << m1;
    std::cout << "m1*m1 = \n" << m1*m1;
}
TEST(Testcase,test11){//element_wise_multiplication exception
    Matrix<double> m1(3,3);
    double a[] = {1.01,1.02,9.03,1.02, 2.03,0.4,1.05,1.11,4.09};
    m1.setMatrix(a);
    cout << "Matrix m1:\n";
    cout << m1;
    Matrix<double> m2(3,3);
    double b[] = {1.09,1.22,9.32,1.00, 1.00, 0,2.09,1.99,6.09};
    m2.setMatrix(b);
    cout << "Matrix m2:\n";
    cout << m2;
    cout << "element_wise_mul:\n";
    cout << m1.element_wise_mul(m2);
}
TEST(Testcase,test12){//element_wise_multiplication
    Matrix<double> m1(3,3);
    double a[] = {1.01,1.02,9.03,1.02, 2.03,0.4,1.05,1.11,4.09};
    m1.setMatrix(a);
    cout << "Matrix m1:\n";
    cout << m1;
    Matrix<double> m2(2,2);
    double b[] = {1.09,1.22,9.32,1.00};
    m2.setMatrix(b);
    cout << "Matrix m2:\n";
    cout << m2;
    cout << "element_wise_mul:\n";
    cout << m1.element_wise_mul(m2);
}
TEST(Testcase,test13){//dot exception
    Matrix<int> m1(2,3);
    int a[] = {1,2,3,4,5,6};
    m1.setMatrix(a);
    cout << "Vector v1:\n";
    cout << m1;
    Matrix<int> m2(3,1);
    int b[] = {1,5,10};
    m2.setMatrix(b);
    cout << "Vector v2:\n";
    cout << m2;
    std::cout << "dot product:\n";
    cout << m1.dot(m2);
}
TEST(Testcase,test14){//dot
    Matrix<int> m1(3,1);
    int a[] = {1,2,2};
    m1.setMatrix(a);
    cout << "Vector v1:\n";
    cout << m1;
    Matrix<int> m2(3,1);
    int b[] = {1,5,10};
    m2.setMatrix(b);
    cout << "Vector v2:\n";
    cout << m2;
    std::cout << "dot product:\n";
    cout << m1.dot(m2);
    GTEST_CHECK_(m1.dot(m2)==31);
}
TEST(Testcase,test15){//vector * matrix
    Matrix<complex<double>> m1(2,4);
    complex<double> a[] = {{0, 3},{2, 1},{3,3},{2,5},
                           {0, 2},{1, 1},{6,3},{1,9}};
    m1.setMatrix(a);
    cout << "Matrix:\n";
    cout << m1;
    Matrix<complex<double>> vector(4,1);
    complex<double> b[] = {{4, 5},{1,3},{3, 5},{4, 2}};
    vector.setMatrix(b);
    cout << "Vector:\n";
    cout << vector;
    std::cout << "vector * matrix\n" << m1*vector;
}
TEST(Testcase,test16){//vector cross vector
    Matrix<int> m1(1,3);
    int a[] = {1,2,2};
    m1.setMatrix(a);
    cout << "Vector1:\n";
    cout << m1;
    Matrix<int> m2(1,3);
    int b[] = {3,1,4};
    m2.setMatrix(b);
    cout << "Vector2:\n";
    cout << m2;
    cout << "Vector1 cross Vector2:\n" << m1.cross(m2);
}
TEST(Testcase,test17){//trace & determinant
    Matrix<double> x(2,2);
    double r[]={1.2,2.3,2,0};
    x.setMatrix(r);
    cout << "Matrix:\n";
    cout << x;
    cout << "trace:\n" << x.trace() << endl;
    cout << "determinant:\n" << x.determinant() << endl;
    Matrix<int> m(4,4);
    int result[] = {0, 3, 0, 6,2, 1, 4, 2,0, 9, 0, 3,6, 3, 2, 1};
    m.setMatrix(result);
    cout << "Matrix:\n";
    cout << m;
    cout << "trace:\n" << m.trace() << endl;
    cout << "determinant:\n" << m.determinant();
}
TEST(Testcase,test18){//sum & average & max & min
    Matrix<int> m1(2,5);
    int a[] ={1,2,3, 4,5, 6,7, 8,9, 10};
    m1.setMatrix(a);
    cout << "Matrix1:\n" << m1;
    cout << "sum of all matrix:"<<m1.sum(-1,-1) << endl;
    cout << "sum of 1st column:"<<m1.sum(-1,1) << endl;
    cout << "average of all element:"<<m1.average(-1,-1) << endl;
    Matrix<int> m2(3,2);
    int b[] ={1, 1, 4,5, 1, 4};
    m2.setMatrix(b);
    cout << "Matrix2:\n" << m2;
    cout << "Max of element of all matrix:"<<m2.max(-1,-1) << endl;
    cout << "Min of 2th column:"<<m2.min(-1,2) << endl;
    cout << "average of all element on 1st column:"<<m2.average(-1,-1) <<endl;
}
TEST(Testcase,test19){//convolution
    Matrix<int> m1(5,5);
    int a[] = {1, 1, 1, 0, 0,
               0, 1, 1, 1, 0,
               0, 0, 1, 1, 1,
               0, 0, 1, 1, 0,
               0, 1, 1, 0, 0};
    m1.setMatrix(a);
    cout << "Matrix1:\n" << m1;
    Matrix<int> kernel(3,3);
    int array[]={0,1,0,1,-4,1,0,1,0};
    kernel.setMatrix(array);
    cout << "kernel:\n"<<kernel;
    cout << "get convolution\n" << m1.convolution(2,1,kernel);
}
TEST(Testcase,test20){// reshape
    Matrix<complex<int>> m(4,6);
    complex<int> a[]={{1,2},{2,3},{3,4},{5,6},{6,7},{7,8},
                      {2,3},{3,4},{5,6},{6,7},{7,8},{9,10},
                      {5,6},{6,7},{7,8},{9,10},{1,4},{2,2},
                      {4,7},{2,6},{1,1},{0,5},{2,8},{5,5}};
    m.setMatrix(a);
    cout << "Matrix:\n" << m;
    cout << "reshape:\n" << m.reshape(3,8);
    cout << "reshape another for 5x5 matrix:\n" << m.reshape(5,5);
}
TEST(Testcase,test21){//slicing
    Matrix<complex<int>> m(4,6);
    complex<int> a[]={{1,2},{2,3},{3,4},{5,6},{6,7},{7,8},
                      {2,3},{3,4},{5,6},{6,7},{7,8},{9,10},
                      {5,6},{6,7},{7,8},{9,10},{1,4},{2,2},
                      {4,7},{2,6},{1,1},{0,5},{2,8},{5,5}};
    m.setMatrix(a);
    cout << "Matrix:\n" << m;
    cout << "slicing:\n" << m.slicing(2,3,2,3);
    cout << "slicing again:\n" << m.slicing(5,2,1,4);
}
TEST(Testcase,test22){//特征值特征向量
    int number =10000;//迭代次数
    double error = 0.00000001;//误差最小值
    int SIZE =3;
    Matrix<double> matrix(SIZE,SIZE);
    double a[]={1,-3,3,3,-5,3,6,-6,4};
    matrix.setMatrix(a);
    Matrix<double>&A =matrix;
    cout<<"The input matrix is "<<endl;
    cout<<matrix;
    Matrix<double> back(SIZE,2);//返回的特征值以及其对应的个数
    Matrix<double>&S =back;
    int count = Matrix_EigenValue(A,number,error,S);//返回为不同特征值的数量
    for(int i=0;i<count;i++){
        cout<<"The eigenVectors with eigenValue = "<<back.data[i][0]<<endl;
        Matrix<double> value(SIZE,back.data[i][1]);
        Matrix<double> &V = value;
        Math_Matrix_EigTor(A,back.data[i],V);
        cout<<V;//在对应特征值时的特征向量
    }
}

TEST(Testcase,testcase23){//求逆
    Matrix<double> m(3,3);
    double a[]={-3.68,-0.41,-2.81,1.79,4.35,0.19,-4.65,0.30,-4.92};
    m.setMatrix(a);
    cout << "Matrix:\n" << m;
    Matrix<double> m1(3,3);
    m1 = m.inverse(3);
    cout << "Inverse matrix m:\n" << m1;
    cout << "Inverse the inverse of matrix is the orginal Matrix:\n" << m1.inverse(3);
}
TEST(Testcase,testcase24){//operator division
    Matrix<double> m1(3,3);
    double a[] = {1.01,1.02,9.03,1.02, 2.03,0.4,1.05,1.11,4.09};
    m1.setMatrix(a);
    cout << "Matrix m1:\n";
    cout << m1;
    cout << "m1/3 is\n" << m1/3;
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
