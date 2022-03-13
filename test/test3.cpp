//
// Created by 19121 on 2021/6/10.
//
#include "spareMatrix.h"
#include <iostream>
using namespace std;
int main(){
    int a1[12] = {1,0,0,0,2,0,0,5,0,4,5,0};
    SparseMatrix<int> sm(a1, 3, 4);
    cout<<sm;
    cout<<"-------------------"<<endl;
    int a2[12] = {0,2,3,0,3,0,3,0,5,1,1,1};
    SparseMatrix<int> sm2(a2, 4, 3);
    cout<<sm2;
    cout<<"-------------------"<<endl;

    SparseMatrix<int> ms = sm.Trans();
    cout<<ms;
    cout<<"-------------------"<<endl;

    SparseMatrix<int> add = sm + sm2;
    cout<<add;
    cout<<"-------------------"<<endl;

    SparseMatrix<int> sub = sm - sm2;
    cout<<sub;
    cout<<"-------------------"<<endl;

    SparseMatrix<int> MulOutput = sm * sm2;
    cout<<MulOutput;
    cout<<"-------------------"<<endl;

    bool flag= sm.value(5,2,2);
    cout<<flag<<endl;//to judge if the position is in the range of SparseMatrix
    cout<<sm;
    cout<<"-------------------"<<endl;

}
