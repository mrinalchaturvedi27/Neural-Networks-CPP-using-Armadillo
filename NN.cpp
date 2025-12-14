#include <iostream>
#include <armadillo>
using namespace std;
using namespace arma;

int main(){
  mat a={{1,2},{2,3}};
  mat b={{1,1},{2,2}};
  a.print();
  cout<<endl;

  b.print();
  vec c = {1, 1};
  mat d = a * c.t();
  cout<<endl;
  c.print();


    return 0;
}