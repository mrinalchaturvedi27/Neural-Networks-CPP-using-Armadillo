#include <iostream>
#include <armadillo>
using namespace std;
using namespace arma;
template <typename T>
class Tensor{
  private:
  vector<T> storage;
  vector<int> shape;
  vector<int> strides;
  
  void compute_strides() {
        strides.resize(shape.size());
        int stride = 1;
        for (int i = shape.size() - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape[i];
        }
    }
  public:
   Mat<T> view() const{
       return Mat<T>(storage.data(), shape[1], shape[0], false, true).t();
    }
    Tensor(const Mat<T> &X){
    shape={(int)X.n_rows,(int)X.n_cols};
    storage.resize(X.n_elem);
    int k = 0;
    for(int i=0; i < (int)X.n_rows; i++) {
     for(int j=0; j < (int)X.n_cols; j++) {
        storage[k++] = X(i, j);
            }
        }
      compute_strides();
    }
    
  
  Tensor(const vector<int>& s) {
    shape = s;
    int total = 1;
    for (int dim : shape) total *= dim;
    storage.assign(total,0);
    compute_strides();
  }
  
    const T& operator()(const vector<int> &id){
    int k = 0;
    for(size_t i = 0; i < id.size(); i++) {
        k += strides[i] * id[i];
    }
    return storage[k];
  }

  Tensor operator+(const Tensor &other){
    if(shape!=other.shape) {
      throw runtime_error("Shape mismatch");
      // cout<<"Shape Does Not Match"<<endl;
      }
    Mat<T> result = this->view() + other.view();
    return Tensor(result);
  }

  Tensor operator-(const Tensor &other){
    if(shape!=other.shape) {
      throw runtime_error("Shape mismatch");
      // cout<<"Shape Does Not Match"<<endl;
      }
    Mat<T> result = this->view() - other.view();
    return Tensor(result);
    return out;
  }

  Tensor operator/(const Tensor &other){
    if(shape!=other.shape) {
      throw runtime_error("Shape mismatch");
      // cout<<"Shape Does Not Match"<<endl;
      }
    Mat<T> result = this->view() / other.view();
    return Tensor(result);
    return out;
  }

  Tensor operator%(const Tensor &other){
    if(shape!=other.shape) {
      throw runtime_error("Shape mismatch");
      // cout<<"Shape Does Not Match"<<endl;
      }
    Mat<T> result = this->view() % other.view();
    return Tensor(result);
    return out;
  }
  
  Tensor dot(const Tensor &other){
    if (shape.size() != 2 || other.shape.size() != 2)
      throw runtime_error("Dot only for 2-D tensors");

    if(shape[1]!=other.shape[0]) {
      throw runtime_error("Shape mismatch: shape[1]!=other.shape[0]");
      // cout<<"Shape Does Not Match"<<endl;
      }
      Mat<T> result=this->view() * other.view();
    return Tensor(result);
  }

  Tensor reshape(const vector<int> &s){
    size_t total = 1;
    for(int dim : s) total *= dim;
    if(total != storage.size()) throw runtime_error("Reshape size mismatch");
    Tensor<T> new_t(*this);
    new_t.shape = s;
    new_t.compute_strides();
    return new_t;
  }

  Tensor transpose(){
      Tensor new_t=*this;
      swap(new_t.shape[0],new_t.shape[1]);
      swap(new_t.strides[0],new_t.strides[1]);
    return new_t;
  }

  Tensor sum(int dim){
    if(dim != 0 && dim != 1) throw runtime_error("Axis must be 0 or 1");
    return Tensor<T>(arma::sum(this->view(), dim));
}

Tensor mean(int dim){
    if(dim != 0 && dim != 1) throw runtime_error("Axis must be 0 or 1");
    return Tensor<T>(arma::mean(this->view(), dim));
}

Tensor MAX(int dim){
    if(dim != 0 && dim != 1) throw runtime_error("Axis must be 0 or 1");
    return Tensor<T>(arma::max(this->view(), dim));
}
  void print() const {
    this->view().print();
  }

//   Tensor add_vector(const Tensor& vec) {
//     if (vec.shape[1] != this->shape[1]) 
//         throw runtime_error("Width mismatch for broadcasting");
//     Mat<T> result = this->view();
//     result.each_row() += vec.x_tensor.row(0);
    
//     return Tensor(result);
// }
};

int main(){
  fmat A={{1,3},{2,4}};
  fmat B={{1,0.5},{0.5,1/3}};
  Tensor<float> a(A);
  Tensor<float> b(B);
  Tensor<float> d=a.transpose();
  a.print();
  cout<<endl;
  d.print();
  return 0;
}
