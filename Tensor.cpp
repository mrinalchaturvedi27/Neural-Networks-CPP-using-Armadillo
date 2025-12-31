#include <iostream>
#include <armadillo>
using namespace std;
using namespace arma;
template <typename T>
class Tensor{
  private:
  Mat<T> x_tensor;
  vector<int> shape;
  vector<int> strides;
  Row<T> data;
  void compute_strides() {
        strides.resize(shape.size());
        int stride = 1;
        for (int i = shape.size() - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape[i];
        }
    }
  public:
  Tensor(const Mat<T> &X){
    x_tensor=X;
    shape={(int)X.n_rows,(int)X.n_cols};
    int size=X.n_rows*X.n_cols;
    data.set_size(size);
    int k=0;
    for(int i=0;i<X.n_rows;i++){
      for(int j=0;j<X.n_cols;j++){
        data[k++]=X(i,j);
      }
    }
    compute_strides();
  }
  Tensor(const vector<int>& s) {
    shape = s;
    int total = 1;
    for (int dim : shape) total *= dim;
    data.set_size(total);
    data.zeros();
    x_tensor.zeros(s[0], s[1]);
    compute_strides();
  }
  const T& operator()(const vector<int> &id){
    if (id.size() != shape.size()) {
        throw runtime_error("Dimension mismatch in indexing");
    }
    int k=0;
    for(size_t i=0;i<id.size();i++){
      if (id[i] < 0 || id[i] >= shape[i]) {
            throw runtime_error("Index out of bounds at dimension " + to_string(i));
        }
      k+=strides[i]*id[i];
    }
    return data[k];
  }

  Tensor operator+(const Tensor &other){
    if(shape!=other.shape) {
      throw runtime_error("Shape mismatch");
      // cout<<"Shape Does Not Match"<<endl;
      }
    Tensor out(shape);
    out.x_tensor=other.x_tensor+x_tensor;
    out.data=data+other.data;
    return out;
  }

  Tensor operator+(const T &scalar) {
    return Tensor(x_tensor + scalar); 
  }

  Tensor operator-(const Tensor &other){
    if(shape!=other.shape) {
      throw runtime_error("Shape mismatch");
      // cout<<"Shape Does Not Match"<<endl;
      }
    Tensor out(shape);
    out.x_tensor=x_tensor-other.x_tensor;
    out.data=data-other.data;
    return out;
  }

  Tensor operator/(const Tensor &other){
    if(shape!=other.shape) {
      throw runtime_error("Shape mismatch");
      // cout<<"Shape Does Not Match"<<endl;
      }
    Tensor out(shape);
    out.x_tensor=x_tensor/other.x_tensor;
    return out;
  }

  Tensor operator%(const Tensor &other){
    if(shape!=other.shape) {
      throw runtime_error("Shape mismatch");
      // cout<<"Shape Does Not Match"<<endl;
      }
    Tensor out(shape);
    out.x_tensor = x_tensor % other.x_tensor; 
    out.data = data % other.data;
    return out;   
  }
  
  Tensor dot(const Tensor &other){
     if (shape.size() != 2 || other.shape.size() != 2)
      throw runtime_error("Dot only for 2-D tensors");

    if(shape[1]!=other.shape[0]) {
      throw runtime_error("Shape mismatch: shape[1]!=other.shape[0]");
      // cout<<"Shape Does Not Match"<<endl;
      }
    Mat<T> result=x_tensor*other.x_tensor;
    return Tensor(result);
  }

  Tensor reshape(const vector<int> &s){
    size_t dim=1;
    for(int k:s) dim*=k;
    if(dim!=data.size()) throw runtime_error("Shape mismatch");
    Mat<T> res=reshape(x_tensor,s[0],s[1]);
    return Tensor(res);
  }

  Tensor transpose(){
    return Tensor(x_tensor.t());
  }

  Tensor sum(int dim){
   if(dim!=0 && dim!=1){
    throw runtime_error("Wrong axis argument: axis=[0,1]");
   }
  Mat<T> result=sum(x_tensor, dim);  
  return Tensor(result);
  }
  
  Tensor mean(int dim){
   if(dim!=0 && dim!=1){
    throw runtime_error("Wrong axis argument: axis=[0,1]");
   }
  Mat<T> result=mean(x_tensor, dim);   
  return Tensor(result);
  }

  Tensor MAX(int dim){
   if(dim!=0 && dim!=1){
    throw runtime_error("Wrong axis argument: axis=[0,1]");
   } 
  Mat<T> result=max(x_tensor, dim);  
  return Tensor(result);
  }
  void print() const {
        x_tensor.print();
  }

  Tensor add_vector(const Tensor& vec) {
    if (vec.shape[1] != this->shape[1]) 
        throw runtime_error("Width mismatch for broadcasting");
    Mat<T> result = x_tensor;
    result.each_row() += vec.x_tensor.row(0); 
    
    return Tensor(result);
}
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
