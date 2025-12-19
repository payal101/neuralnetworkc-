#include<iostream>
#include<vector>
#include<cmath>
#include<cstdlib>
#include<fstream>
using namespace std;

double sigmoid(double x)
{
    return 1.0/(1.0+ exp(-x));
}
double sigmoid_derivative(double x)
{
    return x*(1.0-x);
}

class NeuralNetwork{
    private:
    int input_size;
    int hidden_size;
    int output_size;
    vector<vector<double>> W1;
    vector<vector<double>>W2;

    vector<double> hidden;
    vector<double>output;
    double learning_rate;

public:
NeuralNetwork(int in,int hid,int out,double lr): input_size(in),hidden_size(hid),output_size(out),learning_rate(lr){
    W1.resize(input_size,vector<double>(hidden_size));
    W2.resize(hidden_size,vector<double>(output_size));
for(int i=0;i<input_size;i++)
{
    for(int j=0;j<hidden_size;j++)
    {
        W1[i][j]=((double) rand()/RAND_MAX) -0.5;

    }


}

for( int i=0;i<hidden_size;i++)
{
    for(int j=0;j<output_size;j++)
W2[i][j]=((double)rand()/RAND_MAX)-0.5;

}
hidden.resize(hidden_size);
output.resize(output_size);

}
vector<double>forward(const vector<double>&input)
{
for(int j=0;j<hidden_size;j++)
{
    double sum=0.0;
    for(int i=0;i<input_size;i++)
    {
 sum+=input[i]*W1[i][j];
    }
hidden[j]=sigmoid(sum);
}
for(int k=0;k<output_size;k++)
{
    double sum=0.0;
for(int j=0;j<hidden_size;j++)
{
    sum+=hidden[j]*W2[j][k];
}
output[k]=sigmoid(sum);
}
return output;
}
void backward(const vector<double>& input,const vector<double>& target)
{
 vector<double> out=forward(input);
 double y=out[0];
 double t= target[0];

 double error=t-y;
 double delta_out=error*sigmoid_derivative(y);
 vector<double>delta_hidden(hidden_size);
 {
    for(int j=0;j<hidden_size;j++)
    {
        delta_hidden[j]=W2[j][0]*sigmoid_derivative(hidden[j])*delta_out;
    }
 }
for(int j=0;j<hidden_size;j++)
{
    W2[j][0]+=learning_rate*delta_out*hidden[j];//Here index 0 is there and hence there is W[j][0]  j is for second  hidden layer and 0 is output layer
}
for(int i=0;i<input_size;i++)
{
    for(int j=0;j<hidden_size;j++)
    {
        W1[i][j]+=learning_rate*delta_hidden[j]*input[i];
    }
}


}



};

int main()
{
    srand(time(0));
    NeuralNetwork nn(2,2,1,0.685);
    vector<vector<double>> X={
        {0,0},
        {0,1},
        {1,0},
        {1,1}
    };
    vector<vector<double>> Y={
        {0},
        {1},
        {1},
        {0}
    };
    for(int epoch=0;epoch<10000;epoch++)
    {
        for(int i=0;i<X.size();i++)
        {
            nn.forward(X[i]);
            nn.backward(X[i],Y[i]);
        }
    }
for(int i=0;i<X.size();i++)
{
    vector<double> out=nn.forward(X[i]);
    cout<<X[i][0]<<" XOR  "<<X[i][1]<<" =  "<<out[0]<<endl;
}
return 0;

}

