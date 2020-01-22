// System includes
#include <stdio.h>
#include <cstdlib>
#include <assert.h>
#include <random>
#include <cmath>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <time.h>

// Other c++ libraries

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include "device_launch_parameters.h"
#include <helper_cuda.h>
#include "cudaDmy.cuh"

//#define N 2000
#define PI 3.141592653
#define PREC 20
#define maxNeighbors 6
typedef double4 particle;
typedef double dbl;
typedef double3 dbl3;
typedef double2 dbl2;
//typedef float4 particle;
//typedef float2 dbl2;
//typedef float3 dbl3;
//typedef float dbl;

using namespace std;

enum string_code {
    enDim,
    enumParticles,
    ephi,
    epotentialPower,
    eisFinished,
    enone
};

string_code hashit (string const& inString) {
    if (inString == "nDim") return enDim;
    else if (inString == "numParticles") return enumParticles;
    else if (inString == "phi") return ephi;
    else if (inString == "potentialPower") return epotentialPower;
    else if (inString == "isFinished") return eisFinished;
    else return enone;
}

__device__ void calc_harm_force(dbl2 &force, particle *p, int i, int j, dbl LEshear) {
    //float2 tmp(0.0f,0.0f);
    dbl2 tmpforce;
    dbl xt = p[i].x-p[j].x;
    dbl yt = p[i].y-p[j].y;
    if (xt > .5) {xt -= 1.0; yt -= LEshear;}
    else if (xt < -.5) {xt += 1.0; yt += LEshear;}
    if (yt > .5) yt -= 1.0;
    else if (yt < -.5) yt += 1.0;
    //float mag = sqrtf((xt)*(xt)+(yt)*(yt));
    //might not need mag, wait nvm I need that
    dbl mag = sqrt((xt)*(xt)+(yt)*(yt));
    dbl forcemag = (mag - (p[i].z + p[j].z));
    tmpforce.x = xt*forcemag/mag;
    tmpforce.y = yt*forcemag/mag;
    force.x += tmpforce.x;
    force.y += tmpforce.y;
    return;
}

__device__ void step(particle *parts, dbl2 move, int i) {
    parts[i].x += move.x;
    parts[i].y += move.y;
    return;
}

template<int N>
__global__ void minimization_step(particle *parts, int neighbors[N][maxNeighbors], dbl LEshear) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    dbl2 force = make_double2(0,0);// make_float2(0,0);
    for (int j=0; j < maxNeighbors; j++) {
        if (neighbors[i][j] == -1) break;
        calc_harm_force(force, parts, i, j, LEshear);
    }
    step(parts, force, i);
    return;
}

//template<int N>
__global__ void LEShift(particle *parts, dbl LEShift) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    parts[i].y = parts[i].y + parts[i].x*LEShift;
    return;
}

template <int N>
class Packing {
public:
    // w/ LE boundary conditions
    int nParticles = N;
    particle parts[N] = {0};
    dbl phi = 0.95;
    dbl pPow = 2.5;
    dbl boxsize[2] = {2.0, 2.0};
    dbl LEshear = 0.0;
    // initialize with random positions at a fixed packing fraction
    Packing(int seed=0) {
        srand(seed);
        std::default_random_engine generator;
        std::uniform_real_distribution<double> distribution(0.0,1.0);
        dbl r = sqrt(phi/((double)N)*PI);
        for (int i=0; i < N; i++) {
            parts[i].x = distribution(generator);
            parts[i].y = distribution(generator);
            parts[i].z = r;
            parts[i].w = pPow;
        }
    }

    // initialize from a pyCudaPacking save directory
    Packing(string input_dir) {
        ifstream inFile;
        dbl value, value2;
        string svalue;
        inFile.open(input_dir+"/scalars.dat");
        if (!inFile)
        {
            cout << "\nError opening scalar file.\n";
            return;
        }
        int index = 0;
        while (inFile >> svalue >> setprecision(PREC) >> value) {

            switch (hashit(svalue))
            {
                case enDim: cout << value << ": nDim must be 2" << endl; break;
                case enumParticles: cout << value << ": numParticles must be " << N << endl; break;
                case ephi: phi = value; break;
                case epotentialPower: pPow = value; break;
                case eisFinished: cout << value << ": isFinished should be 1" << endl; break;
                default: break;
            }
            index++;
        }
        for (int i=0; i < N; i++) {
            parts[i].w = pPow;
        }
        inFile.close();
        inFile.clear();
        inFile.open(input_dir+"/boxSize.dat");
        cout << !inFile << endl;
        if (!inFile)
        {
            cout << "\nError opening boxsize file.\n";
            return;
        }
        index = 0;
        while (inFile >> setprecision(PREC) >> value) {
            boxsize[index] = value;
            index++;
        }
        inFile.close();
        inFile.clear();
        inFile.open(input_dir+"/radii.dat");
        if (!inFile)
        {
            cout << "\nError opening radii file.\n";
            return;
        }
        index = 0;
        while (inFile >> setprecision(PREC) >> value) {
            parts[index].z = value;
            index++;
        }
        inFile.close();
        inFile.clear();
        inFile.open(input_dir+"/positions.dat");
        if (!inFile)
        {
            cout << "\nError opening positions file.\n";
            return;
        }
        index = 0;
        while (inFile >> setprecision(PREC) >> value >> setprecision(PREC) >> value2) {
            parts[index].x = value;
            parts[index].y = value2;
            index++;
        }
        return;
        inFile.close();
    }

    void minimizeCuda(int neighbors[N][maxNeighbors]) {
        // allocate memory
        particle *dev_parts = 0;
        int dev_neighbors[N][maxNeighbors] {{0}};
        dbl *dev_LEshear;

        cudaSetDevice(0);
        const clock_t begin_time = clock();
        cudaMalloc((void**)&dev_parts, N * sizeof(particle));
        cudaMemcpy(dev_parts, parts, N * sizeof(particle), cudaMemcpyHostToDevice);
        cudaMalloc((void**)&dev_neighbors, N * maxNeighbors * sizeof(int));
        cudaMemcpy(dev_neighbors, neighbors, N * maxNeighbors * sizeof(int), cudaMemcpyHostToDevice);
        cudaMalloc((void**)&dev_LEshear, sizeof(dbl));
        cudaMemcpy(dev_LEshear, &LEshear, sizeof(dbl), cudaMemcpyHostToDevice);
        // execute kernel
        minimization_step<N><<<N/200,200>>>(dev_parts, dev_neighbors, LEshear);
        // copy back to host
        cudaDeviceSynchronize();
        cudaMemcpy(parts, dev_parts, N * sizeof(particle), cudaMemcpyDeviceToHost);
        std::cout << endl << "BLAHHH " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << endl; 
        cudaFree(dev_parts);
        cudaFree(dev_neighbors);
    }

    void shearInfCuda(dbl inf_LEshear) {
        // allocate memory
        particle *dev_parts = 0;

        cudaSetDevice(0);

        cudaMalloc((void**)&dev_parts, N * sizeof(particle));
        cudaMemcpy(dev_parts, parts, N * sizeof(particle), cudaMemcpyHostToDevice);
        // execute kernel
        LEShift<<<N/200,200>>>(dev_parts, inf_LEshear);
        // copy back to host
        cudaDeviceSynchronize();
        cudaMemcpy(parts, dev_parts, N * sizeof(particle), cudaMemcpyDeviceToHost);

        cudaFree(dev_parts);
        LEshear += inf_LEshear;
    }

    void shearTotalCuda(dbl new_LEshear) {
        // allocate memory
        particle *dev_parts = 0;

        cudaSetDevice(0);

        cudaMalloc((void**)&dev_parts, N * sizeof(particle));
        cudaMemcpy(dev_parts, parts, N * sizeof(particle), cudaMemcpyHostToDevice);
        // execute kernel
        LEShift<<<N/200,200>>>(dev_parts, -LEshear);
        LEShift<<<N/200,200>>>(dev_parts, new_LEshear);
        // copy back to host
        cudaDeviceSynchronize();
        cudaMemcpy(parts, dev_parts, N * sizeof(particle), cudaMemcpyDeviceToHost);

        cudaFree(dev_parts);
        LEshear = new_LEshear;
    }

    int find_neighbors(int neighbors[N][maxNeighbors]) {
        dbl3 tmp;
        int c[N] = {0};
        int max;
        dbl xdim = .5*boxsize[0];
        dbl ydim = .5*boxsize[1];
        dbl shift = LEshear*boxsize[0];
        if (LEshear == 0.0) {
            for (int i=0; i < N; i++) {
                for (int j=i+1; j < N; j++) {
                    tmp.x = parts[i].x - parts[j].x;
                    if (tmp.x > xdim) {tmp.x -= 1; tmp.y -= shift;}
                    else if (tmp.x < -xdim) {tmp.x += 1; tmp.y += shift;}
                    tmp.x *= tmp.x;
                    tmp.y = parts[i].y - parts[j].y;
                    if (tmp.y > ydim) tmp.y -= 1;
                    else if (tmp.y < -ydim) tmp.y += 1;
                    tmp.y *= tmp.y;
                    tmp.z = parts[i].z + parts[j].z;
                    if (tmp.x + tmp.y < tmp.z*tmp.z) {
                        neighbors[i][c[i]] = j;
                        neighbors[j][c[j]] = i;
                        c[i]++;
                        c[j]++;
                        if (c[i] > max) max++;
                    }
                }
            }
            return max;
        }
        else {
            for (int i=0; i < N; i++) {
                for (int j=i+1; j < N; j++) {
                    tmp.x = parts[i].x - parts[j].x;
                    if (tmp.x > xdim) {tmp.x -= 1;}
                    else if (tmp.x < -xdim) {tmp.x += 1;}
                    tmp.x *= tmp.x;
                    tmp.y = parts[i].y - parts[j].y;
                    if (tmp.y > ydim) tmp.y -= 1;
                    else if (tmp.y < -ydim) tmp.y += 1;
                    tmp.y *= tmp.y;
                    tmp.z = parts[i].z + parts[j].z;
                    if (tmp.x + tmp.y < tmp.z*tmp.z) {
                        neighbors[i][c[i]] = j;
                        neighbors[j][c[j]] = i;
                        c[i]++;
                        c[j]++;
                        if (c[i] > max) max++;
                    }
                }
            }
            return max;
        }
    }

};

int main() {
    const int N = 10000;
    Packing<N> p("/home/ian/Projects/initialize_packings/data/2d_N10000_p0.01_monoT_pPow2.0_seed0");
    cout << p.parts[0].x << " " << p.parts[0].z << " " << p.parts[0].w << " " << p.phi << endl;
    int neighbors[N][maxNeighbors] {{0}};
    for (int i=0; i < N; i++) {
        for (int j=0; j<maxNeighbors; j++) {
            neighbors[i][j] = -1;
        }
    }
    //cout << neighbors[2][1] << endl;
    int max;
    max = p.find_neighbors(neighbors);
    int count = 0;
    int Nn = N;
    for (int i=0; i < N; i++) {
        for (int j=0; j<maxNeighbors; j++) {
            cout << neighbors[i][j] << "  ";
            if (neighbors[i][j] == -1) {
                
                if (j == 0) Nn--;
                break;
            }
            count += 1;
        }
        cout << endl;
    }
    cout << max << endl; // max coordination
    cout << (float)count/(float)Nn << endl; // average coordination among non-rattlers
    cout << count << " " << Nn << " " << N << endl;
    cout << N - Nn << endl;
    clock_t begin_time = clock();
    p.shearInfCuda(1e-5);
    std::cout << "Time to shear " << float( clock () - begin_time ) /  CLOCKS_PER_SEC;
    begin_time = clock();
    p.find_neighbors(neighbors);
    std::cout << endl << "Time to find neighbors CPU " << float( clock () - begin_time ) /  CLOCKS_PER_SEC;
    begin_time = clock();
    p.minimizeCuda(neighbors);
    std::cout << endl << "Time to minimize " << float( clock () - begin_time ) /  CLOCKS_PER_SEC;

    begin_time = clock();
    p.shearInfCuda(1e-5);
    std::cout << endl << "Time to shear " << float( clock () - begin_time ) /  CLOCKS_PER_SEC;
    begin_time = clock();
    p.find_neighbors(neighbors);
    std::cout << endl << "Time to find neighbors CPU " << float( clock () - begin_time ) /  CLOCKS_PER_SEC;
    begin_time = clock();
    p.minimizeCuda(neighbors);
    std::cout << endl << "Time to minimize " << float( clock () - begin_time ) /  CLOCKS_PER_SEC;

    begin_time = clock();
    p.shearInfCuda(1e-5);
    std::cout << endl << "Time to shear " << float( clock () - begin_time ) /  CLOCKS_PER_SEC;
    begin_time = clock();
    p.find_neighbors(neighbors);
    std::cout << endl << "Time to find neighbors CPU " << float( clock () - begin_time ) /  CLOCKS_PER_SEC;
    begin_time = clock();
    p.minimizeCuda(neighbors);
    std::cout << endl << "Time to minimize " << float( clock () - begin_time ) /  CLOCKS_PER_SEC;

    begin_time = clock();
    p.shearInfCuda(1e-5);
    std::cout << endl << "Time to shear " << float( clock () - begin_time ) /  CLOCKS_PER_SEC;
    begin_time = clock();
    p.find_neighbors(neighbors);
    std::cout << endl << "Time to find neighbors CPU " << float( clock () - begin_time ) /  CLOCKS_PER_SEC;
    begin_time = clock();
    p.minimizeCuda(neighbors);
    std::cout << endl << "Time to minimize " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << endl;
    begin_time = clock();

}