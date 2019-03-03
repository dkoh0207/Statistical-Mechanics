#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
#include <vector>
#include <string>
#include "boost/multi_array.hpp"
#include "boost/tuple/tuple.hpp"
#include "boost/tuple/tuple_comparison.hpp"
#include <cassert>
#include <cmath>
#include <map>
#include <ctime>

using namespace std;
typedef boost::multi_array<int, 4> Lattice;
typedef Lattice::index coord;
typedef boost::tuple<int, int, int, int> site;

int seed = time(0);
default_random_engine random_engine(seed);

inline int periodic(int i, const int limit, int add) {
    // This implementation of periodic boundary condition is from 
    // the 2D ising model example by Hjorth Jensen.
    return (i + limit + add) % (limit);
}

void read_input(unsigned int&, unsigned int&, unsigned int&, 
double&, double&, string&, vector<double>&, string &);
double compute_free_energy(Lattice &lattice, const unsigned int lsize, 
const unsigned int n_bonds, const double & K);
void initialize(Lattice &lattice, const unsigned int lsize, const unsigned int n_bonds, 
double &K, double& E, const int &coldstart, map<int, double> & weights);
double deltaE(Lattice &lattice, site &bond,
const unsigned int lsize, const unsigned int n_bonds, const double &K);
site choose_random_bond(const unsigned int lsize, const unsigned int n_bonds);
void metropolis(Lattice &lattice, const unsigned int lsize, 
const unsigned int n_bonds, const double &K, double &E);
double compute_wilson_loop(Lattice &lattice, const unsigned int lsize, const double &K);


void read_input(unsigned int &lsize, unsigned int &num_steps, unsigned int &mcs,
  double &init_T, double &final_T, vector<double> &sweep, string &path) {
  // Function to read configurations for Monte Carlo pass.
  cout << "Enter lattice size: " << endl;
  cin >> lsize;
  cout << "Enter initial T: " << endl;
  cin >> init_T;
  cout << "Enter Final T: " << endl;
  cin >> final_T;
  cout << "Enter number of steps: " << endl;
  cin >> num_steps;
  cout << "Enter MCS per site: " << endl;
  cin >> mcs;
  cout << "Enter output path: " << endl;
  cin >> path;
  double T_step = (final_T - init_T) / ((double) num_steps);
  double t = {0.0};
  for (auto i = 0; i != num_steps; ++i) {
    t = init_T + T_step * (double) i;
    sweep.push_back(t);
  }
  cout << "Temperature Sweep Range:" << endl;
  for (auto k : sweep) {
    cout << "T = " << k << endl;
  }
}


double compute_free_energy(Lattice &lattice, 
const unsigned int lsize, const unsigned int n_bonds, const double & K) {

    double E_temp = 0.0;
    double plaquettes = 0.0;

    for (auto i = 0; i != lsize; ++i) {
        for (auto j = 0; j != lsize; ++j) {
            for (auto k = 0; k != lsize; ++k) {
                plaquettes = 0.0;
                // Square plaquette in the (1,2) direction. 
                plaquettes += (double) (lattice[i][j][k][0] * lattice[i][j][k][1]
                * lattice[periodic(i, lsize, 1)][j][k][1] 
                * lattice[i][periodic(j, lsize, 1)][k][0]);
                // Square plaquette in the (2,3) direction.
                plaquettes += (double) (lattice[i][j][k][1] * lattice[i][j][k][2]
                * lattice[i][periodic(j, lsize, 1)][k][2]
                * lattice[i][j][periodic(k, lsize, 1)][1]);
                // Square plaquette in the (1,3) direction.
                plaquettes += (double) (lattice[i][j][k][0] * lattice[i][j][k][2]
                * lattice[periodic(i, lsize, 1)][j][k][2]
                * lattice[i][j][periodic(k, lsize, 1)][0]);
                E_temp += plaquettes;
            }
        }
    }
    return E_temp * K;
}


void initialize(Lattice &lattice, const unsigned int lsize, const unsigned int n_bonds, 
double &K, double& E, const int &coldstart, map<int, double> &weights) {
    /*
    Function for initializing all bonds in the lattice.
    coldstart = 1 gives all spins in +1 config, while 
    coldstart = -1 gives random spin config with +1 or -1. 
    */
    static uniform_int_distribution<int> u(0,1);

    // Initialize bonds on lattice
    for (auto i = 0; i != lsize; ++i) {
        for (auto j = 0; j != lsize; ++j) {
            for (auto k = 0; k != lsize; ++k) {
                for (auto b = 0; b != n_bonds; ++b) {
                    if (u(random_engine) == 1) {
                        lattice[i][j][k][b] = 1;
                    } else {
                        lattice[i][j][k][b] = coldstart;
                    }
                }
            }
        }
    }
    // Initializing weights for metropolis
    weights.insert(pair<int, double>(-12, exp(-12.0 * K)));
    weights.insert(pair<int, double>(-8, exp(-8.0 * K)));
    weights.insert(pair<int, double>(-4, exp(-4.0 * K)));
    weights.insert(pair<int, double>(4, 1.0));
    weights.insert(pair<int, double>(8, 1.0));
    weights.insert(pair<int, double>(12, 1.0));

    // Compute free energy of the whole lattice
    E = compute_free_energy(lattice, lsize, n_bonds, K);
    cout << "Initial E = " << E << endl;
}

double deltaE(Lattice &lattice, site &bond,
const unsigned int lsize, const unsigned int n_bonds, const double &K) {
    double dE = 0.0;
    double plaq_sum = 0.0;
    vector<int> ind(3);
    int b = 0;
    ind[0] = bond.get<0>();
    ind[1] = bond.get<1>();
    ind[2] = bond.get<2>();
    b  = bond.get<3>();
    for (int m = 0; m != n_bonds; ++m) {
        double plaq = 0.0;
        plaq = 0;
        // This particular implementation of computing the energy
        // is borrowed from Michael Cruetz's z2 lattice gauge simulation.
        if (m != b) {
            plaq = lattice[ind[0]][ind[1]][ind[2]][b];
            ind[m] = periodic(ind[m], lsize, -1);
            plaq *= lattice[ind[0]][ind[1]][ind[2]][m] * 
            lattice[ind[0]][ind[1]][ind[2]][b];
            ind[b] = periodic(ind[b], lsize, 1);
            plaq *= lattice[ind[0]][ind[1]][ind[2]][m];
            plaq_sum += plaq;
            ind[m] = periodic(ind[m], lsize, 1);
            plaq = lattice[ind[0]][ind[1]][ind[2]][m];
            ind[m] = periodic(ind[m], lsize, 1);
            ind[b] = periodic(ind[b], lsize, -1);
            plaq *= lattice[ind[0]][ind[1]][ind[2]][b];
            ind[m] = periodic(ind[m], lsize, -1);
            plaq *= lattice[ind[0]][ind[1]][ind[2]][m] * 
            lattice[ind[0]][ind[1]][ind[2]][b];
            plaq_sum += plaq;
        }
    }
    dE = -2 * plaq_sum;
    return dE;
}

site choose_random_bond(const unsigned int lsize, const unsigned int n_bonds) {
    // Helper function for choosing random site.
    static uniform_int_distribution<int> u(0, lsize-1);
    static uniform_int_distribution<int> bond(0, n_bonds-1);
    int idx, idy, idz, b = {0};
    idx = u(random_engine);
    idy = u(random_engine);
    idz = u(random_engine);
    b = bond(random_engine);
    site s(idx, idy, idz, b);
    return s;
}

void metropolis(Lattice &lattice, const unsigned int lsize, const unsigned int n_bonds,
const double &K, double &E, map<int, double> &weights) {
    unsigned int count_flipped = 0;
    unsigned int N = lsize*lsize*lsize*n_bonds;
    //unsigned int N = 1;
    double w = 0.0;
    static uniform_real_distribution<double> u(0,1);
    double dE = 0.0;
    for (auto i = 0; i != N; ++i) {
        int key = 0;
        site s = choose_random_bond(lsize, n_bonds);
        dE = deltaE(lattice, s, lsize, n_bonds, K);
        w = weights[dE];
        cout << "w = " << endl;
        cout << "exp(dE) = " << endl;
        //cout << "dE = " << dE << endl;
        if (u(random_engine) < w) {
            //cout << "Flipped" << endl;
            E += dE;
            count_flipped += 1;
            lattice[s.get<0>()][s.get<1>()][s.get<2>()][s.get<3>()] *= -1;
        }
    }
}

double compute_wilson_loop(Lattice &lattice, const unsigned int lsize, const double &K) {

    double wilson_loop = 1.0;

    for (auto i = 0; i != lsize-1; ++i) {
        wilson_loop *= lattice[i][0][0][0];
    }
    for (auto i = 0; i != lsize-1; ++i) {
        wilson_loop *= lattice[lsize-1][i][0][1];
    }
    for (auto i = lsize-1; i != -1; --i) {
        wilson_loop *= lattice[i][0][0][0];
    }
    for (auto i = lsize-1; i != -1; --i) {
        wilson_loop *= lattice[0][i][0][0];
    }

    return wilson_loop;
}

void write_data(const unsigned int mcs, const unsigned int lsize, 
const unsigned int n_bonds, const string& output_filename,
Lattice& lattice, double& K, double& E, map<int, double> &weights) {
  // Free Energy per site
  vector<double> vec_E;
  vector<double> vec_E2;
  double m = 0.0;
  double energy = 0.0;
  int n = lsize*lsize*lsize*n_bonds;
  for (auto i = 0; i != mcs; ++i) {
    if (i % 1000 == 0) {
      cout << "i = " << i << endl;
      cout << "M = " << m << endl;
      //test_update(lattice, lsize);
    }
    metropolis(lattice, lsize, n_bonds, K, E, weights);
    energy = E / ((double) n);
    vec_E.push_back(energy);
    vec_E2.push_back(energy * energy);
  }
  ofstream result;
  result.open(output_filename);
  result << "E, E2" << endl;
  for (auto i = 0; i != mcs; ++i) {
    result << setprecision(8) << vec_E[i] << "," << vec_E2[i] << endl;
  }
}

int main() {

    // Initializing Step
    unsigned int lsize, num_steps, start = {0};
    double init_K, final_K, M, E, step = {0.0};
    double K, T = 0.0;
    K = 0.5;
    lsize = 10;
    unsigned int N = lsize * lsize * lsize;
    string output_filename;
    vector<double> sweep;
    unsigned int mcs = 2000; // NUmber of Monte Carlo steps per site.
    unsigned int n_bonds = 3;
    read_input(lsize, num_steps, start, init_K, 
    final_K, output_filename, sweep, output_filename);

    // Set the lattice
    Lattice lattice(boost::extents[lsize][lsize][lsize][n_bonds]);
    cout << "Lattice size = " << lattice.size() << endl;

    // Set output file configurations
    ofstream readme;
    string s_readme = "./" + output_filename + "/readme.txt";
    readme.open(s_readme);
    readme << "Number of MCS: " << mcs << endl;
    readme << "Lattice Size: " << lsize << endl;
    readme << "K,T" << endl;

    int coldstart = 1;
    for (auto i = 0; i != num_steps; ++i) {
      T = sweep[i];
      K = 1.0 / T;
      cout << "K = " << K << endl;
      cout << "T = " << T << endl;
      readme << setprecision(8) << K << ",";
      readme << setprecision(8) << T << endl;

      for (coldstart = -1; coldstart < 2; coldstart += 2) {
        cout << "Coldstart: " << coldstart << endl;
        map<int, double> weights;
        initialize(lattice, lsize, n_bonds, K, E, coldstart, weights);
        string ofile;
        string ind = to_string(i);
        ofile = "./" + output_filename + "/";
        if (coldstart == -1){
          ofile += "hot_";
        } else {
          ofile += "cold_";
        }
        ofile = ofile + ind + ".csv";
        write_data(mcs, lsize, n_bonds, ofile, lattice, K, E, weights);
      }
    }

    /*
    initialize(lattice, lsize, n_bonds, K, E, coldstart);
    unsigned int mcs = 200;
    double E_test = 0;
    for (auto i = 0; i != mcs; ++i) {
        metropolis(lattice, lsize, n_bonds, K, E);
        cout << "E = " << E / ((double) lsize*lsize*lsize*3) << endl;
        cout << "W = " << compute_wilson_loop(lattice, lsize, K) << endl;
        //cout << "E_text = " << compute_free_energy(lattice, lsize, n_bonds, K) << endl;
    }
    */
    return 0;
}