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
#include <queue>
#include <set>

using namespace std;
ofstream ofile;
// The boost linear algebra C++ library is used to implement
// multidimensional arrays and tuples efficiently and reliably.
typedef boost::multi_array<int, 3> Lattice;
typedef Lattice::index coord;
typedef boost::tuple<int, int, int > site;

// Set a different seed for the random number generator
// at every run using the current time.
random_device rd;
// The Mersenne Twister 19937 Random Number Generator
mt19937_64 mt(rd());

inline int periodic(int i, const int limit, int add) {
  // This implementation of periodic boundary condition is from
  // the 2D ising model example by Hjorth Jensen.
  return (i + limit + add) % (limit);
}


// ======================Function Declarations======================

void read_input(unsigned int&, unsigned int&, unsigned int&, double&, double&, string&, vector<double>&);
void initialize(Lattice&, map<int, double>&, double&, double&, double&);
site choose_random_site(const unsigned int lsize);
void metropolis();
int deltaE(site s, Lattice& lattice, const unsigned int lsize, const double& K);
void wolff(unsigned int mcs, Lattice &lattice, const unsigned int lsize, const double K,
double& M, double& E);
vector<site> find_cluster(Lattice &lattice, const unsigned int lsize, const double K);
vector<site> find_neighbors(const site, const unsigned int);
void test_update(Lattice& lattice, const unsigned int lsize);


// ======================Function Implementations======================

void read_input(unsigned int &lsize, unsigned int &num_steps, unsigned int &mcs,
  double &init_T, double &final_T, vector<double> &sweep, string &path) {
  /*
  Function to read initial configurations of MC simulation.
  Inputs: 
    lsize: Linear dimension of the lattice
    num_steps: Number of steps between initial T and final T
    used in temeperature sweep
    mcs: Number of Monte Carlo steps per site
    init_T: intial temperature
    final_T: final temperature
    sweep: vector containing values for each temperature used
    during sweep
    path: path to output directory containing MC history files. 

    Returns:
      None (a void function)
  */
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
  // For display to terminal during debugging and monitoring.
  cout << "Temperature Sweep Range:" << endl;
  for (auto k : sweep) {
    cout << "T = " << k << endl;
  }
}


void initialize( Lattice &lattice, map<int, double> &weights, double &K,
double &M, double &E, const int &coldstart) {
  /*
  Function for setting initial state, Boltzmann weights, and observables.

  Inputs: 
    lattice: A 3D multidimensional boost array
    weights: A dictionary (ref to std::map) for saving precalculated weights. 
    K: The value of the current temperature. 
    M: (double) Magnetization
    E: (double) Energy
    coldstart: 1 for setting all sites = 1, -1 for setting all sites randomly 
    between -1 and 1. 
  
  Returns: 
    None (a void function)
  */
  static uniform_int_distribution<int> u(0,1);
  unsigned int lsize = 0;
  lsize = lattice.size();
  // Refresh Magnetization and Energy
  M = 0;
  E = 0;
  // Set initial state for new MC run
  for (auto i = 0; i != lsize; ++i) {
    for (auto j = 0; j != lsize; ++j) {
      for (auto k = 0; k != lsize; ++k) {
          if (u(mt) == 1) {
            lattice[i][j][k] = 1;
          } else {
            lattice[i][j][k] = coldstart;
          }
          // Adjust magnetization accordingly
          M += (double) lattice[i][j][k];
      }
    }
  }
  // After initializing lattice, compute initial energy:
  for (auto i = 0; i != lsize; ++i) {
    for (auto j = 0; j != lsize; ++j) {
      for (auto k = 0; k != lsize; ++k) {
          E += lattice[i][j][k] * (
              lattice[periodic(i, lsize, 1)][j][k] +
              lattice[i][periodic(j, lsize, 1)][k] +
              lattice[i][j][periodic(k, lsize, 1)]);
      }
    }
  }
  /*
  Initializing boltzmann weights to be used in each metropolis step.
  This allows us to avoid computing exp(dE) at every step, which is
  computationally expensive due to the exponential function.
  */
  weights.insert(pair<int, double>(-12, exp(-12.0 * K)));
  weights.insert(pair<int, double>(-8, exp(-8.0 * K)));
  weights.insert(pair<int, double>(-4, exp(-4.0 * K)));
  weights.insert(pair<int, double>(0, 1.0));
  weights.insert(pair<int, double>(4, 1.0));
  weights.insert(pair<int, double>(8, 1.0));
  weights.insert(pair<int, double>(12, 1.0));

  // We check the initial magnetization (per site) and energy.
  cout << "Initial M = " << M / ((double) lsize*lsize*lsize) << endl;
  cout << "Initial E = " << E / ((double) lsize*lsize*lsize) << endl;
}


site choose_random_site(const unsigned int lsize) {
  /*
  A helper function for choosing a random site on the lattice, 
  using periodic boundary conditions.

  Inputs: 
    lsize: Linear dimension of the lattice. 

  Returns: 
    s (site): A randomly selected site object (3D tuple) from the lattice. 
    A site object consists of three coordinate indices. 
  */
  static uniform_int_distribution<int> u(0, lsize-1);
  int idx, idy, idz = {0};
  // Call three random integers, which correspond to coordinates on the lattice.
  idx = u(mt);
  idy = u(mt);
  idz = u(mt);
  site s(idx, idy, idz);
  return s;
}

vector<site> find_neighbors(const site s, const unsigned int lsize) {
  /*
  A helper function for finding the six neighboring sites, given
  a specific site on the lattice.

  Inputs: 
    site s: A site object indicating the current site.
    lsize: Linear dimension of the lattice

  Returns:
    neighbors: A std::vector of sites containing the site objects 
    for the six neighbors of site s. 
  */
  vector<site> neighbors;
  int idx, idy, idz = {0};
  idx = s.get<0>();
  idy = s.get<1>();
  idz = s.get<2>();
  for (auto i = 0; i != 3; ++i) {
    if (i == 0) {
      site n1(periodic(idx, lsize, 1), idy, idz);
      site n2(periodic(idx, lsize, -1), idy, idz);
      neighbors.push_back(n1);
      neighbors.push_back(n2);
    }
    if (i == 1) {
      site n1(idx, periodic(idy, lsize, 1), idz);
      site n2(idx, periodic(idy, lsize, -1), idz);
      neighbors.push_back(n1);
      neighbors.push_back(n2);
    }
    if (i == 2) {
      site n1(idx, idy, periodic(idz, lsize, 1));
      site n2(idx, idy, periodic(idz, lsize, -1));
      neighbors.push_back(n1);
      neighbors.push_back(n2);
    }
  }
  return neighbors;
}


int deltaE(site s, Lattice& lattice, const unsigned int lsize, const double& K) {
  /*
  Helper function for computing the change in free energy that would
  occur if a site is to be flipped. 

  Inputs: 
    site s: The current site in consideration
    lattice: 3D multidimensional boost array modeling the lattice.
    lsize: Linear dimension of the lattice.
    K: Current temperature

  Returns:
    dE: (double) The change in free energy E_f - E_i. 
    The convention is that E > 0, so that the free energy is 
    actually given by -E. 
  */
  int dE = 0;
  vector<site> neighbors = find_neighbors(s, lsize);
  for (auto n : neighbors) {
    dE += lattice[n.get<0>()][n.get<1>()][n.get<2>()];
  }
  dE = dE * lattice[s.get<0>()][s.get<1>()][s.get<2>()];
  dE = -dE * 2;
  return dE;
}

void metropolis(Lattice &lattice, const unsigned int lsize,
const double &K, double& M, double& E, map<int, double>& weights) {
  /*
  Implements the Metropolis Algorithm

  Inputs: 
    lattice: A 3D Boost multidimensional array modeling the Ising Lattice.
    lsize: Linear dimension of the lattice.
    K: Inverse temperature (reference)
    M: Magnetization (reference)
    E: Free Energy (reference)
    weights: A c++ map (key-value pair) use to efficiently calculate
    the Boltzmann weights.
  
  Returns:
    None (a void function)

  */
  unsigned int count_flipped = 0;
  unsigned int n = lsize*lsize*lsize;
  double w = 0.0;
  // The static qualifier is to ensure that each distribution is initialized
  // only once throughout the program. 
  static uniform_real_distribution<double> u(0,1);
  int dE = 0;
  for (auto i = 0; i != n; ++i) {
    int key = 0;
    site s = choose_random_site(lsize);
    dE = deltaE(s, lattice, lsize, K);
    w = weights[dE];
    if (u(mt) < w) {
      count_flipped += 1;
      lattice[s.get<0>()][s.get<1>()][s.get<2>()] *= -1;
      M +=  2 * lattice[s.get<0>()][s.get<1>()][s.get<2>()];
      E += ((double) dE);
    }
  }
}

void test_update(Lattice& lattice, const unsigned int lsize) {
  // Testing if correct M and E
  double testM = 0.0;
  double testE = 0.0;
  int n = lsize*lsize*lsize;
  for (auto i2 = 0; i2 != lsize; ++i2) {
      for (auto j2 = 0; j2 != lsize; ++j2) {
          for (auto k2 = 0; k2 != lsize; ++k2) {
              testM += (double) lattice[i2][j2][k2];
          }
      }
  }
  for (auto i2 = 0; i2 != lsize; ++i2) {
      for (auto j2 = 0; j2 != lsize; ++j2) {
          for (auto k2 = 0; k2 != lsize; ++k2) {
              testE += (double) lattice[i2][j2][k2] * (
                  lattice[periodic(i2, lsize, 1)][j2][k2] +
                  lattice[i2][periodic(j2, lsize, 1)][k2] +
                  lattice[i2][j2][periodic(k2, lsize, 1)]
              );
          }
      }
  }
  cout << "testM = " << testM / (double) n << endl;
  cout << "testE = " << testE / (double) n << endl;
}


vector<site> find_cluster(Lattice &lattice, const unsigned int lsize, const double K) {
  static uniform_real_distribution<double> u(0,1);
  double r = {0.0};
  site s = choose_random_site(lsize);
  queue<site> q;
  vector<site> added;
  q.push(s);
  added.push_back(s);
  while (!q.empty()) {
    s = q.front();
    q.pop();
    vector<site> neighbors = find_neighbors(s, lsize);
    for (auto n : neighbors) {
      double spin_n = (double) lattice[n.get<0>()][n.get<1>()][n.get<2>()];
      double spin_s = (double) lattice[s.get<0>()][s.get<1>()][s.get<2>()];
      double prob = 1 - exp( -2 * K * spin_n * spin_s);
      r = u(mt);
      if (find(added.begin(), added.end(), n) == added.end() && r < prob) {
        q.push(n);
        added.push_back(n);
      }
    }
  }
  return added;
}

void wolff(unsigned int mcs, Lattice &lattice, const unsigned int lsize, const double K,
double& M, double& E) {
  double dE = 0.0;
  for (auto i = 0; i != mcs; ++i) {
    vector<site> clusters = find_cluster(lattice, lsize, K);
    cout << "Cluster size: " << clusters.size() << endl;
    for (auto s : clusters) {
        dE = deltaE(s, lattice, lsize, K);
        lattice[s.get<0>()][s.get<1>()][s.get<2>()] *= -1;
        M += 2 * lattice[s.get<0>()][s.get<1>()][s.get<2>()];
        E += dE;
    }
  }
  cout << "M = " << M << endl;
}

void write_data(const unsigned int mcs, const unsigned int lsize, const string& output_filename,
Lattice& lattice, double& K, double& M, double& E, map<int, double> &weights) {
  // Magnetization per site
  vector<double> vec_M;
  vector<double> vec_M2;
  // Free Energy per site
  vector<double> vec_E;
  vector<double> vec_E2;
  vector<double> vec_absM;
  double m = 0.0;
  double energy = 0.0;
  int n = lsize*lsize*lsize;
  for (auto i = 0; i != mcs; ++i) {
    if (i % 1000 == 0) {
      cout << "i = " << i << endl;
      cout << "M = " << m << endl;
      cout << "E = " << energy << endl;
      test_update(lattice, lsize);
    }
    metropolis(lattice, lsize, K, M, E, weights);
    m = M / ((double) n);
    energy = E / ((double) n);
    vec_M.push_back(m);
    vec_M2.push_back(m * m);
    vec_E.push_back(energy);
    vec_E2.push_back(energy * energy);
    vec_absM.push_back(abs(m));
  }
  ofstream result;
  result.open(output_filename);
  result << "M, M2, E, E2, abs(M)" << endl;
  for (auto i = 0; i != mcs; ++i) {
    result << setprecision(8) << vec_M[i] << "," << vec_M2[i] << ","
    << vec_E[i] << "," << vec_E2[i] << "," << vec_absM[i] << endl;
  }
}

int main() {

    // Initialize Variables
    unsigned int lsize, num_steps, start = {0};
    double init_K, final_K, M, E, step = {0.0};
    double K, T = 0.0;
    K = 0.5;
    lsize = 10;
    unsigned int N = lsize * lsize * lsize;
    string output_filename;
    vector<double> sweep;
    unsigned int mcs = 2000; // NUmber of Monte Carlo steps per site.
    read_input(lsize, num_steps, mcs, init_K, final_K, sweep, output_filename);
    Lattice lattice(boost::extents[lsize][lsize][lsize]);
    cout << "Done Initializing" << endl;

    // Set output file configurations
    ofstream readme;
    string s_readme = "./" + output_filename + "/readme.txt";
    readme.open(s_readme);
    readme << "Number of MCS: " << mcs << endl;
    readme << "Lattice Size: " << lsize << endl;
    readme << "K,T" << endl;

    // To determine thermalization, we must run our simulation
    // under different initial conditions and see if they flatten
    // to the same values (away from K_c).
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
        initialize(lattice, weights, K, M, E, coldstart);
        string ofile;
        string ind = to_string(i);
        ofile = "./" + output_filename + "/";
        if (coldstart == -1){
          ofile += "hot_";
        } else {
          ofile += "cold_";
        }
        ofile = ofile + ind + ".csv";
        write_data(mcs, lsize, ofile, lattice, K, M, E, weights);
        cout << endl;
      }
    }
    /*
    wolff(100000, lattice, lsize, K, M, E);
    double testM, testE = {0.0};
    for (auto i = 0; i != lsize; ++i) {
        for (auto j = 0; j != lsize; ++j) {
            for (auto k = 0; k != lsize; ++k) {
                testM += (double) lattice[i][j][k];
            }
        }
    }
    for (auto i = 0; i != lsize; ++i) {
        for (auto j = 0; j != lsize; ++j) {
            for (auto k = 0; k != lsize; ++k) {
                testE -= (double) lattice[i][j][k] * (
                    lattice[periodic(i, lsize, 1)][j][k] +
                    lattice[i][periodic(j, lsize, 1)][k] +
                    lattice[i][j][periodic(k, lsize, 1)]
                );
            }
        }
    }
    cout << "testM = " << testM / (double) N << endl;
    cout << "testE = " << testE / (double) N << endl;
    */
    return 0;
}
