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
#include <ctime>
#include <set>

using namespace std;
ofstream ofile;
typedef boost::multi_array<int, 3> Lattice;
typedef Lattice::index coord;
typedef boost::tuple<int, int, int > site;

int seed = time(0);
default_random_engine random_engine(seed);

inline int periodic(int i, const int limit, int add) {
  // This implementation of periodic boundary condition is from
  // the 2D ising model example by Hjorth Jensen.
  return (i + limit + add) % (limit);
}

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

void read_input(unsigned int &lsize, unsigned int &num_steps, unsigned int &mcs,
  double &init_T, double &final_T, vector<double> &sweep) {
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

void initialize( Lattice &lattice, map<int, double> &weights, double &K,
double &M, double &E, const int &coldstart) {
  static uniform_int_distribution<int> u(0,1);
  unsigned int lsize = 0;
  lsize = lattice.size();
  M = 0;
  E = 0;
  for (auto i = 0; i != lsize; ++i) {
    for (auto j = 0; j != lsize; ++j) {
      for (auto k = 0; k != lsize; ++k) {
          // Call random number generator and initialize lattice
          if (u(random_engine) == 1) {
            lattice[i][j][k] = 1;
          } else {
            lattice[i][j][k] = coldstart;
          }
          // Adjust magnetization accordingly
          M += (double) lattice[i][j][k];
      }
    }
  }
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
  // Initializing weights for metropolis
  weights.insert(pair<int, double>(-12, exp(-12.0 * K)));
  weights.insert(pair<int, double>(-8, exp(-8.0 * K)));
  weights.insert(pair<int, double>(-4, exp(-4.0 * K)));
  weights.insert(pair<int, double>(4, 1.0));
  weights.insert(pair<int, double>(8, 1.0));
  weights.insert(pair<int, double>(12, 1.0));

  cout << "Initial M = " << M / ((double) lsize*lsize*lsize) << endl;
  cout << "Initial E = " << E / ((double) lsize*lsize*lsize) << endl;
}

site choose_random_site(const unsigned int lsize) {
  // Helper function for choosing random site.
  static uniform_int_distribution<int> u(0, lsize-1);
  int idx, idy, idz = {0};
  idx = u(random_engine);
  idy = u(random_engine);
  idz = u(random_engine);
  site s(idx, idy, idz);
  return s;
}

vector<site> find_neighbors(const site s, const unsigned int lsize) {
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
  int dE = 0;
  vector<site> neighbors = find_neighbors(s, lsize);
  for (auto n : neighbors) {
    //cout << n.get<0>() << "," << n.get<1>() << "," << n.get<2>() << endl;
    dE += lattice[n.get<0>()][n.get<1>()][n.get<2>()];
  }
  dE = dE * lattice[s.get<0>()][s.get<1>()][s.get<2>()];
  dE = -dE * 2;
  return dE;
}

void metropolis(Lattice &lattice, const unsigned int lsize,
const double &K, double& M, double& E, map<int, double>& weights) {
  unsigned int count_flipped = 0;
  unsigned int n = lsize*lsize*lsize;
  double w = 0.0;
  static uniform_real_distribution<double> u(0,1);
  int dE = 0;
  for (auto i = 0; i != n; ++i) {
    int key = 0;
    site s = choose_random_site(lsize);
    dE = deltaE(s, lattice, lsize, K);
    w = weights[dE];
    /*
    if (u(random_engine) < exp(dE)) {
      count_flipped += 1;
      lattice[s.get<0>()][s.get<1>()][s.get<2>()] *= -1;
      M +=  2 * lattice[s.get<0>()][s.get<1>()][s.get<2>()];
      E -= dE;
    }
    */
    if (u(random_engine) < w) {
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
      r = u(random_engine);
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
    read_input(lsize, num_steps, mcs, init_K, final_K, sweep);
    Lattice lattice(boost::extents[lsize][lsize][lsize]);
    cout << "Done Initializing" << endl;

    // Set output file configurations
    ofstream readme;
    readme.open("./output_30/readme.txt");
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
        ofile = "./output_30/";
        if (coldstart == -1){
          ofile += "hot_";
        } else {
          ofile += "cold_";
        }
        ofile = ofile + ind + ".csv";
        write_data(mcs, lsize, ofile, lattice, K, M, E, weights);
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
