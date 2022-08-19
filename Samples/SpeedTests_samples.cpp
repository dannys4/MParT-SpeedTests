/**
TODO: ADD DESCRIPTION
 */


#include <random>
#include <fstream>
#include <stdio.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <Kokkos_Core.hpp>

#include <MParT/ConditionalMapBase.h>
#include <MParT/MapFactory.h>
#include <MParT/MultiIndices/MultiIndexSet.h>
#include <MParT/Utilities/ArrayConversions.h>

#include <random>
#include <chrono>

using namespace std::chrono;

using namespace mpart; 

template<typename MemorySpace>
struct Generator{

    const unsigned int seed = 1000;

    Generator(){
        std::srand(seed);
    }

    Kokkos::View<double*,MemorySpace> Coeffs(unsigned int numCoeffs);
    Kokkos::View<double**, MemorySpace> Points(unsigned int dim, unsigned int numPts);
};

template<>
Kokkos::View<double*,Kokkos::HostSpace> Generator<Kokkos::HostSpace>::Coeffs(unsigned int numCoeffs)
{
    Eigen::VectorXd coeffs_eig = 0.5*Eigen::VectorXd::Random(numCoeffs);
    return VecToKokkos<double>(coeffs_eig);
}

template<>
Kokkos::View<double**,Kokkos::HostSpace> Generator<Kokkos::HostSpace>::Points(unsigned int dim, unsigned int numPts)
{
    Eigen::RowMatrixXd pts_eig = 0.8*Eigen::RowMatrixXd::Random(dim,numPts);
    return MatToKokkos<double>(pts_eig);
}

#if defined(KOKKOS_ENABLE_CUDA )
template<typename MemorySpace>
Kokkos::View<double*,MemorySpace> Generator<MemorySpace>::Coeffs(unsigned int numCoeffs)
{
    Eigen::VectorXd coeffs_eig = 0.5*Eigen::VectorXd::Random(numCoeffs);
    auto coeffs = VecToKokkos<double>(coeffs_eig);
    return ToDevice<MemorySpace,double>(coeffs);
}

template<typename MemorySpace>
Kokkos::View<double**,MemorySpace> Generator<MemorySpace>::Points(unsigned int dim, unsigned int numPts)
{
    Eigen::MatrixXd pts_eig = 0.8*Eigen::MatrixXd::Random(dim,numPts);
    auto pts = MatToKokkos<double>(pts_eig);
    return ToDevice<MemorySpace,double>(pts);
}

#endif 

int main(int argc, char* argv[]){

    assert(argc>=2);
    std::string backend = argv[1];
    
    Kokkos::InitArguments args;
    if(argc==3)
        args.num_threads = std::stoi(argv[2]);
    
    Kokkos::initialize(args);
    {
    MapOptions opts;

    unsigned int nn = 7;

    Generator<Kokkos::DefaultExecutionSpace::memory_space> gen;

    auto tEvalMat_m = Eigen::VectorXd(nn);
    auto tLogDetMat_m = Eigen::VectorXd(nn);
    auto tEvalMat_s = Eigen::VectorXd(nn);
    auto tLogDetMat_s = Eigen::VectorXd(nn);

    unsigned int nk = 50;

    auto tEval = Eigen::MatrixXd(nn,nk);
    auto tLogDet = Eigen::MatrixXd(nn,nk);
    
    for (unsigned int n=0; n<nn;++n){
        unsigned int dim = 5;
        unsigned int order = 2;
        
        auto map = MapFactory::CreateTriangular<Kokkos::DefaultExecutionSpace::memory_space>(dim,dim,order,opts);
        
        auto numCoeffs = map->numCoeffs;
        unsigned int numPts = pow(10,(6-n));

        // auto tEval = Eigen::VectorXd(nk);
        // auto tLogDet = Eigen::VectorXd(nk);

        for(unsigned int k=0; k<nk;++k){
            
            Kokkos::View<double*> coeffs = gen.Coeffs(numCoeffs);            
            map->SetCoeffs(coeffs); 

            Kokkos::View<double**> pts = gen.Points(dim,numPts);
          
            auto start1 = high_resolution_clock::now();
            auto evals = map->Evaluate(pts);
            auto stop1 = high_resolution_clock::now();
            auto duration1 = duration_cast<microseconds>(stop1 - start1);
            tEval(n,k)=duration1.count();

            auto start2 = high_resolution_clock::now();
            auto logDet = map->LogDeterminant(pts);
            auto stop2 = high_resolution_clock::now();
            auto duration2 = duration_cast<microseconds>(stop2 - start2);
            tLogDet(n,k)=duration2.count();
        }
        // tEvalMat_m(n)=tEval.mean();
        // tLogDetMat_m(n)=tLogDet.mean();
        // tEvalMat_s(n)=std::sqrt((tEval - tEval.mean()).square().sum()/(tEval.size()-1));
        // tLogDetMat_s(n)=std::sqrt((tLogDet - tLogDet.mean()).square().sum()/(tLogDet.size()-1));
        std::cout<<n<<std::endl;
    }
    
    {
        std::stringstream filename;
        filename << "ST_CPP_eval_d5_to2_" << backend << ".txt";

        std::ofstream file1(filename.str());  
        if(file1.is_open())  // si l'ouverture a réussi
        {   
        file1 << tEval << "\n";
        }
    }

    {
        std::stringstream filename;
        filename << "ST_CPP_logdet_d5_to2_" << backend << ".txt";

        std::ofstream file2(filename.str());  
        if(file2.is_open())  // si l'ouverture a réussi
        {   
        file2 << tLogDet << "\n";
        }
    }
    

    }
    std::cout<<"FINI !!!"<<std::endl;
    Kokkos::finalize();
	
    return 0;
}