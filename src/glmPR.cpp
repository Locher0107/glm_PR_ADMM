// #include <Rcpp.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<algorithm>

#include <RcppEigen.h>
//[[Rcpp::depends(RcppEigen)]]

// #include<DataStd.h>

using namespace Rcpp;
using namespace Eigen;

using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::ArrayXf;
using Eigen::ArrayXd;
using Eigen::ArrayXXf;

typedef Eigen::SparseVector<float> SpVec;
typedef Eigen::SparseMatrix<float> SpMat;


inline double norm2sum(VectorXd a){
  double sum=0;
  for(int i=0;i<a.rows();i++){
    sum=sum+pow(a(i),2.0);
  }
  return sum;
}
inline double eigenvec_sum(VectorXd a){
  double sum=0;
  for(int i=0;i<a.rows();i++){
    sum=sum+a(i);
  }
  return sum;
}

//for test
void print_table(MatrixXd matr){
  int nrow=matr.rows(), ncol=matr.cols();
  for(int i=0;i<nrow;i++){
    for(int j=0;j<ncol;j++){
      printf("%.4lf ",matr(i,j));
    }
    printf("\n");
  }
  return ;
}

// [[Rcpp::export]]
RcppExport SEXP glmPR(SEXP x_, SEXP y_, SEXP lambda_,SEXP rho_)
{
  BEGIN_RCPP

  //reading parameter
  Rcpp::NumericMatrix xx(x_);
  Rcpp::NumericVector yy(y_);
  //int yrow=yy.size();

  MatrixXd X_temp=as<MatrixXd>(xx);
  int Xnrow=X_temp.rows(), Xncol=X_temp.cols();
  MatrixXd X(Xnrow,Xncol+1);
  //add 1 to X
  for(int i=0;i<Xnrow;i++){
    X(i,0)=1;
    for(int j=0;j<Xncol;j++){
      X(i,j+1)=X_temp(i,j);
    }
  }
  Xncol=Xncol+1;
  //print_table(X);





  VectorXd Y=as<VectorXd>(yy);
  //printf("Vec Y is %d\n",Y.rows());



  double lambda=as<double>(lambda_);
  double rho=as<double>(rho_);
  //printf("Lambda is %lf\n",lambda);
  //printf("Rho is %lf\n",rho);


  //return List::create(Named("lambda")=lambda);

  //initializing parameter
  Rcpp::NumericVector b_gd_temp=rnorm(Xncol,0,1);
  Eigen::VectorXd b_gd=as<VectorXd>(b_gd_temp);
  for(int i=0;i<Xncol;i++){
    printf("%.4f ",b_gd(i));

  }
  printf("\n\n\n**********************\n");

  NumericVector z_admm_temp=rnorm(Xncol,0,1), z_old_temp=z_admm_temp;
  VectorXd z_admm= as<VectorXd>(z_admm_temp);
  VectorXd z_old= as<VectorXd>(z_old_temp);

  NumericVector mu_admm_temp =rnorm(Xncol,0,1);
  VectorXd mu_admm= as<VectorXd>(mu_admm_temp);


  bool terminate=FALSE;
  //learning rate
  double eta=0.01;

  // ADMM
  // GD update X

  int marker = 1;
  //criterion of convergence of x
  //According to paper Distributed Optimization and Statistical.....
  //page42
  double tol =1E-3;

  //condtion of final convergence
  double tol_1 = 1E-2;
  double tol_2 = 1E-2;
  double ABSTOL=1E-4;
  double RELTOL=1E-2;


  int Yrow= Y.rows();
  int Xcol= X.cols();//Xcol=6
  int Xrow=X.rows();//Xrow=100
  while(!terminate){

    //Use Adam algorithm to update x
    VectorXd m=VectorXd::Zero(Xcol);

    VectorXd v=VectorXd::Zero(Xcol);
    //printf("he m lenth is %d\n\n the vector v size is %d\n",m.size(),v.size());
    int conver_mark=1;
    bool converge=FALSE;

    while(!converge){
      VectorXd G=VectorXd::Zero(Xcol);
      //printf("%f",Y.rows());



      //For elementwise multiplication
      // G    <- apply(X*Y,2,sum) - apply(as.numeric(exp( X %*% as.numeric(b_gd))) * X,2,sum) + (rho)*(b_gd-z_admm+mu_admm)
      //For elementwise multiplication
      //caculate the firsterm
      MatrixXd X_ele=X;

      for(int i=0;i<Xrow; i++){
        for(int j=0;j<Xcol; j++){
          X_ele(i,j)=X(i,j)*Y(i);

        }

      }
      VectorXd sum_ele(Xcol);
      //sum_ele =sum(X*Y) elementwise, then sum
      for(int i=0;i<Xcol; i++){
        sum_ele(i)=X_ele.col(i).sum();
      }
      //print_table(sum_ele.transpose());
      // printf("\n\n\n\nthe row of sumele is %d\n",sum_ele.rows());




      //caculate the second term of G
      VectorXd Xbgd;
      Xbgd=X*b_gd;
      //printf("the length of vec Xbgd is %d\n", Xbgd.size());
      MatrixXd X_exp=X;
      for(int i=0;i<Xrow;i++){
        for(int j=0;j<Xcol;j++){
          X_exp(i,j)=X(i,j)*exp(Xbgd(i));
        }
      }
      VectorXd sum_exp(Xcol);
      for(int i=0;i<Xcol;i++){
        sum_exp(i)=X_exp.col(i).sum();
      }



      //caculate the third term of G
      VectorXd Third_vec(Xcol);
      Third_vec=rho*(b_gd-z_admm+mu_admm);


      //Combine G
      G=sum_ele-sum_exp+Third_vec;
      //printf("the length of G is %d\n",G.size());

      //return List::create(Named("lambda")=lambda);

      ////**************
      //A.d.a.m algorithm to converge x_k+1
      ////*******************

      // m <- 0.1 * m + 0.9 * G
      //v <- 0.1 * v + 0.9 * G^2
      //b_gd <- b_gd + eta * m/sqrt(v)


      for(int i=0;i<Xcol;i++){
        m(i)=0.1*m(i)+0.9*G(i);
      }



      VectorXd G_pow(Xcol);
      for(int i=0;i<Xcol;i++){
        G_pow(i)=pow(G(i),2);
        v(i)=0.1*v(i)+0.9*G_pow(i);
      }
      //v=0.1*v+0.9*G_pow;

      VectorXd v_temp(Xcol);
      VectorXd m_temp(Xcol);
      for(int i=0;i<Xcol;i++){
        v_temp(i)=sqrt(v(i)+0.1);
        m_temp(i)=m(i)/v_temp(i);
      }

      for(int i=0;i<Xcol;i++){
        b_gd(i)=b_gd(i)+eta*m_temp(i);
      }


      conver_mark=conver_mark+1;





      printf("\n b_gd value is as follows\n");
      for(int i=0;i<Xcol;i++){
        printf("%.4f ",b_gd(i));

      }
      printf("\n\n\nconverge mark is %d\n",conver_mark);




      //converge condition of update of x_k+1
      bool max_mark=TRUE;
      for(int i=0;i<Xcol;i++){
        double etaG=(eta*G)(i);
        if( etaG>tol  ) {max_mark=FALSE; break;}
      }

      if(max_mark==TRUE || (conver_mark>100) ){
        converge=TRUE;
      }
    }



    //Soft Threshholding
    //z_admm <-pmax(0,b_gd+mu_admm-lambda/rho)-pmax(0,-b_gd-mu_admm-lambda/rho)
    VectorXd lam_rho(Xcol );
    for(int i=0;i<lam_rho.rows();i++){
      lam_rho(i)=lambda/rho;
    }

    VectorXd alpha_kapp;
    alpha_kapp=b_gd+mu_admm-lam_rho;
    VectorXd alpha_pluskapp=b_gd+mu_admm+lam_rho;
    for(int i=0;i<Xcol;i++){
      z_admm(i)=std::max(0.0,double(alpha_kapp(i)) )-std::max(0.0,-double(alpha_pluskapp(i)) );
    }
    //complete the update of z_admm


    //Go update Augmented Variable mu_admm
    mu_admm =mu_admm + b_gd - z_admm;


    //Final convergence condition

    double cond_1=norm2sum(b_gd-z_admm);
    double cond_2=norm2sum(-rho*(z_admm - z_old));
    double cond_3=sqrt(5)*ABSTOL+RELTOL*std::max(norm2sum(b_gd), norm2sum(-z_admm));
    double cond_4=sqrt(5)*ABSTOL+RELTOL*norm2sum(rho*mu_admm);
    if( cond_1 < cond_3 && cond_2 <cond_4 ){
      terminate=TRUE;
    }
    marker=marker+1;
    if(marker>100){
      terminate=TRUE;
    }
    printf("total marker is %lf\n",marker);
  }
  VectorXd beta_hat=b_gd;
  VectorXd lam_rho(Xcol );
  // for(int i=0;i<lam_rho.rows();i++){
  //   lam_rho(i)=lambda/rho;
  // }
  // for(int i=0;i<Xcol;i++){
  //  if(b_gd(i)<lam_rho(i)) beta_hat(i)=0;
  //  else if (beta_hat(i)=b_gd(i));
  // }
  return List::create(Named("lambda")=lambda,
                      Named("beta.hat")=b_gd,
                      Named("niter")=marker );


  END_RCPP

}
