#include <iostream>
#include <iomanip>
#include <Eigen/Eigen>
#include<vector>
#include<cmath>
using namespace std;
using namespace Eigen;

void VisualizzaMatrice (const MatrixXd& A)
{
    cout << setprecision(16) << A << endl << endl<<endl;

}


MatrixXd QR_decomposition(const MatrixXd& A, const MatrixXd& b, int i)  //evitiamo di copiare i vettori e le matrici passati, uso il const& tale da non poter modificare la matrice
    {

    MatrixXd x1 = A.householderQr().solve(b);
    // Eseguo la decomposizione QR con pivoting parziale
    ColPivHouseholderQR<MatrixXd> qr(A);

    // Recupero le matrici Q e R
    MatrixXd Q = qr.householderQ();
    MatrixXd R = qr.matrixQR().triangularView<Upper>();

    //y = Q' * b;     x= R\y;
    // Risolvo il sistema triangolare superiore Rx = Qt*b
    VectorXd Qtb = Q.transpose() * b;
    VectorXd x = R.triangularView<Upper>().solve(Qtb);
    //controllo che le due soluzioni siano uguali
    if (x==x1)
    {
        return x1;
    }
    else
    {
        cerr<<"valutando le soluzioni ottenute nel SISTEMA "<<i<< " con solve e la risoluzione del sistema QR "<<endl;
        cerr<<"non otteniamo lo stesso risultato, troviamo una differenza in norma 1 pari a "<<(x-x1).norm()<<endl<<endl;
        return x1;
    }
}


 MatrixXd PALU_decomposition(const MatrixXd& A, const MatrixXd& b, int i)  //evitiamo di copiare i vettori e le matrici passati, uso il const& tale da non poter modificare la matrice
 {
    MatrixXd x1 = A.lu().solve(b);
    //provo a risolvere il sistema
    // Eseguo la decomposizione LU parziale
    PartialPivLU<MatrixXd> lu(A);

    // Recupero le matrici L, U e P
    MatrixXd L = lu.matrixLU().triangularView<StrictlyLower>();
    MatrixXd U = lu.matrixLU().triangularView<Upper>();
    MatrixXd P = lu.permutationP();

    // Aggiungo l'identità a L
    MatrixXd I = MatrixXd::Identity(A.rows(), A.cols());
    MatrixXd L_with_identity = L + I;

    // Risolvo il sistema PA = LU
    //y = L\(P*b)    x=U\y
    // Moltiplico il termine noto per la matrice di permutazione P
    MatrixXd Pb = P * b;

    // Risolvo il sistema triangolare inferiore Ly = Pb
    VectorXd y = L_with_identity.triangularView<Lower>().solve(Pb);

    // Risolvo il sistema triangolare superiore Ux = y
    VectorXd x = U.triangularView<Upper>().solve(y);

    //controllo che le due soluzioni siano uguali
    if (x==x1)
    {
     return x1;
    }
    else
    {
        cerr<<"valutando le soluzioni ottenute del SISTEMA "<<i<<" con solve e la risoluzione del sistema PALU "<<endl;
        cerr<<"non otteniamo lo stesso risultato, troviamo una differenza tra  i due risultati in norma uno pari a "<<(x-x1).norm()<<endl<<endl;
      return x1;
    }
 }


 double errore_relativo(const MatrixXd& x_esatto, const MatrixXd& x )
 {
    //effettuo la norma L1
    double norma1 = (x_esatto-x).norm();
    double norma2 = x_esatto.norm();
    return norma1/norma2;
 }


int main()
{
    //definisco prima la soluzione di tutti i sistemi
    MatrixXd x_esatto(2, 1);
    x_esatto << -1.000000000000000e+0, -1.000000000000000e+00;

    cout<<"PRIMO SISTEMA"<<endl;
    int i = 1;
    //studio la prima matrice
    MatrixXd A1(2, 2);
    A1 << 5.547001962252291e-01, -3.770900990025203e-02,
          8.320502943378437e-01, -9.992887623566787e-01;
    cout<<"A1:"<<endl;
    VisualizzaMatrice(A1);
    //inizializzo il termine noto
    MatrixXd b1(2, 1);
    cout<<"b1:"<<endl;
    b1 << -5.169911863249772e-01, 1.672384680188350e-01;
    VisualizzaMatrice(b1);

    //effettuo il calcolo della soluzione con la decomposizione QR
    MatrixXd x1QR = QR_decomposition(A1,b1,i);
    cout<<"soluzione ottenuta dalla decomposizione QR:"<<endl;
    VisualizzaMatrice(x1QR);

    //effettuo il calcolo con la decomposizione PA=LU
    MatrixXd x1LU = PALU_decomposition(A1,b1,i);
    cout<<"soluzione ottenuta dalla decomposizione LU risulta:"<<endl;
    VisualizzaMatrice(x1LU);

    cout<<"ERRORI SISTEMA 1:"<<endl;
    //effettuo il calcolo dell'errore relativo
    double err1QR = errore_relativo(x_esatto, x1QR);
    cout<<"l'errore relativo associato alla fattorizzazione QR risulta:  "<<err1QR<<endl<<endl;

    //effettuo il calcolo dell'errore relativo
    double err1LU = errore_relativo(x_esatto, x1LU);
    cout<<"l'errore relativo associato alla fattorizzazione LU risulta:  "<<err1LU<<endl<<endl;;



    cout<<"SECONDO SISTEMA"<<endl;
    i++;
    //studio la seconda matrice
    MatrixXd A2(2, 2);
    A2 << 5.547001962252291e-01, -5.540607316466765e-01,
        8.320502943378437e-01, -8.324762492991313e-01;
    cout<<"A2:"<<endl;
    VisualizzaMatrice(A2);
    //inizializzo il termine noto
    MatrixXd b2(2, 1);
    cout<<"b2:"<<endl;
    b2 << -6.394645785530173e-04, 4.259549612877223e-04;
    VisualizzaMatrice(b2);

    //effettuo il calcolo della soluzione con la decomposizione QR
    MatrixXd x2QR = QR_decomposition(A2,b2,i);
    cout<<"soluzione ottenuta dalla decomposizione QR:"<<endl;
    VisualizzaMatrice(x2QR);


    //effettuo il calcolo con la decomposizione PA=LU
    MatrixXd x2LU = PALU_decomposition(A2,b2,i);
    cout<<"soluzione ottenuta dalla decomposizione LU:"<<endl;
    VisualizzaMatrice(x2LU);


    cout<<"ERRORI SISTEMA 2:"<<endl;
    //effettuo il calcolo dell'errore relativo
    double err2QR = errore_relativo(x_esatto, x2QR);
    cout<<"l'errore relativo associato alla fattorizzazione QR risulta:  "<<err2QR<<endl<<endl;

    //effettuo il calcolo dell'errore relativo
    double err2LU = errore_relativo(x_esatto, x2LU);
    cout<<"l'errore relativo associato alla fattorizzazione LU risulta:  "<<err2LU<<endl<<endl;



    cout<<"TERZO SISTEMA"<<endl;
    i++;
    //studio la terza matrice
    MatrixXd A3(2, 2);
    A3 << 5.547001962252291e-01, -5.547001955851905e-01,
        8.320502943378437e-01, -8.320502947645361e-01;
    cout<<"A3:"<<endl;
    VisualizzaMatrice(A3);
    //inizializzo il termine noto
    MatrixXd b3(2, 1);
    cout<<"b3:"<<endl;
    b3 << -6.400391328043042e-10, 4.266924591433963e-10;
    VisualizzaMatrice(b3);

    //effettuo il calcolo della soluzione con la decomposizione QR
    MatrixXd x3QR = QR_decomposition(A3,b3,i);
    cout<<"soluzione ottenuta dalla decomposizione QR:"<<endl;
    VisualizzaMatrice(x3QR);

    //effettuo il calcolo con la decomposizione PA=LU
    MatrixXd x3LU = PALU_decomposition(A3,b3,i);
    cout<<"soluzione ottenuta dalla decomposizione LU:"<<endl;
    VisualizzaMatrice(x3LU);


    cout<<"ERRORI SISTEMA 3:"<<endl;
    //effettuo il calcolo dell'errore relativo
    double err3QR = errore_relativo(x_esatto, x3QR);
    cout<<"l'errore relativo associato alla fattorizzazione QR risulta:   "<<err3QR<<endl<<endl;;

    //effettuo il calcolo dell'errore relativo
    double err3LU = errore_relativo(x_esatto, x3LU);
    cout<<"l'errore relativo associato alla fattorizzazione LU risulta:   "<<err3LU<<endl<<endl;

  return 0;
}
