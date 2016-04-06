# FariaTcc
Trabalho de Conclusão de Curso de Ciência da Computação na UFRJ

Aluno: Felipe Sepulveda de Faria

Orientadora: Silvana Roseto

Co-orientador: Aloísio Pina
## Objetivo
Desenvolver uma implementação de SVM usando CUDA, o algoritmo implementado é o KAA que utiliza força bruta para encontrar os vetores de suporte e seus pesos.
## Guia
- *FariaSVM/* : O codigo fonte se encontra aqui
- *FariaSVM/FariaSVM.cpp* : main
- *FariaSVM/ParallelSVM.cu* : versão paralela
- *FariaSVM/SequentialSVM.cu* : versão sequencial
- *FariaSVM/Data/* : conjuntos de dados analisados
- *FariaTccTests/* : Testes unitários
- *Latex/* : projeto escrito em Latex

## Comandos do programa
- -c : Constraint for softmargin. Default is 999Default is: 999
- -d : DataSet to use, i | a[1-9] | w[1-8]Default is: i
- -f : Folds used in cross validation. Default is 10Default is: 10
- -g : Gamma value used for gaussian kernel, default varies by dataSet.Default is: 0.5
- -h : Shows options available.Default is:
- -l : Define log level, {a:all, r:results, e:only errors, n:none} Default is rDefault is: r
- -mi : Threads Per Block used for cuda kernels. Default is: 128Default is: 128
- -p : Precision of double values. Default is 1e-10Default is: 1e-010
- -sd : Seed used for random number generator. Default is: time(nullptr)Default is: 1459862050
- -st : Size of first step is algorithm. Default is 1Default is: 1
- -svm : Type of SVM to use, 'p' for parallel, 's' for sequential. Default is: sDefault is: s
- -t : Threads Per Block used for cuda kernels. Default is: 128Default is: 128
