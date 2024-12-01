# Projeto PCD: Simulação e Análise de Modelos de Difusão de Contaminantes em Água

## Descrição do Projeto

Objetivo: Criar uma simulação que modele a difusão de contaminantes em um corpo d'água (como um lago ou rio), aplicando conceitos de
paralelismo para acelerar o cálculo e observar o comportamento de poluentes ao longo do tempo. O projeto investigará o impacto de OpenMP,
CUDA e MPI no tempo de execução e na precisão do modelo.

## Estrutura do Repositório

- `src/`: Contém o códigos fonte das implementações em OpenMP do formato serial e paralelo em liguagem C.
- `data/`: Arquivo .xlsx com as especificações do computador usado para testes, e a tabela contendo as comparações de tempos de execuções com as variaçoes de quantidade de threads.
- `article/`: Versão final do artigo científico no padrão IEEE.

## Como Executar o Projeto

1. **Clone o repositório**:
   ```bash
   git clone https://github.com/Esteves31/ProjetoFinalPCD.git

2. **Vá para o diretório /src**
    ```bash
    cd src

3. **Compile e execute o código**
    ```bash
    gcc -fopenmp -o parallel parellel.c
    ./parallel

## Tecnologias Utilizadas

<a href="https://skillicons.dev"><img src="https://skillicons.dev/icons?i=c" /></a>
<a href="https://skillicons.dev"><img src="https://skillicons.dev/icons?i=vscode" /></a>
<a href="https://skillicons.dev"><img src="https://skillicons.dev/icons?i=git" /></a>
<a href="https://skillicons.dev"><img src="https://skillicons.dev/icons?i=github" /></a>