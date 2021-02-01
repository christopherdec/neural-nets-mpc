# Neural Nets MPC

Programa em Python para identificação de sistemas com redes neurais do tipo DLP, com o intuito de posterior utilização dos modelos obtidos como modelo de processo por um controlador preditivo (MPC).

Desenvolvido como estágio obrigatório do curso de Eng. de Controle e Automação, fazendo parte do projeto SCoPI dentro do projeto de pesquisa "Desenvolvimento de Algoritmos de Controle Preditivo Não Linear para Plataformas de Produção de Petróleo: Fase 2".

#### Principais tecnologias utilizadas

- [Python](https://www.python.org/)
- [TensorFlow](https://www.tensorflow.org/api_docs/python/tf/keras)
- [Pandas](https://pandas.pydata.org/)

## Pré-requisitos

Recomenda-se a utilização de um gerenciador de pacotes como o [Anaconda](https://www.anaconda.com/), visto que possibilita a criação de environments isolados para cada projeto. No início do desenvolvimento, a versão do Python instalada em meu computador ainda não era suportada pelo TensorFlow, e o Anaconda permitiu a criação de um environment específico para o projeto, utilizando uma versão mais antiga da Python.

O arquivo requirements.txt contém todas as bibliotecas utilizadas no environment do projeto. Após a instalação do Anaconda, um novo environment pode ser criado com estes requirements com o seguinte comando:

```bash
conda create --name neural-nets-mpc --file requirements.txt
```

Para execução do código, é recomendado utilizar o Spyder IDE, que já vem com o Anaconda. Sua interface lembra a do Matlab, e a vantagem é o Variable Explorer, indispensável para inspeção dos dicionários de treinamento e análise gerados pelo código.

## Relatório e mudanças

É possível ler meu relatório do estágio, que explica todo o estudo feito na área de identificação de sistemas com redes neurais. A seção 5 tem uma breve explicação sobre o código. Porém, desde então, algumas mudanças estruturais foram feitas, como:

- adequação à PEP8
- organização dos arquivos auxiliares em no package "utils"
- criação de unused_functions.py para guardar funções não mais utilizadas
- documentação do projeto com Docstrings e Sphinx
- desenvolvimento da versão decremental do algoritmo de input selection

## Documentação

Todas as funções e módulos do código foram feitos com a utilização de docstrings. A tool Sphinx foi utilizada para extrair todos esses docstrings numa página de HTML, que pode ser utilizada para consulta rápida.

Para visualizar a documentação, abrir o arquivo docs\\_build\html\index.html.

Para atualizar a documentação, ir na pasta docs com o terminal e digitar make html

Tutorial de utilização do Sphinx pode ser visto neste [link](https://www.youtube.com/watch?v=b4iFyrLQQh4)

É importante que não haja nenhum código "solto" na hora de realizar a documentação, pois o Sphinx importa todos os arquivos, e código solto seria executado. Por isso o protective wrapper na main.py.

## Organização do projeto

### \data\

Na pasta "data" é onde estão guardados os dados que podem ser utilizados para obtenção de modelos neurais. 

- Batches 1, 2 e 3 são conjuntos de dados do sistema real de compressão de gás.

- Dados de "emso_simulator" foram gerados numa iteração prévia do projeto em um simulador de um sistema de compressão de gás. 

- Dados de "simulink" foram gerados num sistema simples na ferramente de simulação [Simulink](https://www.mathworks.com/products/simulink.html), do [Matlab](https://www.mathworks.com/products/matlab.html). com o intuito de validar os algoritmos de seleção de estrutura com dados gerados por um modelo totalmente conhecido. O arquivo .slx está nessa pasta e fornece um ponto de partida para a geração de um novo conjunto de dados.

Mais informações sobre os dados podem ser encontradas na documentação da função "load_data()", do módulo analysis_utils, inclusive o formato adequado para a utilização de novos conjuntos de dados.

### \analysis\

A pasta analysis é onde são guardados os três dicionários (de treinamento, análise e modelos), juntamente com os plots. Ao trocar o conjunto de dados ou parâmetros de execução, é recomendado criar manualmente uma pasta backup aqui e arrastar o conteúdo já obtido para dentro, evitando a perda já que os resultados serão sobrescritos.

### \docs\

Contém a documentação do projeto. Todos os arquivos foram gerados pelo Sphinx. Ver a seção de documentação para mais informações.

### \misc\

Contém alguns programas de testes, dentre eles o comparison_test.py, que permite comparar o desempenho de modelos neurais com tipos de redes e horizontes diferentes para um mesmo conjunto de dados e regressores.

### \utils\

Pacote de utilidades. Contém as funcionalidades que são chamadas pelo arquivo principal (main.py) para manipulação de dados, construção e análise de modelos.

## Exemplo de utilização

Neste exemplo, vamos criar um modelo de rede neural para os dados levantados com o modelo de testes no Simulink.

### Preparação

- Verificar que a pasta \analysis\ está vazia (só com os folders de backup). Ao final da execução, é esperada a criação dos três dicionários (training, analysis e model), e duas figuras de plots.

- Abrir o main.py no Spyder

- Verificar o carregamento de dados com du.load_data("simulink"). Isso carrega os dados do simulink. Esse sistema tem apenas uma saída 'y', portanto apenas um modelo será criado.

- training_dictionary será inicializado como um objeto vazio, pois ainda não existe um dicionario.pickle salvo na pasta analysis

- Nas configurações de execução, certificar de que "create models" = True. O resto das configurações devem ficar como estão.

A criação de modelos tem duas etapas: primeiro, a seleção de inputs é chamada, que define o melhor conjunto de variáveis e ordens para usar como input do modelo. Os dados do simulink foram gerados com um objetivo específico de testar esse algoritmo. O modelo original é uma função do tipo:

<center><bold>
y1(k) = u1(k-1) + u4(k-1) + u4(k-2) + y1(k-1) + y1(k-2) + y1(k-3)
</bold></center>

Logo é esperado que os inputs selecionados como entradas do modelo sejam representados por um "regressor" do tipo:

<center><bold>
regressor = [u1 = 1, u2 = 0, u3 = 0, u4 = 2, y1 = 3]
</bold></center>

Como os valores de u2 e u3 são zero constante, o algoritmo não consegue selecionar a ordem zero para estar variáveis. Este é um detalhe específico desse conjunto de dados. O que importa para o teste ser bem sucedido é que as ordens para as variáveis u1, u4 e y1 estejam corretas.

Tendo definidos os inputs, a segunda etapa é a determinação do número de neurônios na camada oculta, denominado K. O modelo retornado na segunda etapa já é o modelo final treinado para a saída em questão. Esse processo deve ser efetuado para cada saída do sistema. No caso do modelo do simulink, como só tem uma saída, só é criado o modelo neural para y1.

Após a criação dos modelos, tem-se a etapa de análise, onde o dicionário de análise é criado. Este pode ser inspecionado com o Variable Inspector do Spyder para ver informações úteis sobre os modelos obtidos.

O dicionário de modelos também é criado. Este não é relevante dentro do programa, e serve apenas para ser exportado ao programa que fará a leitura dos modelos obtidos e converterá em XML para poder rodar no MPA.

A última etapa será o plot dos modelos. Primeiro é feito os singleplots, onde um plot é feito para cada um dos modelos que foram criados (que estiverem no training_dictionary). Em seguida, multiplots são feitos, onde várias respostas aparecem junto no mesmo plot. As respostas são as mesmas, a única razão de fazer multiplots é para compactar os plots caso se deseje mostrar vários de uma vez.
