# Neural Nets MPC

Programa escrito em Python para identificação de sistemas com redes neurais do tipo DLP, com o intuito de utilizar o resultado como modelo de processo em um controlador preditivo (MPC).

Desenvolvido como projeto de estágio obrigatório do curso de Eng. de Controle e Automação - UFSC.

<!--
#### Stack

- [Python](https://www.python.org/)
- [TensorFlow](https://www.tensorflow.org/api_docs/python/tf/keras)
- [Pandas](https://pandas.pydata.org/)
-->

## Pré-requisitos

Recomenda-se a utilização de um gerenciador de pacotes como o [Anaconda](https://www.anaconda.com/)/[Miniconda](https://docs.conda.io/en/latest/miniconda.html), visto que possibilitam a criação de ambientes de desenvolvimento isolados para cada projeto.

<!--
No início do desenvolvimento, a versão do Python instalada em meu computador ainda não era suportada pelo TensorFlow, e o Anaconda permitiu a criação de um environment específico para o projeto, utilizando uma versão mais antiga da Python.
-->

O arquivo ```requirements.txt``` contém uma lista com todas as bibliotecas utilizadas no projeto. Utilizando o conda, crie um novo environment com o seguinte comando:

```bash
conda create --name neural-nets-mpc --file requirements.txt
```

Para analisar o código e executar o programa, recomendo o [Spyder IDE](https://www.spyder-ide.org/), instalável pelo próprio Anaconda. Sua interface lembra a do Matlab, e a ferramente Variable Explorer facilita bastante a inspeção das variáveis no decorrer do treinamento dos modelos.

## Relatório e mudanças

Meu relatório do estágio está salvo na pasta ```docs```. Nele, explico em detalhes o estudo que foi efetuado na área de identificação de sistemas com redes neurais, e justifico a escolha de múltiplos DLPs como estratégia de identificação de sistema. Desde a escrita do relatório, foram feitas as seguintes mudanças no código:

- adequação ao guia [PEP8](https://www.python.org/dev/peps/pep-0008/);
- organização dos arquivos auxiliares em no package ```utils```;
- desenvolvimento da versão decremental do algoritmo de input selection, que substituiu a versão anterior (agora guardada em ```unused_functions.py```);
- utilização da [correlação](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html) entre as variáveis de entrada e saída no início do algoritmo de input selection, para filtragem inicial de variáveis.
- ```dependency.py``` para adição/remoção de dependências de uma saída

## Documentação

O código foi documentado com docstrings. A ferramenta [Sphinx](https://www.sphinx-doc.org/en/master/index.html) foi utilizada para gerar uma página html a partir desses docstrings. Para visualizá-la, basta clicar no atalho ```Abrir Documentação```, que está na pasta ```docs```.

Para gerar a documentação novamente, basta abrir o terminal do Anaconda na pasta ```docs``` e executar o comando ```make html```. Também é necessário instalar o tema com:

```bash
pip install sphinx-rtd-theme
```

É importante que não haja nenhum código "solto" na hora de realizar a documentação, pois o Sphinx importa todos os arquivos, e tudo que está solto é executado. Por isso, foi necessário adicionar o "protective wrapper" na ```main.py```.

<center><a href="https://www.youtube.com/watch?v=b4iFyrLQQh4">Tutorial do Sphinx</a></center>

## Organização do projeto

#### \data

Nesta pasta, estão os logs dos sensores, que são utilizados para obtenção de modelos neurais.

- Batches 1, 2 e 3 são conjuntos de dados do sistema real de compressão de gás.

- ```emso_simulator``` é um conjunto que foi gerado numa iteração prévia do projeto, com um simulador do sistema.

- ```simulink``` é um conjunto gerado a partir de um sistema simples desenvolvido na ferramente de simulação [Simulink](https://www.mathworks.com/products/simulink.html), do [Matlab](https://www.mathworks.com/products/matlab.html), com o intuito de validar o algoritmo de input selection ao utilizar um modelo cuja estrutura é conhecida.

A documentação da função ```analysis_utils.load_data``` contém mais informações sobre esses conjuntos, e também indica como pode ser feito para importação de novos conjutnos de dados.

#### \analysis

Todos os resultados são guardados nessa pasta, incluindo os três dicionários (de treinamento, análise e modelos) e os plots que forem efetuados.

Também é possível, por exemplo, criar os modelos de y1 até y5, analisar os resultados e depois continuar de y6 em diante sem perder os resultados, porém se recriar o modelo de y1, o modelo anterior é sobrescrito. Logo, ao trocar o conjunto de dados ou parâmetros de execução, é necessário criar uma pasta backup e arrastar para dentro os resultados anteriores, caso não queira sobrescrevê-los.

#### \docs

Contém a documentação do projeto. Ver a seção sobre a documentação para mais informações.

#### \misc

Contém alguns programas extras, dentre eles o ```comparison_test.py```, que permite comparar o desempenho de modelos neurais treinados em horizontes diferentes e com um mesmo conjunto de dados e regressores.

### \utils

Pacote de utilidades. Contém as funcionalidades que são chamadas pelo arquivo principal ```main.py``` para manipulação de dados, construção e análise de modelos.

## Exemplo de utilização

Neste exemplo, cria-se um modelo de rede neural para o conjunto de dados levantados com o modelo de testes no Simulink.

### Preparação

- Verificar que a pasta ```analysis``` está vazia, com exceção das pastas de backup. Ao final da execução, é esperado a criação dos três dicionários (training, analysis e model) e duas figuras de plots;

- Abrir o ```main.py``` no Spyder;

- Verificar o carregamento de dados com ```du.load_data("simulink")```. Esse conjunto de dados tem apenas uma saída (y1), portanto apenas um modelo DLP será criado;

- ```training_dictionary``` será inicializado como um objeto vazio, pois ainda não existe um ```dicionario.pickle``` salvo na pasta ```analysis```;

- Nas configurações de execução, certificar-se de que ```"create models" = True```. Aconselho deixar o resto das configurações como estão no commit, mas ler os comentários sobre cada uma. Depois, pode-se testar configurações diferentes;

- Executar.

### Execução

A primeira etapa da criação de um modelo é a seleção de inputs, que define o melhor conjunto de variáveis e ordens para usar como entrada do modelo. Os dados do Simulink foram gerados com um objetivo específico de testar essa etapa. O modelo original segue uma função do tipo:

```bash
y1(k) = u1(k-1) + u4(k-1) + u4(k-2) + y1(k-1) + y1(k-2) + y1(k-3)
# coeficientes desconhecidos (omitidos) multiplicam cada uma
# destas variáveis, eles são obtidos com o treinamento da rede
```

Logo, é esperado que os inputs selecionados pelo algoritmo sejam representados por um ```regressor``` contendo os seguintes valores:

```bash
regressor: [u1 = 1, u2 = 0, u3 = 0, u4 = 2, y1 = 3]
# ou ainda
regressor: [u1 = 1, u4 = 2, y1 = 3]
# esse regressor simboliza exatamente a função anterior
```

O número associado a cada variável de entrada indica a ordem desta, achada pelo algoritmo. Ter ordem igual a 0 significa que a variável é desconsiderada, sendo suprimida do ```regressor```.

Caso se tenha o Matlab instalado, é possível abrir o modelo .slx do Simulink, que está salvo na pasta ```data\simulink```. Lá é possível ver que as variáveis u2 e u3 não estão conectadas ao sistema. Isso foi feito de propósito, para ver se o algoritmo conseguiria detectar isso e regredir as ordens destas variáveis até 0, eliminando-as da seleção de inputs.


Tendo definidos os inputs, a segunda etapa é a determinação do número de neurônios na camada oculta (K). Esse processo é explicado na documentação da própria função, sendo bem straight-forward. O modelo retornado ao final já é o modelo definitivo para a saída em questão.

Todo esse processo, de input selection e K selection, deve ser efetuado para cada saída do sistema (se este for MIMO), já que utilizam-se DLPs. No caso do modelo do Simulink, como só tem uma saída, só é criado o modelo para y1.

Após a criação dos modelos, vem a etapa de análise, onde o dicionário de análise é criado. Este pode ser inspecionado com o Variable Inspector do Spyder para ver informações úteis sobre os modelos obtidos, como tempo de execução, participação das variáveis, etc.

O dicionário de modelos também é criado. Este não é relevante dentro do programa (todos os parâmetros dos modelos criados já estão no dicionário de treinamento), servindo apenas para exportação a um programa externo responsável por convertê-lo em um arquivo XML para rodar no MPA.

A última etapa são os plots. Primeiro são feitos os singleplots, onde um plot é feito para cada um dos modelos que foram criados (que estiverem no ```training_dictionary```). Em seguida, são feitos os multipltos, que são basicamente alguns singleplots aparecendo junto no mesmo plot, o que pode ser útil para apresentações.

#### Explicação sobre os inputs do input selection

Suponha que um sistema tem 10 variáveis de entrada e 10 variáveis de saída. Portanto, o conjunto de dados ```raw_data``` terá 20 colunas, u1...u10, y1...y10. Para cada saída, uma rede neural será criada. O input selection precisa decidir, para uma determinada rede de uma saída, quais variáveis do sistema deverão ser consideradas na entrada desta rede.

No procedimento empírico utilizado nesse programa, considerar todas as 20 variáveis como possíveis candidatas seria inviavelmente lento. Por isso, são disponibilizadas algumas opções para pré-filtrar esse conjunto inicial de variáveis e auxiliar o trabalho do input selection.

- O modo padrão é considerar como variáveis candidatas apenas as variáveis de entrada (u) e valores passados (regressores) da própria saída do sistema. No caso do suposto sistema, o conjunto de variáveis candidatas teria 10 variáveis 'u' + 1 variável referente a própria saída.

- O segundo modo é simplesmente considerar todas as 20 variáveis. Dependendo do sistema e do tempo disponível, isso pode ser viável. Para usar esta opção, basta simplesmente alterar ```use all variables``` para ```true``` no dicionário de configurações de execução.

- O terceiro modo é informando explicitamente qual conjunto de variáveis considerar para o input selection de uma determinada saída. Esse conjunto é referenciado aqui como "dependência", sendo simplesmente um array de strings que fica guardado no dicionário de treinamento e utilizado para filtrar ```raw_data``` antes do input selection começar para uma saída.

Para adicionar/remover uma dependência, basta rodar o arquivo ```dependency.py```, que está na pasta de utilidades. A implementação é muito simples e pode ser entendida inspecionando o código.

Um recurso que pode acelerar bastante a etapa de input selection é a filtragem de variáveis por correlação, independente de qual das três opções anteriores é utilizada.Basta alterar o parâmetro ```min abs corr``` no dicionário de execução para um valor de 0 até 1.

A ideia é eliminar as variáveis cuja correlação absulta com a saída do sistema, calculada usando o próprio conjunto de dados, seja menor do que o valor informado. Apesar de correlação não implicar em causalidade, a ausência de correlação implica em ausência de causalidade.
