- [ ] Prunning das árvores
- [ ] Modelo particionado
   1. [x] Quando é caso de atraso/cancelamento/noshow
   2. Outro(s) para o resto
      1. Direito de arrependimento e Downgrade viram modelo de médias
- [ ] Analizar os casos removidos e testá-los no modelo final
  - [ ] Criar outro modelo separado incluindo os casos removidos
- [ ] Relatório de performance (tanto nas métricas quanto performance computacional)
- [ ] Pesquisar mais como interpretar o gráfico SHAP

# Talvez

- [ ] Fazer fit de hiperparâmetros para todos os modelos.
- [ ] Tornar o repositório capaz de rodar múltiplos testes diferentes parametrizáveis
  - [x] Testar diferentes Modelos de uma vez
  - [ ] Testar diferentes formatações e seleções de dados de uma vez
- [ ] Criar uma Wiki compreensiva do repositório
- [ ] Correção Monetária dos Valores para 2025
    1. Talvez usando IPCA
- [ ] Features de casos especiais
    1. Juiz
    2. Pandemia
    3. Outros Períodos excepcionais
- [ ] Weight Classification para substituir o balanceamento
    1. [How to Remove Outliers for Machine Learning - MachineLearningMastery.com](https://machinelearningmastery.com/how-to-use-statistics-to-identify-outliers-in-data/)
    2. Dar pesos para as instâncias, ou para o erro

# Implementadas

- [x] Plotar a distribuição do erro nos modelos
  - [x] Estratificar em intervalos de 250 reais e observar a frequência
- [x] Combinar extravio definitivo e intervalo de extravio
  - Só piora muito todos os modelos.
  - Da mesma forma para avaria
  - Ou para tentar combinar overbooking com atraso.
- [x] Escalonar os intervalos
  - Não melhora a regressão linear nem redes neurais
  - Elevar ao quadrado também não ajuda
- [x] Testar mais modelos
  - [x] Linear Regression
  - [x] GradientBoost
  - [x] Redes Neurais
  - [x] SVM
  - [x] naive bayes
- [x] Feature Selection
  - [x] Usando Variancia abaixo de 5% como critério de eliminação
  - [x] Os features não são apenas ignorados, mas todas as entradas onde os features removidos são diferentes de 0 são eliminadas
  - [x] Técnicas de Feature Selection (wrappers \+ Boruta)
- [x] Cross-Validation
      - Mas, para cada variação, é necessário encontrar o melhor conjunto de parâmetros, modelos e hiperparâmetros.
      - Depois achar as combinações que performam melhor na média
- [x] Desenvolver um gerador automático de logs
- [x] Calcular algo similar ao MAPE, mas baseado no valor predito ao invés do real.
- [x] O balanceamento e deve acontecer só no conjunto de treino
- [x] Fazer gráficos de distribuição do valor de dano moral
    1. [Exploração (Orange)](https://drive.google.com/drive/u/0/folders/1DvvW3kKk3EywpfABm--hK27BixCdlpYg)
- [x] Inserir a data como variável
    1. Colocando o ano, semestre,  \*trimestre
    2. ⇒ trimestre contado a partir de 2010
    3. Não é escalável
- [x] [https://github.com/ajaymache/machine-learning-yearning](https://github.com/ajaymache/machine-learning-yearning)
- [x] Pesquisar métodos de avaliação
    1. ⇒ MAP (porcentagem), mas só para visualizar
- [x] Testes com variações de formatação
    [x] Mudar o cancelamento para len(atrasos) \+ 1
       1. Não muda
    [x] Assistência → Desamparo
    [x] Com e \*sem cofactors
    [x] Com e \*sem binárias de intervalo
- [x] Balanceamento por faixa de dano moral
    [x] faixas decididas pelo intervalo desejável
    [x] oversampling
- [x] Testar outras estratégias de balanceamento
    1. not majority foi a melhor
- [x] Tornar con-factors perguntas para o usuário
  - [x] Remover do treinamento e prompt
- [x] Usar valores categóricos para as variáveis de intervalo
- [x] Remover “não se aplica” da assistência da companhia aérea
- [x] Remover as binárias de atraso e extravio
- [x] Fazer divisão (holdout) estratificada 80/20
- [x] Não otimizar o random seed (42)