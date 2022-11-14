# ARCEL-DS

#Description
Sistema de recomendação que gera aprendizado personalizado para alunos. Um algoritmo que ultiliza históricos escolares para indicar textos para o aluno

Pré-requisitos #

- Clonar o github 
- Rodar no console - pip install requirements.txt
- python run app.py

A aplicação está ultizando flask como servidor http e o docker atuando como container. 

O modelo utiliza XGBoostRegressor para criar relações entre as informações do aluno como histórico, vivência social e comportamento para predizer como será seu desempenho futuro, e a partir disso conseguir disponibilizar textos/livros de acordo com cada dificuldade. 

O conceito é baseado em estudo personalizado, onde nós temos algortimos estatistico apoiando o ensino de forma adaptável e gerando métricas para os alunos e instituições. 

App - é a aplicação que possui o servidor HTTP 
Dockerfile - Configuração do container
requisitos - bibliotecas utilizadas e necessária para utilizar o código

Dash

Failure - quantidade de reprovações do dataset
Mapa de calor - correlação das features
Validate model - Relações do treinamento com dados reais e o quão distante estão nossas features em relação ao erro quadrático médio


