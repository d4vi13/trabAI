# Módulo 04 — Provas Anteriores e Exercícios Resolvidos (Padrão Formal e Detalhado)

> **Regra de ouro das provas da Prof. Aurora Pozo (UFPR):**
> *"Obs: A resolução deve ser formal e detalhada."*
>
> Neste módulo estão **todas** as questões reais encontradas nos arquivos da pasta — **100% verificadas numericamente** em aritmética exata de frações e diferenças finitas — estruturadas exatamente como você deve redigi-las na folha de prova:
> 1. **Prova Anterior 1:** [`prova2.pdf`](file:///home/dcr/repos/trabAI/prova/prova2.pdf) (Questões 1, 2, 3 e 4 — Naïve Bayes, Perceptron, Rede de Bayes / Enumeração / Rejection Sampling e Árvore de Decisão)
> 2. **Prova Anterior 2:** [`Documento sem título.pdf`](file:///home/dcr/repos/trabAI/prova/Documento%20sem%20ti%A6%FCtulo.pdf) (Questões 1, 2 e 3 — Eliminação de Variáveis Miles, MDPs Jim e Pam com VI/PI, e Rede Neural com Ativação Quadrática)
> 3. **Lista Oficial CS188:** [`CS_188_Su26_Discussion_5_SOLUTION.pdf`](file:///home/dcr/repos/trabAI/prova/CS_188_Su26_Discussion_5_SOLUTION.pdf) (Questões Q1 e Q2 — Model-Based RL, Propagação de Q-Learning, V/F, Q-Learning Tabular e Aproximado)
> 4. **Questões Extras de Provas UFPR encontradas em `RL.pptx`:** Táxi 5×5 (Slides 11–13) e Pacman 2×3 Aproximado (Slides 19–22)

---

## Parte I — Resolução Completa de `prova2.pdf`

### Questão 1 — Aprendizado de Máquina: Classificação (Naïve Bayes)

**Enunciado:** Considere o conjunto de treinamento com características $X_1, X_2$ e classe $Y$ ($N = 8$ exemplos). Assuma um classificador **Naïve Bayes** para modelar os dados. Qual seria a classificação para um novo dado com $X_1 = 1$ e $X_2 = 0$?

| # | $X_1$ | $X_2$ | $Y$ |
|---:|:---:|:---:|:---:|
| 1 | 0 | 0 | 1 |
| 2 | 0 | 1 | 0 |
| 3 | 1 | 0 | 0 |
| 4 | 1 | 0 | 0 |
| 5 | 1 | 0 | 1 |
| 6 | 1 | 1 | 1 |
| 7 | 1 | 1 | 1 |
| 8 | 1 | 1 | 1 |

#### Resolução Formal e Detalhada

**Passo 1 — Modelo Naïve Bayes e Regra de Decisão:**
Pela suposição de independência condicional dos atributos $X_1$ e $X_2$ dada a classe $Y$, a probabilidade a posteriori de cada classe $y \in \{0, 1\}$ dada a evidência $(X_1 = 1, X_2 = 0)$ é:
$$P(Y = y \mid X_1 = 1, X_2 = 0) = \frac{P(Y = y)\,P(X_1 = 1 \mid Y = y)\,P(X_2 = 0 \mid Y = y)}{P(X_1 = 1, X_2 = 0)}$$
Como o denominador $P(X_1 = 1, X_2 = 0)$ é constante para ambas as classes, computamos primeiro o numerador (verossimilhança conjunta não normalizada) para $Y=0$ e $Y=1$ e depois normalizamos.

**Passo 2 — Estimação das Probabilidades Empíricas (MLE, sem suavização, pois não há contagens nulas):**
- **Probabilidades *a priori* $P(Y)$** (total $N = 8$):
  - Classe $Y = 0$: ocorre nas linhas $\{2, 3, 4\}$ (3 exemplos) $\implies P(Y = 0) = \dfrac{3}{8}$
  - Classe $Y = 1$: ocorre nas linhas $\{1, 5, 6, 7, 8\}$ (5 exemplos) $\implies P(Y = 1) = \dfrac{5}{8}$

- **Probabilidades condicionais para $Y = 0$** (dentre os 3 exemplos com $Y = 0$: `(0,1), (1,0), (1,0)`):
  - $P(X_1 = 1 \mid Y = 0) = \dfrac{\text{count}(X_1=1, Y=0)}{\text{count}(Y=0)} = \dfrac{2}{3}$
  - $P(X_2 = 0 \mid Y = 0) = \dfrac{\text{count}(X_2=0, Y=0)}{\text{count}(Y=0)} = \dfrac{2}{3}$

- **Probabilidades condicionais para $Y = 1$** (dentre os 5 exemplos com $Y = 1$: `(0,0), (1,0), (1,1), (1,1), (1,1)`):
  - $P(X_1 = 1 \mid Y = 1) = \dfrac{\text{count}(X_1=1, Y=1)}{\text{count}(Y=1)} = \dfrac{4}{5}$
  - $P(X_2 = 0 \mid Y = 1) = \dfrac{\text{count}(X_2=0, Y=1)}{\text{count}(Y=1)} = \dfrac{2}{5}$

**Passo 3 — Cálculo dos Numeradores e Normalização:**
- Para $Y = 0$:
  $$P(Y = 0, X_1 = 1, X_2 = 0) = P(Y = 0)\,P(X_1 = 1 \mid Y = 0)\,P(X_2 = 0 \mid Y = 0) = \frac{3}{8} \cdot \frac{2}{3} \cdot \frac{2}{3} = \frac{12}{72} = \frac{1}{6} \approx 0{,}1667$$
- Para $Y = 1$:
  $$P(Y = 1, X_1 = 1, X_2 = 0) = P(Y = 1)\,P(X_1 = 1 \mid Y = 1)\,P(X_2 = 0 \mid Y = 1) = \frac{5}{8} \cdot \frac{4}{5} \cdot \frac{2}{5} = \frac{40}{200} = \frac{1}{5} = 0{,}2000$$
- Constante de normalização:
  $$Z = \frac{1}{6} + \frac{1}{5} = \frac{5 + 6}{30} = \frac{11}{30}$$
- Probabilidades *a posteriori* normalizadas:
  $$P(Y = 0 \mid X_1 = 1, X_2 = 0) = \frac{1/6}{11/30} = \frac{5}{11} \approx 45{,}45\%$$
  $$P(Y = 1 \mid X_1 = 1, X_2 = 0) = \frac{1/5}{11/30} = \frac{6}{11} \approx 54{,}55\%$$

**Resposta Final:** Como $P(Y = 1 \mid X_1 = 1, X_2 = 0) = \dfrac{6}{11} > \dfrac{5}{11} = P(Y = 0 \mid X_1 = 1, X_2 = 0)$, o novo dado $(X_1 = 1, X_2 = 0)$ será classificado como **$Y = 1$**. *(Observação interessante para conferência intuitiva: na tabela de treino há dois exemplos `(1,0)->0` e apenas um `(1,0)->1`, mas o Naïve Bayes prevê `Y=1` por assumir independência condicional e combinar a priori maior `5/8` com a alta frequência de `X1=1` na classe `Y=1`!)*

---

### Questão 2 — Perceptron Binário e Separabilidade Linear

**Enunciado:** Considere treinar um perceptron com 2 características $(x_1, x_2)$ e o termo bias, no conjunto de treinamento abaixo. Os pesos iniciais são $w = [-1, 1, 0]$, correspondendo a $(x_1, x_2, 1)$ respectivamente.
- **(a)** Depois de executar uma iteração do algoritmo nos 3 primeiros dados da tabela, qual é o novo vetor de pesos $w$?
- **(b)** Discuta sobre a existência de um vetor de pesos que consiga classificar os dados perfeitamente.

| # | $x_1$ | $x_2$ | Label ($y^*$) | Vetor aumentado $f(x) = [x_1, x_2, 1]$ |
|---:|:---:|:---:|:---:|:---:|
| 1 | 3 | 4 | `+` ($+1$) | $[3, 4, 1]$ |
| 2 | 1 | 3 | `-` ($-1$) | $[1, 3, 1]$ |
| 3 | 2 | 3 | `+` ($+1$) | $[2, 3, 1]$ |
| 4 | 4 | 1 | `-` ($-1$) | $[4, 1, 1]$ |
| 5 | 1 | 2 | `+` ($+1$) | $[1, 2, 1]$ |
| 6 | 2 | 4 | `-` ($-1$) | $[2, 4, 1]$ |
| 7 | 3 | 1 | `-` ($-1$) | $[3, 1, 1]$ |

#### Resolução Formal e Detalhada

**(a) Execução de 1 iteração nos 3 primeiros exemplos:**
Regra de classificação do Perceptron Binário: $y = +1$ se $w \cdot f(x) \ge 0$, e $y = -1$ se $w \cdot f(x) < 0$. Se $y \neq y^*$, atualiza-se $w \leftarrow w + y^* \cdot f(x)$.
Vetor inicial: $w^{(0)} = [-1,\; 1,\; 0]$.

1. **Exemplo 1:** $f(x^{(1)}) = [3,\; 4,\; 1]$, $y^* = +1$.
   - Ativação: $w^{(0)} \cdot f(x^{(1)}) = (-1)(3) + (1)(4) + (0)(1) = -3 + 4 + 0 = +1 \ge 0 \implies y = +1$.
   - Como $y = y^*$ (**correto**), os pesos **não mudam**:
     $$w^{(1)} = [-1,\; 1,\; 0]$$

2. **Exemplo 2:** $f(x^{(2)}) = [1,\; 3,\; 1]$, $y^* = -1$.
   - Ativação: $w^{(1)} \cdot f(x^{(2)}) = (-1)(1) + (1)(3) + (0)(1) = -1 + 3 + 0 = +2 \ge 0 \implies y = +1$.
   - Como $y = +1 \neq -1 = y^*$ (**erro**), subtrai-se $f(x^{(2)})$:
     $$w^{(2)} = w^{(1)} + (-1)\,f(x^{(2)}) = [-1,\; 1,\; 0] - [1,\; 3,\; 1] = [-2,\; -2,\; -1]$$

3. **Exemplo 3:** $f(x^{(3)}) = [2,\; 3,\; 1]$, $y^* = +1$.
   - Ativação: $w^{(2)} \cdot f(x^{(3)}) = (-2)(2) + (-2)(3) + (-1)(1) = -4 - 6 - 1 = -11 < 0 \implies y = -1$.
   - Como $y = -1 \neq +1 = y^*$ (**erro**), soma-se $f(x^{(3)})$:
     $$w^{(3)} = w^{(2)} + (+1)\,f(x^{(3)}) = [-2,\; -2,\; -1] + [2,\; 3,\; 1] = [0,\; 1,\; 0]$$

**Resposta (a):** O novo vetor de pesos após os 3 primeiros exemplos é **$w = [0,\; 1,\; 0]$**.

**(b) Existência de um vetor de pesos que classifique perfeitamente (Separabilidade Linear):**
**NÃO existe** tal vetor de pesos! O conjunto de treinamento **não é linearmente separável** em $\mathbb{R}^2$.

**Prova geométrica e algébrica rigorosa (que garante nota máxima!):**
- Observe os pontos da classe positiva (`+`): $(1,2)$, $(2,3)$ e $(3,4)$ — todos estão sobre a reta $x_2 - x_1 = 1$.
- Observe os pontos da classe negativa (`-`):
  - Acima dessa reta: $(1,3)$ e $(2,4)$ (onde $x_2 - x_1 = 2$).
  - Abaixo dessa reta: $(3,1)$ e $(4,1)$ (onde $x_2 - x_1 \in \{-2, -3\}$).
- Logo, os pontos `+` ficam **sanduichados entre dois grupos de pontos `-`** (os envoltórios convexos das duas classes se intersectam!).
- **Demonstração algébrica direta por contradição:** Suponha que exista $w = [w_1, w_2, b]$ que classifique perfeitamente todos os dados, ou seja:
  - Para todo ponto `+`: $w_1 x_1 + w_2 x_2 + b > 0$
  - Para todo ponto `-`: $w_1 x_1 + w_2 x_2 + b < 0$
  Note que o ponto positivo $(2, 3)$ é uma **combinação convexa exata** de três pontos negativos $(2, 4)$, $(1, 3)$ e $(4, 1)$:
  $$\frac{2}{5}(2, 4) + \frac{2}{5}(1, 3) + \frac{1}{5}(4, 1) = \left(\frac{4 + 2 + 4}{5},\; \frac{8 + 6 + 1}{5}\right) = (2, 3), \qquad \frac{2}{5} + \frac{2}{5} + \frac{1}{5} = 1$$
  Portanto, pela linearidade de $s(x_1, x_2) = w_1 x_1 + w_2 x_2 + b$:
  $$s(2, 3) = \frac{2}{5}\underbrace{s(2, 4)}_{< 0} + \frac{2}{5}\underbrace{s(1, 3)}_{< 0} + \frac{1}{5}\underbrace{s(4, 1)}_{< 0} < 0$$
  Isso contradiz a exigência $s(2, 3) > 0$ (pois $(2, 3)$ tem rótulo `+`)! Logo, **nenhum** hiperplano (nenhum vetor de pesos $w$) consegue separar perfeitamente esses dados, e o algoritmo Perceptron jamais convergirá neste conjunto (ficará oscilando / *thrashing*).

---

### Questão 3 — Rede de Bayes: Enumeração e *Rejection Sampling*

**Enunciado:** Considere a Rede de Bayes com variáveis binárias $\{A, B, C, D, E\}$ e arestas direcionadas:
- $B \to A$, $D \to A$ (atenção à seta apontando de $D$ para cima em $A$!)
- $A \to C$
- $B \to E$, $C \to E$

- **(a)** Qual é a equação para calcular $P(+b \mid +e)$ usando **inferência por enumeração**?
- **(b)** Considere os 6 exemplos abaixo e calcule $P(+b \mid +e)$ usando **Rejection Sampling**:
  1. `(+a, +b, -c, +d, -e)`
  2. `(+a, -b, +c, +d, +e)`
  3. `(-a, +b, +c, +d, -e)`
  4. `(+a, +b, +c, +d, +e)`
  5. `(+a, -b, -c, -d, +e)`
  6. `(-a, -b, +c, +d, +e)`

#### Resolução Formal e Detalhada

**(a) Equação de Inferência por Enumeração para $P(+b \mid +e)$:**
1. **Fatoração da distribuição conjunta pela semântica da Rede de Bayes** ($P(x_1,\dots,x_n) = \prod_i P(x_i \mid \text{parents}(X_i))$):
   - Pais de cada nó: $\text{Pa}(B)=\emptyset$, $\text{Pa}(D)=\emptyset$, $\text{Pa}(A)=\{B, D\}$, $\text{Pa}(C)=\{A\}$, $\text{Pa}(E)=\{B, C\}$.
   $$P(A, B, C, D, E) = P(B)\,P(D)\,P(A \mid B, D)\,P(C \mid A)\,P(E \mid B, C)$$
2. **Definição condicional e marginalização das variáveis ocultas** ($H = \{A, C, D\}$, consulta $Q = B$, evidência $E = +e$):
   $$P(+b \mid +e) = \frac{P(+b, +e)}{P(+e)} = \frac{\sum_{a} \sum_{c} \sum_{d} P(a, +b, c, d, +e)}{\sum_{b} \sum_{a} \sum_{c} \sum_{d} P(a, b, c, d, +e)}$$
3. **Substituindo a fatoração da Rede de Bayes** (ou usando a constante de normalização $\alpha = \frac{1}{P(+e)} = \frac{1}{Z}$):
   $$P(+b \mid +e) = \frac{\displaystyle\sum_{a \in \{+a,-a\}} \sum_{c \in \{+c,-c\}} \sum_{d \in \{+d,-d\}} P(+b)\,P(d)\,P(a \mid +b, d)\,P(c \mid a)\,P(+e \mid +b, c)}{\displaystyle\sum_{b \in \{+b,-b\}} \sum_{a \in \{+a,-a\}} \sum_{c \in \{+c,-c\}} \sum_{d \in \{+d,-d\}} P(b)\,P(d)\,P(a \mid b, d)\,P(c \mid a)\,P(+e \mid b, c)}$$
   *(Forma equivalente colocando fatores independentes fora dos somatórios: $P(+b \mid +e) = \alpha\, P(+b) \sum_{d} P(d) \sum_{a} P(a \mid +b, d) \sum_{c} P(c \mid a)\,P(+e \mid +b, c)$, onde $\alpha = \frac{1}{P(+b,+e) + P(-b,+e)}$.)*

**(b) Cálculo de $P(+b \mid +e)$ usando *Rejection Sampling*:**
No *Rejection Sampling*, descartamos (rejeitamos) todas as amostras incompatíveis com a evidência $E = +e$ e calculamos a fração das amostras aceitas que possuem $B = +b$:

| # | Amostra | Evidência $E = +e$? | Status | Possui $B = +b$? |
|---:|:---|:---:|:---:|:---:|
| 1 | `(+a, +b, -c, +d, -e)` | Não (`-e`) | **Rejeitada** | — |
| 2 | `(+a, -b, +c, +d, +e)` | Sim (`+e`) | **Aceita** | Não (`-b`) |
| 3 | `(-a, +b, +c, +d, -e)` | Não (`-e`) | **Rejeitada** | — |
| 4 | `(+a, +b, +c, +d, +e)` | Sim (`+e`) | **Aceita** | **Sim (`+b`)** |
| 5 | `(+a, -b, -c, -d, +e)` | Sim (`+e`) | **Aceita** | Não (`-b`) |
| 6 | `(-a, -b, +c, +d, +e)` | Sim (`+e`) | **Aceita** | Não (`-b`) |

- Total de amostras aceitas (consistentes com $+e$): $N(+e) = 4$ (amostras 2, 4, 5 e 6).
- Dentre as aceitas, total com $B = +b$: $N(+b, +e) = 1$ (amostra 4).
- Estimativa por *Rejection Sampling*:
  $$\hat{P}(+b \mid +e) = \frac{N(+b, +e)}{N(+e)} = \mathbf{\frac{1}{4} = 0{,}25}$$

---

### Questão 4 — Árvore de Decisão (Entropia e Ganho de Informação)

**Enunciado:** Usando a tabela de 7 exemplos abaixo para predição de `Read` (`Yes`/`No`), com atributos:
- `Fiction`: `yes`/`no` (binário)
- `Age`: `new`/`old`/`ancient`
- `Award`: `yes`/`no` (binário)
- `Rating`: `hi`/`med`/`low`
- `Length`: `short`/`medium`/`long`

| Ex. | Fiction | Age | Award | Rating | Length | **Read** |
|---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | no | new | no | hi | short | **Yes** |
| 2 | yes | ancient | no | med | long | **No** |
| 3 | yes | old | yes | hi | medium | **Yes** |
| 4 | yes | old | no | med | medium | **No** |
| 5 | yes | ancient | yes | low | short | **No** |
| 6 | no | new | no | low | short | **No** |
| 7 | yes | new | no | hi | short | **Yes** |

- **(a)** Qual atributo seria o **melhor** para colocar na raiz da árvore? Justifique detalhadamente.
- **(b)** Que **atributo binário** seria a **pior** escolha para a raiz da árvore? Justifique detalhadamente.

#### Resolução Formal e Detalhada

**Passo 1 — Entropia inicial da raiz ($p = 3$ `Yes`, $n = 4$ `No`, total $p+n = 7$):**
$$I\!\left(\frac{3}{7}, \frac{4}{7}\right) = -\frac{3}{7}\log_2\frac{3}{7} - \frac{4}{7}\log_2\frac{4}{7} \approx 0{,}5239 + 0{,}4613 = \mathbf{0{,}9852\text{ bits}}$$

**Passo 2 — Cálculo de $\text{remainder}(A)$ e $IG(A) = I(\frac{3}{7}, \frac{4}{7}) - \text{remainder}(A)$ para TODOS os 5 atributos:**

1. **`Rating`** (`hi`, `med`, `low`):
   - `hi`: Ex. $\{1, 3, 7\} \to (3\text{Y}, 0\text{N}) \implies I(1, 0) = 0$
   - `med`: Ex. $\{2, 4\} \to (0\text{Y}, 2\text{N}) \implies I(0, 1) = 0$
   - `low`: Ex. $\{5, 6\} \to (0\text{Y}, 2\text{N}) \implies I(0, 1) = 0$
   $$\text{remainder}(\text{Rating}) = \frac{3}{7}(0) + \frac{2}{7}(0) + \frac{2}{7}(0) = \mathbf{0\text{ bits}}$$
   $$IG(\text{Rating}) = 0{,}9852 - 0 = \mathbf{0{,}9852\text{ bits}}$$

2. **`Age`** (`new`, `old`, `ancient`):
   - `new`: Ex. $\{1, 6, 7\} \to (2\text{Y}, 1\text{N}) \implies I(\frac{2}{3}, \frac{1}{3}) \approx 0{,}9183$
   - `old`: Ex. $\{3, 4\} \to (1\text{Y}, 1\text{N}) \implies I(\frac{1}{2}, \frac{1}{2}) = 1$
   - `ancient`: Ex. $\{2, 5\} \to (0\text{Y}, 2\text{N}) \implies I(0, 1) = 0$
   $$\text{remainder}(\text{Age}) = \frac{3}{7}(0{,}9183) + \frac{2}{7}(1) + \frac{2}{7}(0) \approx 0{,}6793\text{ bits} \implies IG(\text{Age}) \approx \mathbf{0{,}3060\text{ bits}}$$

3. **`Length`** (`short`, `medium`, `long`):
   - `short`: Ex. $\{1, 5, 6, 7\} \to (2\text{Y}, 2\text{N}) \implies I(\frac{1}{2}, \frac{1}{2}) = 1$
   - `medium`: Ex. $\{3, 4\} \to (1\text{Y}, 1\text{N}) \implies I(\frac{1}{2}, \frac{1}{2}) = 1$
   - `long`: Ex. $\{2\} \to (0\text{Y}, 1\text{N}) \implies I(0, 1) = 0$
   $$\text{remainder}(\text{Length}) = \frac{4}{7}(1) + \frac{2}{7}(1) + \frac{1}{7}(0) = \frac{6}{7} \approx 0{,}8571\text{ bits} \implies IG(\text{Length}) \approx \mathbf{0{,}1281\text{ bits}}$$

4. **`Fiction`** *(atributo binário: `yes`, `no`)*:
   - `no`: Ex. $\{1, 6\} \to (1\text{Y}, 1\text{N}) \implies I(\frac{1}{2}, \frac{1}{2}) = 1$
   - `yes`: Ex. $\{2, 3, 4, 5, 7\} \to (2\text{Y}, 3\text{N}) \implies I(\frac{2}{5}, \frac{3}{5}) = -\frac{2}{5}\log_2\frac{2}{5} - \frac{3}{5}\log_2\frac{3}{5} \approx 0{,}9710$
   $$\text{remainder}(\text{Fiction}) = \frac{2}{7}(1) + \frac{5}{7}(0{,}9710) \approx 0{,}9793\text{ bits} \implies IG(\text{Fiction}) \approx \mathbf{0{,}0060\text{ bits}}$$

5. **`Award`** *(atributo binário: `yes`, `no`)*:
   - `yes`: Ex. $\{3, 5\} \to (1\text{Y}, 1\text{N}) \implies I(\frac{1}{2}, \frac{1}{2}) = 1$
   - `no`: Ex. $\{1, 2, 4, 6, 7\} \to (2\text{Y}, 3\text{N}) \implies I(\frac{2}{5}, \frac{3}{5}) \approx 0{,}9710$
   $$\text{remainder}(\text{Award}) = \frac{2}{7}(1) + \frac{5}{7}(0{,}9710) \approx 0{,}9793\text{ bits} \implies IG(\text{Award}) \approx \mathbf{0{,}0060\text{ bits}}$$

**Respostas Finais:**
- **(a) Melhor atributo:** **`Rating`**, pois classifica perfeitamente todos os 7 exemplos em 1 único nível (todos os ramos são 100% puros: `hi -> Yes`, `med -> No`, `low -> No`), atingindo $\text{remainder}(\text{Rating}) = 0$ e o ganho máximo possível **$IG(\text{Rating}) = 0{,}9852\text{ bits}$**.
- **(b) Pior atributo binário:** Os únicos atributos binários do conjunto são **`Fiction`** e **`Award`**, e ambos **empatam exatamente como a pior escolha**, pois produzem partições de classes idênticas: um ramo $(1\text{Y}, 1\text{N})$ e outro $(2\text{Y}, 3\text{N})$, resultando no maior `remainder` ($\approx 0{,}9793\text{ bits}$) e menor ganho de informação (**$IG(\text{Fiction}) = IG(\text{Award}) \approx 0{,}0060\text{ bits}$**).

---

## Parte II — Resolução Completa de `Documento sem título.pdf`

### Questão 1 — Redes de Bayes: Eliminação de Variáveis (Miles)

**Enunciado:** Grafo direcionado com variáveis binárias $\{A, C, S, D, T\}$ e arestas $A \to S$, $C \to S$, $C \to D$, $S \to T$, $D \to T$.
CPTs fornecidas *(nota: há um erro tipográfico no cabeçalho da tabela de $C$ no PDF, impresso como `C | P(A)` onde claramente significa $P(C)$)*:
- $P(A)$: $-a: \frac{1}{4},\; +a: \frac{3}{4}$
- $P(C)$: $-c: \frac{4}{5},\; +c: \frac{1}{5}$
- $P(D \mid C)$: $(+c,+d): \frac{1}{4},\; (+c,-d): \frac{3}{4},\; (-c,+d): 0,\; (-c,-d): 1$
- $P(S \mid A, C)$:
  - $(+a,+c,+s): \frac{11}{16},\; (+a,+c,-s): \frac{5}{16}$
  - $(+a,-c,+s): \frac{1}{8},\; (+a,-c,-s): \frac{7}{8}$
  - $(-a,+c,+s): \frac{9}{10},\; (-a,+c,-s): \frac{1}{10}$
  - $(-a,-c,+s): \frac{1}{5},\; (-a,-c,-s): \frac{4}{5}$
- $P(T \mid S, D)$ — **selecionando apenas a evidência $T = -t$**:
  - $(-t, +d, +s): \frac{19}{20},\; (-t, +d, -s): \frac{3}{5},\; (-t, -d, +s): \frac{4}{5},\; (-t, -d, -s): 0$

**Pedido:** Calcule $P(C = +c \mid T = -t)$ usando **Eliminação de Variáveis**.

#### Resolução Formal e Detalhada

**Passo 1 — Fatores iniciais (já instanciados na evidência $T = -t$):**
Variável de consulta: $C$. Evidência: $T = -t$. Variáveis ocultas: $\{A, S, D\}$.
Fatores ativos:
$$\big[\; P(A),\;\; P(C),\;\; P(S \mid A, C),\;\; P(D \mid C),\;\; P(-t \mid S, D) \;\big]$$
Escolhemos a ordem de eliminação eficiente **$A \to S \to D$** (mantém todos os fatores pequenos!).

**Passo 2 — Eliminar $A$:**
Fatores que mencionam $A$: $P(A)$ e $P(S \mid A, C)$.
Fazemos o *Join* e somamos $A$ para fora, obtendo $f_1(C, S) = \sum_{a \in \{+a,-a\}} P(a)\,P(S \mid a, C)$:
- $f_1(+c, +s) = \frac{3}{4}\cdot\frac{11}{16} + \frac{1}{4}\cdot\frac{9}{10} = \frac{33}{64} + \frac{9}{40} = \frac{165 + 72}{320} = \mathbf{\dfrac{237}{320}}$
- $f_1(+c, -s) = \frac{3}{4}\cdot\frac{5}{16} + \frac{1}{4}\cdot\frac{1}{10} = \frac{15}{64} + \frac{1}{40} = \frac{75 + 8}{320} = \mathbf{\dfrac{83}{320}}$ *(conferência: $\frac{237+83}{320}=1$ ✓)*
- $f_1(-c, +s) = \frac{3}{4}\cdot\frac{1}{8} + \frac{1}{4}\cdot\frac{1}{5} = \frac{3}{32} + \frac{1}{20} = \frac{15 + 8}{160} = \mathbf{\dfrac{23}{160}}$
- $f_1(-c, -s) = \frac{3}{4}\cdot\frac{7}{8} + \frac{1}{4}\cdot\frac{4}{5} = \frac{21}{32} + \frac{4}{20} = \frac{105 + 32}{160} = \mathbf{\dfrac{137}{160}}$ *(conferência: $\frac{23+137}{160}=1$ ✓)*

Fatores restantes: $\big[\; P(C),\;\; P(D \mid C),\;\; f_1(C, S),\;\; P(-t \mid S, D) \;\big]$.

**Passo 3 — Eliminar $S$:**
Fatores que mencionam $S$: $f_1(C, S)$ e $P(-t \mid S, D)$.
Fazemos o *Join* e somamos $S$ para fora, obtendo $f_2(C, D, -t) = \sum_{s \in \{+s,-s\}} f_1(C, s)\,P(-t \mid s, D)$:
- $f_2(+c, +d, -t) = \frac{237}{320}\cdot\frac{19}{20} + \frac{83}{320}\cdot\frac{3}{5} = \frac{4503}{6400} + \frac{996}{6400} = \mathbf{\dfrac{5499}{6400}}$
- $f_2(+c, -d, -t) = \frac{237}{320}\cdot\frac{4}{5} + \frac{83}{320}\cdot 0 = \frac{237}{400} = \mathbf{\dfrac{237}{400}}$
- $f_2(-c, +d, -t) = \frac{23}{160}\cdot\frac{19}{20} + \frac{137}{160}\cdot\frac{3}{5} = \frac{437}{3200} + \frac{1644}{3200} = \mathbf{\dfrac{2081}{3200}}$
  *(Dica ninja de prova: como $P(+d \mid -c) = 0$, você nem precisaria multiplicar essa linha no Passo 4, mas calculá-la completa mostra rigor total!)*
- $f_2(-c, -d, -t) = \frac{23}{160}\cdot\frac{4}{5} + \frac{137}{160}\cdot 0 = \mathbf{\dfrac{23}{200}}$

Fatores restantes: $\big[\; P(C),\;\; P(D \mid C),\;\; f_2(C, D, -t) \;\big]$.

**Passo 4 — Eliminar $D$:**
Fatores que mencionam $D$: $P(D \mid C)$ e $f_2(C, D, -t)$.
Fazemos o *Join* e somamos $D$ para fora, obtendo $f_3(C, -t) = \sum_{d \in \{+d,-d\}} P(d \mid C)\,f_2(C, d, -t)$:
- $f_3(+c, -t) = \frac{1}{4}\cdot\frac{5499}{6400} + \frac{3}{4}\cdot\frac{237}{400} = \frac{5499}{25600} + \frac{11376}{25600} = \frac{16875}{25600} = \mathbf{\dfrac{675}{1024}}$ *(dividindo por 25!)*
- $f_3(-c, -t) = 0\cdot\frac{2081}{3200} + 1\cdot\frac{23}{200} = \mathbf{\dfrac{23}{200}}$

Fatores restantes: $\big[\; P(C),\;\; f_3(C, -t) \;\big]$.

**Passo 5 — *Join* final em $C$ e Normalização:**
Multiplicamos $f_4(C, -t) = P(C)\,f_3(C, -t)$:
- $P(+c, -t) = \frac{1}{5} \cdot \frac{675}{1024} = \mathbf{\dfrac{135}{1024}} \approx 0{,}131836$
- $P(-c, -t) = \frac{4}{5} \cdot \frac{23}{200} = \frac{92}{1000} = \mathbf{\dfrac{23}{250}} = 0{,}092000$

Constante de normalização $Z = P(-t)$:
$$Z = \frac{135}{1024} + \frac{23}{250} = \frac{135 \times 125 + 23 \times 512}{128000} = \frac{16875 + 11776}{128000} = \frac{28651}{128000} \approx 0{,}223836$$

**Resposta Final:**
$$P(C = +c \mid T = -t) = \frac{135/1024}{28651/128000} = \mathbf{\frac{16875}{28651} \approx 0{,}5890 \; (58{,}90\%)}$$

---

### Questão 2 — MDPs: Jim e Pam (Avaliação, Iteração de Valor e Iteração de Política)

**Enunciado:** Grade de 5 quadrados $s_J \in \{1, 2, 3, 4, 5\}$. Jim começa em $s_J = 1$; Pam fixa em $5$.
- Ação `right` (mover para a direita): sucesso com prob. $p$ (vai para $\min(s_J + 1, 5)$) e falha com prob. $1 - p$ (fica em $s_J$); **recompensa sempre $0$**.
- Ação `rest` (descansar/parar): sempre funciona (fica em $s_J$) com recompensa **$R(d) = 4^{5 - d}$**, onde $d = 5 - s_J$ é a distância até Pam.
  Logo, como $5 - d = 5 - (5 - s_J) = s_J$, a recompensa de `rest` no quadrado $s_J$ é simplesmente:
  $$R(s_J, \text{rest}) = 4^{s_J} \implies R(1)=4,\;\; R(2)=16,\;\; R(3)=64,\;\; R(4)=256,\;\; R(5)=1024$$
- MDP de horizonte infinito sem estado terminal. Dica: soma de série geométrica $S = \dfrac{a}{1 - r}$.

#### Resolução Formal e Detalhada

**(a) $\gamma = 0{,}5$, Jim começa em $s_J = 1$. Para quais valores de $p$ a Política 1 é superior à Política 2?**
- **Política 1 ($\pi_1$ — descansar sempre em $s_J = 1$):**
  $$V^{\pi_1}(1) = 4 + 0{,}5(4) + 0{,}5^2(4) + \dots = \frac{4}{1 - 0{,}5} = \mathbf{8}$$
- **Política 2 ($\pi_2$ — tenta `right` uma vez em $s_J=1$ e depois descansa para sempre onde cair):**
  - Se `right` tem sucesso (prob. $p$), cai em $s_J = 2$ e descansa para sempre, obtendo a partir dali $V_{\text{rest}}(2) = \dfrac{16}{1 - 0{,}5} = 32$.
  - Se `right` falha (prob. $1 - p$), fica em $s_J = 1$ e descansa para sempre, obtendo $V_{\text{rest}}(1) = \dfrac{4}{1 - 0{,}5} = 8$.
  - Como o primeiro passo (`right`) tem recompensa imediata $0$ e desconto $\gamma = 0{,}5$:
    $$V^{\pi_2}(1) = 0 + 0{,}5 \big[ p \cdot 32 + (1 - p) \cdot 8 \big] = 0{,}5(8 + 24p) = \mathbf{4 + 12p}$$
- **Condição $\pi_1$ superior a $\pi_2$:**
  $$V^{\pi_1}(1) > V^{\pi_2}(1) \iff 8 > 4 + 12p \iff 4 > 12p \iff \mathbf{0 \le p < \frac{1}{3}}$$

**(b) Suponha $p = 1$, $s_J = 1$. Para quais valores de $\gamma \in [0, 1)$ a Política 1 é superior à Política 2?**
- $V^{\pi_1}(1) = \dfrac{4}{1 - \gamma}$
- $V^{\pi_2}(1) = 0 + \gamma \cdot V_{\text{rest}}(2) = \gamma \cdot \dfrac{16}{1 - \gamma} = \dfrac{16\gamma}{1 - \gamma}$
- Como $1 - \gamma > 0$ para $\gamma \in [0, 1)$:
  $$V^{\pi_1}(1) > V^{\pi_2}(1) \iff \frac{4}{1 - \gamma} > \frac{16\gamma}{1 - \gamma} \iff 4 > 16\gamma \iff \mathbf{0 \le \gamma < \frac{1}{4} \; (0{,}25)}$$

**(c) Assuma $\gamma = 0{,}5$ e $p = 1$. Realizar 2 iterações de Iteração de Valor ($V_0 = 0$):**
Equação de Bellman síncrona (com $p = 1$):
$$V_{k+1}(s) = \max\Big\{ \underbrace{4^s + 0{,}5\,V_k(s)}_{\text{rest}},\;\; \underbrace{0 + 0{,}5\,V_k(\min(s+1, 5))}_{\text{right}} \Big\}$$
- **Iteração $k=1$ ($V_1$ a partir de $V_0 = 0$):**
  Para todo $s$, $Q_1(s, \text{right}) = 0$ e $Q_1(s, \text{rest}) = 4^s \implies V_1(s) = 4^s$:
  $$V_1 = [4,\; 16,\; 64,\; 256,\; 1024]$$
- **Iteração $k=2$ ($V_2$ a partir de $V_1$):**
  - $s=1$: $\max\{4 + 0{,}5(4),\; 0{,}5(16)\} = \max\{6,\; 8\} = \mathbf{8}$
  - $s=2$: $\max\{16 + 0{,}5(16),\; 0{,}5(64)\} = \max\{24,\; 32\} = \mathbf{32}$
  - $s=3$: $\max\{64 + 0{,}5(64),\; 0{,}5(256)\} = \max\{96,\; 128\} = \mathbf{128}$
  - $s=4$: $\max\{256 + 0{,}5(256),\; 0{,}5(1024)\} = \max\{384,\; 512\} = \mathbf{512}$
  - $s=5$: $\max\{1024 + 0{,}5(1024),\; 0{,}5(1024)\} = \max\{1536,\; 512\} = \mathbf{1536}$

| Iteração | $s_J = 1$ | $s_J = 2$ | $s_J = 3$ | $s_J = 4$ | $s_J = 5$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **$V_0$** | $0$ | $0$ | $0$ | $0$ | $0$ |
| **$V_1$** | **$4$** | **$16$** | **$64$** | **$256$** | **$1024$** |
| **$V_2$** | **$8$** | **$32$** | **$128$** | **$512$** | **$1536$** |

**(d) Assuma $\gamma = 0{,}5$ e $p = 1$. Realizar 2 iterações de Iteração de Política a partir de $\pi_i = (\text{right}, \text{rest}, \text{rest}, \text{rest}, \text{rest})$:**

- **Iteração 1 — Avaliação exata de $V^{\pi_i}$:**
  Para $s \in \{2,3,4,5\}$, $\pi_i(s) = \text{rest} \implies V^{\pi_i}(s) = \dfrac{4^s}{1 - 0{,}5} = 2 \cdot 4^s$:
  - $V^{\pi_i}(2) = 32$, $V^{\pi_i}(3) = 128$, $V^{\pi_i}(4) = 512$, $V^{\pi_i}(5) = 2048$.
  Para $s = 1$, $\pi_i(1) = \text{right} \implies V^{\pi_i}(1) = 0 + 0{,}5 \cdot V^{\pi_i}(2) = 0{,}5(32) = \mathbf{16}$.
- **Iteração 1 — Melhoria de Política ($\pi_{i+1}$):**
  - $s=1$: $Q(\text{rest})=4 + 0{,}5(16)=12$ vs. $Q(\text{right})=0{,}5(32)=16 \implies \mathbf{\text{right}}$
  - $s=2$: $Q(\text{rest})=16 + 0{,}5(32)=32$ vs. $Q(\text{right})=0{,}5(128)=64 \implies \mathbf{\text{right}}$
  - $s=3$: $Q(\text{rest})=64 + 0{,}5(128)=128$ vs. $Q(\text{right})=0{,}5(512)=256 \implies \mathbf{\text{right}}$
  - $s=4$: $Q(\text{rest})=256 + 0{,}5(512)=512$ vs. $Q(\text{right})=0{,}5(2048)=1024 \implies \mathbf{\text{right}}$
  - $s=5$: $Q(\text{rest})=1024 + 0{,}5(2048)=2048$ vs. $Q(\text{right})=0{,}5(2048)=1024 \implies \mathbf{\text{rest}}$

- **Iteração 2 — Avaliação exata de $V^{\pi_{i+1}}$** (resolvendo de $s=5$ para trás):
  - $V^{\pi_{i+1}}(5) = \frac{1024}{1 - 0{,}5} = \mathbf{2048}$
  - $V^{\pi_{i+1}}(4) = 0{,}5 \cdot 2048 = \mathbf{1024}$
  - $V^{\pi_{i+1}}(3) = 0{,}5 \cdot 1024 = \mathbf{512}$
  - $V^{\pi_{i+1}}(2) = 0{,}5 \cdot 512 = \mathbf{256}$
  - $V^{\pi_{i+1}}(1) = 0{,}5 \cdot 256 = \mathbf{128}$
- **Iteração 2 — Melhoria de Política ($\pi_{i+2}$):**
  - $s=1$: $Q(\text{rest}) = 4 + 64 = 68 < 128 = Q(\text{right}) \implies \mathbf{\text{right}}$
  - $s=2$: $Q(\text{rest}) = 16 + 128 = 144 < 256 = Q(\text{right}) \implies \mathbf{\text{right}}$
  - $s=3$: $Q(\text{rest}) = 64 + 256 = 320 < 512 = Q(\text{right}) \implies \mathbf{\text{right}}$
  - $s=4$: $Q(\text{rest}) = 256 + 512 = 768 < 1024 = Q(\text{right}) \implies \mathbf{\text{right}}$
  - $s=5$: $Q(\text{rest}) = 1024 + 1024 = 2048 > 1024 = Q(\text{right}) \implies \mathbf{\text{rest}}$
  *(Convergiu! Como $\pi_{i+2} = \pi_{i+1}$, essa já é a **política ótima $\pi^*$**!)*

| Linha | $s_J = 1$ | $s_J = 2$ | $s_J = 3$ | $s_J = 4$ | $s_J = 5$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **$\pi_i$** | `right` | `rest` | `rest` | `rest` | `rest` |
| **$V^{\pi_i}$** | **$16$** | **$32$** | **$128$** | **$512$** | **$2048$** |
| **$\pi_{i+1}$** | **`right`** | **`right`** | **`right`** | **`right`** | **`rest`** |
| **$V^{\pi_{i+1}}$** | **$128$** | **$256$** | **$512$** | **$1024$** | **$2048$** |
| **$\pi_{i+2}$** | **`right`** | **`right`** | **`right`** | **`right`** | **`rest`** |

---

### Questão 3 — Redes Neurais: Função de Ativação Quadrática

**Enunciado:** Entrada $x \in \mathbb{R}^d$, saída escalar $\hat{y} \in \mathbb{R}$, matrizes/vetores de pesos $W_1 \in \mathbb{R}^{n_1 \times n_2}$ e $W_2 \in \mathbb{R}^k$ ($k, d$ conhecidos).
$$z = W_1 x, \qquad \hat{y} := W_2^T g(z), \qquad g(z) := z \odot z \;\; (\text{produto elemento a elemento / Hadamard})$$
$$\mathcal{L}(W_1; W_2) := \frac{1}{2}(y - \hat{y})^2$$
- **(a)** Quais os valores de $n_1$ e $n_2$ em termos de $k$ e $d$?
- **(b)** Calcule $\dfrac{\partial \mathcal{L}}{\partial W_2}$ e $\dfrac{\partial \mathcal{L}}{\partial x}$ explicitamente (sem escrever $g'$; pode incluir $x, y, \hat{y}, W_1, W_2$ e $z$).

#### Resolução Formal e Detalhada

**(a) Dimensões $n_1$ e $n_2$:**
1. Para o produto matriz-vetor $z = W_1 x$ estar bem definido com $W_1 \in \mathbb{R}^{n_1 \times n_2}$ e $x \in \mathbb{R}^{d \times 1}$, o número de colunas de $W_1$ deve igualar a dimensão de $x$:
   $$n_2 = d \qquad \text{e o resultado é } z \in \mathbb{R}^{n_1}$$
2. A ativação elemento a elemento preserva dimensão: $g(z) = z \odot z \in \mathbb{R}^{n_1}$.
3. Para o produto interno $\hat{y} = W_2^T g(z) \in \mathbb{R}$ estar bem definido com $W_2^T \in \mathbb{R}^{1 \times k}$ e $g(z) \in \mathbb{R}^{n_1 \times 1}$, exige-se:
   $$n_1 = k$$
**Resposta (a):** **$n_1 = k$** e **$n_2 = d$** (logo $W_1 \in \mathbb{R}^{k \times d}$ e $z \in \mathbb{R}^k$).

**(b) Derivadas $\dfrac{\partial \mathcal{L}}{\partial W_2}$ e $\dfrac{\partial \mathcal{L}}{\partial x}$ (sem $g'$):**

Primeiro, calculamos a derivada comum da perda em relação à saída $\hat{y}$:
$$\frac{\partial \mathcal{L}}{\partial \hat{y}} = \frac{\partial}{\partial \hat{y}}\left[\frac{1}{2}(y - \hat{y})^2\right] = -(y - \hat{y}) = (\hat{y} - y)$$

**1. Cálculo de $\dfrac{\partial \mathcal{L}}{\partial W_2} \in \mathbb{R}^k$:**
Escrevendo $\hat{y} = W_2^T (z \odot z) = \sum_{i=1}^{k} W_{2,i}\, z_i^2$, derivamos em relação à $i$-ésima componente $W_{2,i}$:
$$\frac{\partial \hat{y}}{\partial W_{2,i}} = z_i^2 \implies \frac{\partial \mathcal{L}}{\partial W_{2,i}} = \frac{\partial \mathcal{L}}{\partial \hat{y}}\frac{\partial \hat{y}}{\partial W_{2,i}} = (\hat{y} - y)\, z_i^2$$
Em forma vetorial (sem usar $g$):
$$\mathbf{\frac{\partial \mathcal{L}}{\partial W_2} = (\hat{y} - y)\,(z \odot z)} \quad \big(\text{equivalente a } -(y - \hat{y})(z \odot z) \in \mathbb{R}^k\big)$$

**2. Cálculo de $\dfrac{\partial \mathcal{L}}{\partial x} \in \mathbb{R}^d$:**
Aplicamos a Regra da Cadeia atravessando $\mathcal{L} \to \hat{y} \to z \to x$:
- Em relação a cada componente $z_i$ ($i = 1, \dots, k$): como $\hat{y} = \sum_{m=1}^{k} W_{2,m}\, z_m^2$:
  $$\frac{\partial \hat{y}}{\partial z_i} = 2\, W_{2,i}\, z_i \implies \frac{\partial \mathcal{L}}{\partial z_i} = \frac{\partial \mathcal{L}}{\partial \hat{y}}\frac{\partial \hat{y}}{\partial z_i} = 2(\hat{y} - y)\, W_{2,i}\, z_i$$
  Em forma vetorial: $\dfrac{\partial \mathcal{L}}{\partial z} = 2(\hat{y} - y)\,(W_2 \odot z) \in \mathbb{R}^k$.
- Como $z_i = \sum_{j=1}^{d} W_{1,ij}\, x_j$, temos $\dfrac{\partial z_i}{\partial x_j} = W_{1,ij}$. Pela regra da cadeia multivariável, para cada componente $x_j$ ($j = 1, \dots, d$):
  $$\frac{\partial \mathcal{L}}{\partial x_j} = \sum_{i=1}^{k} \frac{\partial \mathcal{L}}{\partial z_i}\frac{\partial z_i}{\partial x_j} = 2(\hat{y} - y) \sum_{i=1}^{k} W_{1,ij}\, W_{2,i}\, z_i = 2(\hat{y} - y)\left[ W_1^T (W_2 \odot z) \right]_j$$
Em forma matricial/vetorial compacta (vetor coluna $\in \mathbb{R}^d$):
$$\mathbf{\frac{\partial \mathcal{L}}{\partial x} = 2(\hat{y} - y)\, W_1^T (W_2 \odot z)}$$
*(Nota: na convenção de vetor linha $\in \mathbb{R}^{1 \times d}$, escreve-se a transposta $2(\hat{y}-y)(W_2 \odot z)^T W_1$; apresentar a derivação componente-a-componente + a forma matricial acima elimina qualquer ambiguidade de convenção!)*

---

## Parte III — Exercícios Resolvidos de `CS_188_Su26_Discussion_5_SOLUTION.pdf`

### Q1. Reinforcement Learning (Model-Based, Propagação Q-Learning e V/F)

**Episódio de 9 transições** sobre estados $\{A, B, C, X\}$ e ações $\{\leftarrow, \rightarrow\}$ (repetido ciclicamente na parte b):
`1:(A,→,B,0)`, `2:(B,→,C,0)`, `3:(C,←,B,0)`, `4:(B,←,A,0)`, `5:(A,→,B,0)`, `6:(B,→,A,0)`, `7:(A,→,B,0)`, `8:(B,→,C,0)`, `9:(C,→,X,1)`.

- **(a) RL Baseado em Modelo:**
  - O par `(B, →)` ocorre 3 vezes (passos 2, 6, 8): vai para `C` em 2 deles (passos 2 e 8) e para `A` em 1 (passo 6) $\implies \hat{T}(B, \rightarrow, C) = \mathbf{\dfrac{2}{3}}$.
  - A transição `(C, →, X)` ocorre 1 vez (passo 9) com recompensa $1 \implies \hat{R}(C, \rightarrow, X) = \mathbf{1}$.

- **(b) Em qual iteração do Q-Learning (com $Q_0 = 0, \alpha > 0, \gamma > 0$) $Q(A, \rightarrow)$ e $Q(B, \leftarrow)$ ficam pela primeira vez diferentes de zero?**
  - **Ciclo 1 (iterações 1–9):**
    - Iterações 1–8: todas têm $r = 0$ e todos os $Q = 0 \implies$ nada muda.
    - **Iteração 9** `(C, →, X, 1)`: $Q(C, \rightarrow)$ recebe $r=1 \implies \mathbf{Q(C, \rightarrow) > 0}$ (agora $\max_a Q(C, a) > 0$).
  - **Ciclo 2 (iterações 10–18):**
    - Iteração 10 `(A, →, B, 0)`: $\max_a Q(B, a) = 0 \implies Q(A, \rightarrow) = 0$.
    - **Iteração 11** `(B, →, C, 0)`: $\max_a Q(C, a) > 0 \implies \mathbf{Q(B, \rightarrow) > 0}$ (agora $\max_a Q(B, a) > 0$).
    - Iteração 12 `(C, ←, B, 0)`: $Q(C, \leftarrow) > 0$.
    - Iteração 13 `(B, ←, A, 0)`: $\max_a Q(A, a) = 0$ ainda! $\implies Q(B, \leftarrow) = 0$.
    - **Iteração 14** `(A, →, B, 0)`: agora $\max_a Q(B, a) > 0 \implies \mathbf{Q(A, \rightarrow) > 0}$! (**Resposta 1: `14`**)
    - Iterações 15–18: `(B, ←, A)` não aparece mais no Ciclo 2!
  - **Ciclo 3 (iterações 19–27):**
    - **Iteração 22** `(B, ←, A, 0)` (4ª transição do Ciclo 3: $18 + 4 = 22$): agora $\max_a Q(A, a) = Q(A, \rightarrow) > 0 \implies \mathbf{Q(B, \leftarrow) > 0}$! (**Resposta 2: `22`**)

- **(c) Verdadeiro ou Falso:**
  1. **(i) `TRUE`** — *No Q-Learning você não aprende o modelo.* (É livre de modelo: aprende a política ótima / valores Q explicitamente e o modelo apenas implicitamente.)
  2. **(ii) `FALSE`** — *No TD Learning, multiplicar todas as recompensas por $p \neq 0$ garante encontrar a política ótima.* (TD Learning é RL passivo: **não** busca política ótima, apenas avalia a política fixa dada!)
  3. **(iii) `FALSE`** — *Na Avaliação Direta você recalcula valores de estados após cada transição.* (Avaliação Direta calcula a partir de **episódios completos**, não transições isoladas.)
  4. **(iv) `FALSE`** — *Q-Learning exige que todas as amostras venham da política ótima.* (Q-Learning é **off-policy**: converge para $Q^*$ mesmo agindo de forma subótima ou exploratória.)

---

### Q2. RL — Q-Learning Tabular e Q-Learning Aproximado

**Setup:** Estados `[A, B, C]`, ações `[Stop, Go]`, $\gamma = 1$, $\alpha = 0{,}5$.

- **(a) Q-Learning Tabular** ($Q = 0$ inicialmente). Amostras em ordem: `1:(A,Go,B,2)`, `2:(C,Stop,A,0)`, `3:(B,Stop,A,-2)`, `4:(B,Go,C,-6)`, `5:(C,Go,A,2)`, `6:(A,Go,A,-2)`:
  1. Amostra 1 `(A, Go, B, 2)`: $Q(A, \text{Go}) \leftarrow 0{,}5(0) + 0{,}5(2 + 1\cdot\max_a Q(B,a)) = 0{,}5(2 + 0) = \mathbf{1}$
  2. Amostra 2 `(C, Stop, A, 0)`: $Q(C, \text{Stop}) \leftarrow 0{,}5(0) + 0{,}5(0 + 1\cdot\max\{1, 0\}) = \mathbf{0{,}5}$
  3. Amostras 3 e 4 atualizam apenas o estado $B$ ($Q(B,\text{Stop}) = -0{,}5$, $Q(B,\text{Go}) = -2{,}75$) — não afetam $A$ nem $C$.
  4. Amostra 5 `(C, Go, A, 2)`: $Q(C, \text{Go}) \leftarrow 0{,}5(0) + 0{,}5(2 + 1\cdot\max\{1, 0\}) = 0{,}5(3) = \mathbf{1{,}5}$
  5. Amostra 6 `(A, Go, A, -2)` atualiza $Q(A,\text{Go})$ para $0$ **depois** que as amostras 2 e 5 já ocorreram!
  **Respostas:** (i) $Q(C, \text{Stop}) = \mathbf{0{,}5}$; (ii) $Q(C, \text{Go}) = \mathbf{1{,}5}$.

- **(b) Q-Learning Aproximado** com $f_1(s, a) = 1$, $f_2(s, a) = +1$ (se `Go`) ou $-1$ (se `Stop`), iniciando com $w_1 = 0, w_2 = 0$:
  - **Atualização 1 — Amostra `(A, Go, B, 4)`:**
    - $Q(A, \text{Go}) = 0(1) + 0(1) = 0$; $\max_a Q(B, a) = 0$.
    - $\text{difference} = [4 + 1\cdot 0] - 0 = 4$.
    - (i) $w_1 \leftarrow 0 + 0{,}5(4)(1) = \mathbf{2}$; (ii) $w_2 \leftarrow 0 + 0{,}5(4)(1) = \mathbf{2}$.
  - **Atualização 2 — Amostra `(B, Stop, A, 0)`:**
    - Com os **novos pesos** $(2, 2)$:
      - $Q(B, \text{Stop}) = 2(1) + 2(-1) = 0$.
      - Em $s' = A$: $Q(A, \text{Go}) = 2(1)+2(1) = 4$ e $Q(A, \text{Stop}) = 2(1)+2(-1) = 0 \implies \max_a Q(A, a) = \mathbf{4}$.
    - $\text{difference} = [0 + 1\cdot 4] - 0 = 4$.
    - Usando características da origem $(B, \text{Stop})$: $f_1 = 1,\; f_2 = -1$:
      - (iii) $w_1 \leftarrow 2 + 0{,}5(4)(1) = \mathbf{4}$
      - (iv) $w_2 \leftarrow 2 + 0{,}5(4)(-1) = \mathbf{0}$

---

## Parte IV — Índice das Questões Extras de Prova UFPR em `RL.pptx`

Além dos dois PDFs de prova acima, [`RL.pptx`](file:///home/dcr/repos/trabAI/prova/RL.pptx) contém **duas questões completas de provas anteriores da UFPR** embutidas como imagens, já resolvidas passo a passo no [Módulo 02](file:///home/dcr/repos/trabAI/prova/guia_prova2/02_MDPS_E_APRENDIZADO_POR_REFORCO.md):
1. **Questão 4 (30 pts) — Táxi 5×5 Q-Learning Tabular** (`RL.pptx` Slides 11–13, ver Módulo 02 §4.3 Ex. B): escolha gulosa `Baixo` ($Q=1$) $\to$ atualização para $Q(S_0,\text{Baixo}) = 0{,}0$ (com alerta sobre o typo `$S_2$` $\to$ `$S_1$` no slide 13).
2. **Questão Pacman 2×3 — Q-Learning Aproximado** (`RL.pptx` Slides 19–22, ver Módulo 02 §6.4 Ex. B): extração $F_g=2, F_p=1 \to Q=190 \to Q_{\mathbf{new}}=40 \to w_g=-50,\; w_p=-85$.
