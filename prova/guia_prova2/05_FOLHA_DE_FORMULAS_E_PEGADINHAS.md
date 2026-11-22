# Módulo 05 — Folha de Fórmulas, Algoritmos em 3 Linhas e Pegadinhas de Prova

> **Use este arquivo para sua revisão final amanhã (14:00–16:30, antes da prova às 17:00).** Contém **todas** as fórmulas essenciais na notação exata dos slides, os algoritmos condensados, tabelas de decisão rápida e o catálogo de pegadinhas e erratas dos slides.

---

## 1. Fórmulas de Incerteza e Redes Bayesianas (Módulo 01)

| Conceito | Fórmula Exata (Notação dos Slides) |
| :--- | :--- |
| **Axiomas de Probabilidade** | $\forall x:\; P(X=x) \ge 0 \quad \text{e} \quad \sum_x P(X=x) = 1$ |
| **Marginalização (*Sum out*)** | $P(X_1 = x_1) = \sum_{x_2} P(X_1=x_1, X_2=x_2)$ |
| **Condicional & Produto** | $P(a \mid b) = \dfrac{P(a, b)}{P(b)} \iff P(x, y) = P(y)\,P(x \mid y)$ |
| **Normalização ($Z$ / $\alpha$)** | $P(Q \mid \mathbf{e}) = \dfrac{1}{Z} P(Q, \mathbf{e}) = \alpha P(Q, \mathbf{e}), \quad Z = \sum_q P(q, \mathbf{e}) = P(\mathbf{e})$ |
| **Regra da Cadeia** | $P(x_1, \dots, x_n) = \prod_{i=1}^n P(x_i \mid x_1 \dots x_{i-1})$ |
| **Regra de Bayes** | $P(x \mid y) = \dfrac{P(y \mid x)\,P(x)}{P(y)}$ |
| **Independência Absoluta & Condicional** | $X \perp\!\!\!\perp Y \iff P(x,y)=P(x)P(y)$ · $X \perp\!\!\!\perp Y \mid Z \iff P(x,y\mid z)=P(x\mid z)P(y\mid z) \iff P(x\mid z,y)=P(x\mid z)$ |
| **Semântica Global de RB** | $P(x_1, \dots, x_n) = \prod_{i=1}^n P\big(x_i \mid \text{parents}(X_i)\big)$ |
| **Tamanho da RB vs. Conjunta** | Conjunta: $O(d^n)$ / $2^N$ · RB (máx. $k$ pais): $O(n\cdot 2^k)$ parâmetros / $O(N\cdot 2^{k+1})$ entradas |
| **Cobertura de Markov (*Markov Blanket*)** | $\text{MB}(X) = \text{Pais}(X) \cup \text{Filhos}(X) \cup \text{Outros pais dos filhos}(X)$ |
| **Ou-Ruidoso (*Noisy-OR*)** | $P(\neg y \mid x_1\dots x_k) = \prod_{i:\,x_i=\text{T}} q_i$, $\quad P(y \mid \cdot) = 1 - \prod_{i:\,x_i=\text{T}} q_i$ ($O(k)$ parâmetros) |
| **Enumeração em RB** | $P(Q \mid \mathbf{e}) = \alpha \sum_{\mathbf{h}} \prod_{i=1}^n P(x_i \mid \text{parents}(X_i))$ — Tempo $O(2^n)$, Espaço $O(n)$ |
| **Factor Zoo (somas)** | $P(X,Y) \to 1$ · $P(x,Y) \to P(x)$ · $P(Y\mid x) \to 1$ · $P(X\mid Y) \to \|Y\|$ · $P(y\mid X) \to \text{quem sabe!}$ |
| **Likelihood Weighting** | $w = \prod_{i=1}^m P(e_i \mid \text{Parents}(E_i))$; $S_{WS}(\mathbf{z},\mathbf{e})\cdot w(\mathbf{z},\mathbf{e}) = P(\mathbf{z},\mathbf{e})$ |
| **Gibbs (Reamostragem Local)** | $P(X_i \mid \text{todos outros}) \propto P(X_i \mid \text{Parents}(X_i)) \prod_{Y \in \text{Filhos}(X_i)} P(y \mid \text{Parents}(Y))$ |

### D-Separação em 10 Segundos
- **Cadeia Causal ($A \to B \to C$)** e **Causa Comum ($A \leftarrow B \to C$)**: **ATIVA** se $B$ oculto; **BLOQUEADA** se $B$ observado.
- **Efeito Comum / V-Estrutura ($A \to B \leftarrow C$)**: **BLOQUEADA** se nem $B$ nem descendente observado; **ATIVA** se $B$ **ou qualquer descendente de $B$** for observado!
- **Regra do caminho:** **1 tripla inativa bloqueia o caminho.** **Todos os caminhos bloqueados $\implies$ d-separados ($X \perp\!\!\!\perp Y \mid E$).**

### Algoritmos de Inferência em 3 Linhas
- **Eliminação de Variáveis (VE):** (1) Instancie evidências nas TPCs; (2) Para cada variável oculta $H$: **Join** todos os fatores com $H$, depois **Sum out** $H$; (3) **Join** fatores restantes e **Normalize**. *(Complexidade = maior fator $d^k$; ordem ótima é NP-Completo; inferência geral é NP-Hard via 3-SAT; em poli-árvores é polinomial.)*
- **Prior Sampling:** Sorteie cada $X_i \sim P(X_i\mid \text{Parents}(X_i))$ em ordem topológica; conte frequências.
- **Rejection Sampling:** Igual ao Prior, mas aborte imediatamente se qualquer evidência sorteada $x_i \neq e_i$; normalize as aceitas.
- **Likelihood Weighting:** Em ordem topológica, fixe evidências ($w \leftarrow w \cdot P(e_i \mid \text{Parents}(E_i))$) e sorteie não-evidências; some os pesos $w$ por valor de $Q$ e normalize.
- **Gibbs Sampling:** Fixe evidências; inicialize o resto aleatoriamente; repita reamostrando 1 variável $X_i$ por vez dada sua **Cobertura de Markov**; conte e normalize.

---

## 2. Fórmulas de MDPs e Aprendizado por Reforço (Módulo 02)

| Conceito | Fórmula Exata (Notação dos Slides) |
| :--- | :--- |
| **Série Geométrica Infinita** | $S = a + a\gamma + a\gamma^2 + \dots = \dfrac{a}{1 - \gamma} \quad (0 \le \gamma < 1)$ |
| **Iteração de Valor (MDP)** | $V_{k+1}(s) \leftarrow \max_a \sum_{s'} T(s,a,s')\big[R(s,a,s') + \gamma V_k(s')\big]$ |
| **Avaliação de Política (MDP)** | $V^\pi(s) = \sum_{s'} T(s,\pi(s),s')\big[R(s,\pi(s),s') + \gamma V^\pi(s')\big]$ |
| **Melhoria de Política (MDP)** | $\pi_{i+1}(s) = \arg\max_a \sum_{s'} T(s,a,s')\big[R(s,a,s') + \gamma V^{\pi_i}(s')\big]$ |
| **Iteração de Valor Q** | $Q_{k+1}(s,a) \leftarrow \sum_{s'} T(s,a,s')\big[R(s,a,s') + \gamma \max_{a'} Q_k(s',a')\big]$ |
| **RL Baseado em Modelo (ADP)** | $\hat{T}(s,a,s') = \dfrac{\text{count}(s,a,s')}{\text{count}(s,a)}$, $\quad \hat{R}(s,a,s') =$ média das recompensas observadas |
| **Avaliação Direta (Passivo MF)** | $V^\pi(s) =$ média da soma descontada de recompensas a partir de cada visita a $s$ até o fim do episódio |
| **TD Learning (Passivo MF)** | $\text{sample} = R(s,\pi(s),s') + \gamma V^\pi(s')$; $\quad V^\pi(s) \leftarrow (1-\alpha)V^\pi(s) + \alpha \cdot \text{sample}$ |
| **Expansão EMA do TD** | $V_k^\pi(s) = \alpha\big[(1-\alpha)^{k-1}\text{sample}_1 + \dots + (1-\alpha)\text{sample}_{k-1} + \text{sample}_k\big]$ *(pesos antigos decaem!)* |
| **Q-Learning Tabular (CS188)** | $\text{sample} = R(s,a,s') + \gamma \max_{a'} Q(s',a')$; $\quad Q(s,a) \leftarrow (1-\alpha)Q(s,a) + \alpha \cdot \text{sample}$ |
| **Q-Learning (R&N / Determinístico)** | R&N: $Q(a,s) \leftarrow Q(a,s) + \alpha[R(s) + \gamma\max_{a'}Q(a',s') - Q(a,s)]$ · Det. ($\alpha=1$): $Q(s_t,a_t) \leftarrow r + \gamma\max_{a'}Q(s_{t+1},a')$ |
| **Função de Exploração** | $f(u, n) = u + \dfrac{k}{n} \iff f(s,a) = Q(s,a) + \dfrac{k}{N(s,a)}$; $\quad Q(s,a) \leftarrow_\alpha R(s,a,s') + \gamma \max_{a'} f(Q(s',a'), N(s',a'))$ |
| **Q-Learning Aproximado** | $Q(s,a) = \sum_{i=1}^n w_i f_i(s,a)$; $\;\; \text{difference} = [r + \gamma\max_{a'} Q(s',a')] - Q(s,a)$; $\;\; w_i \leftarrow w_i + \alpha[\text{difference}] f_i(s,a)$ |

---

## 3. Fórmulas de Aprendizado de Máquina (Módulo 03)

| Conceito | Fórmula Exata (Notação dos Slides) |
| :--- | :--- |
| **Entropia (Árvore de Decisão)** | $I(P(v_1),\dots,P(v_n)) = \sum_{i=1}^n -P(v_i)\log_2 P(v_i)$ · Binária: $I(\frac{p}{p+n}, \frac{n}{p+n})$ |
| **Remainder & Ganho de Informação** | $\text{remainder}(A) = \sum_{i=1}^v \dfrac{p_i+n_i}{p+n} I\!\left(\dfrac{p_i}{p_i+n_i}, \dfrac{n_i}{p_i+n_i}\right)$; $\quad IG(A) = I\!\left(\dfrac{p}{p+n}, \dfrac{n}{p+n}\right) - \text{remainder}(A)$ |
| **Naïve Bayes (Nominal)** | $\Pr[H \mid E] = \dfrac{\Pr[E_1\mid H]\cdots \Pr[E_n\mid H]\Pr[H]}{\Pr[E]}$ *(normalizar os numeradores!)* |
| **Suavização Laplace (+1) e $\mu$** | Laplace $+1$: $\dfrac{c + 1}{N_h + v}$ · Modificada ($\mu$): $\dfrac{c + \mu/v}{N_h + \mu}$ *(em `missing value`: omite o atributo!)* |
| **Naïve Bayes Gaussiano** | $\mu = \dfrac{1}{n}\sum_{i=1}^n x_i$, $\;\; \sigma = \sqrt{\dfrac{1}{n-1}\sum_{i=1}^n (x_i-\mu)^2}$, $\;\; f(x) = \dfrac{1}{\sqrt{2\pi}\,\sigma} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ |
| **Perceptron Binário ($\{+1,-1\}$)** | $y = +1$ se $w\cdot f(x) \ge 0$, senão $-1$; se $y \neq y^*$: $w \leftarrow w + y^*\cdot f(x)$ *(geometria: $w\cdot f = \|w\|\|f\|\cos\theta$)* |
| **Perceptron ($\{0,1\}$ R&N)** | $\mathbf{w} \leftarrow \mathbf{w} + \alpha\big(y - h_{\mathbf{w}}(\mathbf{x})\big)\mathbf{x}$ |
| **Perceptron Multiclasse** | $y = \arg\max_y w_y \cdot f(x)$; se $y \neq y^*$: $w_y \leftarrow w_y - f(x)$ e $w_{y^*} \leftarrow w_{y^*} + f(x)$ |
| **Limite de Erros (Perceptron)** | $\text{mistakes} < \dfrac{k}{\delta^2}$ (se linearmente separável com margem $\delta$) |
| **Regressão Logística Binária (MLE)** | $\phi(z) = \dfrac{1}{1+e^{-z}}$; $\; P(y=+1\mid x;w) = \phi(w\cdot f(x))$; $\; \max_w ll(w) = \max_w \sum_i \log P(y^{(i)}\mid x^{(i)}; w)$ |
| **Regressão Logística Multiclasse** | $P(y^{(i)}\mid x^{(i)}; w) = \dfrac{e^{w_{y^{(i)}}\cdot f(x^{(i)})}}{\sum_y e^{w_y\cdot f(x^{(i)})}}$ (*Softmax*) |
| **Regressão Linear (Equações Normais)** | $\text{Loss} = \dfrac{1}{2}\|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2 \implies \hat{\mathbf{w}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ |
| **Gradiente Ascendente & Variantes** | $w \leftarrow w + \alpha \nabla_w g(w)$ (Batch: $\sum_i$; SGA: 1 aleatório $j$; Mini-Batch: $\sum_{j\in J}$). *Descendente: sinal $-$!* |
| **Camada de Rede Neural (DNN)** | $z_i^{(k)} = g\!\left(\sum_j W_{i,j}^{(k-1,k)} z_j^{(k-1)}\right)$ |
| **Ativações & Derivadas** | **Sigmoid:** $g' = g(1-g)$ · **Tanh:** $\frac{e^z-e^{-z}}{e^z+e^{-z}}$, $g' = 1-g^2$ · **ReLU:** $\max(0,z)$, $g' = \mathbf{1}_{z>0}$ · **Softplus:** $\log(1+e^z)$ |
| **Regra da Cadeia / Backprop** | $f(x)=g(h(x)) \implies f'(x) = g'(h(x))h'(x)$ · Passagem *forward* (salva cache) $\to$ *backward* (multiplica derivadas locais) |

---

## 4. Tabela de Decisão Rápida ("Qual Método Usar?")

- **Quer inferência exata em RB pequena ou média?** $\to$ **Eliminação de Variáveis (VE)** (nunca monte a conjunta inteira!).
- **Quer testar se $X$ e $Y$ são independentes dado $E$ olhando só o grafo?** $\to$ **D-Separação** (cheque TODOS os caminhos não direcionados!).
- **RB gigante/densa com evidência rara nas folhas?** $\to$ **Amostragem de Gibbs** (Prior/Rejection desperdiçam quase tudo; Likelihood Weighting não condiciona nós *upstream* e gera pesos minúsculos).
- **MDP com $T, R$ conhecidos?** $\to$ **Iteração de Valor** ou **Iteração de Política** (use série geométrica $\frac{a}{1-\gamma}$ para avaliar política fixa de horizonte infinito).
- **RL sem $T, R$, política fixa?** $\to$ **TD Learning** (online passo a passo, mais rápido que Avaliação Direta; ou ADP se quiser estimar $\hat{T}, \hat{R}$).
- **RL sem $T, R$, quer aprender $\pi^*$?** $\to$ **Q-Learning Tabular** (se $|S|$ pequeno) ou **Q-Learning Aproximado** com features $f_i(s,a)$ (se $|S|$ gigante).
- **Classificação probabilística generativa rápida?** $\to$ **Naïve Bayes** (com Laplace se houver contagem $0$; Gaussiano se atributo for numérico).
- **Fronteira linear determinística vs. probabilística vs. não-linear?** $\to$ **Perceptron** (determinística, falha em não-separáveis) $\to$ **Regressão Logística** (probabilística, prefere maior margem) $\to$ **Rede Neural Profunda** (aprende características e funções não-lineares).

---

## 5. Top 15 Pegadinhas Clássicas de Prova (e Erratas dos Slides!)

1. **V-estrutura + Descendente observado:** Em $A \to B \leftarrow C$ com $B \to D$, observar **$D$** (filho/neto de $B$) **ativa** a V-estrutura em $B$!
2. **Seta invertida no grafo da prova:** Sempre confira a ponta de CADA seta antes de fatorar! Em `prova2.pdf` Q3, a seta é **`D -> A`** (para cima), logo $A$ tem **dois pais** $\{B, D\}$!
3. **Eliminação de Variáveis antes do Join completo:** Jamais some uma variável $H$ antes de multiplicar **todos** os fatores ativos que contêm $H$!
4. **Iteração de Valor vs. Iteração de Política:** Na Iteração de Valor, $V_{k+1}$ dá 1 passo usando $V_k$ (síncrono!). Na Avaliação de Política dentro da Iteração de Política, calcula-se $V^{\pi_i}$ **exato** (horizonte infinito / série geométrica / sistema linear)!
5. **Argumentos de Q-Learning Aproximado:** Na atualização $w_i \leftarrow w_i + \alpha[\text{difference}] f_i(s, a)$, multiplique pelas características da **origem $(s, a)$**, NUNCA de $(s', a')$! E na amostra seguinte, **recalcule** $\max_{a'} Q(s', a')$ com os **novos pesos**!
6. **TD Learning NÃO acha política ótima:** Ele é passivo e *on-policy* (só avalia a política fixa $\pi$). Já Q-Learning é *off-policy* (acha $Q^*$ mesmo agindo subótimo).
7. **Avaliação Direta NÃO atualiza a cada transição:** Só calcula médias sobre **episódios completos** (`disc5` Q1c-iii).
8. **Naïve Bayes — denominador da condicional:** $\Pr[X_j=a \mid Y=y]$ divide por **$\text{count}(Y=y)$**, nunca pelo total $N$! E nunca esqueça de multiplicar pela **priori $\Pr[Y=y]$**!
9. **Posição do bias no Perceptron:** Confira sempre no enunciado se o bias está na posição 0 (`[1, x1, x2]`, como em `perceptrons.pdf` p. 19) ou na última posição (`[x1, x2, 1]`, como em `prova2.pdf` Q2)!
10. **Separabilidade linear com pontos colineares / sanduichados:** Um ponto `+` que é combinação convexa de pontos `-` prova algebricamente que o conjunto **não** é linearmente separável!
11. **Entropia em Árvores:** Use **$\log_2$** (bits), nunca $\ln$; multiplique cada ramo pelo peso $\frac{p_i+n_i}{p+n}$; ramo puro tem $I=0$ e ramo 50/50 tem $I=1$.
12. **Regra da cadeia matricial em Redes Neurais:** Ao derivar $g(z) = z \odot z$, lembre que $g'(z) = 2z$ elemento a elemento (logo $\frac{\partial (W_2^T(z\odot z))}{\partial z} = 2(W_2 \odot z)$), e confira as dimensões de cada matriz ($W_1^T$ tem dimensão $d \times k$ para devolver um vetor em $\mathbb{R}^d$!).
13. **Erratas conhecidas nos slides oficiais (não decore errado!):**
    - `perceptrons.pdf` p. 23 inverte Tanh ($\frac{e^{2x}+1}{e^{2x}-1}$) $\to$ o correto é $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$ (`redesneurais.pdf` p. 8).
    - `naivebayes.pdf` p. 15 omite $\sqrt{\cdot}$ na fórmula do desvio padrão $\sigma$ $\to$ o correto tem raiz quadrada!
    - `CS 188 ML I` p. 60 imprime `IG(Patrons) = .0541` $\to$ o correto é **`0.541 bits`**.
    - `RB4.pptx` Slide 24 imprime `-l: 0.886` $\to$ o correto é **`0.866`** (corrigido no Slide 28).
    - `Bayes.pptx` Slide 40 omite $P(e)$ na fórmula escrita de $P(B\mid j,m)$ (presente na árvore do Slide 41).
    - `RL.pptx` Slide 13 escreve `$s' = S_2$` no gabarito vermelho, mas usa os números de **$S_1$** (resultado `0.0` está certo para $S_1$).
    - `naivebayes.pdf` p. 4 imprime `0,002` antes de escrever certo `0,02%` (`0,0002`); págs. 35–38 traduzem `CPTs` como *"tubos de raios catódicos"*.
14. **Traduções literais em `RB4`/`RB5`:** *Adesão/Junção* = Join; *Resumir* = Sum out; *Provas* = Evidence; *Articulação* = Joint; *Condicionamento de corte* = Cutset conditioning; *Número de capitais* = Nº de maiúsculas.
15. **Sinal do Gradiente:** **Maximizar** log-verossimilhança $ll(w) \implies$ **Gradiente Ascendente ($+$)**; **Minimizar** função de perda $\mathcal{L}(w) \implies$ **Gradiente Descendente ($-$)**!
