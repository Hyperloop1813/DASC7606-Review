# DASC7606 Final - Prompting - summarized by hyperloop

# Section 0 Exam Scope

![](./Scope.png)

---

# Section 1 CROFTC & COSTAR

![1764945580793](image/Revision-Prompting/1764945580793.png)
## Prompting：什么是 Prompt（提示词）？

- **定义**：Prompt 是用户提供给模型的输入，用来**描述任务目标**。
- **组成**：通常包含 **指令 / 问题 / 示例** 等内容。
- **作用**：通过 Prompt **引导模型完成特定任务**（如写作、问答、生成图片等）。

## 重要结论：Prompting 的特点
- **偏经验主义**：效果像做实验一样，需要不断尝试。
- **模型差异大**：同一 Prompt 在不同模型上表现可能不同。
- **需要试错与技巧**：常靠大量实验 + 启发式规则来优化 Prompt。

![1764945585134](image/Revision-Prompting/1764945585134.png)
![1764945589290](image/Revision-Prompting/1764945589290.png)

---

## Effective Prompts：CO-STAR 框架（写出高质量提示词）

CO-STAR 用来把 Prompt 写得**清晰、可控、可复用**：

- **C — Context（背景）**：提供任务相关背景信息/产品信息/约束环境  
- **O — Objective（目标）**：明确要模型完成的任务（产出什么、达成什么效果）  
- **S — Style（写作风格）**：指定参考风格/表达方式（如“像某品牌文案”）  
- **T — Tone（语气）**：指定态度与情绪（如说服、正式、幽默）  
- **A — Audience（受众）**：说明写给谁看（年龄层、偏好、关注点）  
- **R — Response（输出格式）**：规定输出形式（长度、结构、是否列表、是否含标题等）

## CROFTC vs CO-STAR：两个常用提示词结构对比

### CROFTC（更偏“角色 + 约束”）
- **C — Context（背景）**
- **R — Role（角色）**：给 AI 设定身份（例：你是营销文案写手）
- **O — Objective（目标）**
- **F — Form（形式）**：规定输出的结构/排版（例：编号列表）
- **T — Tone（语气）**
- **C — Conditions（条件/边界）**：明确限制（例：少于10词、避免术语）

### CO-STAR（更偏“面向受众 + 输出”）
- **C — Context（背景）**
- **O — Objective（目标）**
- **S — Style（风格）**
- **T — Tone（语气）**
- **A — Audience（受众）**
- **R — Response（输出格式）**

### 对应关系（便于记忆）
- **Role ≈ Style**：角色设定常会影响写作风格  
- **Form ≈ Response**：都是在规定输出结构/格式  
- **Conditions**：在 CO-STAR 中可作为补充约束加入（更“硬”的边界条件）  
- **Audience**：是 CO-STAR 的显式组件（强调“写给谁”）

---
![1764945772611](image/Revision-Prompting/1764945772611.png)
![1764945712656](image/Revision-Prompting/1764945712656.png)
## Prompting：写出更有效 Prompt 的进一步技巧

- **像和人对话一样**：模型可能误解你的意图，需要你补充说明  
  - 建议用**交互式、多轮**方式：逐步追问、分步 уточ化需求
- **先“搭舞台”再提问**：先定义**角色/背景/任务**（可用 CO-STAR）  
  - 例：说明你是谁、现状是什么、目标是什么（如“新手跑者，6个月内想完赛马拉松，如何准备？”）
- **让 AI 扮演特定身份/职业**：例如老师、市场经理、记者、家长等  
  - 也可用更具象的创作设定来引导风格（提升可控性与一致性）
- **让 AI 帮你“写 Prompt”**：直接要求它生成/优化提示词（Prompt creation）
- **让 ChatGPT 保持在轨，减少幻觉**：用追问逼它给依据  
  - 如：**“你为什么这么认为？”**、**“你的证据是什么？”**
- **Negative Prompting（反向约束）**：明确“不需要什么/不要输出什么”  
  - 如：不要冒犯/不要有害/不要事实错误等（用于减少不想要的内容）
- **大胆试错与重述**：同一问题不同写法可能得到不同输出；不理想就换表述再问


## Prompt Style Matter?（提示词语气/风格有用吗？）

- 实验任务：要求输出“1000 个宝宝名字”的编号列表  
- 对比多种语气：
  - **中性**（强制完成、不要缩短）
  - **正向礼貌**（请求、感谢、夸赞）
  - **负向威胁**（恐吓式措辞）
- 结论：
  - **正向礼貌**不一定比中性提示更能提升结果
  - **负向威胁**会显著让输出更差
  - 核心点：**“威胁 ChatGPT 不起作用”**（Threatening ChatGPT Doesn’t Work）

---

# Section 2 In-Context Learning

![1764945894306](image/Revision-Prompting/1764945894306.png)
![1764945905792](image/Revision-Prompting/1764945905792.png)

### 1) ICL 是什么？
- **ICL = few-shot prompting**：把**少量示例（demonstrations）**直接写进 prompt 里（自然语言格式），模型在**不更新参数**的情况下，依据这些例子完成新任务。
- 本质：利用“当前上下文”来**引导模型行为**（prompt formulation 阶段完成适配）。

### 2) ICL vs Fine-tuning（微调）
- **方法论**：
  - ICL：**不改模型参数**，不需要在特定数据集上训练。
  - Fine-tuning：需要用数据训练，**更新参数**。
- **灵活性**：
  - ICL：**即用即走**，换任务只要改 prompt/示例即可。
  - Fine-tuning：换任务通常要重新训练或再微调。
- **资源需求**：
  - ICL：主要成本在推理时的上下文长度与计算。
  - Fine-tuning：需要**额外算力 + 标注数据**来训练。

### 3) 推理时 LLM 如何做到 ICL？（仍在研究中）
- 观点：ICL 可能是**预训练中涌现**出的能力。
- 一些解释方向：
  - 模型在前向计算中对示例产生类似“**meta-gradients**”的效应，
    借助注意力机制**隐式地做类似梯度下降/优化**的行为。
  - ICL 通过 prompt 去“**定位/唤起**”预训练中学到的相关概念与模式，从而完成任务。

### 4) 图示要点（情感分类例子）
- 在 prompt 中给出 **k 个示例**（Review → Sentiment 标签），再给一个新句子让模型补全标签。
- 模型参数 **冻结（Parameter Freeze）**，仅靠上下文里示例的模式完成预测（例如输出 Positive/Negative）。

---

# Section 3 Chain of Thought/Draft
![1764946093728](image/Revision-Prompting/1764946093728.png)
## Chain of Thought（CoT，思维链提示）

- **概念**：让 LLM 先生成一串“中间推理步骤”，再给最终答案（常用于数学/逻辑等复杂推理）。
- **为什么有用**：把“怎么想”的过程显式化，通常能显著提升复杂推理的正确率。
- **来源/现象**：
  - LLM（尤其在大量代码/推理数据上训练后）更擅长多步推理与 CoT 提示。
  - CoT 能力在大模型中常表现为一种“自然涌现”的能力。
- **对比（图示）**：
  - **标准提示**：直接要答案 → 容易算错。
  - **CoT 提示**：给一个带推理步骤的示例（**One-shot CoT**）→ 模型更可能沿着同样的推理格式解后续题目、正确率更高。

---

![1764946098502](image/Revision-Prompting/1764946098502.png)
## Self-consistency CoT（自一致性思维链）

- **动机**：复杂问题往往存在多条合理推理路径；单次 CoT 可能走偏。
- **做法**：对同一问题进行**多路径解码**（生成多条不同 CoT/答案），得到一组候选最终答案。
- **决策**：对候选答案做**多数投票（majority voting）**，选出“最一致”的答案作为输出。
- **优点**：通常比单条 CoT 更稳、更准（尤其在多步推理任务上）。
- **局限**：需要更多计算成本（常见要采样 **5–10** 条路径）。


---

![1764946200225](image/Revision-Prompting/1764946200225.png)
## Zero-shot CoT（零样本思维链）：“Let’s think step by step.”

- **核心思想**：不给示例（zero-shot），只在提示词里加一句**引导推理**的话（如“让我们一步一步想”），也能显著提升模型的多步推理表现。
- **图中例子**：
  - 直接问（Zero-shot）：模型可能输出错误答案（示例中给了 8 ✗）。
  - 加一句 “Let’s think step by step.”（Zero-shot CoT）：模型会展开中间推理并得到正确答案（16个球 → 8个高尔夫球 → 4个蓝色高尔夫球 ✓）。

### Zero-shot CoT 的两步结构
1. **原问题输入 X**：照常给出任务/问题（Q: [X]）
2. **推理触发句 T**：追加一句引导推理的话（如 “Let’s think step by step.”）
→ 模型更倾向于先写推理过程，再给最终答案 Z。

### 其他常见的推理触发句（同类 T）
- “Let’s think about this logically.”
- “Let’s solve this problem by splitting it into steps.”
- “Let’s think like a detective step by step before we dive into the answer.”

---

![1764946251657](image/Revision-Prompting/1764946251657.png)
![1764946255643](image/Revision-Prompting/1764946255643.png)
## Prompting – Chain of Draft（CoD，草稿链）

- **核心思想**：不像 CoT 要写很长推理过程，CoD 让模型**“写更少也能想清楚”**：每步只保留**极简草稿**（例如每步 ≤5 个词），最后再给答案（常用分隔符 `####`）。
- **动机**：推理能力≠长输出。通过减少中间文字，做到**更少 token、更低延迟**，同时保持较高准确率。


## Standard vs CoT vs CoD（图与例子想表达什么）

### 1) 例子对比（棒棒糖）
- **Standard**：直接给答案（可能不稳）
- **CoT**：写一大段逐步推理（更稳但很长）
- **CoD**：只写关键式子/最小草稿，例如 `20 - x = 12 → x = 8`，然后 `#### 8`

### 2) 效果趋势（准确率 & Token）
- **准确率**：CoT、CoD 通常明显高于 Standard；CoD 往往接近 CoT。
- **Token/成本**：CoT 的 token 数暴涨；CoD token 明显更少（接近“短推理”），因此**更快**。


## 实验设置（提示词模板）
- **Standard**：直接回答，不要解释/推理
- **CoT**：逐步思考，最后在 `####` 后给答案
- **CoD**：逐步思考，但**每步只保留最少草稿**（如每步 ≤5 词），最后在 `####` 后给答案


## Few-shot vs Zero-shot 的观察
- **Few-shot（给少量示例）**：CoD 能在 **高准确率**下显著降低 token 和延迟（相对 CoT）。
- **Zero-shot（不给示例）**：CoD 效果可能下降（尤其某些模型更明显）。
  - 文中假设原因：训练数据里 **CoD 风格推理模式更稀缺**，缺少示例时更难学会“又短又准”的草稿写法。

## 一句话总结
- **CoD 证明：LLM 的有效推理不一定需要冗长输出；用“最小草稿”也能保持推理质量，同时显著节省 token 与时间。**

---
# Section 4 P-tuning
![1764946363969](image/Revision-Prompting/1764946363969.png)
## Prompt-tuning / P-tuning 是什么？

- **Prompt tuning ≠ prompt engineering**
  - Prompt engineering：手工写/改提示词（离散 token）。
  - Prompt-tuning / P-tuning：一种**参数高效微调（PEFT）**方法，学习一小段可训练的“软提示”（soft prompt）。

## 基本流程（图示含义）
- 输入的自然语言 prompt 先经过一个**小的可训练模块（P-tuning model）**
- 该模块生成**任务相关的虚拟 token / 软提示向量（virtual tokens）**
- 然后把这些虚拟 token **拼接到原 prompt 的其他部分**一起送入 **LLM**
- 训练时通常**冻结 LLM 主体参数**，只训练这小段 soft prompt（动机之一：**token budget / 上下文长度受限**）

## 优点（Advantages）
- **参数效率高**：只训练很小一部分 soft prompts，比全量 fine-tuning 省参数/省资源。
- **支持多任务切换**：推理时更换不同任务的 soft prompt，就能适配不同任务。
- **推理效率潜力**：训练好后，P-tuning 模块可用**查表（lookup table）**替代，进一步加速推理。

---

# Section 5 Ultimate ChatGPT Prompt
![1764946400551](image/Revision-Prompting/1764946400551.png)
## The Ultimate ChatGPT Prompt（“终极提示词”）核心内容

### 1) 终极 Prompt 的特征（Characteristics）
1. **少于 25 个词**（短）
2. **容易记住**（不必每次复制粘贴）
3. **比常规提示更有效**
4. **为推理型模型优化**（新一代 reasoning models）
5. **尽量一次就达成目标**（one try every time）

### 2) 给出的“定义性 Prompt”（definitive prompt）
> **请输出我请求的每一个维度的概览；找出不确定点；然后尽可能多地问我澄清问题。**

（英文原意：Output an overview of every single dimension of my request. Find points of uncertainty. Then, ask me as many clarifying questions as possible.)

### 3) 使用这个 Prompt 的好处（Benefits）
1. **提升我们对需求的理解**（把问题拆全）
2. **迫使我们补齐 AI 所需背景信息**
3. **固定上下文**：让 AI 更清楚“可能的输出长什么样”
4. **更可能得到全面、覆盖面更广的回答**

---

# Section 6 Retrieval Augmented Generation
![1764946567608](image/Revision-Prompting/1764946567608.png)
## Retrieval Augmented Generation（RAG，检索增强生成）要点

### 1) 为什么需要 RAG（LLM 的常见弱点）
- **幻觉**：模型可能“以为自己知道”，但其实不知道/会编造。
- **知识过期**：仅依赖训练数据，信息可能不更新。
- **来源不可靠**：可能生成看似合理但并非真实的“假数据/假引用”。

### 2) RAG 的核心思想（R + A + G）
- **Retrieval（检索）**：从外部数据源/知识库中**实时找相关信息**。
- **Augmented（增强）**：把检索到的**最新/领域特定上下文**喂给模型。
- **Generation（生成）**：LLM 基于“问题 + 检索片段”生成更可靠答案。
> 目标：用外部事实支撑，提升回答的**准确性与可信度**。

---

![1764946570870](image/Revision-Prompting/1764946570870.png)

## Basic RAG Pipeline（基础流程）

### A. Ingestion（入库/构建索引）
1. 收集 **Documents（文档）**
2. 切分为 **Chunks（小段文本）**
3. 为每个 chunk 计算 **Embeddings（向量表示）**
4. 存入 **Index（向量索引/数据库）**

### B. Retrieval（检索）
- 将用户 **Query** 也转成 embedding
- 在 Index 里做相似度搜索，取 **Top-K** 相关 chunks

### C. Synthesis（综合生成）
- 把 **Query + 检索到的 chunks** 作为 prompt/context 输入 LLM
- LLM 进行总结、归纳、引用并输出答案

## 评估关注点（Evaluation）
- **Query → Context 是否相关？**（检索到的上下文是否真正支撑问题）
- **Context → Answer 是否可支撑？**（答案是否基于检索内容而非编造）

---

# Section 7 Hallucination
![1764946699782](image/Revision-Prompting/1764946699782.png)
## 为什么 GPT 会“幻觉”（Hallucination）？

### 1) 根本原因
- **基础语言模型的目标**：生成“最可能/最顺的续写”，**不等于**生成“最真实的事实”。  
  → 所以即使不知道，也可能给出听起来合理的答案。

### 2) 具体成因
- **训练数据本身有问题**：可能包含错误、也可能缺失最近发生的事件（信息过时）。
- **缺少内置的“真假判别器”**：模型没有天然的 truth vs falsehood 感知；  
  当信息稀疏/模糊时，仍倾向于**继续回答/猜测**，甚至在“其实没有唯一真答案”时也给结论。

## 如何降低幻觉（常见缓解手段）
- **RAG（检索增强生成）**：从数据库/网页等外部知识源检索相关事实，把检索结果作为上下文再生成答案。
- **训练/微调更“事实敏感”**：让模型更贴合领域知识与真实行为，例如学会说“我不知道”。
- **提示词约束**：如要求“解释你的推理”“不确定就别猜”。
- **让模型自检**：让模型回看并审查自己的输出，常能发现错误并修正，提高最终准确性。
