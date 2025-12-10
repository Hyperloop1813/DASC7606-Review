DASC7606Final 2025 复习知识点自查

1.	Difference of loss and cost?

2.	Expression of MSE? Sigmoid? Softmax?

3.	Difference between Gradient Descent? Stochastic Gradient Decent? Mini-batch SGD?

4.	How to prevent overfitting?early stopping,dropout,weight regularzation

5.	Expression of L1,L2 regularzation?

6.	What is not random dropout?

7.	what is the time that we should apply early stopping?

8.	What is the process of convolution? What is the size of feature map after convolution?

9.	How to calculate computational cost?(times of multiplication)

10.	What is the process of pooling? How many parameters are there in the process of pooling?

11.	what is dense layer? How to calculate the number of parameters and computational cost?

12.	How can we visualizing the process of cnn?
(probility heat maps,gradient ascent,maximally activating neurons)


13.	What is occlusion? What is occlusion sensitivity? 
(The ability that masking an important part of the feature map doesnt impact the result of classify)

14.	What is probability heat maps/saliency maps?

15.	What can we observe by applying gradient ascent visualization?
(Input an random pic, use gradient ascent to maximize the activity of some neurons, we can get what feature the neurons is focusing on)

16.	what is maximally ativating patches? Difference between this and gradient ascent?
(pick a neuron to observe what kind of picture it activates the most, by input a series of image)

17.	First/shallow layer and last/deep layer extracts what kinds of features?
(some simple features like edges, lines, low receptive field/ specific and compleced features)

18.	What is an adversarial training sample?
(add distortion the human cannot find but cnn would be fooled to output fault output)

19.	What is FGSM(fast gradient sign method)?
(an easy way by add distortion in original input x by maximize the loss and let model to misclassify)

20.	How to prevent adversarial image attack?
(traning with adversarial images to improve robustness)

21.	What is DeepDream?the thought of deepdream?
(Input the picture to the network, augment some features and change the input by maximizing some activation values of specific neurons, maximize the activation value (neuron^2) by gradient ascent to change the input picture X(not the weight W)

22.	Waht is GAN?

23.	The working progress of GAN?

24.	GAN Loss function? How to determine whether update discriminator or generator?
（在输入样本时就决定更新G还是D，然后根据输出来更新（固定另一半）。通常训练多轮D后训练一次G。
eg:假如这次要更新G，生成一个fake，然后D完全没分辨出来（输出D（G（z）=1），那么也不更新D。

25.	What is the ideal result of GAN training?
	(discriminator cannot distinguish real and fake samples(D output=0.5)，
	 generator generates realistic and diversed picturess)

26.	What is the detailed training progress of GAN?

27.	What problems may happen in GAN training progress?
(mode collapse,non-convergence,generation of distorted figures)

28.	Structure of CBOW? Structure of skip-gram? advantages and disadvantages?


29.	5 kinds of approach of word embedding? 
	onehot,cbow,skipgram,elmo,bert

30.	What is elmo?
	bidirectional deep LSTM, can identify semantic difference of a same word occurs at diffrent place


31.	Difference of BERT and ELMO？

32.	onehot-CBOW/skipgram-ELMO-BERT的进化过程，每一步分别解决了什么问题？

33.	How Self-attention works？What is KQV？

34.	Size of each matrix of KQV？What is the ralationship of KQV size and multihead count？

35.	Major difference betwween the architecture BERT and gpt3?
	(Encoder only/Decoder only)

36.	Advantages of transformer architecture?
(parallel training,information transform do not rely on sequence, flexible embedding dims)

37.	Why we say trasnformer is parallel?

38.	What kind of training skill can we apply in Attention to achieve parallel?
	(mask)

39.	Two kinds of training progress of bert?
(1.mask words and predict
2.next sentance prediction)

40.	Popular application of BERT? 
	(embedding/classifying(直接分类任务)/identity classification(给句子中的词打词性标签),question and answer)

41.	How BERT do tokenize？advantage？
（bert分词根据整词－前缀－字母的顺序进行匹配，这样在vocabulary中没有的单词可一定程度上通过前缀/后缀试图理解部分语义）

42.	LLM training important factors?
 model scale,dataset quality,compute budget
43.	what is the ability of emergent?
模型涌现能力：同样的架构和设计，参数量越大效果越好

44.	After predict topk possible next words, how does LLM select next word?(greedy/beam search/probability distribution random)

45.	how to build a target model in specific realm based on basemodel?
(base model + designed dataset + fine-tuning)

46.	what is RLHF?what are the steps?


47.	LLM limitations?
hallucination,sensitive to input,excessively verbose,guess user intends but never deny, output harmful and biased content

48.	Popular prompt template?
(CROFTC/CO-STAR)

49.	What is In Context Learning?

50.	What is COT? How to achieve?

51.	What is prompt tuning/P-tuning?

52. why we use 1*1 conv on some conditions?
a)adjust the numbers of channel
b)combine and mixture the information of input feature maps
c)add nonlinear transformation
d)kind of substitution of full-connected layer

53.	Why sqrt(dk) in attention formula?

54.	How to predict next token when GPTs inferencing? What is KV cache?


