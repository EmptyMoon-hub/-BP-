# -BP-
为了简单理解，采用matlab写的

$$
\begin{align}
E &= \frac{1}{2}e^Te=\frac{1}{2}(l-y)^T(l-y)\\
y &= \phi(W^T_2z+b)\\
z &= \phi(W^T_1x+b)\\
\\
\nabla_{W_2}E &= -z\cdot [\phi^{'}(W^T_2z+b_2)\circ e]^T\\
\nabla_{b_2}E &= -\phi^{'}(W^T_2z+b_2)\circ e\\
-\nabla_{z}E &= W_2\cdot [\phi^{'}(W^T_2z+b_2)\circ e]\\ \\
\nabla_{W_1}E &= -x\cdot [\phi^{'}(W^T_1x+b_1)\circ (-\nabla_{z}E)]^T\\
\nabla_{b_1}E &= -\phi^{'}(W^T_1x+b_1)\circ (-\nabla_{z}E)\\
-\nabla_{x}E &= W_1\cdot [\phi^{'}(W^T_1x+b_1)\circ  \nabla_{z}E]\\
\end{align}
$$

对于第$i$个隐层

$$
\begin{align}
\nabla_{b_i}E &= -\phi^{'}(W^T_ix_i+b_i)\circ (-\nabla_{\phi(W^T_ix_i+b_i)}E)\\
			  & = \phi^{'}(W^T_ix_i+b_i)\circ \nabla_{\phi(W^T_ix_i+b_i)}E\\
\nabla_{W_i}E &= -x_i\cdot [\phi^{'}(W^T_ix_i+b_i)\circ (-\nabla_{\phi(W^T_ix_i+b_i)}E)]^T\\
		      &= x_i\cdot \nabla_{b_i}E^T\\
-\nabla_{x_i}E &= W_i\cdot [\phi^{'}(W^T_ix_i+b_i)\circ \nabla_{\phi(W^T_ix_i+b_i)}E]\\
			   &=W_i\cdot \nabla_{b_i}E^T\\
\end{align}

$$
为了验证该推导，我使用Matlab编写了多层感知机程序，输入为两个服从不同正态分布的点集，标签为0、1，输入需要包含隐藏层的个数，学习率，迭代次数。
