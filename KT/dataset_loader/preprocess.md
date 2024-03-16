### 一些规定

在我们的任务中，总是假定每次交互对应一个问题若干个知识点，而使用知识点代替习题编号的这种处理方法我们视为一种数据降维的手段

对于任意一个raw dataset
我们会生成以下文件：

skill_id_remapped.txt: [remmapped_id, original_id, skill_name]
question_id_remapped.txt: [remapped_id,original_id,quesiton_name]
重新映射后的sid/qid的范围为[1~sn] / [1~qn]

qs_matrix.npz: q矩阵，为了方便，维数设定为[qn+1,sn+1](因为之前预处理的下标从1开始的)
interaction.txt