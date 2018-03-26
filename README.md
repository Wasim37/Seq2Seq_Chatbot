首先需要获取大量的对话数据，数据可以使用网上公开的数据集，也可以自己爬取。
当然你可以抽取电影里的对话字幕，途径很多。
本项目中的dgk_shooter_min.conv文件是事先按某种格式处理好的问答数据集。
得到预处理数据集后，还需根据每个问答句子的长度，将数据划分为不同的bucket。
比如bucket_5_15.db这个文件存储的问答内容，question的长度小于等于5，answer的长度小于等于15。

- 特征工程
- 模型训练、调优
- 模型诊断

- decode_conv.py
读取预处理好的问答数据集dgk_shooter_min.conv文件，存放到./bd目录下的conversation.db文件中

- data_utils
读取conversation.db文件，按问答的不同长度，划分为不同的bucket。
最终在./bucket_dbs目录下会生成bucket_5_15.db、bucket_10_20.db、bucket_15_25.db和bucket_20_30.db四个文件。

- s2s.py
读取./bucket_dbs下的db数据进行训练与测试

### 文章收录
- [tensor flow dynamic_rnn 与rnn有啥区别？](https://www.zhihu.com/question/52200883/answer/136317118)