- decode_conv.py
读取dgk_shooter_min.conv文件生成./bd下的conversation.db

- data_utils
读取./bd下的conversation.db生成./bucket_dbs下的bucket_5_15.db、bucket_10_20.db、bucket_15_25.db、bucket_20_30.db

- s2s.py
读取./bucket_dbs下的db数据进行训练与测试

### 文章收录
- [tensor flow dynamic_rnn 与rnn有啥区别？](https://www.zhihu.com/question/52200883/answer/136317118)