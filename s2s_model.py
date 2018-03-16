#encoding=utf8
import pdb
import random
import copy

import numpy as np
import tensorflow as tf

import data_utils

class S2SModel(object):
    
    def __init__(self,
                source_vocab_size, # dictionary.json 字典总长度6865
                target_vocab_size,
                buckets, # [(5, 15), (10, 20), (15, 25), (20, 30)]
                size, # 512 LSTM每层神经元数量
                dropout, # 1.0 每层输出DROPOUT的大小
                num_layers, # 2 LSTM的层数
                max_gradient_norm, # 5.0 梯度最大阈值
                batch_size,  # 64 批量梯度下降的批量大小
                learning_rate,  # 0.0003 学习率
                num_samples, # 512 分批softmax的样本量
                forward_only=False,
                dtype=tf.float32):
        
        # init member variales
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.buckets = buckets
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # LSTM cells
        cell = tf.contrib.rnn.BasicLSTMCell(size)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)
        cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers)
        
        output_projection = None # 权重与偏置
        softmax_loss_function = None

        # 如果vocabulary太大，我们还是按照vocabulary来sample的话，内存会爆
        if num_samples > 0 and num_samples < self.target_vocab_size:
            print('开启投影：{}'.format(num_samples))
            w_t = tf.get_variable(
                "proj_w",
                [self.target_vocab_size, size],
                dtype=dtype
            )
            w = tf.transpose(w_t) # 转置
            b = tf.get_variable(
                "proj_b",
                [self.target_vocab_size],
                dtype=dtype
            )
            output_projection = (w, b)

            def sampled_loss(labels, logits):
                labels = tf.reshape(labels, [-1, 1])
                # 因为选项有选fp16的训练，这里统一转换为fp32
                # tf.cast 即数据格式转
                local_w_t = tf.cast(w_t, tf.float32)
                local_b = tf.cast(b, tf.float32)
                local_inputs = tf.cast(logits, tf.float32)
                return tf.cast(
                    tf.nn.sampled_softmax_loss(
                        weights=local_w_t,
                        biases=local_b,
                        labels=labels,
                        inputs=local_inputs,
                        num_sampled=num_samples,
                        num_classes=self.target_vocab_size
                    ),
                    dtype
                )
            softmax_loss_function = sampled_loss

        # seq2seq_f
        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            # Encoder.先将cell进行deepcopy，因为seq2seq模型是两个相同的模型，但是模型参数不共享，所以encoder和decoder要使用两个不同的RnnCell
            tmp_cell = copy.deepcopy(cell)
            
                #cell:                RNNCell常见的一些RNNCell定义都可以用.
                #num_encoder_symbols: source的vocab_size大小，用于embedding矩阵定义
                #num_decoder_symbols: target的vocab_size大小，用于embedding矩阵定义
                #embedding_size:      embedding向量的维度
                #num_heads:           Attention头的个数，就是使用多少种attention的加权方式，用更多的参数来求出几种attention向量
                #output_projection:   输出的映射层，因为decoder输出的维度是output_size，所以想要得到num_decoder_symbols对应的词还需要增加一个映射层，参数是W和B，W:[output_size, num_decoder_symbols],b:[num_decoder_symbols]
                #feed_previous:       是否将上一时刻输出作为下一时刻输入，一般测试的时候置为True，此时decoder_inputs除了第一个元素之外其他元素都不会使用。
                #initial_state_attention: 默认为False, 初始的attention是零；若为True，将从initial state和attention states开始。
            
            #tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                #encoder_inputs, 
                #decoder_inputs, 
                #cell, 
                #num_encoder_symbols, 
                #num_decoder_symbols, 
                #embedding_size, 
                #num_heads=1, 
                #output_projection=None, 
                #feed_previous=False, 
                #dtype=None, 
                #scope=None, 
                #initial_state_attention=False)    
            return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                encoder_inputs,# tensor of input seq
                decoder_inputs,# tensor of decoder seq
                tmp_cell,#自定义的cell,可以是GRU/LSTM, 设置multilayer等
                num_encoder_symbols=source_vocab_size,# 词典大小 40000
                num_decoder_symbols=target_vocab_size,# 目标词典大小 40000
                embedding_size=size,# embedding 维度
                output_projection=output_projection,# 不设定的话输出维数可能很大(取决于词表大小)，设定的话投影到一个低维向量
                feed_previous=do_decode,# 
                dtype=dtype
            )

        # inputs
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.decoder_weights = []
        # encoder_inputs 这个列表对象中的每一个元素表示一个占位符，其名字分别为encoder0, encoder1,…,encoder39，encoder{i}的几何意义是编码器在时刻i的输入。
        # buckets中的最后一个是最大的（即第“-1”个）
        for i in range(buckets[-1][0]):
            self.encoder_inputs.append(tf.placeholder(
                tf.int32,
                shape=[None],
                name='encoder_input_{}'.format(i)
            ))
        # 输出比输入大 1，这是为了保证下面的targets可以向左shift 1位
        for i in range(buckets[-1][1] + 1):
            self.decoder_inputs.append(tf.placeholder(
                tf.int32,
                shape=[None],
                name='decoder_input_{}'.format(i)
            ))
            self.decoder_weights.append(tf.placeholder(
                dtype,
                shape=[None],
                name='decoder_weight_{}'.format(i)
            ))
            #target_weights 是一个与 decoder_outputs 大小一样的 0-1 矩阵。该矩阵将目标序列长度以外的其他位置填充为标量值 0。
                # Our targets are decoder inputs shifted by one.
        
        targets = [
            self.decoder_inputs[i + 1] for i in range(buckets[-1][1])
        ]
        
        if forward_only:# 测试阶段
            
            #跟language model类似，targets变量是decoder inputs平移一个单位的结果，
            #encoder_inputs: encoder的输入，一个tensor的列表。列表中每一项都是encoder时的一个词（batch）。
            #decoder_inputs: decoder的输入，同上
            #targets:        目标值，与decoder_input只相差一个<EOS>符号，int32型
            #weights:        目标序列长度值的mask标志，如果是padding则weight=0，否则weight=1
            #buckets:        就是定义的bucket值，是一个列表：[(5，10), (10，20),(20，30)...]
            #seq2seq:        定义好的seq2seq模型，可以使用后面介绍的embedding_attention_seq2seq，embedding_rnn_seq2seq，basic_rnn_seq2seq等
            #softmax_loss_function: 计算误差的函数，(labels, logits)，默认为sparse_softmax_cross_entropy_with_logits
            #per_example_loss: 如果为真，则调用sequence_loss_by_example，返回一个列表，其每个元素就是一个样本的loss值。如果为假，则调用sequence_loss函数，对一个batch的样本只返回一个求和的loss值，具体见后面的分析
            #name: Optional name for this operation, defaults to "model_with_buckets".            
            
            #def model_with_buckets(encoder_inputs,
                               #decoder_inputs,
                               #targets,
                               #weights,
                               #buckets,
                               #seq2seq,
                               #softmax_loss_function=None,
                               #per_example_loss=False,
                               #name=None):            
            
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.encoder_inputs,
                self.decoder_inputs,
                targets,
                self.decoder_weights,
                buckets,
                lambda x, y: seq2seq_f(x, y, True),
                softmax_loss_function=softmax_loss_function
            )
            if output_projection is not None:
                for b in range(len(buckets)):
                    self.outputs[b] = [
                        tf.matmul(
                            output,
                            output_projection[0]
                        ) + output_projection[1]
                        for output in self.outputs[b]
                    ]
        else:#训练阶段
            
            #将输入长度分成不同的间隔，这样数据的在填充时只需要填充到相应的bucket长度即可，不需要都填充到最大长度。
            #比如buckets取[(5，10), (10，20),(20，30)...]（每个bucket的第一个数字表示source填充的长度，
            #第二个数字表示target填充的长度，eg：‘我爱你’-->‘I love you’，应该会被分配到第一个bucket中，
            #然后‘我爱你’会被pad成长度为5的序列，‘I love you’会被pad成长度为10的序列。其实就是每个bucket表示一个模型的参数配置），
            #这样对每个bucket都构造一个模型，然后训练时取相应长度的序列进行，而这些模型将会共享参数。
            #其实这一部分可以参考现在的dynamic_rnn来进行理解，dynamic_rnn是对每个batch的数据将其pad至本batch中长度最大的样本，
            #而bucket则是在数据预处理环节先对数据长度进行聚类操作。明白了其原理之后我们再看一下该函数的参数和内部实现：
            # 
            #encoder_inputs: encoder的输入，一个tensor的列表。列表中每一项都是encoder时的一个词（batch）。
            #decoder_inputs: decoder的输入，同上
            #targets:        目标值，与decoder_input只相差一个<EOS>符号，int32型
            #weights:        目标序列长度值的mask标志，如果是padding则weight=0，否则weight=1
            #buckets:        就是定义的bucket值，是一个列表：[(5，10), (10，20),(20，30)...]
            #seq2seq:        定义好的seq2seq模型，可以使用后面介绍的embedding_attention_seq2seq，embedding_rnn_seq2seq，basic_rnn_seq2seq等
            #softmax_loss_function: 计算误差的函数，(labels, logits)，默认为sparse_softmax_cross_entropy_with_logits
            #per_example_loss: 如果为真，则调用sequence_loss_by_example，返回一个列表，其每个元素就是一个样本的loss值。如果为假，则调用sequence_loss函数，对一个batch的样本只返回一个求和的loss值，具体见后面的分析
            #name: Optional name for this operation, defaults to "model_with_buckets".   
            
            #tf.contrib.legacy_seq2seq.model_with_buckets(encoder_inputs, 
                                                        #decoder_inputs, 
                                                        #targets, 
                                                        #weights, 
                                                        #buckets, 
                                                        #seq2seq, 
                                                        #softmax_loss_function=None, 
                                                        #per_example_loss=False, 
                                                        #name=None)
            
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.encoder_inputs,
                self.decoder_inputs,
                targets,
                self.decoder_weights,
                buckets,
                lambda x, y: seq2seq_f(x, y, False),
                softmax_loss_function=softmax_loss_function
            )

        # tf.trainable_variables 返回的是需要训练的变量列表
        # tf.all_variables 返回的是所有变量的列表
        # 是否是训练变量，通过trainable定义，例如 tf.Variable(tf.constant(5, shape=[1], dtype=tf.float32), name='global_step', trainable=False)  
        params = tf.trainable_variables()
        opt = tf.train.AdamOptimizer(
            learning_rate=learning_rate
        )

        if not forward_only:# 只有训练阶段才需要计算梯度和参数更新
            self.gradient_norms = [] # 梯度归一化
            self.updates = [] # 用来更新参数
            # zip可以同时迭代多个对象 http://www.runoob.com/python/python-func-zip.html
            # 如果各个迭代器的元素个数不一致，则返回的列表长度与最短的对象相同
            for output, loss in zip(self.outputs, self.losses):# 用梯度下降法优化
                gradients = tf.gradients(loss, params)
                clipped_gradients, norm = tf.clip_by_global_norm(
                    gradients,
                    max_gradient_norm
                )
                self.gradient_norms.append(norm)
                self.updates.append(opt.apply_gradients(
                    zip(clipped_gradients, params)
                ))
        
        # self.saver = tf.train.Saver(tf.all_variables())
        self.saver = tf.train.Saver(
            tf.all_variables(),
            write_version=tf.train.SaverDef.V2
        )

    def step(
        self,
        session,
        encoder_inputs,
        decoder_inputs,
        decoder_weights,
        bucket_id,
        forward_only
    ):
        encoder_size, decoder_size = self.buckets[bucket_id]
        if len(encoder_inputs) != encoder_size:
            raise ValueError(
                "Encoder length must be equal to the one in bucket,"
                " %d != %d." % (len(encoder_inputs), encoder_size)
            )
        if len(decoder_inputs) != decoder_size:
            raise ValueError(
                "Decoder length must be equal to the one in bucket,"
                " %d != %d." % (len(decoder_inputs), decoder_size)
            )
        if len(decoder_weights) != decoder_size:
            raise ValueError(
                "Weights length must be equal to the one in bucket,"
                " %d != %d." % (len(decoder_weights), decoder_size)
            )

        # 输入
        input_feed = {}
        for i in range(encoder_size):
            input_feed[self.encoder_inputs[i].name] = encoder_inputs[i]
        for i in range(decoder_size):
            input_feed[self.decoder_inputs[i].name] = decoder_inputs[i]
            input_feed[self.decoder_weights[i].name] = decoder_weights[i]

        # 理论上decoder inputs和decoder target都是n位
        # 但是实际上decoder inputs分配了n+1位空间
        # 不过inputs是第[0, n)，而target是[1, n+1)，刚好错开一位
        # 最后这一位是没东西的，所以要补齐最后一位，填充0
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        # 输出
        if not forward_only:
            output_feed = [
                self.updates[bucket_id], # 更新参数
                self.gradient_norms[bucket_id], # 梯度归一化
                self.losses[bucket_id] # 损失
            ]
            output_feed.append(self.outputs[bucket_id][i])
        else:
            output_feed = [self.losses[bucket_id]]
            for i in range(decoder_size):
                output_feed.append(self.outputs[bucket_id][i])
        
        # 运行
        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], outputs[2], outputs[3:]
        else:
            return None, outputs[0], outputs[1:]

    def get_batch_data(self, bucket_dbs, bucket_id):
        ''' 拿出batch_size个问答对 data, data_in 问答倒转
        '''
        data = []
        data_in = []
        bucket_db = bucket_dbs[bucket_id]
        for _ in range(self.batch_size):
            ask, answer = bucket_db.random()
            data.append((ask, answer))
            data_in.append((answer, ask))
        return data, data_in

    def get_batch(self, bucket_dbs, bucket_id, data):
        '''将data转换为模型训练可接受的格式
        '''
        encoder_size, decoder_size = self.buckets[bucket_id]
        # bucket_db = bucket_dbs[bucket_id]
        encoder_inputs, decoder_inputs = [], []
        for encoder_input, decoder_input in data:
            # encoder_input, decoder_input = random.choice(data[bucket_id])
            # encoder_input, decoder_input = bucket_db.random()
            #把输入句子转化为字典里的id，即在字典里的位置
            encoder_input = data_utils.sentence_indice(encoder_input)
            decoder_input = data_utils.sentence_indice(decoder_input)
            
            # Encoder 
            # 句子填充为固定长度
            encoder_pad = [data_utils.PAD_ID] * (
                encoder_size - len(encoder_input)
            )
            # 输入反转，将填充的pad放到句子前面，降低填充内容的权重，增强句子向量的语义信息，增强模型的准确率。
            # 下面的解码不需要反转。要提高准确率可以采用attention机制
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))
            
            # Decoder
            # 为什么要减2？ 因为解码有个开始符和结束符，对应本项目声明的 GO_ID 和 EOS_ID
            decoder_pad_size = decoder_size - len(decoder_input) - 2
            decoder_inputs.append(
                [data_utils.GO_ID] + decoder_input +
                [data_utils.EOS_ID] +
                [data_utils.PAD_ID] * decoder_pad_size
            )
        
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []
        # batch encoder
        for i in range(encoder_size):
            batch_encoder_inputs.append(np.array(
                [encoder_inputs[j][i] for j in range(self.batch_size)],
                dtype=np.int32
            ))
        # batch decoder
        for i in range(decoder_size):
            batch_decoder_inputs.append(np.array(
                [decoder_inputs[j][i] for j in range(self.batch_size)],
                dtype=np.int32
            ))
            # batch_weights 20行(decoder_size) 64列(batch_size)            
            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            for j in range(self.batch_size):
                if i < decoder_size - 1:
                    target = decoder_inputs[j][i + 1]
                if i == decoder_size - 1 or target == data_utils.PAD_ID:
                    batch_weight[j] = 0.0
            batch_weights.append(batch_weight)
            # 10*64 20*64 20*64
        return batch_encoder_inputs, batch_decoder_inputs, batch_weights
