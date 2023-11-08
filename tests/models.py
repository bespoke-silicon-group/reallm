from structs.Model import Model

gpt2 = Model(name = 'gpt2',
             num_layers = 48,
             d = 1600,
             num_heads = 25,
             )

megatron = Model(name = 'megatron',
                 num_layers = 72,
                 d = 3072,
                 num_heads = 32,
                 )

gpt3 = Model(name = 'gpt3',
             num_layers = 96,
             d = 12288,
             num_heads = 96,
             )

gopher = Model(name = 'gopher',
               num_layers = 80,
               d = 16384,
               num_heads = 128,
               )

mtnlg = Model(name = 'mtnlg',
              num_layers = 105,
              d = 20480,
              num_heads = 128,
              )

bloom = Model(name = 'bloom',
              num_layers = 70,
              d = 14336,
              num_heads = 112,
              )

palm = Model(name = 'palm',
             num_layers = 118,
             d = 18432,
             num_heads = 48,
             heads_per_kv_cache=48
             )

llama2 = Model(name = 'llama2',
               num_layers = 80,
               d = 8192,
               num_heads = 64,
               heads_per_kv_cache=8
               )
