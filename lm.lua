require 'nn'

-- 1. ****** --

cmd = torch.CmdLine()
cmd:text()
cmd:text('Language Modeling ')
cmd:text('Example:')
cmd:text('$> th lm.lua --train train.txt')
cmd:text('Options:')
cmd:option('--train', 'train.txt', 'training file')
cmd:option('--vocab', 'vocab.txt', 'vocabulary file')
cmd:text()

opt = cmd:parse(arg or {})
print(opt)

-----------------------------------------------------------
vocab = {}
vocab_byIndex = {}

function load_vocab(vocab_file)
    print("vocab_file : " .. vocab_file)
    f_vocab = io.open(vocab_file, "r")

    idx_voc = 1
    for line in io.lines(vocab_file) do
        word = line
        vocab[word] = idx_voc
        table.insert(vocab_byIndex, word)
        idx_voc = idx_voc + 1
    end

    f_vocab:close()
end
-----------------------------------------------------------
train_sents = {}

function load_data(train_file)
    print("train_file : " .. train_file)    
    f_train = io.open(train_file, "r")

    for line in io.lines(train_file) do
        print(line)
        local sent = {}
        for word in string.gmatch(line, "[^%s]+") do
            print(word)
            --table.insert(sent, word)
            table.insert(sent, vocab[word])
        end
        table.insert(train_sents, sent)
    end

    f_train:close()
end	
------------------------------------------------------------

load_vocab(opt.vocab)
load_data(opt.train)

print(train_sents)
print(vocab)
print(vocab_byIndex)

trainData = torch.IntTensor(train_sents)
trainData = trainData:transpose(2,1)
print(trainData)


