#!/usr/bin/env th

require 'nn'
_ = require 'underscore'

-- 1. ****** --

cmd = torch.CmdLine()
cmd:text()
cmd:text('Language Modeling ')
cmd:text('Example:')
cmd:text('$> th lm.lua --train train.txt --valid valid.txt --test test.txt --vocab vocab.txt --n 3')
cmd:text('Options:')
cmd:option('--train', 'train.txt', 'training file')
cmd:option('--valid', 'valid.txt', 'validation file')
cmd:option('--test', 'test.txt', 'test file')
cmd:option('--vocab', 'vocab.txt', 'vocabulary file')
cmd:option('--n', 3, '"n" for n-gram')
cmd:text()

opt = cmd:parse(arg or {})
print(opt)

-----------------------------------------------------------

function load_vocab(vocab_file)
    print("vocab_file : " .. vocab_file)
    
    vocab ={}
    vocab_ByIndex = {}
    
    f_vocab = io.open(vocab_file, "r")

    idx_voc = 1
    for line in io.lines(vocab_file) do
        word = line
        vocab[word] = idx_voc
        table.insert(vocab_ByIndex, word)
        idx_voc = idx_voc + 1
    end

    f_vocab:close()
    
    return vocab, vocab_ByIndex
end
-----------------------------------------------------------
function ngrams(res_sents, n, sent)
    --print(#sent)
    for i, word in pairs(sent) do
        if i <= ((#sent) - (n-1)) then
            ngram = _.slice(sent, i, n)
            --print(ngram)
            table.insert(res_sents, ngram)
        end
    end
    --print()
end

function load_data(data_file, n, vocab)
    print("data_file : " .. data_file)    

    res_sents = {}
    
    f_data = io.open(data_file, "r")

    for line in io.lines(data_file) do
        --print(line)
        local sent = {}
        for word in string.gmatch(line, "[^%s]+") do
            --print(word)
            --table.insert(sent, word)
            table.insert(sent, vocab[word])
        end
        ngrams(res_sents, n, sent)
    end

    f_data:close()

    return res_sents
end	
------------------------------------------------------------

vocab, vocab_ByIndex = load_vocab(opt.vocab)
trainData = load_data(opt.train, opt.n, vocab)
validData = load_data(opt.valid, opt.n, vocab)
testData = load_data(opt.test, opt.n, vocab)

--print(trainData)
--print(validData)
--print(testData)
--print(vocab)
--print(vocab_ByIndex)

trainData = torch.IntTensor(trainData)
trainData = trainData:transpose(2,1)
print(trainData)

validData = torch.IntTensor(validData)
validData = validData:transpose(2,1)
print(validData)

testData = torch.IntTensor(testData)
testData = testData:transpose(2,1)
print(testData)


