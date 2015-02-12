_ = require 'underscore'

function load_vocab(vocab_file)
    print("vocab_file : " .. vocab_file)

    vocab ={} 
    vocab_ByIndex = {}
    vocab_size = 0

    f_vocab = io.open(vocab_file, "r")

    idx_voc = 1 
    for line in io.lines(vocab_file) do
        word = line
        vocab[word] = idx_voc
        table.insert(vocab_ByIndex, word)

        vocab_size = idx_voc
        idx_voc = idx_voc + 1
    end 

    f_vocab:close()

    return vocab, vocab_ByIndex, vocab_size
end

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

function load_datas(data_file, n, vocab)
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

function load_data(N)
--[[% This method loads the training, validation and test set.
% It also divides the training set into mini-batches.
% Inputs:
%   N: Mini-batch size.
% Outputs:
%   train_input: An array of size D X N X M, where
%                 D: number of input dimensions (in this case, 3).
%                 N: size of each mini-batch (in this case, 100).
%                 M: number of minibatches.
%   train_target: An array of size 1 X N X M.
%   valid_input: An array of size D X number of points in the validation set.
%   test: An array of size D X number of points in the test set.
%   vocab: Vocabulary containing index to word mapping.
]]--

vocab, vocab_ByIndex, vocab_size = load_vocab(opt.vocab)

trainData = load_datas(opt.train, opt.n, vocab)
validData = load_datas(opt.valid, opt.n, vocab)
testData = load_datas(opt.test, opt.n, vocab)

trainData = torch.IntTensor(trainData)
trainData = trainData:transpose(2,1)

validData = torch.IntTensor(validData)
validData = validData:transpose(2,1)

testData = torch.IntTensor(testData)
testData = testData:transpose(2,1)

--testData = data['data']['testData'];
--trainData = data['data']['trainData'];
--validData = data['data']['validData'];

numdims = (#trainData)[1];
D = numdims - 1;
M = math.floor((#trainData)[2] / N);
print((#trainData))
print(numdims, D, M)

train_input = torch.reshape(trainData[{ {1,D},{1,N*M} }],D,N,M);
train_target = torch.reshape(trainData[{ {D + 1},{1,N * M} }], 1, N, M);
valid_input = validData[{ {1,D},{} }];
valid_target = validData[{ {D + 1},{} }];
test_input = testData[{ {1,D},{} }];
test_target = testData[{ {D + 1}, {} }];


return train_input, train_target, valid_input, valid_target, test_input, test_target, vocab, vocab_size, vocab_ByIndex

end
