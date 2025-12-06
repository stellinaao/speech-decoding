
modelName = 'ae_gru'

args = {}
args['outputDir'] = '../logs/speech_logs/' + modelName
args['datasetPath'] = '../competitionData/ptDecoder_ctc'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 128
args['lrStart'] = 0.05
args['lrEnd'] = 0.02
args['nUnits'] = 256
args['nBatch'] = 10000 #3000
args['nLayers'] = 5 
args['seed'] = 0
args['nClasses'] = 40
args['nInputFeatures'] = 173 # 192 # 142 # 256
args['dropout'] = 0.2
args['whiteNoiseSD'] = 0.8
args['constantOffsetSD'] = 0.2
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = False
args['time_mask_p'] = 0.2
args['n_time_masks'] = 2
args['channel_mask_p'] = 0.075
args['n_channel_masks'] = 5
args['l2_decay'] = 1e-5
args['use_layernorm'] = True
args['rnn'] = 'GRU'

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)